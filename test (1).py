import os
import sys
import gc
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ClothSwapper")


class Config:
    SEGFORMER_MODEL       = "fashn-ai/fashn-human-parser"
    SD_INPAINT_MODEL      = "SG161222/Realistic_Vision_V5.0_noVAE"
    CONTROLNET_POSE_MODEL = "lllyasviel/control_v11p_sd15_openpose"
    CONTROLNET_DEPTH_MODEL= "lllyasviel/control_v11f1p_sd15_depth"
    MIDAS_MODEL           = "Intel/dpt-large"

    SD_WIDTH  = 512
    SD_HEIGHT = 512

    DEFAULT_PROMPT = (
    "a young woman wearing a classic sports bra and fitted athletic shorts, "
    "natural body proportions, soft studio lighting, "
    "photorealistic skin texture, detailed anatomy, 8k ultra high resolution, "
    "professional photography, sharp focus, intricate detail"
)
    NEGATIVE_PROMPT = (
        "deformed body, twisted torso, extra limbs, extra arms, extra legs, "
        "duplicate body parts, floating clothing, wrong anatomy, bad proportions, "
        "disfigured, mutated, melting body, double person, cloned person, "
        "cartoon, anime, painting, blurry, low quality, watermark, text, logo, "
        "artifacts, glitch, shape alteration, body reshaping, "
        "different body shape, slimmed body, altered figure"
    )

    GUIDANCE_SCALE        = 9.0
    NUM_STEPS             = 50
    STRENGTH              = 0.98
    CONTROLNET_POSE_SCALE = 0.65
    CONTROLNET_DEPTH_SCALE= 0.55
    SEED                  = 42

    MASK_DILATE_PX        = 15
    EDGE_FEATHER_PX       = 14
    CROP_PAD              = 30

    DEPTH_BLEND_STRENGTH  = 0.35
    SHAPE_REGIONS = {
        "chest", "breast", "bust", "upper-clothes", "torso",
        "back", "buttock", "hip", "lower-clothes"
    }

    UPPER_LABELS = {
        "top", "upper-clothes", "dress", "coat", "blouse",
        "shirt", "jacket", "sweater", "vest", "hoodie", "cardigan"
    }
    LOWER_LABELS = {
        "pants", "lower-clothes", "skirt", "trousers",
        "shorts", "jeans", "leggings"
    }


class DepthExtractor:
    def __init__(self, device: str):
        self.device = device
        self._available = False
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            self.processor = DPTImageProcessor.from_pretrained(Config.MIDAS_MODEL)
            self.model = DPTForDepthEstimation.from_pretrained(
                Config.MIDAS_MODEL
            ).to(device)
            self.model.eval()
            self._available = True
        except Exception as e:
            log.warning(f"Depth model unavailable ({e}) — shape preservation reduced")

    @property
    def available(self) -> bool:
        return self._available

    @torch.no_grad()
    def extract(self, pil_image: Image.Image) -> Optional[Image.Image]:
        if not self._available:
            return None
        try:
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth.squeeze()
            depth_np = depth.cpu().numpy()
            depth_resized = cv2.resize(
                depth_np,
                (pil_image.width, pil_image.height),
                interpolation=cv2.INTER_LINEAR
            )
            d_min, d_max = depth_resized.min(), depth_resized.max()
            if d_max > d_min:
                depth_norm = ((depth_resized - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_resized, dtype=np.uint8)
            depth_rgb = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(depth_rgb)
        except Exception as e:
            log.warning(f"  Depth extraction failed: {e}")
            return None

    def free(self):
        if self._available:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class FashionSegmenter:
    def __init__(self, device: str):
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(
            Config.SEGFORMER_MODEL, use_fast=False
        )
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            Config.SEGFORMER_MODEL
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def segment(self, pil_image: Image.Image) -> Tuple[np.ndarray, dict]:
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        upsampled = F.interpolate(
            outputs.logits,
            size=pil_image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        predicted_map = upsampled.argmax(dim=1)[0].cpu().numpy()
        return predicted_map, self.model.config.id2label

    def build_clothing_mask(self, pil_image: Image.Image) -> np.ndarray:
        predicted_map, id2label = self.segment(pil_image)
        H, W = predicted_map.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        for label_id, label_name in id2label.items():
            if label_id == 0:
                continue
            ll = label_name.lower()
            if not (any(u in ll for u in Config.UPPER_LABELS) or
                    any(l in ll for l in Config.LOWER_LABELS)):
                continue
            region = (predicted_map == label_id).astype(np.uint8) * 255
            if region.sum() < 500:
                continue
            mask = np.maximum(mask, region)
        k5  = np.ones((5, 5),  np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)
        dk = np.ones((Config.MASK_DILATE_PX * 2 + 1,
                      Config.MASK_DILATE_PX * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, dk, iterations=1)
        return mask

    def build_body_shape_mask(self, pil_image: Image.Image) -> np.ndarray:
        predicted_map, id2label = self.segment(pil_image)
        H, W = predicted_map.shape
        shape_mask = np.zeros((H, W), dtype=np.uint8)
        for label_id, label_name in id2label.items():
            ll = label_name.lower()
            if any(s in ll for s in Config.SHAPE_REGIONS):
                region = (predicted_map == label_id).astype(np.uint8) * 255
                if region.sum() > 200:
                    shape_mask = np.maximum(shape_mask, region)
        dk = np.ones((9, 9), np.uint8)
        shape_mask = cv2.dilate(shape_mask, dk, iterations=1)
        return shape_mask

    def free(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PoseExtractor:
    def __init__(self):
        try:
            from controlnet_aux import OpenposeDetector
            self._detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def extract(self, pil_image: Image.Image) -> Optional[Image.Image]:
        if not self._available:
            return None
        try:
            return self._detector(pil_image)
        except Exception as e:
            log.warning(f"  Pose extraction failed: {e}")
            return None


class SDInpainter:
    def __init__(self, device: str, use_pose: bool = True, use_depth: bool = True):
        from diffusers import (
            StableDiffusionInpaintPipeline,
            StableDiffusionControlNetInpaintPipeline,
            ControlNetModel,
        )
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.use_controlnet = False
        self.dual_controlnet = False

        if use_pose and use_depth:
            try:
                cn_pose = ControlNetModel.from_pretrained(
                    Config.CONTROLNET_POSE_MODEL, torch_dtype=dtype
                )
                cn_depth = ControlNetModel.from_pretrained(
                    Config.CONTROLNET_DEPTH_MODEL, torch_dtype=dtype
                )
                self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    Config.SD_INPAINT_MODEL,
                    controlnet=[cn_pose, cn_depth],
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to(device)
                self.use_controlnet = True
                self.dual_controlnet = True
            except Exception as e:
                log.warning(f"Dual ControlNet failed ({e}) — trying single pose ControlNet")
                use_depth = False

        if use_pose and not self.dual_controlnet:
            try:
                cn_pose = ControlNetModel.from_pretrained(
                    Config.CONTROLNET_POSE_MODEL, torch_dtype=dtype
                )
                self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    Config.SD_INPAINT_MODEL,
                    controlnet=cn_pose,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to(device)
                self.use_controlnet = True
                self.dual_controlnet = False
            except Exception as e:
                log.warning(f"ControlNet load failed ({e}) — using standard inpainting")

        if not self.use_controlnet:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                Config.SD_INPAINT_MODEL,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)

        self.pipe.enable_attention_slicing()
        if device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
        pose_image: Optional[Image.Image] = None,
        depth_image: Optional[Image.Image] = None,
    ) -> Image.Image:

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
            height=Config.SD_HEIGHT,
            width=Config.SD_WIDTH,
        )

        if self.use_controlnet:
            if self.dual_controlnet and pose_image is not None and depth_image is not None:
                pose_r  = pose_image.resize((Config.SD_WIDTH, Config.SD_HEIGHT), Image.LANCZOS)
                depth_r = depth_image.resize((Config.SD_WIDTH, Config.SD_HEIGHT), Image.LANCZOS)
                kwargs["control_image"] = [pose_r, depth_r]
                kwargs["controlnet_conditioning_scale"] = [
                    Config.CONTROLNET_POSE_SCALE,
                    Config.CONTROLNET_DEPTH_SCALE,
                ]
            elif pose_image is not None:
                pose_r = pose_image.resize((Config.SD_WIDTH, Config.SD_HEIGHT), Image.LANCZOS)
                kwargs["control_image"] = pose_r
                kwargs["controlnet_conditioning_scale"] = Config.CONTROLNET_POSE_SCALE
            elif depth_image is not None:
                depth_r = depth_image.resize((Config.SD_WIDTH, Config.SD_HEIGHT), Image.LANCZOS)
                kwargs["control_image"] = depth_r
                kwargs["controlnet_conditioning_scale"] = Config.CONTROLNET_DEPTH_SCALE

        return self.pipe(**kwargs).images[0]

    def free(self):
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CropInpaintEngine:
    def __init__(self, inpainter: SDInpainter):
        self.inpainter = inpainter

    @staticmethod
    def get_padded_bbox(
        mask: np.ndarray, pad: int, img_h: int, img_w: int
    ) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0, img_w, img_h
        x1 = max(0,     int(xs.min()) - pad)
        y1 = max(0,     int(ys.min()) - pad)
        x2 = min(img_w, int(xs.max()) + pad)
        y2 = min(img_h, int(ys.max()) + pad)
        return x1, y1, x2, y2

    def run(
        self,
        orig_pil: Image.Image,
        mask_np: np.ndarray,
        pose_pil: Optional[Image.Image],
        depth_pil: Optional[Image.Image],
        body_shape_mask: Optional[np.ndarray],
        prompt: str,
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> Image.Image:

        orig_w, orig_h = orig_pil.size
        orig_np = np.array(orig_pil)

        x1, y1, x2, y2 = self.get_padded_bbox(mask_np, Config.CROP_PAD, orig_h, orig_w)
        crop_w = x2 - x1
        crop_h = y2 - y1

        img_crop   = orig_pil.crop((x1, y1, x2, y2))
        mask_crop  = mask_np[y1:y2, x1:x2]

        erased_crop = self._remove_clothes_with_inpaint(img_crop, mask_crop)

        pose_crop  = pose_pil.crop((x1, y1, x2, y2)) if pose_pil is not None else None
        depth_crop = depth_pil.crop((x1, y1, x2, y2)) if depth_pil is not None else None

        shape_mask_crop = None
        if body_shape_mask is not None:
            shape_mask_crop = body_shape_mask[y1:y2, x1:x2]

        img_sd, mask_sd, pose_sd, depth_sd, pad_info = self._resize_with_padding(
            erased_crop, mask_crop, pose_crop, depth_crop,
            Config.SD_WIDTH, Config.SD_HEIGHT
        )
        mask_sd_pil = Image.fromarray(mask_sd).convert("L")

        generated_sd = self.inpainter.generate(
            image=img_sd,
            mask=mask_sd_pil,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            pose_image=pose_sd,
            depth_image=depth_sd,
        )

        generated_crop = self._remove_padding(generated_sd, pad_info, crop_w, crop_h)
        soft_alpha = self._feathered_alpha(mask_crop, Config.EDGE_FEATHER_PX)

        if depth_pil is not None and shape_mask_crop is not None:
            depth_np_crop = np.array(depth_crop.convert("L")).astype(np.float32) / 255.0
            grad_x = cv2.Sobel(depth_np_crop, cv2.CV_32F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(depth_np_crop, cv2.CV_32F, 0, 1, ksize=5)
            curvature = np.sqrt(grad_x**2 + grad_y**2)
            curvature = np.clip(curvature / (curvature.max() + 1e-6), 0, 1)
            shape_weight = curvature * (shape_mask_crop.astype(np.float32) / 255.0)
            shape_weight = gaussian_filter(shape_weight, sigma=3.0)
            shape_weight = np.clip(shape_weight * Config.DEPTH_BLEND_STRENGTH, 0, 0.5)
            shape_weight_3 = shape_weight[:, :, np.newaxis]
            gen_np   = np.array(generated_crop).astype(np.float32)
            orig_np_ = np.array(img_crop).astype(np.float32)
            depth_blended = gen_np * (1 - shape_weight_3) + orig_np_ * shape_weight_3
            generated_crop = Image.fromarray(np.clip(depth_blended, 0, 255).astype(np.uint8))

        soft_alpha_3  = soft_alpha[:, :, np.newaxis]
        gen_np        = np.array(generated_crop).astype(np.float32)
        img_crop_np   = np.array(img_crop).astype(np.float32)
        blended_crop  = gen_np * soft_alpha_3 + img_crop_np * (1.0 - soft_alpha_3)
        blended_crop  = np.clip(blended_crop, 0, 255).astype(np.uint8)

        result_np = orig_np.copy()
        result_np[y1:y2, x1:x2] = blended_crop

        return Image.fromarray(result_np)

    def _remove_clothes_with_inpaint(self, img_pil: Image.Image, mask_np: np.ndarray) -> Image.Image:
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        if mask_np.dtype != np.uint8:
            mask_np = mask_np.astype(np.uint8)
        inpainted_bgr = cv2.inpaint(img_bgr, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _resize_with_padding(
        img_pil: Image.Image,
        mask_np: np.ndarray,
        pose_pil: Optional[Image.Image],
        depth_pil: Optional[Image.Image],
        target_w: int,
        target_h: int
    ) -> Tuple[Image.Image, np.ndarray, Optional[Image.Image], Optional[Image.Image], dict]:
        iw, ih = img_pil.size
        scale  = min(target_w / iw, target_h / ih)
        new_w  = int(iw * scale)
        new_h  = int(ih * scale)
        pad_left = (target_w - new_w) // 2
        pad_top  = (target_h - new_h) // 2

        img_r    = img_pil.resize((new_w, new_h), Image.LANCZOS)
        mask_r   = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        img_padded  = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        mask_padded = np.zeros((target_h, target_w), dtype=np.uint8)

        img_padded.paste(img_r, (pad_left, pad_top))
        mask_padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = mask_r

        pose_padded  = None
        if pose_pil is not None:
            pose_r      = pose_pil.resize((new_w, new_h), Image.LANCZOS)
            pose_padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            pose_padded.paste(pose_r, (pad_left, pad_top))

        depth_padded = None
        if depth_pil is not None:
            depth_r      = depth_pil.resize((new_w, new_h), Image.LANCZOS)
            depth_padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            depth_padded.paste(depth_r, (pad_left, pad_top))

        pad_info = dict(
            pad_left=pad_left, pad_top=pad_top,
            new_w=new_w, new_h=new_h,
            orig_w=iw, orig_h=ih
        )
        return img_padded, mask_padded, pose_padded, depth_padded, pad_info

    @staticmethod
    def _remove_padding(
        generated_sd: Image.Image,
        pad_info: dict,
        target_w: int,
        target_h: int
    ) -> Image.Image:
        pl = pad_info["pad_left"]
        pt = pad_info["pad_top"]
        nw = pad_info["new_w"]
        nh = pad_info["new_h"]
        content = generated_sd.crop((pl, pt, pl + nw, pt + nh))
        return content.resize((target_w, target_h), Image.LANCZOS)

    @staticmethod
    def _feathered_alpha(mask_uint8: np.ndarray, feather_px: int) -> np.ndarray:
        alpha  = mask_uint8.astype(np.float32) / 255.0
        ksize  = feather_px * 2 + 1
        blurred = cv2.GaussianBlur(alpha, (ksize, ksize), feather_px * 0.5)
        return np.clip(blurred, 0.0, 1.0)


class SkinTonePreserver:
    @staticmethod
    def sample(image_bgr: np.ndarray, clothes_mask: np.ndarray) -> np.ndarray:
        H, W = image_bgr.shape[:2]
        non_clothes = (clothes_mask == 0)
        face_zone = np.zeros((H, W), bool)
        face_zone[:H // 3, W // 4: 3 * W // 4] = True
        skin_zone = non_clothes & face_zone
        if skin_zone.sum() < 50:
            skin_zone = non_clothes
        pixels = image_bgr[skin_zone].astype(np.float32)
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3)
                           .astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        skin_filter = (h < 25) & (s > 20) & (s < 175) & (v > 50)
        filtered = pixels[skin_filter]
        if len(filtered) < 10:
            filtered = pixels
        return filtered.mean(axis=0).astype(np.float32)

    @staticmethod
    def apply(
        generated_bgr: np.ndarray,
        original_bgr: np.ndarray,
        clothes_mask: np.ndarray,
        blend: float = 0.80
    ) -> np.ndarray:
        orig_skin = SkinTonePreserver.sample(original_bgr, clothes_mask)
        gen_skin  = SkinTonePreserver.sample(generated_bgr, clothes_mask)
        result    = generated_bgr.astype(np.float32).copy()
        non_cloth = (clothes_mask == 0)
        gen_mean  = gen_skin + 1e-6
        scale     = np.clip(orig_skin / gen_mean, 0.5, 2.0)
        result[non_cloth] = np.clip(result[non_cloth] * scale, 0, 255)
        bm     = non_cloth[:, :, np.newaxis].astype(np.float32)
        result = (result * (1 - blend * bm) +
                  generated_bgr.astype(np.float32) * (blend * bm))
        return np.clip(result, 0, 255).astype(np.uint8)


class ClothSwapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {self.device.upper()}")
        self.depth_extractor = DepthExtractor(self.device)
        self.segmenter       = FashionSegmenter(self.device)
        self.pose_extractor  = PoseExtractor()
        use_pose  = self.pose_extractor.available
        use_depth = self.depth_extractor.available
        self.inpainter   = SDInpainter(self.device,
                                       use_pose=use_pose,
                                       use_depth=use_depth)
        self.crop_engine = CropInpaintEngine(self.inpainter)

    def run(
        self,
        input_path: str,
        output_path: str,
        prompt: str           = Config.DEFAULT_PROMPT,
        negative_prompt: str  = Config.NEGATIVE_PROMPT,
        num_steps: int        = Config.NUM_STEPS,
        guidance_scale: float = Config.GUIDANCE_SCALE,
        seed: Optional[int]   = Config.SEED,
        save_debug: bool      = False,
    ) -> str:

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        orig_pil  = Image.open(input_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        orig_bgr  = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)

        debug_prefix = str(Path(output_path).with_suffix("")) if save_debug else None

        depth_pil = self.depth_extractor.extract(orig_pil)
        if save_debug and depth_pil is not None:
            depth_pil.save(f"{debug_prefix}_0_depth.png")
        self.depth_extractor.free()

        clothes_mask    = self.segmenter.build_clothing_mask(orig_pil)
        body_shape_mask = self.segmenter.build_body_shape_mask(orig_pil)

        if clothes_mask.max() == 0:
            log.error("No clothing detected — aborting.")
            sys.exit(1)

        if save_debug:
            Image.fromarray(clothes_mask).save(f"{debug_prefix}_1_mask.png")
            Image.fromarray(body_shape_mask).save(f"{debug_prefix}_1_shape_mask.png")

        self.segmenter.free()

        pose_pil = self.pose_extractor.extract(orig_pil)
        if save_debug and pose_pil is not None:
            pose_pil.save(f"{debug_prefix}_2_pose.png")

        result_pil = self.crop_engine.run(
            orig_pil=orig_pil,
            mask_np=clothes_mask,
            pose_pil=pose_pil,
            depth_pil=depth_pil,
            body_shape_mask=body_shape_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=Config.STRENGTH,
            seed=seed,
        )

        if save_debug:
            result_pil.save(f"{debug_prefix}_3_after_inpaint.png")

        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        result_bgr = SkinTonePreserver.apply(result_bgr, orig_bgr, clothes_mask)
        result_pil = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result_pil.save(output_path, quality=97)

        self.inpainter.free()
        return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced AI Cloth Swapper — Body Shape Preservation Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python cloth_swap_shape_preserve.py --input person.jpg --output result.jpg\n  python cloth_swap_shape_preserve.py --input person.jpg --prompt \"red bikini, form-fitting\"\n  python cloth_swap_shape_preserve.py --input person.jpg --steps 60 --guidance 10 --debug\n  python cloth_swap_shape_preserve.py --input person.jpg --depth-scale 0.7 --debug"
    )
    p.add_argument("--input",        "-i", default="input.jpg")
    p.add_argument("--output",       "-o", default="output.jpg")
    p.add_argument("--prompt",       "-p", default=Config.DEFAULT_PROMPT)
    p.add_argument("--neg",          "-n", default=Config.NEGATIVE_PROMPT)
    p.add_argument("--steps",        "-s", type=int,   default=Config.NUM_STEPS)
    p.add_argument("--guidance",     "-g", type=float, default=Config.GUIDANCE_SCALE)
    p.add_argument("--seed",               type=int,   default=Config.SEED)
    p.add_argument("--depth-scale",        type=float, default=Config.CONTROLNET_DEPTH_SCALE,
                   help="Depth ControlNet weight [0-1]. Higher = more body shape preserved.")
    p.add_argument("--pose-scale",         type=float, default=Config.CONTROLNET_POSE_SCALE,
                   help="Pose ControlNet weight [0-1]. Higher = stricter body pose lock.")
    p.add_argument("--depth-blend",        type=float, default=Config.DEPTH_BLEND_STRENGTH,
                   help="Compositing blend strength for shape preservation [0-1].")
    p.add_argument("--debug",        "-d", action="store_true",
                   help="Save intermediate debug images (depth, mask, pose, etc.)")
    return p


def main():
    args = build_parser().parse_args()
    Config.CONTROLNET_DEPTH_SCALE = args.depth_scale
    Config.CONTROLNET_POSE_SCALE  = args.pose_scale
    Config.DEPTH_BLEND_STRENGTH   = args.depth_blend
    swapper = ClothSwapper()
    swapper.run(
        input_path     = args.input,
        output_path    = args.output,
        prompt         = args.prompt,
        negative_prompt= args.neg,
        num_steps      = args.steps,
        guidance_scale = args.guidance,
        seed           = args.seed,
        save_debug     = args.debug,
    )


if __name__ == "__main__":
    main()