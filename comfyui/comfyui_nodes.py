"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import gc
import os

import comfy.model_management as mm
import cv2
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler)
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from ..ruyi.data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ..ruyi.models.autoencoder_magvit import AutoencoderKLMagvit
from ..ruyi.models.transformer3d import HunyuanTransformer3DModel
from ..ruyi.pipeline.pipeline_ruyi_inpaint import RuyiInpaintPipeline
from ..ruyi.utils.lora_utils import merge_lora, unmerge_lora
from ..ruyi.utils.utils import get_image_to_video_latent

# The directory of scripts
script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Ruyi_LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Ruyi-Mini-7B", 
                    ],
                    { "default": "Ruyi-Mini-7B", }
                ),
                "auto_download": (
                    ["yes", "no"],
                    {
                        "default": "yes"
                    }
                ),
                "auto_update": (
                    ["yes", "no"],
                    {
                        "default": "yes"
                    }
                ),

                "fp8_quant_mode": (
                    [ "none", "lite", "strong", "extreme"],
                    { "default": "none", "tooltip": 'Mode "extreme" is not recommended for good quality'}
                ),
                "fp8_data_type": (
                    [ "auto", "fp8_e4m3fn", "fp8_e5m2"],
                    { "default": "auto"}
                ),
            },
        }

    RETURN_TYPES = ("RUYI_MODEL",)
    RETURN_NAMES = ("ruyi_model",)
    FUNCTION = "load_model"
    CATEGORY = "Ruyi"

    RUYI_MODEL_TYPE_DICT = {
        "Ruyi-Mini-7B": "Inpaint"
    }

    def get_model_type(self, model_name):
        return self.RUYI_MODEL_TYPE_DICT.get(model_name, "Inpaint")

    def try_setup_pipeline(self, model_path, weight_dtype, config):
        try:
            # Init processbar
            pbar = ProgressBar(6)

            # Get Vae
            vae = AutoencoderKLMagvit.from_pretrained(
                model_path, 
                subfolder="vae"
            ).to(weight_dtype)
            # Update pbar
            pbar.update(1)

            # Get Transformer
            transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
            transformer = HunyuanTransformer3DModel.from_pretrained_2d(
                model_path, 
                subfolder="transformer",
                transformer_additional_kwargs=transformer_additional_kwargs
            ).to(weight_dtype)
            # Update pbar
            pbar.update(1)

            if self.fp8_quant_mode != 'none':
                count_f8 =0
                fp8_type = torch.float8_e5m2 if self.fp8_data_type == 'fp8_e5m2' else torch.float8_e4m3fn
                if self.fp8_quant_mode != 'extreme':

                    shape_size = 2816 ** 2
                    if self.fp8_quant_mode == 'strong': shape_size -= 1

                    for module in transformer.modules():
                        if module.__class__.__name__ in ["Linear"]:
                            x,y = module.weight.shape
                            if x * y > shape_size:
                                module.to(fp8_type)
                                count_f8 += 1
                else:
                    for module in transformer.modules():
                        if len(list(module.modules())) > 1: continue
                        elif module.__class__.__name__ not in ["Embedding"]:
                            module.to(fp8_type)
                            count_f8 += 1

                print (f'FP8: {count_f8} layers converted to {fp8_type}')
            # Update pbar
            pbar.update(1)

            # Load Clip
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path, subfolder="image_encoder"
            ).to(weight_dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                model_path, subfolder="image_encoder"
            )
            # Update pbar
            pbar.update(1)

            # Load sampler and create pipeline
            Choosen_Scheduler = DDIMScheduler
            scheduler = Choosen_Scheduler.from_pretrained(
                model_path, 
                subfolder="scheduler"
            )
            pipeline = RuyiInpaintPipeline.from_pretrained(
                model_path,
                vae=vae,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=weight_dtype,
                clip_image_encoder=clip_image_encoder,
                clip_image_processor=clip_image_processor,
            )

            # Load embeddings
            embeddings = load_safetensors(os.path.join(model_path, "embeddings.safetensors"))
            pipeline.embeddings = embeddings
            # Update pbar
            pbar.update(1)

            return pipeline
        except Exception as e:
            print("[Ruyi] Setup pipeline failed:", e)
            return None
    
    def load_model(self, model, auto_download, auto_update, fp8_quant_mode='none', fp8_data_type='auto'):
        self.fp8_quant_mode = fp8_quant_mode
        self.fp8_data_type = fp8_data_type

        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()

        # Init model name and type
        model_name = model
        model_type = self.get_model_type(model_name)
        weight_dtype = torch.bfloat16

        # Load config
        config_path = os.path.join(script_directory, "config", "default.yaml")
        config = OmegaConf.load(config_path)

        # Check for update
        repo_id = f"IamCreateAI/{model_name}"
        model_path = os.path.join(folder_paths.models_dir, "Ruyi", model_name)
        if auto_download == "yes" and auto_update == "yes":
            print(f"Checking for {model_name} updates ...")
            
            # Download the model
            snapshot_download(repo_id=repo_id, local_dir=model_path)

        # Init model
        pipeline = self.try_setup_pipeline(model_path, weight_dtype, config)
        if pipeline is None and auto_download == "yes":
            mm.soft_empty_cache()
            gc.collect()

            # Download the model
            snapshot_download(repo_id=repo_id, local_dir=model_path)

            pipeline = self.try_setup_pipeline(model_path, weight_dtype, config)
        
        if pipeline is None:
            message = (f"[ Load Model {model_name} Failed ]\n"
                       f"Please download Ruyi model from huggingface repo '{repo_id}',\n"
                       f"And put it into '{model_path}'.")
            if auto_download == "no":
                message += "\n\nOr just set auto_download to 'yes'."
            raise FileNotFoundError(message)
        
        pipeline.enable_model_cpu_offload()
        ruyi_model = {
            'pipeline': pipeline,
            'dtype': weight_dtype,
            'model_path': model_path,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
        }
        return (ruyi_model,)


class Ruyi_LoadLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ruyi_model": ("RUYI_MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("RUYI_MODEL",)
    RETURN_NAMES = ("ruyi_model",)
    FUNCTION = "load_lora"
    CATEGORY = "Ruyi"

    def load_lora(self, ruyi_model, lora_name, strength_model):
        if lora_name is not None:
            return (
                {
                    'pipeline': ruyi_model["pipeline"], 
                    'dtype': ruyi_model["dtype"],
                    'model_path': ruyi_model["model_path"],
                    'model_type': ruyi_model["model_type"],
                    'loras': ruyi_model.get("loras", []) + [folder_paths.get_full_path("loras", lora_name)],
                    'strength_model': ruyi_model.get("strength_model", []) + [strength_model],
                }, 
            )
        else:
            return (ruyi_model,)


class Ruyi_I2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ruyi_model": (
                    "RUYI_MODEL", 
                ),
                "video_length": (
                    "INT", {"default": 72, "min": 8, "max": 120, "step": 8}
                ),
                "base_resolution": (
                    "INT", {"default": 512, "min": 384, "max": 1024, "step": 16}
                ),
                "seed": (
                    "INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 25, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                ),
                "motion": (
                    [ "1", "2", "3", "4", "auto" ],
                    { "default": "2" }
                ),
                "camera_direction": (
                    [ "static", "left", "right", "up", "down", "auto" ],
                    { "default": "static" }
                ),
                "GPU_memory_mode": (
                    ["normal_mode", "low_memory_mode"],
                    {
                        "default": "normal_mode",
                    }
                ),
                "GPU_offload_steps": (
                    [ "0", "1", "5", "7", "10" ],
                    { "default": "0" }
                ),
                "start_img": (
                    "IMAGE",
                ),
            },
            "optional":{
                "end_img": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "Ruyi"

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

    def numpy2pil(self, image):
        return Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))

    def to_pil(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            return self.tensor2pil(image)
        if isinstance(image, np.ndarray):
            return self.numpy2pil(image)
        raise ValueError(f"Cannot convert {type(image)} to PIL.Image")

    def get_control_embeddings(self, pipeline, aspect_ratio, motion, camera_direction):
        # Default keys
        p_default_key = "p.default"
        n_default_key = "n.default"

        # Load embeddings
        if motion == "auto":
            motion = "0"
        p_key = f"p.{aspect_ratio.replace(':', 'x')}movie{motion}{camera_direction}"
        embeddings = pipeline.embeddings

        # Get embeddings
        positive_embeds = embeddings.get(f"{p_key}.emb1", embeddings[f"{p_default_key}.emb1"])
        positive_attention_mask = embeddings.get(f"{p_key}.mask1", embeddings[f"{p_default_key}.mask1"])
        positive_embeds_2 = embeddings.get(f"{p_key}.emb2", embeddings[f"{p_default_key}.emb2"])
        positive_attention_mask_2 = embeddings.get(f"{p_key}.mask2", embeddings[f"{p_default_key}.mask2"])

        negative_embeds = embeddings[f"{n_default_key}.emb1"]
        negative_attention_mask = embeddings[f"{n_default_key}.mask1"]
        negative_embeds_2 = embeddings[f"{n_default_key}.emb2"]
        negative_attention_mask_2 = embeddings[f"{n_default_key}.mask2"]

        return {
            "positive_embeds": positive_embeds,
            "positive_attention_mask": positive_attention_mask,
            "positive_embeds_2": positive_embeds_2,
            "positive_attention_mask_2": positive_attention_mask_2,

            "negative_embeds": negative_embeds,
            "negative_attention_mask": negative_attention_mask,
            "negative_embeds_2": negative_embeds_2,
            "negative_attention_mask_2": negative_attention_mask_2,
        }

    def get_scheduler(self, model_path, scheduler_name):
        if scheduler_name == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
        elif scheduler_name == "Euler":
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
        elif scheduler_name == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
        elif scheduler_name == "PNDM":
            noise_scheduler = PNDMScheduler.from_pretrained(model_path, subfolder='scheduler')
        elif scheduler_name == "DDIM":
            noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
        
        return noise_scheduler

    def process(
            self, ruyi_model, video_length, base_resolution, seed, steps, cfg, scheduler, 
            motion, camera_direction, GPU_memory_mode, GPU_offload_steps,
            start_img, end_img=None
        ):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        start_img = [self.to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
        end_img = [self.to_pil(_end_img) for _end_img in end_img] if end_img is not None else None

        # Count most suitable height and width
        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        aspect_ratio = "16:9"  # Do not change, currently "16:9" works better
        
        # Get Pipeline
        pipeline = ruyi_model['pipeline']
        model_path = ruyi_model['model_path']

        # Set GPU memory mode
        if GPU_memory_mode == "low_memory_mode" and pipeline.model_cpu_offload_flag:
            # Switch to low_memory_mode
            pipeline.enable_sequential_cpu_offload()
        elif GPU_memory_mode == "normal_mode" and not pipeline.model_cpu_offload_flag:
            # Switch to normal_mode
            pipeline.enable_model_cpu_offload()

        # Set GPU offload steps
        pipeline.transformer.hidden_cache_size = int(GPU_offload_steps)

        # Load Sampler
        pipeline.scheduler = self.get_scheduler(model_path, scheduler)

        # Set random seed
        generator= torch.Generator(device).manual_seed(seed)

        # Load control embeddings
        embeddings = self.get_control_embeddings(pipeline, aspect_ratio, motion, camera_direction)

        # Inference
        with torch.no_grad(), torch.autocast(str(device), dtype = pipeline.transformer.dtype):
            video_length = int(video_length // pipeline.vae.mini_batch_encoder * pipeline.vae.mini_batch_encoder) if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            for _lora_path, _lora_weight in zip(ruyi_model.get("loras", []), ruyi_model.get("strength_model", [])):
                pipeline = merge_lora(pipeline, _lora_path, _lora_weight)
            
            sample = pipeline(
                prompt_embeds = embeddings["positive_embeds"],
                prompt_attention_mask = embeddings["positive_attention_mask"],
                prompt_embeds_2 = embeddings["positive_embeds_2"],
                prompt_attention_mask_2 = embeddings["positive_attention_mask_2"],

                negative_prompt_embeds = embeddings["negative_embeds"],
                negative_prompt_attention_mask = embeddings["negative_attention_mask"],
                negative_prompt_embeds_2 = embeddings["negative_embeds_2"],
                negative_prompt_attention_mask_2 = embeddings["negative_attention_mask_2"],

                video_length = video_length,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                clip_image   = clip_image, 
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            for _lora_path, _lora_weight in zip(ruyi_model.get("loras", []), ruyi_model.get("strength_model", [])):
                pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)

        return (videos,)


NODE_CLASS_MAPPINGS = {
    "Ruyi_LoadModel": Ruyi_LoadModel,
    "Ruyi_LoadLora": Ruyi_LoadLora,
    "Ruyi_I2VSampler": Ruyi_I2VSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ruyi_LoadModel": "Load Model",
    "Ruyi_LoadLora": "Load LoRA",
    "Ruyi_I2VSampler": "Sampler for Image to Video",
}
