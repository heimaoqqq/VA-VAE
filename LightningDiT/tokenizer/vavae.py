"""
Vision Foundation Model Aligned VAE.
It has exactly the same architecture as the LDM VAE (or VQGAN-KL).
Here we first provide its inference implementation with diffusers. 
The training code will be provided later. 

"LightningDiT + VA_VAE" achieves state-of-the-art Latent Diffusion System
with 0.27 rFID and 1.35 FID on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from tokenizer.autoencoder import AutoencoderKL

class VA_VAE:
    """Vision Foundation Model Aligned VAE Implementation"""
    
    def __init__(self, config, img_size=256, horizon_flip=0.5, fp16=True):
        """Initialize VA_VAE
        Args:
            config: Configuration dict containing img_size, horizon_flip and fp16 parameters
        """
        self.config = OmegaConf.load(config)
        self.embed_dim = self.config.model.params.embed_dim
        self.ckpt_path = self.config.ckpt_path
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.load()

    def load(self):
        """Load and initialize VAE model"""
        self.model = AutoencoderKL(
            embed_dim=self.embed_dim,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=self.ckpt_path
        ).cuda().eval()
        return self
    
    def img_transform(self, p_hflip=0, img_size=None):
        """Image preprocessing transforms
        Args:
            p_hflip: Probability of horizontal flip
            img_size: Target image size, use default if None
        Returns:
            transforms.Compose: Image transform pipeline
        """
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images):
        """Encode images to latent representations
        Args:
            images: Input image tensor
        Returns:
            torch.Tensor: Encoded latent representation
        """
        with torch.no_grad():
            posterior = self.model.encode(images.cuda())
            return posterior.sample()

    def decode_to_images(self, z):
        """Decode latent representations to images with proper micro-Doppler colormap
        Args:
            z: Latent representation tensor
        Returns:
            np.ndarray: Decoded image array with proper colormap
        """
        with torch.no_grad():
            # 解码到[-1, 1]范围
            images = self.model.decode(z.cuda())

            # 调试信息
            print(f"🔍 VA-VAE解码调试:")
            print(f"  解码前潜在特征范围: [{z.min():.3f}, {z.max():.3f}]")
            print(f"  解码后图像范围: [{images.min():.3f}, {images.max():.3f}]")

            # 强制归一化到[-1, 1]范围 (处理超出范围的情况)
            images = 2.0 * (images - images.min()) / (images.max() - images.min()) - 1.0
            images = torch.clamp(images, -1.0, 1.0)
            print(f"  归一化后图像范围: [{images.min():.3f}, {images.max():.3f}]")

            # 对于微多普勒时频图，应用正确的颜色映射
            processed_images = []
            for i in range(images.shape[0]):
                img = images[i]  # (C, H, W)

                # 如果是RGB图像，转换为灰度作为强度
                if img.shape[0] == 3:
                    # 使用标准RGB到灰度的权重
                    intensity = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                else:
                    intensity = img[0]  # 使用第一个通道

                # 归一化到[0, 1]
                intensity = (intensity + 1.0) / 2.0
                intensity = torch.clamp(intensity, 0, 1)
                print(f"  强度图范围: [{intensity.min():.3f}, {intensity.max():.3f}]")

                # 应用微多普勒时频图的颜色映射 (类似matplotlib的jet colormap)
                colored_img = self._apply_micro_doppler_colormap(intensity)
                print(f"  颜色映射后范围: [{colored_img.min():.3f}, {colored_img.max():.3f}]")
                processed_images.append(colored_img)

            # 转换为numpy数组
            result = torch.stack(processed_images).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            print(f"  最终图像范围: [{result.min()}, {result.max()}]")
            print(f"  最终图像形状: {result.shape}")

        return result

    def _apply_micro_doppler_colormap(self, intensity):
        """Apply micro-Doppler specific colormap (blue to red)
        Args:
            intensity: Normalized intensity tensor (H, W) in [0, 1]
        Returns:
            torch.Tensor: RGB image (3, H, W) in [0, 255]
        """
        # 创建类似jet colormap的微多普勒颜色映射
        # 低强度 -> 深蓝色，高强度 -> 红色

        # 定义颜色映射的关键点 (强度 -> RGB)
        # 0.0: 深蓝 (0, 0, 128)
        # 0.25: 蓝色 (0, 0, 255)
        # 0.5: 青色 (0, 255, 255)
        # 0.75: 黄色 (255, 255, 0)
        # 1.0: 红色 (255, 0, 0)

        h, w = intensity.shape
        rgb_img = torch.zeros(3, h, w, device=intensity.device)

        # Red channel
        rgb_img[0] = torch.where(intensity < 0.5,
                                0,
                                torch.where(intensity < 0.75,
                                           (intensity - 0.5) * 4 * 255,
                                           255))

        # Green channel
        rgb_img[1] = torch.where(intensity < 0.25,
                                0,
                                torch.where(intensity < 0.5,
                                           (intensity - 0.25) * 4 * 255,
                                           torch.where(intensity < 0.75,
                                                      255,
                                                      255 - (intensity - 0.75) * 4 * 255)))

        # Blue channel
        rgb_img[2] = torch.where(intensity < 0.25,
                                128 + intensity * 4 * 127,  # 深蓝到蓝
                                torch.where(intensity < 0.5,
                                           255 - (intensity - 0.25) * 4 * 255,
                                           0))

        return torch.clamp(rgb_img, 0, 255)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


if __name__ == "__main__":
    vae = VA_VAE('tokenizer/configs/vavae_f16d32_vfdinov2.yaml')
    vae.load()