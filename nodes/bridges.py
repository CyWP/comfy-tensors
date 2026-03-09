"""IMAGE/LATENT/MASK format conversion nodes for ComfyUI."""

import torch


class Img2Latent:
    """Converts IMAGE tensors to LATENT format.

    Transforms IMAGE tensors from BHWC (batch, height, width, channels)
    to LATENT format BCHW (batch, channels, height, width).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("IMAGE",),
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensor: torch.Tensor, permute: bool) -> tuple[dict[str, torch.Tensor]]:
        """
        Args:
            tensor: IMAGE tensor in BHWC format (or HWC if single image).
            permute: Whether to permute BHWC -> BCHW (default True).

        Returns:
            - out: LATENT dict with 'samples' key containing tensor in BCHW format.
        """
        if isinstance(tensor, list):
            imgs = tensor
        else:
            imgs = [tensor]

        processed = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)  # () -> (1, H, W, C)

            if img.ndim != 4:
                raise ValueError(f"Expected tensor with 3 or 4 dims, got {img.shape}")

            if permute:
                img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW

            processed.append(img)

        tensor_out = torch.cat(processed, dim=0)

        return ({"samples": tensor_out},)


class Latent2Img:
    """Converts LATENT tensors to IMAGE format.

    Transforms LATENT tensors from BCHW (batch, channels, height, width)
    to IMAGE format BHWC (batch, height, width, channels).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(
        self, latent: dict[str, torch.Tensor] | list[dict], permute: bool
    ) -> tuple[torch.Tensor]:
        """
        Args:
            latent: LATENT dict(s) with 'samples' key in BCHW format.
            permute: Whether to permute BCHW -> BHWC (default True).

        Returns:
            - out: IMAGE tensor in BHWC format.
        """
        if isinstance(latent, list):
            latents = latent
        else:
            latents = [latent]

        processed = []
        for l in latents:
            if not isinstance(l, dict) or "samples" not in l:
                raise ValueError("Expected LATENT dictionary with 'samples' key")

            x = l["samples"]

            if x.ndim != 4:
                raise ValueError(f"Expected latent samples with 4 dims, got {x.shape}")

            if permute:
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC

            processed.append(x)

        tensor_out = torch.cat(processed, dim=0)

        return (tensor_out,)


class Img2Mask:
    """Extracts a mask channel from IMAGE tensor.

    Reduces IMAGE tensor to a mask by extracting a specific channel
    (r/g/b/a) or computing the mean across channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reduction": (
                    "STRING",
                    {"default": "mean", "choices": ["r", "g", "b", "a", "mean"]},
                ),
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(
        self, image: torch.Tensor, reduction: str, permute: bool
    ) -> tuple[torch.Tensor]:
        """
        Args:
            image: IMAGE tensor in BHWC format.
            reduction: Channel to extract ('r', 'g', 'b', 'a') or 'mean' across channels.
            permute: Whether to permute BHWC -> BCHW (default True).

        Returns:
            - out: IMAGE tensor with single channel dimension.
        """
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        processed = []
        for im in images:
            if im.ndim == 3:  # HWC
                im = im.unsqueeze(0)  # (H, W, C) -> (1, H, W, C)
            elif im.ndim != 4:  # B, H, W, C
                raise ValueError(f"Expected IMAGE with 3 or 4 dims, got {im.shape}")

            if permute:
                im = im.permute(0, 3, 1, 2)  # BHWC -> BCHW

            if reduction in ["r", "g", "b", "a"]:
                ch_map = {"r": 0, "g": 1, "b": 2, "a": 3}
                ch_idx = ch_map[reduction]
                if im.shape[1] <= ch_idx:
                    raise ValueError(f"Image does not have channel '{reduction}'")
                mask = im[:, ch_idx : ch_idx + 1, :, :]  # (B, 1, H, W)
            elif reduction == "mean":
                mask = im.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

            processed.append(mask)

        out = torch.cat(processed, dim=0)

        return (out,)


class Mask2Img:
    """Expands a mask tensor to an IMAGE tensor with configurable channels.

    Takes a mask tensor and expands it to image format with the specified
    number of channels by repeating the mask data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
                "num_channels": ("INT", {"default": 1, "min": 1}),
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(
        self, mask: torch.Tensor, num_channels: int, permute: bool
    ) -> tuple[torch.Tensor]:
        """
        Args:
            mask: Mask tensor in IMAGE format (2D, 3D, or 4D).
            num_channels: Number of channels in output image.
            permute: Whether to permute BCHW -> BHWC (default True).

        Returns:
            - out: IMAGE tensor with num_channels dimensions.
        """
        if isinstance(mask, list):
            masks = mask
        else:
            masks = [mask]

        processed = []
        for m in masks:
            if m.ndim == 2:  # H, W
                m = m.unsqueeze(0).unsqueeze(-1)  # (H, W) -> (1, H, W, 1)
            elif m.ndim == 3:  # H, W, C
                m = m.unsqueeze(0)  # (H, W, C) -> (1, H, W, C)
            elif m.ndim == 4:  # B, H, W, C
                pass
            else:
                raise ValueError(f"Expected mask with 2,3,4 dims, got {m.shape}")

            if permute:
                m = m.permute(0, 3, 1, 2)  # BHWC -> BCHW

            if m.shape[1] != num_channels:
                m = m.repeat(1, num_channels // m.shape[1] + 1, 1, 1)
                m = m[:, :num_channels]

            processed.append(m)

        out = torch.cat(processed, dim=0)

        return (out,)


class Latent2Mask:
    """Converts LATENT samples to IMAGE mask format.

    Takes LATENT samples tensor and converts it to IMAGE format
    (BHWC) for use as a mask in ComfyUI workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(
        self, latent: dict[str, torch.Tensor] | list[dict], permute: bool
    ) -> tuple[torch.Tensor]:
        """
        Args:
            latent: LATENT dict(s) with 'samples' key in BCHW format.
            permute: Whether to permute BCHW -> BHWC (default True).

        Returns:
            - out: IMAGE tensor in BHWC format.
        """
        if isinstance(latent, list):
            latents = latent
        else:
            latents = [latent]

        processed = []
        for l in latents:
            if not isinstance(l, dict) or "samples" not in l:
                raise ValueError("Expected LATENT dictionary with 'samples' key")

            x = l["samples"]

            if x.ndim != 4:
                raise ValueError(
                    f"Expected latent samples with 4 dims [B,C,H,W], got {x.shape}"
                )

            if permute:
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC

            processed.append(x)

        tensor_out = torch.cat(processed, dim=0)

        return (tensor_out,)


class Mask2Latent:
    """Converts a mask tensor to LATENT samples format.

    Takes an IMAGE-format mask tensor and converts it to LATENT
    samples format (BCHW) for use in latent operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, mask: torch.Tensor, permute: bool) -> tuple[dict[str, torch.Tensor]]:
        """
        Args:
            mask: Mask tensor in IMAGE format (2D, 3D, or 4D).
            permute: Whether to permute BHWC -> BCHW (default True).

        Returns:
            - out: LATENT dict with 'samples' key containing tensor in BCHW format.
        """
        if isinstance(mask, list):
            masks = mask
        else:
            masks = [mask]

        processed = []
        for m in masks:
            if m.ndim == 2:  # H, W
                m = m.unsqueeze(0).unsqueeze(-1)  # (H, W) -> (1, H, W, 1)
            elif m.ndim == 3:  # H, W, C
                m = m.unsqueeze(0)  # (H, W, C) -> (1, H, W, C)
            elif m.ndim == 4:  # B, H, W, C
                pass
            else:
                raise ValueError(f"Expected mask with 2,3,4 dims, got {m.shape}")

            if permute:
                m = m.permute(0, 3, 1, 2)  # BHWC -> BCHW

            processed.append(m)

        tensor_out = torch.cat(processed, dim=0)

        return ({"samples": tensor_out},)
