import torch

class Img2Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("IMAGE",),
                "permute": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensor, permute):

        # Normalize to list
        if isinstance(tensor, list):
            imgs = tensor
        else:
            imgs = [tensor]

        processed = []

        for img in imgs:

            if img.ndim == 3:
                img = img.unsqueeze(0)

            if img.ndim != 4:
                raise ValueError(f"Expected tensor with 3 or 4 dims, got {img.shape}")

            if permute:
                # BHWC → BCHW
                img = img.permute(0, 3, 1, 2)

            processed.append(img)

        tensor_out = torch.cat(processed, dim=0)

        return ({"samples": tensor_out},)
    
class Latent2Img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "permute": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, latent, permute):

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
                # BCHW → BHWC
                x = x.permute(0, 2, 3, 1)

            processed.append(x)

        tensor_out = torch.cat(processed, dim=0)

        return (tensor_out,)
    
class Img2Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reduction": ("STRING", {"default": "mean", "choices": ["r","g","b","a","mean"]}),
                "permute": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, image, reduction, permute):

        # Normalize to list
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        processed = []

        for im in images:

            if im.ndim == 3:  # HWC
                im = im.unsqueeze(0)
            elif im.ndim != 4:  # B,H,W,C
                raise ValueError(f"Expected IMAGE with 3 or 4 dims, got {im.shape}")

            # Permute to BCHW
            if permute:
                im = im.permute(0, 3, 1, 2)

            # Apply reduction
            if reduction in ["r","g","b","a"]:
                ch_map = {"r":0, "g":1, "b":2, "a":3}
                ch_idx = ch_map[reduction]
                if im.shape[1] <= ch_idx:
                    raise ValueError(f"Image does not have channel '{reduction}'")
                mask = im[:, ch_idx:ch_idx+1, :, :]  # keep dims
            elif reduction == "mean":
                mask = im.mean(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

            processed.append(mask)

        out = torch.cat(processed, dim=0)
        return (out,)
    
class Mask2Img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),            # input mask(s)
                "num_channels": ("INT", {"default": 1, "min": 1}),
                "permute": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, mask, num_channels, permute):

        # Normalize to list
        if isinstance(mask, list):
            masks = mask
        else:
            masks = [mask]

        processed = []

        for m in masks:
            if m.ndim == 2:  # H,W
                m = m.unsqueeze(0).unsqueeze(-1)  # add batch & channel
            elif m.ndim == 3:  # H,W,C
                m = m.unsqueeze(0)  # add batch
            elif m.ndim == 4:  # B,H,W,C
                pass
            else:
                raise ValueError(f"Expected mask with 2,3,4 dims, got {m.shape}")

            # Optionally permute: BHWC → BCHW
            if permute:
                m = m.permute(0, 3, 1, 2)

            # Expand / reduce channels
            if m.shape[1] != num_channels:
                m = m.repeat(1, num_channels // m.shape[1] + 1, 1, 1)
                m = m[:, :num_channels]

            processed.append(m)

        out = torch.cat(processed, dim=0)
        return (out,)
    
class Latent2Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),  # input latent(s)
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, latent, permute):

        # Normalize input
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
                raise ValueError(f"Expected latent samples with 4 dims [B,C,H,W], got {x.shape}")

            # Permute BCHW → BHWC
            if permute:
                x = x.permute(0, 2, 3, 1)

            processed.append(x)

        tensor_out = torch.cat(processed, dim=0)
        return (tensor_out,)
    
class Mask2Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),  # input mask(s)
                "permute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, mask, permute):

        # Normalize input to list
        if isinstance(mask, list):
            masks = mask
        else:
            masks = [mask]

        processed = []

        for m in masks:
            # Add batch if needed
            if m.ndim == 2:  # H,W
                m = m.unsqueeze(0).unsqueeze(-1)
            elif m.ndim == 3:  # H,W,C
                m = m.unsqueeze(0)
            elif m.ndim == 4:  # B,H,W,C
                pass
            else:
                raise ValueError(f"Expected mask with 2,3,4 dims, got {m.shape}")

            # Permute BHWC → BCHW
            if permute:
                m = m.permute(0, 3, 1, 2)

            processed.append(m)

        tensor_out = torch.cat(processed, dim=0)

        return ({"samples": tensor_out},)