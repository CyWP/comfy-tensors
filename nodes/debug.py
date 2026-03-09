import torch


class TensorInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("LATENT",),  # or IMAGE/LATENT
            }
        }

    RETURN_TYPES = ("STRING",)  # now returns a string
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    @staticmethod
    def summarize_tensor(tensor):
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "sum": tensor.sum().item(),
            "norm": tensor.norm().item(),
            "requires_grad": tensor.requires_grad,
        }

    def run(self, tensor):
        stats = {}

        # Normalize input to a list
        tensors = (
            [tensor["samples"]]
            if not isinstance(tensor, list)
            else [t["samples"] for t in tensor]
        )
        lines = []
        for i, t in enumerate(tensors):
            lines.append(f"----------{i}----------")
            for k, v in self.summarize_tensor(t).items():
                lines.append(f"{k}: {v}")

        return ("\n".join(lines),)  # ComfyUI expects a tuple for RETURN_TYPES
