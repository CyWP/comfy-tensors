"""Debug node for inspecting tensor statistics in ComfyUI."""

import torch


class TensorInspector:
    """Returns detailed statistics for a LATENT tensor as a string.

    Computes shape, dtype, device, min, max, mean, std, sum, norm,
    and requires_grad for inspection during workflow debugging.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("LATENT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    @staticmethod
    def summarize_tensor(tensor: torch.Tensor) -> dict[str, float | str | bool]:
        """
        Args:
            tensor: The tensor to summarize.

        Returns:
            Dictionary containing tensor statistics:
            - shape: tuple of dimensions
            - dtype: data type as string
            - device: device as string
            - min: minimum value
            - max: maximum value
            - mean: mean value
            - std: standard deviation
            - sum: sum of all elements
            - norm: L2 norm
            - requires_grad: whether gradient is tracked
        """
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

    def run(
        self, tensor: dict[str, torch.Tensor] | list[dict]
    ) -> tuple[str]:
        """
        Args:
            tensor: LATENT dict(s) with 'samples' key.

        Returns:
            - out: String containing formatted tensor statistics.
        """
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

        return ("\n".join(lines),)
