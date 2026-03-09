"""Tensor aggregation nodes for ComfyUI."""

import torch


class TensorCat:
    """Concatenates a list of tensors along a specified dimension.

    Validates that all tensors have the same number of dimensions before
    concatenating along the specified axis.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensors": ("TENSOR",),
                "dim": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensors: list[torch.Tensor], dim: int) -> tuple[torch.Tensor]:
        """
        Args:
            tensors: List of tensors to concatenate.
            dim: Dimension along which to concatenate.

        Returns:
            - out: Concatenated tensor.

        Raises:
            ValueError: If tensors have different numbers of dimensions.
        """
        if not isinstance(tensors, list):
            tensors = [tensors]

        ref_shape = tensors[0].shape
        for t in tensors[1:]:
            if len(t.shape) != len(ref_shape):
                raise ValueError(
                    f"All tensors must have the same number of dimensions, "
                    f"got {t.shape} vs {ref_shape}"
                )

        out = torch.cat(tensors, dim=dim)

        return (out,)


class TensorStack:
    """Stacks a list of tensors along a new dimension.

    Validates that all tensors have identical shapes before stacking
    along a new dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensors": ("TENSOR",),
                "dim": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensors: list[torch.Tensor], dim: int) -> tuple[torch.Tensor]:
        """
        Args:
            tensors: List of tensors to stack. All must have identical shapes.
            dim: Dimension along which to stack.

        Returns:
            - out: Stacked tensor with shape (N, *original_shape) at stacking dim.

        Raises:
            ValueError: If tensors have different shapes.
        """
        if not isinstance(tensors, list):
            tensors = [tensors]

        ref_shape = tensors[0].shape
        for t in tensors[1:]:
            if t.shape != ref_shape:
                raise ValueError(
                    f"All tensors must have the same shape to stack, "
                    f"got {t.shape} vs {ref_shape}"
                )

        out = torch.stack(tensors, dim=dim)

        return (out,)
