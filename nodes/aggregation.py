import torch


class TensorCat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensors": ("TENSOR",),  # accept a list of tensors
                "dim": ("INT", {"default": 0}),  # dimension to concatenate along
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensors, dim):

        # Ensure list of tensors
        if not isinstance(tensors, list):
            tensors = [tensors]

        # Validate all tensors have the same rank except along `dim`
        ref_shape = tensors[0].shape
        for t in tensors[1:]:
            if len(t.shape) != len(ref_shape):
                raise ValueError(
                    f"All tensors must have the same number of dimensions, got {t.shape} vs {ref_shape}"
                )

        out = torch.cat(tensors, dim=dim)
        return (out,)


class TensorStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensors": ("TENSOR",),  # accept a list of tensors
                "dim": ("INT", {"default": 0}),  # dimension to stack along
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensors, dim):

        # Ensure list of tensors
        if not isinstance(tensors, list):
            tensors = [tensors]

        # Validate all tensors have the same shape
        ref_shape = tensors[0].shape
        for t in tensors[1:]:
            if t.shape != ref_shape:
                raise ValueError(
                    f"All tensors must have the same shape to stack, got {t.shape} vs {ref_shape}"
                )

        out = torch.stack(tensors, dim=dim)
        return (out,)
