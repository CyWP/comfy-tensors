"""comfy-tensors: ComfyUI custom nodes for tensor operations, format conversions, and symbolic expression evaluation.

License: GNU GPL v3

Exposes:
- TorchSymbolic: Symbolic expression evaluator for LATENT tensors.
- Img2Latent, Latent2Img, Mask2Latent, Latent2Mask, Img2Mask, Mask2Img: Format conversion nodes.
- TensorCat, TensorStack: Tensor aggregation nodes.
- TensorInspector: Debug node for tensor statistics.

Internal:
- nodes/bridges.py: IMAGE/LATENT/MASK format conversion nodes.
- nodes/aggregation.py: TensorCat and TensorStack.
- nodes/symbolic.py: TorchSymbolic expression evaluator.
- nodes/debug.py: TensorInspector debug node.
- utils/symbolic/parser.py: Base recursive-descent expression parser.
- utils/symbolic/torch_parser.py: PyTorch-specific parser with tensor functions.
"""

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Symbolic Parser": TorchSymbolic,
    "Image2Latent": Img2Latent,
    "Latent2Image": Latent2Img,
    "Mask2Latent": Mask2Latent,
    "Latent2Mask": Latent2Mask,
    "Image2Mask": Img2Mask,
    "Mask2Image": Mask2Img,
    "Concatenate": TensorCat,
    "Stack": TensorStack,
    "Inspect": TensorInspector,
}
