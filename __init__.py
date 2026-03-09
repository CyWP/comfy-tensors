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
