import torch

from ..utils.symbolic import TorchParser


def generate_symbol_names(n):
    """
    Generates variable names:
    a,b,c,...,z,aa,ba,ca,...,ab,bb,...
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    names = []

    i = 0
    while len(names) < n:
        name = ""
        k = i

        while True:
            name += alphabet[k % 26]
            k //= 26
            if k == 0:
                break

        names.append(name)
        i += 1

    return names


class TorchSymbolic:

    _parser = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("LATENT",),  # can be changed to any tensor type you want
                "expr": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(self, tensor, expr):

        # Normalize tensor input to list
        if isinstance(tensor, list):
            tensors = [t["samples"] for t in tensor]
        else:
            tensors = [tensor["samples"]]

        # Assign variable names
        names = generate_symbol_names(len(tensors))
        variables = dict(zip(names, tensors))

        # Create parser
        parser = (
            TorchParser() if TorchSymbolic._parser is None else TorchSymbolic._parser
        )
        TorchSymbolic._parser = parser

        # Evaluate expression
        result = parser.compute(expr, variables)

        return ({"samples": result},)
