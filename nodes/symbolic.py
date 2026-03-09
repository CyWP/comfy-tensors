"""Symbolic expression evaluator for LATENT tensors in ComfyUI."""

import torch

from ..utils.symbolic import TorchParser


def generate_symbol_names(n: int) -> list[str]:
    """Generates variable names for tensor inputs.

    Produces sequential names: a, b, c, ..., z, aa, ba, ca, ..., ab, bb, ...

    Args:
        n: Number of variable names to generate.

    Returns:
        List of variable names.
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
    """Evaluates symbolic expressions on LATENT tensors.

    Parses a multiline string expression and evaluates it against
    LATENT tensor inputs. Input tensors are auto-named with sequential
    letters (a, b, c, ...).

    Attributes:
        _parser: Cached TorchParser instance for expression evaluation.

    Notes:
        - Uses TorchParser which supports PyTorch tensor operations.
        - Parser is cached as class variable for performance.

    Example:
        Input: a + b
        Result: Adds two latent tensors and returns the result.
    """

    _parser: TorchParser | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("LATENT",),
                "expr": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Tensor Ops"

    def run(
        self,
        tensor: dict[str, torch.Tensor] | list[dict],
        expr: str,
    ) -> tuple[dict[str, torch.Tensor]]:
        """
        Args:
            tensor: LATENT dict(s) with 'samples' key.
            expr: Multiline symbolic expression to evaluate.

        Returns:
            - out: LATENT dict with 'samples' key containing result tensor.

        Notes:
            - Supports operations: +, -, *, /, **, //, %, @, etc.
            - Supports functions: mean, std, max, min, relu, tanh, etc.
            - Supports slicing: a[:, :, :10, :10]
        """
        if isinstance(tensor, list):
            tensors = [t["samples"] for t in tensor]
        else:
            tensors = [tensor["samples"]]

        names = generate_symbol_names(len(tensors))
        variables = dict(zip(names, tensors))

        parser = TorchParser() if TorchSymbolic._parser is None else TorchSymbolic._parser
        TorchSymbolic._parser = parser

        result = parser.compute(expr, variables)

        return ({"samples": result},)
