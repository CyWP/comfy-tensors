import torch
import torch.nn.functional as F

try:
    from .parser import Parser
except:
    from parser import Parser


def idx(x, *indices):
    """
    Parse tuples for indexing.
    """
    parsed = []

    for i in indices:
        if isinstance(i, tuple):
            # interpret tuple as slice(start, stop, step)
            parsed.append(slice(*i))
        elif i is None:
            parsed.append(slice(i))
        else:
            # integer / tensor / boolean mask
            parsed.append(i)
    print("Parsed: ", parsed)
    return x[tuple(parsed)]


class TorchParser(Parser):

    _torch_functions = {
        # Mathematical functions
        "abs": lambda x: x.abs(),
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "soft": lambda x: F.softmax(x, dim=-1),
        "sig": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "normalize": lambda x: F.normalize(x, dim=-1),
        "norm": lambda x: x.norm(),
        "max": lambda x: x.max(),
        "maximum": lambda x, y: torch.maximum(x, y),
        "min": lambda x: x.min(),
        "minimum": lambda x, y: torch.minimum(x, y),
        "argmax": lambda x: x.argmax(),
        "argmin": lambda x: x.argmin(),
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "var": lambda x: x.var(),
        "floor": lambda x: x.floor(),
        "ceil": lambda x: x.ceil(),
        "round": lambda x: x.round(),
        "clamp": lambda x, a, b: torch.clamp(x, min=a, max=b),
        "clip": lambda x, a, b: torch.clamp(x, min=a, max=b),
        # Shape ops
        "T": lambda x: x.T,
        "transpose": lambda x, dim0, dim1: x.transpose(dim0, dim1),
        "permute": lambda x, *dims: x.permute(*dims),
        "reshape": lambda x, *dims: x.reshape(*dims),
        "view": lambda x, *dims: x.view(*dims),
        "flatten": lambda x, start_dim=0, end_dim=-1: x.flatten(
            start_dim=start_dim, end_dim=end_dim
        ),
        "unsqueeze": lambda x, dim: x.unsqueeze(dim),
        "squeeze": lambda x, dim=None: x.squeeze() if dim is None else x.squeeze(dim),
        "repeat": lambda x, *dims: x.repeat(*dims),
        "expand": lambda x, *dims: x.expand(*dims),
        "chunk": lambda x, chunks, dims=None: x.chunk(x, chunks, dims),
        "split": lambda x, i, dim=0: x.tensor_split(x, i, dim),
        # combinations
        "cat": lambda *args: torch.cat(args[:-1], dim=args[-1]),
        "stack": lambda *args: torch.stack(args[:-1], dim=args[-1]),
        # conversions
        "float": lambda x: x.float(),
        "half": lambda x: x.half(),
        "long": lambda x: x.long(),
        "bool": lambda x: x.bool(),
        "requires_grad": lambda x, val: x.requires_grad(val),
        # Indexing
        "idx": idx,
    }

    def __init__(self):
        super().__init__(functions=TorchParser._torch_functions)


if __name__ == "__main__":
    vars = {
        "x": torch.randn((3, 3, 3)),
        "y": torch.zeros((3, 3, 3)),
    }
    expr = "(x + y)**2/x-x + repeat(unsqueeze(idx(y, 0, :, :), 0), 3, 1, 1)"
    parser = TorchParser()
    print(parser.compute(expr, vars))
    print("done, Should be very close to zero.")
