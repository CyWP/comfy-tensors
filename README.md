# Comfy Tensors

Tensor operations, format conversions (IMAGE/LATENT/MASK), aggregation utilities, and symbolic expression evaluation for ComfyUI.

---

## Installation

```bash
comfy node install tensors
```
or find in extensions marketplace.

---

## Nodes

### Format Conversion

#### Image2Latent

Converts IMAGE tensors from BHWC to LATENT format BCHW.

**Inputs:**
- `tensor`: IMAGE tensor in BHWC format (or HWC for single image)
- `permute`: Whether to permute BHWC → BCHW (default: True)

**Outputs:**
- `LATENT` dict with `samples` key

---

#### Latent2Image

Converts LATENT tensors from BCHW to IMAGE format BHWC.

**Inputs:**
- `latent`: LATENT dict(s) with `samples` key
- `permute`: Whether to permute BCHW → BHWC (default: True)

**Outputs:**
- `IMAGE` tensor in BHWC format

---

#### Image2Mask

Extracts a mask from an IMAGE tensor by reducing across channels.

**Inputs:**
- `image`: IMAGE tensor in BHWC format
- `reduction`: Channel to extract — `r`, `g`, `b`, `a`, or `mean` (default: `mean`)
- `permute`: Whether to permute BHWC → BCHW (default: True)

**Outputs:**
- `IMAGE` tensor with single channel

---

#### Mask2Image

Expands a mask tensor to an IMAGE with configurable channels.

**Inputs:**
- `mask`: Mask tensor (2D, 3D, or 4D)
- `num_channels`: Number of channels in output (default: 1)
- `permute`: Whether to permute BCHW → BHWC (default: True)

**Outputs:**
- `IMAGE` tensor with `num_channels` channels

---

#### Latent2Mask

Converts LATENT samples to IMAGE mask format.

**Inputs:**
- `latent`: LATENT dict(s) with `samples` key
- `permute`: Whether to permute BCHW → BHWC (default: True)

**Outputs:**
- `IMAGE` tensor in BHWC format

---

#### Mask2Latent

Converts a mask tensor to LATENT samples format.

**Inputs:**
- `mask`: Mask tensor (2D, 3D, or 4D)
- `permute`: Whether to permute BHWC → BCHW (default: True)

**Outputs:**
- `LATENT` dict with `samples` key

---

### Aggregation

#### Concatenate

Concatenates a list of tensors along a specified dimension.

**Inputs:**
- `tensors`: List of tensors to concatenate
- `dim`: Dimension along which to concatenate (default: 0)

**Outputs:**
- `TENSOR`: Concatenated tensor

**Notes:**
- All tensors must have the same number of dimensions.

---

#### Stack

Stacks a list of tensors along a new dimension.

**Inputs:**
- `tensors`: List of tensors to stack (all must have identical shapes)
- `dim`: Dimension along which to stack (default: 0)

**Outputs:**
- `TENSOR`: Stacked tensor

**Notes:**
- All tensors must have identical shapes.

---

### Debug

#### Inspect

Returns statistics for a LATENT tensor as a string.

**Inputs:**
- `tensor`: LATENT dict(s) with `samples` key

**Outputs:**
- `STRING` containing: shape, dtype, device, min, max, mean, std, sum, norm, requires_grad

---

### Symbolic Expression

#### Symbolic Parser

Evaluates symbolic expressions on LATENT tensors.

**Inputs:**
- `tensor`: LATENT dict(s) with `samples` key
- `expr`: Multiline string expression to evaluate

**Outputs:**
- `LATENT` dict with `samples` key containing the result

---

## Symbolic Parser Usage

Input tensors are automatically assigned variable names: `a`, `b`, `c`, ... `z`, `aa`, `ba`, ...

### Operators

| Operator | Description |
|----------|-------------|
| `+`, `-`, `*`, `/` | Add, subtract, multiply, divide |
| `//`, `%` | Floor division, modulo |
| `**` | Power |
| `@` | Matrix multiplication |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Comparison |
| `&`, `\|`, `^` | Logical and, or, xor |
| `neg`, `not` | Unary negate, invert |

### Functions

**Math:**
`abs`, `sin`, `cos`, `tan`, `exp`, `log`, `sig`, `tanh`, `relu`, `soft`, `normalize`, `norm`, `mean`, `std`, `var`, `max`, `min`, `argmax`, `argmin`, `clamp`, `floor`, `ceil`, `round`

**Shape:**
`T`, `transpose`, `permute`, `reshape`, `view`, `flatten`, `unsqueeze`, `squeeze`, `repeat`, `expand`, `chunk`, `split`

**Combinators:**
`cat`, `stack`

**Indexing:**
`idx(x, i)` — index at i
`idx(x, (start, stop))` — slice
`idx(x, (start, stop, step))` — strided slice

**Constants:**
`pi`, `e`, `eps` (1e-8)

### Examples

**Add two latents:**
```
a + b
```

**Blend with weight:**
```
a * 0.5 + b * 0.5
```

**Activation:**
```
relu(a - b)
```

**Normalize:**
```
normalize(a)
```

**Slice first 10 along dim 2:**
```
idx(a, (0, 10), :, :)
```

**Reshape to flatten:**
```
flatten(reshape(a, 1, -1))
```

**Tile a slice:**
```
repeat(unsqueeze(idx(a, 0, :, :), 0), 1, 1, 1, 3)
```

### Complex Examples

**Standardize to [-1, 1]:**
```
(a - mean(a)) / (std(a) + eps) * 0.5
```

**Soft blend three latents:**
```
soft(cat(a, b, c, dim=1)) * cat(a, b, c, dim=1)
```

**Residual with activation:**
```
a + relu(a - b) * 0.1
```

**Tanh gating:**
```
a * tanh(b)
```

**Sigmoid mask:**
```
a * sig(b) * 2 - 1
```

**Split, scale, recombine:**
```
cat(chunk(a, 2, 0)[0] * 2, chunk(a, 2, 0)[1] * 0.5, dim=0)
```

**Attention-like:**
```
normalize(a) @ transpose(normalize(b), 0, 1)
```

**Chained shape transforms:**
```
flatten(view(permute(a, 0, 2, 3, 1), 0, -1))
```

**Broadcast multiply:**
```
a * reshape(b, (1, -1, 1, 1))
```