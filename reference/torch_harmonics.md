# torch_harmonics reference

General, non-task-specific notes on the `torch_harmonics` package (v0.8.0),
gathered by reading its source. Written so future sessions can look things up
here instead of re-reading the library. All facts below were verified
numerically, not just read off docstrings.

Package: `import torch_harmonics as harmonics`. Top-level exports include
`RealSHT`, `InverseRealSHT`, `RealVectorSHT`, `InverseRealVectorSHT`,
`DiscreteContinuousConvS2`, `AttentionS2`, `ResampleS2`, plus submodules
`quadrature`, `legendre`, `random_fields`, `filter_basis`, `resample`,
`attention`, `convolution`.

## Grids and quadrature (`torch_harmonics.quadrature`)

Three supported grids, each with its own quadrature rule and default `lmax`
when not given explicitly to a transform:

| grid            | weights function             | default lmax (if omitted) |
|-----------------|-------------------------------|----------------------------|
| `equiangular`   | `clenshaw_curtiss_weights`    | `nlat`                     |
| `legendre-gauss`| `legendre_gauss_weights`      | `nlat`                     |
| `lobatto`       | `lobatto_weights`             | `nlat - 1`                 |

Each `*_weights(n, a, b)` returns `(cost, weights)`: `cost` are the `n`
quadrature nodes in `cos(colatitude)` space over `[a, b]` (pass `-1, 1`), and
`weights` are the matching integration weights such that
`sum(f(cost_i) * weights_i) ≈ ∫_{-1}^{1} f(x) dx`.

To turn nodes into a latitude array: `lats = -arcsin(cost)` (after flipping
node order, as the transforms internally do via `torch.flip`). Longitudes are
a plain uniform grid of `nlon` points spanning `[0, 2*pi)`.

Clenshaw-Curtiss (equiangular) quadrature is only formally exact up to degree
`nlat`, but is empirically accurate well beyond that in this package.

## `RealSHT` / `InverseRealSHT` (scalar transform pair)

```python
RealSHT(nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True)
InverseRealSHT(nlat, nlon, lmax=None, mmax=None, grid="equiangular", norm="ortho", csphase=True)
```

- Both are `nn.Module`s with only non-persistent buffers (precomputed
  associated-Legendre/quadrature weight tensors) — no learnable parameters.
  Cheap-ish to construct on the fly, but not free (building the weight
  tensors involves nontrivial precomputation), so prefer constructing once
  and reusing over a hot loop rather than rebuilding per call.
- `mmax` defaults to `nlon // 2 + 1` (all real-FFT bins) if not given; this
  codebase always passes `mmax = lmax` explicitly for a triangular spectral
  truncation independent of grid resolution.
- `norm="ortho"` (the default) means the real spherical harmonics `Y_l^m` are
  **orthonormal**: `∫ Y_l^m Y_l'^m'^* dΩ = δ_ll' δ_mm'` over the unit sphere.
- `csphase` toggles the Condon-Shortley phase `(-1)^m` in the associated
  Legendre functions. Two transform pairs must agree on `csphase` (and
  `grid`) to be inverses of each other / to interoperate — a mismatch
  silently flips the sign of odd-`m` coefficients rather than erroring.
- `RealSHT.forward(x)`: `x` has shape `(..., nlat, nlon)` (real). Internally:
  `rfft` along longitude (`torch.fft.rfft(..., norm="forward")` scaled by
  `2*pi`), then a Legendre contraction. Returns complex tensor of shape
  `(..., lmax, mmax)` — **axis order is `(l, m)`**, degree first, order
  second (don't assume it matches the internal `(m, l, k)`-shaped weight
  buffer's axis order).
- `InverseRealSHT.forward(x)`: `x` has shape `(..., lmax, mmax)` (complex),
  returns `(..., nlat, nlon)` (real). Explicitly zeroes the imaginary part of
  the DC (`m=0`) and Nyquist bins before `irfft`, since real fields require
  those.
- Only non-negative `m` (`0 .. mmax-1`) are stored; negative-`m` coefficients
  are implicit via the real-field conjugate-symmetry relation. Coefficients
  with `m > l` are **not meaningful** (no such spherical harmonic exists) —
  code that assembles/edits a spectral tensor by hand should `torch.tril()`
  it to zero out the `m > l` entries.
- **The `m=0` column must be real** (zero imaginary part) for the tensor to
  represent an actual real field; verified experimentally that leaving a
  stray imaginary part on `m=0` desyncs the reconstructed grid field from
  what Parseval's theorem (below) predicts.

### Useful closed-form identities (orthonormal real SHT, verified numerically)

Let `a_lm` be the spectral coefficients of a real, band-limited field `u`
(shape `(lmax, mmax)`, complex, `m=0` column real, `m>l` entries zero).

- **Domain mean** (sphere-average of `u`), no inverse transform needed:
  `mean(u) = a_{0,0}.real / sqrt(4*pi)`
  (since `Y_0^0 = 1/sqrt(4*pi)` is the constant function).
- **Mean square / Parseval's theorem**:
  `mean(u**2) = (1/(4*pi)) * ( sum_l |a_{l,0}|**2 + 2 * sum_{l, m>0} |a_{l,m}|**2 )`
  i.e. sum every `|a_lm|^2`, counting `m=0` once and each `m>0` column
  **twice** (it stands in for the `+m`/`-m` conjugate pair). This is exact
  for a band-limited field, not a quadrature approximation — cheaper and
  more accurate than reconstructing the grid and integrating.
- **Variance**: `var(u) = mean(u**2) - mean(u)**2`; the `l=0` term cancels
  exactly between the two, so equivalently `var(u)` is the same double-sum
  above restricted to `l >= 1`.

These identities are why, e.g., a domain-mean and an RMS-deviation-from-mean
diagnostic can both be read directly off spectral coefficients without ever
calling `InverseRealSHT`.

## `RealVectorSHT` / `InverseRealVectorSHT` (vector transform pair)

Same constructor signature as the scalar pair. Operates on a **stacked
2-component field**, e.g. `(u, v)` wind:

- `RealVectorSHT.forward(x)`: `x` shape `(..., 2, nlat, nlon)` →
  `(..., 2, lmax, mmax)` complex.
- `InverseRealVectorSHT.forward(x)`: the reverse.
- The 2 output/input channels are the **spheroidal** (curl-free / potential,
  i.e. divergence-related) and **toroidal** (divergence-free / rotational,
  i.e. vorticity-related) parts of the vector field's harmonic decomposition
  — channel 0 is spheroidal, channel 1 is toroidal. This is the standard
  Helmholtz-type decomposition of a tangent vector field on the sphere, not
  a per-component (x, y) transform.
- To get actual vorticity/divergence spectra from a raw `(u, v)` field, the
  spheroidal/toroidal output still needs the eigenvalue of the surface
  Laplacian applied (`-l(l+1)/radius**2`, standard spherical-harmonic
  identity, not something the library does for you) — see how this repo's
  solver builds `vrtdivspec` from `RealVectorSHT` output for a worked
  example, but the scaling itself is general spherical-harmonics theory, not
  torch_harmonics-specific behavior.

## Dtype / device notes

- Real-space tensors are float32/float64; spectral tensors are the matching
  complex dtype (`complex64`/`complex128`) — the forward transform builds
  this via `torch.view_as_real` / `torch.view_as_complex` internally.
- All four transform classes are ordinary `nn.Module`s, so `.to(device)`
  moves their (non-persistent) buffers; nothing else is needed to run them
  on GPU.
