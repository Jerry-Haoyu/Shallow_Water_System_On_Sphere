import torch

def l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    # integrate_grid reduces (nlat, nlon) only and keeps the channel axis
    # (..., channel) - left alone here (no .sum(dim=-1)) so geopotential/
    # vorticity/divergence, which can differ by orders of magnitude, don't
    # get combined into one norm before being weighted (see
    # spectral_l2loss_sphere below for the same reasoning).
    loss = solver.integrate_grid((prd - tar)**2, dimensionless=True)  # (..., channel)

    if relative:
        # each channel normalized by its OWN target norm - an automatic,
        # magnitude-agnostic per-channel weighting.
        tar_norm = solver.integrate_grid(tar**2, dimensionless=True)  # (..., channel)
        loss = loss / tar_norm.clamp_min(1e-12)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()  # average over batch AND channel, so each channel counts equally

    return loss

def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    # compute coefficients, keeping the channel axis separate from the degree
    # axis (summed below) so geopotential/vorticity/divergence - which can
    # differ by orders of magnitude depending on the input normalization -
    # don't get combined into one norm before being weighted.
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=-1)  # (..., channel)

    if relative:
        # each channel normalized by its OWN target spectral norm - an
        # automatic, magnitude-agnostic per-channel weighting, so e.g.
        # geopotential's larger physical scale can't swamp vorticity/
        # divergence in the combined loss.
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=-1)  # (..., channel)
        loss = loss / tar_norm2.clamp_min(1e-12)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()  # average over batch AND channel, so each channel counts equally

    return loss


# loss_type -> callable, keyed by the same short name used in the
# neural-operator checkpoint tree's loss_<> node (see
# src/helpers/run_model.py's _neural_config_nodes/neural_model_path).
LOSS_FUNCTIONS = {
    "grid": l2loss_sphere,
    "spectral": spectral_l2loss_sphere,
}