import torch


# ---------------------------------------------------------------------------
# Potential-vorticity mass-conservation diagnostic
# ---------------------------------------------------------------------------
# For an inviscid (geostrophic) shallow-water flow, potential vorticity q is
# materially conserved: Dq/Dt = 0, so fluid parcels never cross PV contours.
# A region of fluid whose PV lies in a fixed band [q_a, q_b) is therefore a
# *material* region, and by mass conservation the mass it carries is invariant
# in time.  We exploit this: partition the domain into PV level sets defined by
# the *initial* frame, track the mass in each level set, and measure how far it
# drifts from its initial value.  A perfect solver gives zero error.
# ---------------------------------------------------------------------------


def _as_tensor(frame):
    """Accept either a torch tensor or a numpy array (as returned by
    data.get_var_field) and return a torch tensor."""
    if isinstance(frame, torch.Tensor):
        return frame
    return torch.as_tensor(frame)


def _cos_lat_weight(lats, nlon, dtype, device):
    """Build a (nlat, nlon) area weight ∝ cos(lat) for a lat-lon grid.

    On an equiangular grid the cell area is dA = cos(lat)·Δlat·Δlon; the constant
    Δlat·Δlon cancels in the normalized error, so only the cos(lat) factor (which
    varies with row) is kept here. `lats` is the latitude of each row in radians,
    as returned by solver.lats / data.get_var_field.
    """
    lats = _as_tensor(lats).to(dtype=dtype, device=device).reshape(-1)
    w = torch.cos(lats).clamp_min(0.0)            # (nlat,)
    return w.unsqueeze(1).expand(-1, nlon)        # (nlat, nlon)


def __get_cutoff(initial_vort_frame, N_c):
    """Compute the bin edges that split the vorticity range into N_c + 1 level sets.

    N_c contours partition the range [q_min, q_max] into N_c + 1 level sets, which
    requires N_c + 2 equally spaced edges.

    Args :
        N_c : number of contours
        initial_vort_frame: A tensor of shape nlats * nlons
    Return:
        A 1-D tensor of N_c + 2 monotonically increasing cutoffs (bin edges),
        cutoffs[0] == q_min and cutoffs[-1] == q_max.
    """
    initial_vort_frame = _as_tensor(initial_vort_frame)
    q_max, q_min = torch.max(initial_vort_frame), torch.min(initial_vort_frame)
    if torch.isnan(q_max) or torch.isnan(q_min):
        raise RuntimeError("NaN encountered in partitioning relative vorticity into level sets")
    delta_q = (q_max - q_min) / (N_c + 1)
    if delta_q == 0:
        raise RuntimeError("constant vorticity field: cannot partition into level sets")
    # N_c + 1 equal-width level sets  ->  N_c + 2 edges spanning [q_min, q_max].
    ks = torch.arange(N_c + 2, device=initial_vort_frame.device, dtype=initial_vort_frame.dtype)
    cutoffs = q_min + ks * delta_q
    return cutoffs


def __partition__(vort_frame, cutoffs):
    """Given a gridded potential vorticity, partition by the cutoffs

    Args:
        vort_frame : A tensor of shape nlats * nlons
        cutoffs : The cutoff values (bin edges) of potential vorticity between
            different level sets, as produced by __get_cutoff on the *initial*
            frame (the bins must stay fixed in time so masses are comparable).
    Return:
        A list level_sets of length (N_c + 1), level_sets[i] is a tensor
        [(lat_j, lon_j)]_j of shape (M_i, 2) holding the grid coordinates of all
        the voxels in that level set. The level set straddling zero potential
        vorticity is omitted (set to None) so it is skipped consistently across
        every frame.
    """
    vort_frame = _as_tensor(vort_frame)
    n_bins = len(cutoffs) - 1

    # bin index of every voxel: value v in [cutoffs[i], cutoffs[i+1]) -> i.
    # right=True gives  cutoffs[i-1] <= v < cutoffs[i] -> i, so subtract 1.
    idx = torch.bucketize(vort_frame, cutoffs, right=True) - 1
    idx = idx.clamp_(0, n_bins - 1)  # fold the q_min / q_max endpoints into the edge bins

    # Identify the level set that contains q = 0, but only if 0 is actually
    # inside the partitioned range (otherwise nothing is omitted).
    zero_bin = -1
    if cutoffs[0] <= 0 <= cutoffs[-1]:
        zero_bin = int(torch.bucketize(torch.zeros(1, dtype=cutoffs.dtype, device=cutoffs.device),
                                       cutoffs, right=True).item()) - 1
        zero_bin = min(max(zero_bin, 0), n_bins - 1)

    level_sets = []
    for b in range(n_bins):
        if b == zero_bin:
            level_sets.append(None)            # omit the zero-PV level set, keep alignment
            continue
        coords = torch.nonzero(idx == b, as_tuple=False)  # (M, 2): (lat_idx, lon_idx)
        level_sets.append(coords)
    return level_sets


def __compute_mass__(vort_frame, height_frame, cutoffs, area_weight=None):
    """ Compute a list mass, where mass[j] gives the mass of level set j:
            ∑_{(x,y)⋲level[j]} height_frame[x,y]

    Args:
        vort_frame : A tensor of shape nlats * nlons giving potential vorticity
        height_frame : A tensor of shape nlats * nlons giving height
        cutoffs : The fixed bin edges from __get_cutoff (initial frame).
        area_weight : Optional tensor of shape nlats * nlons giving the area of
            each grid cell. On a lat-lon grid the physical mass is ∫ h dA, so
            pass cos(lat)·Δlat·Δlon here for the true spherical mass. When None
            the unweighted sum of heights is used (as written in the spec).
    Return:
        A 1-D tensor of length (N_c + 1); the omitted zero-PV level set holds 0.
    """
    height_frame = _as_tensor(height_frame)
    level_sets = __partition__(vort_frame, cutoffs)

    if area_weight is not None:
        area_weight = _as_tensor(area_weight)
        contribution = height_frame * area_weight
    else:
        contribution = height_frame

    mass = torch.zeros(len(level_sets), dtype=contribution.dtype, device=contribution.device)
    for j, coords in enumerate(level_sets):
        if coords is None or coords.numel() == 0:
            continue
        mass[j] = contribution[coords[:, 0], coords[:, 1]].sum()
    return mass


def __compute_error__(current_mass, initial_mass, h_avg, A_dom):
    """ Compute the vorticity conservation error of the vort_frame
            ε = (1/(h_avg * A_dom)) * √( (1/(N_c+1)) ∑_j (m_j(t)-m_j(0))^2 )
        Do not count the level set with zero potential vorticity
    Args:
        current_mass : A tensor giving m_j(t) for all j
        initial_mass : A tensor giving m_j(0) for all j
        h_avg : domain-mean height
        A_dom : domain area
    """
    diff = current_mass - initial_mass
    rms = torch.sqrt(torch.sum(diff ** 2) / len(current_mass))
    return rms / (h_avg * A_dom)


def compute_conservation_error(vort_video, height_video, N_c, lats=None, h_avg=None,
                               A_dom=None, area_weight=None):
    """Compute the PV mass-conservation error for every frame of a trajectory.

    The PV level sets are defined once from the first frame and held fixed, so
    the per-level-set masses are directly comparable across time. Mass is area
    weighted by cos(lat) by default (built from `lats`), since on a lat-lon grid
    the conserved quantity is ∫ h dA and level sets migrate in latitude over
    time; an unweighted sum would report spurious error even for a perfect run.

    Args:
        vort_video   : sequence of (nlat, nlon) potential-vorticity frames;
                       frame 0 is the reference state.
        height_video : sequence of (nlat, nlon) height frames, aligned with
                       vort_video.
        N_c          : number of contours (=> N_c + 1 PV level sets).
        lats         : latitude of each grid row in radians (solver.lats /
                       data.get_var_field). Used to build the default cos(lat)
                       area weight. If omitted (and no area_weight given) the
                       mass falls back to an unweighted sum, with a warning.
        h_avg        : domain-mean height. Defaults to the (area-weighted) mean
                       of the initial height frame.
        A_dom        : domain area. Defaults to sum(area_weight) when weighting
                       is in effect, otherwise the number of grid cells.
        area_weight  : optional (nlat, nlon) cell-area tensor; overrides the
                       cos(lat) weight built from `lats`.
    Return:
        A 1-D tensor of length T with the conservation error of each frame
        (error[0] == 0 by construction).
    """
    if len(vort_video) != len(height_video):
        raise ValueError("vort_video and height_video must have the same number of frames")

    vort0 = _as_tensor(vort_video[0])
    height0 = _as_tensor(height_video[0])
    nlon = height0.shape[1]

    # Area weight by cos(lat) by default; an explicit area_weight wins.
    if area_weight is None:
        if lats is not None:
            area_weight = _cos_lat_weight(lats, nlon, height0.dtype, height0.device)
        else:
            print("warning: no `lats` given -> using unweighted mass; "
                  "level sets migrate in latitude, so this reports spurious error.")

    if h_avg is None:
        if area_weight is not None:
            h_avg = (height0 * area_weight).sum() / area_weight.sum()
        else:
            h_avg = height0.mean()
    if A_dom is None:
        A_dom = area_weight.sum() if area_weight is not None else float(height0.numel())

    cutoffs = __get_cutoff(vort0, N_c)
    initial_mass = __compute_mass__(vort0, height0, cutoffs, area_weight)

    errors = []
    for vort_frame, height_frame in zip(vort_video, height_video):
        current_mass = __compute_mass__(vort_frame, height_frame, cutoffs, area_weight)
        errors.append(__compute_error__(current_mass, initial_mass, h_avg, A_dom))
    return torch.stack(errors)
