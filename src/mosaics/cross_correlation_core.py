"""Cross-correlation functions for the MOSAICS package."""

import torch

# from leopard_em.backend.utils import normalize_template_projections
from torch_fourier_slice import extract_central_slices_rfft_3d


def _cross_correlate_different_size(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    fourier_slice: torch.Tensor,  # (N, h, w)
) -> torch.Tensor:  # (N, H-h+1, W-w+1)
    """Cross-correlate a stack of particle images against a template when the template is smaller than the images."""
    image_h, image_w = particle_stack_images.shape[-2:]
    template_h, template_w = fourier_slice.shape[-2], fourier_slice.shape[-1] * 2 - 2
    valid_slice = slice(0, image_h - template_h + 1), slice(0, image_w - template_w + 1)

    projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))

    particle_stack_images_fft = torch.fft.rfftn(particle_stack_images, dim=(-2, -1))
    fourier_slice = torch.fft.rfftn(projections, dim=(-2, -1), s=(image_h, image_w))

    cross_correlation_fft = particle_stack_images_fft * fourier_slice.conj()
    cross_correlation = torch.fft.irfftn(cross_correlation_fft, dim=(-2, -1))

    return cross_correlation[:, valid_slice[0], valid_slice[1]]


def _filtered_fourier_slices(
    template_dft: torch.Tensor,  # (d, h, w // 2 + 1)
    rotation_matrices: torch.Tensor,  # (n, 3, 3)
    projective_filters: torch.Tensor,  # (n, h, w // 2 + 1)
) -> torch.Tensor:  # (n, h, w // 2 + 1)
    """Filtered Fourier slices of a template, exactly as used for cross-correlation.

    The DC component is zeroed, the contrast is flipped, and the per-particle
    projective filters (whitening, CTF, ...) are applied. Every step is linear in
    the template, so slices of (t* + h) equal the sum of the slices of t* and h.
    """
    fourier_slice = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        rotation_matrices=rotation_matrices,
    )
    fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
    fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slice *= -1  # flip contrast
    fourier_slice *= projective_filters

    return fourier_slice


def _slices_to_projections(fourier_slice: torch.Tensor) -> torch.Tensor:
    """Convert filtered Fourier slices to real-space (centered) projections."""
    projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    return torch.fft.ifftshift(projections, dim=(-2, -1))


def project_filtered_templates(
    template_dft: torch.Tensor,  # (d, h, w // 2 + 1)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w // 2 + 1)
    batch_size: int = 1024,
) -> torch.Tensor:  # (N, h, w)
    """Real-space filtered projections of a template at each particle pose.

    These are exactly the templates whose inner products with the particle images
    give the MOSAICS cross-correlations, so any inner products between them (e.g.
    the template Gram matrix) are consistent with the correlation values.

    Parameters
    ----------
    template_dft : torch.Tensor
        Template volume, real-Fourier transformed and fftshifted (see
        `cross_correlate_particle_stack`).
    rotation_matrices : torch.Tensor
        Per-particle rotation matrices, shape (N, 3, 3).
    projective_filters : torch.Tensor
        Per-particle projective filters, shape (N, h, w // 2 + 1).
    batch_size : int, optional
        Number of projections to compute at once. Default is 1024.

    Returns
    -------
    torch.Tensor
        The filtered real-space projections, shape (N, h, w).
    """
    num_particles = rotation_matrices.shape[0]
    template_h = template_dft.shape[-2]
    template_w = 2 * (template_dft.shape[-1] - 1)
    if batch_size == -1:
        batch_size = num_particles

    projections = torch.zeros(
        (num_particles, template_h, template_w), device=template_dft.device
    )
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        fourier_slice = _filtered_fourier_slices(
            template_dft,
            rotation_matrices[batch_slice],
            projective_filters[batch_slice],
        )
        projections[batch_slice] = _slices_to_projections(fourier_slice)

    return projections


# pylint: disable=too-many-locals,too-many-arguments
def alternate_template_statistics(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    default_projections: torch.Tensor,  # (N, h, w)
    alternate_template_dft: torch.Tensor,  # (d, h, w // 2 + 1)
    removed_template_dft: torch.Tensor,  # (d, h, w // 2 + 1)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w // 2 + 1)
    batch_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """Correlation and template Gram entries for one alternate template.

    With t the default template, t* the alternate template and h = t - t* the
    removed structure (all filtered projections at each particle's pose), this
    computes, per particle,

    - ``cross_correlation``: c* = <I, t*>
    - ``alternate_overlap``: <t, t*>
    - ``alternate_autocorrelation``: s_{t*} = ||t*||^2
    - ``removed_autocorrelation``: s_h = ||h||^2
    - ``alternate_removed_overlap``: s_{t*h} = <t*, h>

    The last three fully describe the bivariate Gaussian model of (c, c*), since
    s_t = s_{t*} + 2 s_{t*h} + s_h and <t, t*> = s_{t*} + s_{t*h}.

    Parameters
    ----------
    particle_stack_images : torch.Tensor
        Pre-filtered particle images, shape (N, H, W).
    default_projections : torch.Tensor
        Filtered projections of the default template from
        `project_filtered_templates`, shape (N, h, w).
    alternate_template_dft : torch.Tensor
        The alternate (truncated) template volume, real-Fourier transformed and
        fftshifted.
    removed_template_dft : torch.Tensor
        The removed structure (default minus alternate) volume, real-Fourier
        transformed and fftshifted.
    rotation_matrices : torch.Tensor
        Per-particle rotation matrices, shape (N, 3, 3).
    projective_filters : torch.Tensor
        Per-particle projective filters, shape (N, h, w // 2 + 1).
    batch_size : int, optional
        Number of particles to process at once. Default is 1024.

    Returns
    -------
    dict[str, torch.Tensor]
        The statistics listed above. Each has shape (N,), except
        ``cross_correlation`` which is (N, H - h + 1, W - w + 1) when the images are
        larger than the templates.
    """
    device = particle_stack_images.device
    num_particles, image_h, image_w = particle_stack_images.shape
    template_h, template_w = default_projections.shape[-2:]
    _is_same_size = (image_h == template_h) and (image_w == template_w)
    if batch_size == -1:
        batch_size = num_particles

    if _is_same_size:
        cc_shape: tuple[int, ...] = (num_particles,)
    else:
        cc_shape = (num_particles, image_h - template_h + 1, image_w - template_w + 1)

    out = {"cross_correlation": torch.zeros(cc_shape, device=device)}
    for key in (
        "alternate_overlap",
        "alternate_autocorrelation",
        "removed_autocorrelation",
        "alternate_removed_overlap",
    ):
        out[key] = torch.zeros(num_particles, device=device)

    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        rotmats = rotation_matrices[batch_slice]
        filters = projective_filters[batch_slice]

        alt_slice = _filtered_fourier_slices(alternate_template_dft, rotmats, filters)
        rem_slice = _filtered_fourier_slices(removed_template_dft, rotmats, filters)
        alt_proj = _slices_to_projections(alt_slice)
        rem_proj = _slices_to_projections(rem_slice)
        t_proj = default_projections[batch_slice]

        if _is_same_size:
            out["cross_correlation"][batch_slice] = torch.sum(
                particle_stack_images[batch_slice] * alt_proj, dim=(-2, -1)
            )
        else:
            out["cross_correlation"][batch_slice] = _cross_correlate_different_size(
                particle_stack_images[batch_slice], alt_slice
            )

        dims = (-2, -1)
        out["alternate_overlap"][batch_slice] = torch.sum(t_proj * alt_proj, dim=dims)
        out["alternate_autocorrelation"][batch_slice] = torch.sum(alt_proj**2, dims)
        out["removed_autocorrelation"][batch_slice] = torch.sum(rem_proj**2, dim=dims)
        out["alternate_removed_overlap"][batch_slice] = torch.sum(
            alt_proj * rem_proj, dim=dims
        )

    return out


# pylint: disable=too-many-locals
def cross_correlate_particle_stack(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w)
    perfect_projection_images: torch.Tensor | None = None,  # (N, h, w)
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Cross-correlate a stack of particle images against a template.

    Parameters
    ----------
    particle_stack_images : torch.Tensor
        The stack of pre-filtered particle images with shape (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted with shape (D, H, W) in real-space (cubic).
    rotation_matrices : torch.Tensor
        The orientations of the particles to take the Fourier slices of, as a long
        list of rotation matrices. Shape of (N, 3, 3).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    perfect_projection_images : torch.Tensor | None, optional
        Expected perfect projection images for each particle. If provided, used to
        calculate the overlap between teh new template and the perfect projection.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation values for each particle image. Shape of (N,).
    """
    # Helpful constants for later use
    device = particle_stack_images.device
    num_particles, image_h, image_w = particle_stack_images.shape
    _, template_h, template_w = template_dft.shape
    template_w = 2 * (template_w - 1)

    # Can use faster approach if height and width are the same
    _is_same_size = (image_h == template_h) and (image_w == template_w)

    if batch_size == -1:
        batch_size = num_particles

    if _is_same_size:
        out_correlation = torch.zeros(num_particles, device=device)
        out_correlation_perfect = (
            torch.zeros(num_particles, device=device)
            if perfect_projection_images is not None
            else None
        )
    else:
        out_correlation = torch.zeros(
            (num_particles, image_h - template_h + 1, image_w - template_w + 1),
            device=device,
        )
        out_correlation_perfect = (
            torch.zeros(
                (num_particles, image_h - template_h + 1, image_w - template_w + 1),
                device=device,
            )
            if perfect_projection_images is not None
            else None
        )

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        batch_particles_images = particle_stack_images[batch_slice]
        batch_rotation_matrices = rotation_matrices[batch_slice]
        batch_projective_filters = projective_filters[batch_slice]

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = _filtered_fourier_slices(
            template_dft, batch_rotation_matrices, batch_projective_filters
        )

        if _is_same_size:
            projections = _slices_to_projections(fourier_slice)
            tmp = torch.sum(batch_particles_images * projections, dim=(-2, -1))
            if perfect_projection_images is not None:
                perfect_projections = perfect_projection_images[batch_slice]
                tmp_perfect = torch.sum(perfect_projections * projections, dim=(-2, -1))
        else:
            tmp = _cross_correlate_different_size(batch_particles_images, fourier_slice)
            if perfect_projection_images is not None:
                perfect_projections = perfect_projection_images[batch_slice]
                tmp_perfect = _cross_correlate_different_size(
                    perfect_projections, fourier_slice
                )

        out_correlation[batch_slice] = tmp
        if perfect_projection_images is not None:
            out_correlation_perfect[batch_slice] = tmp_perfect

    return out_correlation, out_correlation_perfect
