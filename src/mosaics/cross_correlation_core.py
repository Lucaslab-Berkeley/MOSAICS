"""Cross-correlation functions for the MOSAICS package."""

import torch

from .projection import TemplateProjector, slices_to_projections


def _cross_correlate_different_size(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    projections: torch.Tensor,  # (N, h, w)
) -> torch.Tensor:  # (N, H-h+1, W-w+1)
    """Cross-correlate particle images against smaller (centered) templates."""
    image_h, image_w = particle_stack_images.shape[-2:]
    template_h, template_w = projections.shape[-2:]
    valid_slice = slice(0, image_h - template_h + 1), slice(0, image_w - template_w + 1)

    particle_stack_images_fft = torch.fft.rfftn(particle_stack_images, dim=(-2, -1))
    projections_fft = torch.fft.rfftn(projections, dim=(-2, -1), s=(image_h, image_w))

    cross_correlation_fft = particle_stack_images_fft * projections_fft.conj()
    cross_correlation = torch.fft.irfftn(cross_correlation_fft, dim=(-2, -1))

    return cross_correlation[:, valid_slice[0], valid_slice[1]]


def _correlate(
    particle_stack_images: torch.Tensor,  # (n, H, W)
    projections: torch.Tensor,  # (n, h, w)
) -> torch.Tensor:  # (n,) or (n, H-h+1, W-w+1)
    """Inner products <I, t> (same size) or valid cross-correlations (larger images).

    Parameters
    ----------
    particle_stack_images : torch.Tensor
        The stack of pre-filtered particle images with shape (N, H, W).
    projections : torch.Tensor
        Filtered real-space template projections (centered), shape (N, h, w),
        e.g. from `project_filtered_templates`.

    Returns
    -------
    torch.Tensor
        The cross-correlation values, shape (N,) when the images and templates have
        the same size, otherwise the valid correlation maps (N, H - h + 1, W - w + 1).
    """
    if particle_stack_images.shape[-2:] == projections.shape[-2:]:
        return torch.sum(particle_stack_images * projections, dim=(-2, -1))
    return _cross_correlate_different_size(particle_stack_images, projections)


def _correlation_shape(
    particle_stack_images: torch.Tensor, template_shape: tuple[int, int]
) -> tuple[int, ...]:
    """Get shape of resultant cross-correlation output tensor."""
    num_particles, image_h, image_w = particle_stack_images.shape
    template_h, template_w = template_shape
    if (image_h, image_w) == (template_h, template_w):
        return (num_particles,)
    return (num_particles, image_h - template_h + 1, image_w - template_w + 1)


def project_filtered_templates(
    projector: TemplateProjector,
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w // 2 + 1)
    atom_indices: torch.Tensor | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:  # (N, h, w)
    """Real-space filtered template projections at each particle pose.

    These are exactly the templates whose inner products with the particle images give
    the MOSAICS cross-correlations, so any inner products between them (e.g. the
    template Gram matrix) are consistent with the correlation values.

    Parameters
    ----------
    projector : TemplateProjector
        Analytic projector holding the template atoms.
    rotation_matrices : torch.Tensor
        Per-particle rotation matrices, shape (N, 3, 3).
    projective_filters : torch.Tensor
        Per-particle projective filters, shape (N, h, w // 2 + 1).
    atom_indices : torch.Tensor | None, optional
        Atoms to project. If None (default), all atoms of the projector.
    batch_size : int, optional
        Number of particles processed at once. Default is 1024. If -1, all at once.

    Returns
    -------
    torch.Tensor
        The filtered real-space projections, shape (N, h, w).
    """
    num_particles = rotation_matrices.shape[0]
    template_h, template_w = projector.template_shape
    if batch_size == -1:
        batch_size = num_particles

    projections = torch.zeros(
        (num_particles, template_h, template_w), device=projective_filters.device
    )
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        projections[batch_slice] = projector.projections(
            rotation_matrices[batch_slice],
            projective_filters[batch_slice],
            atom_indices,
        ).to(projections.device)

    return projections


def cross_correlate_particle_stack(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    projections: torch.Tensor,  # (N, h, w)
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against per-particle templates.

    Parameters
    ----------
    particle_stack_images : torch.Tensor
        The stack of pre-filtered particle images with shape (N, H, W).
    projections : torch.Tensor
        Filtered real-space template projections (centered), shape (N, h, w), e.g.
        from `project_filtered_templates`.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        If -1, then the entire stack will be cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation values, shape (N,) when the images and templates have
        the same size, otherwise the valid correlation maps (N, H - h + 1, W - w + 1).
    """
    num_particles = particle_stack_images.shape[0]
    if batch_size == -1:
        batch_size = num_particles

    out = torch.zeros(
        _correlation_shape(particle_stack_images, projections.shape[-2:]),
        device=particle_stack_images.device,
    )
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        out[batch_slice] = _correlate(
            particle_stack_images[batch_slice], projections[batch_slice]
        )

    return out


# pylint: disable=too-many-locals,too-many-arguments
def alternate_template_statistics(
    particle_stack_images: torch.Tensor,  # (N, H, W)
    default_projections: torch.Tensor,  # (N, h, w)
    projector: TemplateProjector,
    removed_atom_indices: torch.Tensor,
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w // 2 + 1)
    batch_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """Correlation and template Gram entries for one alternate template.

    With t the default template, h the removed structure and t* = t - h the
    alternate template (all filtered projections at each particle's pose), this
    computes, per particle,

    - ``cross_correlation``: c* = <I, t*>
    - ``alternate_overlap``: <t, t*>
    - ``alternate_autocorrelation``: s_{t*} = ||t*||^2
    - ``removed_autocorrelation``: s_h = ||h||^2
    - ``alternate_removed_overlap``: s_{t*h} = <t*, h>

    The last three fully describe the bivariate Gaussian model of (c, c*), since
    s_t = s_{t*} + 2 s_{t*h} + s_h and <t, t*> = s_{t*} + s_{t*h}. Only the removed
    atoms are projected; t* follows exactly from linearity of the projection.

    Parameters
    ----------
    particle_stack_images : torch.Tensor
        Pre-filtered particle images, shape (N, H, W).
    default_projections : torch.Tensor
        Filtered projections of the default template from
        `project_filtered_templates`, shape (N, h, w).
    projector : TemplateProjector
        The analytic projector used for the default projections.
    removed_atom_indices : torch.Tensor
        Indices of the atoms removed from the default template.
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
    num_particles = particle_stack_images.shape[0]
    if batch_size == -1:
        batch_size = num_particles

    out = {
        "cross_correlation": torch.zeros(
            _correlation_shape(particle_stack_images, default_projections.shape[-2:]),
            device=device,
        )
    }
    for key in (
        "alternate_overlap",
        "alternate_autocorrelation",
        "removed_autocorrelation",
        "alternate_removed_overlap",
    ):
        out[key] = torch.zeros(num_particles, device=device)

    dims = (-2, -1)
    for i in range(0, num_particles, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_particles))
        rem_proj = slices_to_projections(
            projector.fourier_slices(
                rotation_matrices[batch_slice],
                projective_filters[batch_slice],
                removed_atom_indices,
            )
        ).to(device)
        t_proj = default_projections[batch_slice]
        alt_proj = t_proj - rem_proj

        out["cross_correlation"][batch_slice] = _correlate(
            particle_stack_images[batch_slice], alt_proj
        )
        out["alternate_overlap"][batch_slice] = torch.sum(t_proj * alt_proj, dim=dims)
        out["alternate_autocorrelation"][batch_slice] = torch.sum(alt_proj**2, dims)
        out["removed_autocorrelation"][batch_slice] = torch.sum(rem_proj**2, dim=dims)
        out["alternate_removed_overlap"][batch_slice] = torch.sum(
            alt_proj * rem_proj, dim=dims
        )

    return out
