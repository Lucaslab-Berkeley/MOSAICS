"""Analytic template projections computed directly from atomic models.

Each template projection is computed by rotating the atoms to the particle pose and
analytically integrating their scattering potential along the beam direction with
torch-calculate-electrostatic-potential. This avoids simulating a 3D volume and
interpolating a central slice from its Fourier transform, which introduces errors of
several percent in the projected templates.

The physics matches ttsim3d (used previously to simulate template volumes):

1. Peng (1996) elastic scattering factors with per-atom B-factors.
2. Pixel-averaged potential on a grid upsampled by an integer factor.
3. Cumulative dose filter, Fourier cropping back to the template pixel size, and the
   detector MTF, applied in 2D (these filters are radial, so they are identical to
   the central slice of the corresponding 3D filters).
4. The same overall scale as ttsim3d volumes projected along the beam direction.

On CUDA devices the potential calculation is compiled with `torch.compile`, which
fuses its element-wise operations (about 4.5x faster, identical results up to float
rounding) without any changes to torch-calculate-electrostatic-potential.
"""

import math
import warnings
from typing import Annotated, Any, Callable

import torch
from pydantic import Field
from torch_calculate_electrostatic_potential import (
    PENG_SCATTERING_TO_POTENTIAL,
    GridConfig,
    calculate_scattering_potential_2d,
)
from torch_ctf import calculate_relativistic_electron_wavelength
from torch_fourier_filter.mtf import make_mtf_grid
from ttsim3d.grid_coords import get_upsampling
from ttsim3d.models import Simulator
from ttsim3d.scattering_potential import get_a_param, get_b_param
from ttsim3d.simulate3d import (
    BOND_SCALING_FACTOR,
    MAX_SIZE,
    calculate_simulation_dose_filter_3d,
)

# Default radius (Angstroms) of the per-atom pixel stencil.
DEFAULT_STENCIL_RADIUS = 3.5

# Warn when the stencil truncates more than this fraction of the template potential
MAX_TRUNCATED_FRACTION = 1e-3

# Default cap on the (particle, atom, stencil pixel) triples evaluated at once.
DEFAULT_MAX_BATCH_ELEMENTS = 2**24


class MosaicsSimulator(Simulator):
    """ttsim3d Simulator with the settings of the MOSAICS analytic projections.

    Attributes
    ----------
    stencil_radius : float
        Radius in Angstroms of the per-atom pixel stencil over which each atom's
        projected potential is evaluated. Default is 3.5 Angstroms.
    """

    stencil_radius: Annotated[float, Field(gt=0.0)] = DEFAULT_STENCIL_RADIUS


_COMPILED_POTENTIAL_2D: Callable[..., torch.Tensor] | None = None


def _compiled_potential_2d() -> Callable[..., torch.Tensor]:
    """Lazily compiled `calculate_scattering_potential_2d` (shared by projectors)."""
    global _COMPILED_POTENTIAL_2D  # pylint: disable=global-statement
    if _COMPILED_POTENTIAL_2D is None:
        _COMPILED_POTENTIAL_2D = torch.compile(
            calculate_scattering_potential_2d, dynamic=True
        )
    return _COMPILED_POTENTIAL_2D


def slices_to_projections(fourier_slice: torch.Tensor) -> torch.Tensor:
    """Convert filtered Fourier slices to real-space projections."""
    projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
    return torch.fft.fftshift(projections, dim=(-2, -1))


def _crop_rfft_2d(dft: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Fourier crop a (non-shifted) 2D rfft to a smaller (height, width) image."""
    num_pos = height - height // 2
    rows = torch.cat(
        [dft[..., :num_pos, :], dft[..., dft.shape[-2] - height // 2 :, :]], dim=-2
    )
    return rows[..., : width // 2 + 1]


class TemplateProjector:
    """Filtered template projections computed analytically from atoms.

    Parameters
    ----------
    atom_positions_zyx : torch.Tensor
        Atom positions in Angstroms, shape (N, 3). The origin is the template center
        (array index (h // 2, w // 2) of each projection).
    atom_ids : list[str]
        Uppercase element symbols of the atoms.
    atom_b_factors : torch.Tensor
        Per-atom B-factors in the convention of `ttsim3d.simulate3d.simulate3d`, i.e.
        as returned by `Simulator.get_scale_atom_b_factors`. Shape (N,).
    pixel_size : float
        Template pixel size in Angstroms.
    template_shape : tuple[int, int]
        Shape (h, w) of the template projections.
    voltage : float, optional
        Beam energy in keV, used for the overall potential scale. Default 300.
    upsampling : int, optional
        Integer upsampling factor of the projection grid before Fourier cropping. If
        -1 (default), chosen as in ttsim3d.
    apply_dose_weighting : bool, optional
        Whether to apply the cumulative dose filter. Default True.
    dose_start, dose_end : float, optional
        Exposure range in e-/A^2 for the dose filter. Default 0 and 30.
    dose_filter_modify_signal : str, optional
        Dose filter modification ('None', 'sqrt', 'rel_diff'). Default 'None'.
    dose_filter_critical_bfactor : float, optional
        Critical B-factor of the dose filter, -1 for critical exposure. Default -1.
    apply_dqe : bool, optional
        Whether to apply the detector MTF. Default False.
    mtf_frequencies, mtf_amplitudes : torch.Tensor | None, optional
        The MTF curve, required if `apply_dqe` is True.
    stencil_radius : float, optional
        Radius (Angstroms) of the per-atom pixel stencil. Default 3.5. A warning is
        raised if the stencil truncates more than `MAX_TRUNCATED_FRACTION` of the
        template's projected potential (see `truncated_fraction`).
    device : torch.device | str, optional
        Device for the computation. Default 'cpu'.
    max_batch_elements : int, optional
        Cap on (particle, atom, stencil pixel) triples evaluated at once; controls
        memory use. Default `DEFAULT_MAX_BATCH_ELEMENTS`.
    compile_potential : bool | None, optional
        Whether to compile the potential calculation with `torch.compile`. If None
        (default), compile on CUDA devices only (CPU compilation needs a C++
        toolchain and gives little benefit).
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        atom_positions_zyx: torch.Tensor,
        atom_ids: list[str],
        atom_b_factors: torch.Tensor,
        pixel_size: float,
        template_shape: tuple[int, int],
        voltage: float = 300.0,
        upsampling: int = -1,
        apply_dose_weighting: bool = True,
        dose_start: float = 0.0,
        dose_end: float = 30.0,
        dose_filter_modify_signal: str = "None",
        dose_filter_critical_bfactor: float = -1,
        apply_dqe: bool = False,
        mtf_frequencies: torch.Tensor | None = None,
        mtf_amplitudes: torch.Tensor | None = None,
        stencil_radius: float = DEFAULT_STENCIL_RADIUS,
        device: torch.device | str = "cpu",
        max_batch_elements: int = DEFAULT_MAX_BATCH_ELEMENTS,
        compile_potential: bool | None = None,
    ):
        self.device = torch.device(device)
        self.template_shape = (int(template_shape[0]), int(template_shape[1]))
        self.max_batch_elements = int(max_batch_elements)
        height, width = self.template_shape

        if upsampling == -1:
            upsampling = get_upsampling(pixel_size, max(height, width), MAX_SIZE)
        if upsampling < 1:
            raise ValueError("upsampling must be a positive integer or -1 (auto).")
        self.upsampling = int(upsampling)
        self.upsampled_shape = (height * self.upsampling, width * self.upsampling)
        upsampled_pixel = pixel_size / self.upsampling

        # Per-atom data
        self.positions_xyz = atom_positions_zyx.to(self.device).float().flip(-1)
        self.params_a = get_a_param(list(atom_ids)).to(self.device).float()
        self.params_b = get_b_param(list(atom_ids)).to(self.device).float()
        self.b_factors = atom_b_factors.to(self.device).float()

        if stencil_radius <= 0:
            raise ValueError("stencil_radius must be positive.")
        self.stencil_radius = float(stencil_radius)
        truncated = self.truncated_fraction() if self.num_atoms > 0 else 0.0
        if truncated > MAX_TRUNCATED_FRACTION:
            warnings.warn(
                f"stencil_radius={self.stencil_radius:.2f} A truncates "
                f"{100 * truncated:.2g}% of the template's projected "
                "potential; consider a larger radius.",
                UserWarning,
                stacklevel=2,
            )

        if compile_potential is None:
            compile_potential = self.device.type == "cuda"
        self._potential_2d = (
            _compiled_potential_2d()
            if compile_potential
            else calculate_scattering_potential_2d
        )

        # Grid with the origin at pixel (H // 2, W // 2) as in ttsim3d volumes
        up_h, up_w = self.upsampled_shape
        self.grid = GridConfig(
            grid_shape=self.upsampled_shape,
            voxel_size=(upsampled_pixel, upsampled_pixel),
            left_bottom_point=(
                -(up_h // 2) * upsampled_pixel,
                -(up_w // 2) * upsampled_pixel,
            ),
            right_upper_point=(
                (up_h - 1 - up_h // 2) * upsampled_pixel,
                (up_w - 1 - up_w // 2) * upsampled_pixel,
            ),
            sublattice_radius=self.stencil_radius,
            equal_length=False,
            device=self.device,
        )
        self.stencil_size = int(self.grid.stencil.coordinates.shape[0])

        # Scale so projections match the line integrals of ttsim3d volumes, whose
        # voxels hold BOND_SCALING_FACTOR * wavelength / pixel_size^2 times the
        # fraction of each atom's (normalized) potential inside the voxel.
        wavelength = float(calculate_relativistic_electron_wavelength(voltage * 1e3))
        ttsim3d_scale = BOND_SCALING_FACTOR * wavelength * 1e10 / pixel_size**2
        scale = ttsim3d_scale * upsampled_pixel**2 / PENG_SCATTERING_TO_POTENTIAL

        # Combined 2D filter on the template rfft grid
        if apply_dose_weighting:
            dose = calculate_simulation_dose_filter_3d(
                shape=self.upsampled_shape,
                pixel_spacing=upsampled_pixel,
                dose_start=dose_start,
                dose_end=dose_end,
                critical_bfactor=dose_filter_critical_bfactor,
                modify_signal=dose_filter_modify_signal,
                rfft=True,
                fftshift=False,
                device=self.device,
            )
            dose = _crop_rfft_2d(dose, height, width)
        else:
            dose = torch.ones(height, width // 2 + 1, device=self.device)

        if apply_dqe:
            if mtf_frequencies is None or mtf_amplitudes is None:
                raise ValueError("MTF frequencies and amplitudes needed for the DQE.")
            mtf = make_mtf_grid(
                image_shape=self.template_shape,
                mtf_frequencies=mtf_frequencies.to(self.device),
                mtf_amplitudes=mtf_amplitudes.to(self.device),
                rfft=True,
                fftshift=False,
                device=self.device,
            )
        else:
            mtf = torch.ones_like(dose)

        # Move the origin from upsampled pixel (H // 2, W // 2) to index 0. Applied
        # on the (integer) upsampled frequencies retained by the crop.
        ky = torch.fft.fftfreq(up_h, d=1 / up_h, device=self.device)[:, None]
        ky = _crop_rfft_2d(ky.expand(up_h, up_w // 2 + 1), height, width)
        kx = torch.arange(width // 2 + 1, device=self.device)[None, :]
        phase = torch.exp(
            2j * math.pi * (ky * (up_h // 2) / up_h + kx * (up_w // 2) / up_w)
        )

        # Contrast flip (-1) and zero DC as for the MOSAICS correlation templates
        self.filter = -scale * dose * mtf * phase
        self.filter[0, 0] = 0

    @classmethod
    def from_simulator(
        cls, simulator: Simulator, device: torch.device | str, **kwargs: Any
    ) -> "TemplateProjector":
        """Projector with the atoms and simulation settings of a ttsim3d Simulator.

        The stencil radius is taken from `MosaicsSimulator.stencil_radius` when
        available (and not overridden in ``kwargs``).
        """
        config = simulator.simulator_config
        if isinstance(simulator, MosaicsSimulator):
            kwargs.setdefault("stencil_radius", simulator.stencil_radius)
        mtf_frequencies, mtf_amplitudes = (
            config.mtf_tensors if config.apply_dqe else (None, None)
        )
        return cls(
            atom_positions_zyx=simulator.atom_positions_zyx,
            atom_ids=list(simulator.atom_identities),
            atom_b_factors=simulator.get_scale_atom_b_factors(),
            pixel_size=simulator.pixel_spacing,
            template_shape=tuple(simulator.volume_shape[-2:]),
            voltage=config.voltage,
            upsampling=config.upsampling,
            apply_dose_weighting=config.apply_dose_weighting,
            dose_start=config.dose_start,
            dose_end=config.dose_end,
            dose_filter_modify_signal=config.dose_filter_modify_signal,
            dose_filter_critical_bfactor=config.crit_exposure_bfactor,
            apply_dqe=config.apply_dqe,
            mtf_frequencies=mtf_frequencies,
            mtf_amplitudes=mtf_amplitudes,
            device=device,
            **kwargs,
        )

    def truncated_fraction(self) -> float:
        """Fraction of the template's projected potential outside the atom stencils.å"""
        sigma = torch.sqrt((self.params_b + self.b_factors[:, None]) / (8 * math.pi**2))
        inside = torch.erf(self.stencil_radius / (math.sqrt(2) * sigma)) ** 2
        lost = ((1 - inside) * self.params_a).sum() / self.params_a.sum()
        return float(lost)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms held by the projector."""
        return int(self.positions_xyz.shape[0])

    def fourier_slices(
        self,
        rotation_matrices: torch.Tensor,  # (n, 3, 3)
        projective_filters: torch.Tensor,  # (n, h, w // 2 + 1)
        atom_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (n, h, w // 2 + 1)
        """Filtered Fourier slices of the given atoms at each particle pose.

        Parameters
        ----------
        rotation_matrices : torch.Tensor
            Rotation matrices (left-multiplying xyz column vectors, as used by
            torch-fourier-slice) for each particle, shape (n, 3, 3).
        projective_filters : torch.Tensor
            Per-particle projective filters, shape (n, h, w // 2 + 1).
        atom_indices : torch.Tensor | None, optional
            Atoms to project. If None (default), all atoms.

        Returns
        -------
        torch.Tensor
            Complex filtered slices, shape (n, h, w // 2 + 1).
        """
        height, width = self.template_shape
        num = rotation_matrices.shape[0]
        out = torch.zeros(
            (num, height, width // 2 + 1), dtype=torch.complex64, device=self.device
        )

        if atom_indices is None:
            xyz, a, b, bf = (
                self.positions_xyz,
                self.params_a,
                self.params_b,
                self.b_factors,
            )
        else:
            idx = torch.as_tensor(atom_indices, device=self.device).long()
            xyz, a, b, bf = (
                self.positions_xyz[idx],
                self.params_a[idx],
                self.params_b[idx],
                self.b_factors[idx],
            )
        num_atoms = xyz.shape[0]
        if num_atoms == 0 or num == 0:
            return out

        # Split work so (particles x atoms x stencil pixels) stays under the cap
        atom_batch = max(
            1, min(num_atoms, self.max_batch_elements // self.stencil_size)
        )
        particle_batch = max(
            1, min(num, self.max_batch_elements // (atom_batch * self.stencil_size))
        )

        rotation_matrices = rotation_matrices.to(self.device).float()
        for start in range(0, num, particle_batch):
            batch = slice(start, min(start + particle_batch, num))
            # Image coordinates of each atom: R^T r, projected along z
            img_xyz = torch.einsum("bji,nj->bni", rotation_matrices[batch], xyz)
            img_yx = img_xyz[..., [1, 0]]

            image = self._potential_2d(
                img_yx,
                bf,
                a,
                b,
                self.grid,
                per_voxel_averaging=True,
                batch_size=atom_batch,
            )
            dft = torch.fft.rfftn(image, dim=(-2, -1))
            dft = _crop_rfft_2d(dft, height, width) * self.filter
            out[batch] = dft * projective_filters[batch].to(self.device)

        return out

    def projections(
        self,
        rotation_matrices: torch.Tensor,
        projective_filters: torch.Tensor,
        atom_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:  # (n, h, w)
        """Filtered real-space projections, centered at (h // 2, w // 2)."""
        return slices_to_projections(
            self.fourier_slices(rotation_matrices, projective_filters, atom_indices)
        )
