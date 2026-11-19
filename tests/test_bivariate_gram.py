"""Tests for the analytic template projections and the bivariate Gram entries."""

from pathlib import Path
from types import SimpleNamespace

import mmdf
import numpy as np
import pytest
import roma
import torch
from pydantic import ValidationError
from ttsim3d.models import SimulatorConfig
from ttsim3d.simulate3d import simulate3d

from mosaics.cross_correlation_core import (
    alternate_template_statistics,
    cross_correlate_particle_stack,
    project_filtered_templates,
)
from mosaics.mosaics_dataset import MosaicsDataset
from mosaics.mosaics_manager import MosaicsManager
from mosaics.mosaics_result import AlternateTemplateResult, MosaicsResult
from mosaics.projection import (
    DEFAULT_STENCIL_RADIUS,
    MosaicsSimulator,
    TemplateProjector,
    slices_to_projections,
)
from mosaics.template_iterator import instantiate_template_iterator

PDB_PATH = Path(__file__).resolve().parents[1] / "docs" / "_static" / "1mh1.pdb"

BOX = 32
PIXEL = 1.0
NUM_ATOMS = 60
NUM_REMOVED = 8
SIM_KWARGS = {  # shared simulation settings for the projector and ttsim3d
    "voltage": 300.0,
    "upsampling": 2,
    "apply_dose_weighting": True,
    "dose_start": 0.0,
    "dose_end": 30.0,
    "apply_dqe": True,
}


def ip(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image inner products."""
    return (a * b).sum((-2, -1))


@pytest.fixture(scope="module")
def setup():
    gen = torch.Generator().manual_seed(0)
    positions = 4.0 * torch.randn(NUM_ATOMS, 3, generator=gen)
    atom_ids = [["C", "N", "O", "S"][i % 4] for i in range(NUM_ATOMS)]
    b_factors = 10.0 + 10.0 * torch.rand(NUM_ATOMS, generator=gen)
    mtf_freq, mtf_amp = SimulatorConfig(mtf_reference="k3_300kV_FL2").mtf_tensors
    projector = TemplateProjector(
        atom_positions_zyx=positions,
        atom_ids=atom_ids,
        atom_b_factors=b_factors,
        pixel_size=PIXEL,
        template_shape=(BOX, BOX),
        mtf_frequencies=mtf_freq,
        mtf_amplitudes=mtf_amp,
        **SIM_KWARGS,
    )
    num = 8
    rotmats = torch.cat([torch.eye(3)[None], roma.random_rotmat(num - 1).float()])
    filters = 0.5 + torch.rand(num, BOX, BOX // 2 + 1, generator=gen)
    images = torch.randn(num, BOX, BOX, generator=gen)
    removed = torch.arange(NUM_REMOVED)
    kept = torch.arange(NUM_REMOVED, NUM_ATOMS)
    return {
        "projector": projector,
        "positions": positions,
        "atom_ids": atom_ids,
        "b_factors": b_factors,
        "mtf": (mtf_freq, mtf_amp),
        "rotmats": rotmats,
        "filters": filters,
        "images": images,
        "removed": removed,
        "kept": kept,
    }


def _proj(s, atoms=None):
    return project_filtered_templates(
        s["projector"], s["rotmats"], s["filters"], atom_indices=atoms, batch_size=3
    )


def test_projection_linearity(setup):
    """t = t* + h holds for the analytic projections."""
    s = setup
    t, ts, h = _proj(s), _proj(s, s["kept"]), _proj(s, s["removed"])
    scale = t.abs().max()
    torch.testing.assert_close(t / scale, (ts + h) / scale, rtol=0, atol=1e-5)


def test_projection_batching_invariant(setup):
    """Splitting the work into small batches does not change the projections."""
    s = setup
    p = s["projector"]
    full = p.fourier_slices(s["rotmats"], s["filters"])
    max_elements = p.max_batch_elements
    try:
        p.max_batch_elements = 7 * p.stencil_size  # 7 atoms, 1 particle per batch
        small = p.fourier_slices(s["rotmats"], s["filters"])
    finally:
        p.max_batch_elements = max_elements
    scale = full.abs().max()
    torch.testing.assert_close(small / scale, full / scale, rtol=0, atol=1e-5)


# ttsim3d calls a wavelength helper deprecated in torch-fourier-filter
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_matches_ttsim3d_fourier_slice(setup):
    """Analytic projections agree with projections of ttsim3d volumes.

    This guards the conventions shared with the previous volume-based pipeline:
    rotation direction, projection origin, overall scale, B-factors, dose filter,
    Fourier cropping of the upsampled grid and the MTF. At the identity pose the
    exact projection is the sum of the volume along z; other poses are compared
    against 2x padded Fourier slices, which are accurate to ~1%. The remaining
    ~0.5% difference is ttsim3d truncating each atom at a small voxel neighborhood.
    """
    torch_fourier_slice = pytest.importorskip("torch_fourier_slice")
    s = setup
    volume = simulate3d(
        atom_positions_zyx=s["positions"],
        atom_ids=s["atom_ids"],
        atom_b_factors=s["b_factors"],
        beam_energy_kev=SIM_KWARGS["voltage"],
        sim_pixel_spacing=PIXEL,
        sim_volume_shape=(BOX, BOX, BOX),
        requested_upsampling=SIM_KWARGS["upsampling"],
        apply_dose_weighting=SIM_KWARGS["apply_dose_weighting"],
        dose_start=SIM_KWARGS["dose_start"],
        dose_end=SIM_KWARGS["dose_end"],
        apply_dqe=SIM_KWARGS["apply_dqe"],
        mtf_frequencies=s["mtf"][0],
        mtf_amplitudes=s["mtf"][1],
        device="cpu",
    )
    rotmats = s["rotmats"]
    assert torch.equal(rotmats[0], torch.eye(3))
    raw = torch.cat(
        [
            volume.sum(0)[None],
            torch_fourier_slice.project_3d_to_2d(volume, rotmats[1:], pad_factor=2.0),
        ]
    )
    # Apply the MOSAICS template filtering: origin to index 0, zero DC, flip, filter
    dft = torch.fft.rfftn(torch.fft.ifftshift(raw, dim=(-2, -1)), dim=(-2, -1))
    dft[..., 0, 0] = 0
    expected = slices_to_projections(-dft * s["filters"])

    analytic = _proj(s)
    cos = (
        ip(analytic, expected)
        / (ip(analytic, analytic) * ip(expected, expected)).sqrt()
    )
    ratio = analytic.norm(dim=(-2, -1)) / expected.norm(dim=(-2, -1))
    assert cos[0] > 0.9999 and abs(ratio[0] - 1) < 0.01, (cos[0], ratio[0])
    assert torch.all(cos[1:] > 0.999), cos
    torch.testing.assert_close(ratio[1:], torch.ones_like(ratio[1:]), rtol=0, atol=0.03)


def test_cross_correlation_same_and_larger_images(setup):
    """Correlations equal brute-force inner products, also for larger images."""
    s = setup
    t = _proj(s)
    cc = cross_correlate_particle_stack(s["images"], t, batch_size=3)
    torch.testing.assert_close(cc, ip(s["images"], t), rtol=1e-5, atol=1e-4)

    gen = torch.Generator().manual_seed(2)
    large = torch.randn(t.shape[0], BOX + 6, BOX + 4, generator=gen)
    cc_map = cross_correlate_particle_stack(large, t, batch_size=3)
    assert cc_map.shape == (t.shape[0], 7, 5)
    for dy, dx in [(0, 0), (3, 2), (6, 4)]:
        window = large[:, dy : dy + BOX, dx : dx + BOX]
        torch.testing.assert_close(
            cc_map[:, dy, dx], ip(window, t), rtol=1e-4, atol=1e-3
        )


def test_alternate_statistics(setup):
    """Gram entries agree with brute-force inner products and the identities."""
    s = setup
    t, ts, h = _proj(s), _proj(s, s["kept"]), _proj(s, s["removed"])
    out = alternate_template_statistics(
        s["images"], t, s["projector"], s["removed"], s["rotmats"], s["filters"], 3
    )
    expected = {
        "cross_correlation": ip(s["images"], ts),
        "alternate_overlap": ip(t, ts),
        "alternate_autocorrelation": ip(ts, ts),
        "removed_autocorrelation": ip(h, h),
        "alternate_removed_overlap": ip(ts, h),
    }
    for key, value in expected.items():
        scale = value.abs().max()
        torch.testing.assert_close(out[key] / scale, value / scale, rtol=0, atol=1e-4)

    # Identities used throughout the derivation (eq. gram_identities)
    s_t = ip(t, t)
    s_ts, s_h = out["alternate_autocorrelation"], out["removed_autocorrelation"]
    s_tsh = out["alternate_removed_overlap"]
    torch.testing.assert_close(s_t, s_ts + 2 * s_tsh + s_h, rtol=1e-4, atol=0)
    torch.testing.assert_close(
        out["alternate_overlap"], s_ts + s_tsh, rtol=1e-4, atol=0
    )
    # Cauchy-Schwarz: Gram determinant D >= 0
    assert torch.all(s_ts * s_h - s_tsh**2 >= 0)


@pytest.mark.parametrize("removed_only", [True, False])
def test_inner_loop_atom_index_convention(setup, removed_only):
    """Kept-atom indices (sim_removed_atoms_only=False) give the same statistics."""
    s = setup
    t = _proj(s)
    manager = MosaicsManager.model_construct(sim_removed_atoms_only=removed_only)
    out = manager._mosaics_inner_loop(
        particle_images=s["images"],
        full_length_projections=t,
        projector=s["projector"],
        rotation_matrices=s["rotmats"],
        projective_filters=s["filters"],
        atom_indices=s["removed"] if removed_only else s["kept"],
        batch_size=3,
    )
    expected = alternate_template_statistics(
        s["images"], t, s["projector"], s["removed"], s["rotmats"], s["filters"], 3
    )
    for key, value in expected.items():
        torch.testing.assert_close(out[key], value)


@pytest.mark.parametrize("removed_only", [True, False])
def test_alternates_removing_no_or_all_atoms(setup, removed_only):
    """Alternates removing no atoms (t* = t) or all atoms (t* = 0), in both index
    conventions of the template iterator."""
    s = setup
    num_atoms = s["projector"].num_atoms
    nothing, everything = torch.arange(0), torch.arange(num_atoms)
    # Iterator indices are the removed atoms, or the kept atoms if not removed_only
    alternates = [
        ([None], [None], nothing if removed_only else everything),
        ([None], [None], everything if removed_only else nothing),
    ]
    manager = MosaicsManager.model_construct(
        sim_removed_atoms_only=removed_only, particle_positions_are_centered=True
    )
    inputs = {
        "rotation_matrices": s["rotmats"],
        "projective_filters": s["filters"],
        "particle_images": s["images"],
    }
    default_cc, s_t, (none_removed, all_removed) = manager._process_particle_chunk(
        inputs, s["projector"], alternates, batch_size=3, desc="test"
    )
    zeros = np.zeros_like(s_t)

    expected_none = {
        "cross_correlation": default_cc,
        "alternate_overlap": s_t,
        "alternate_autocorrelation": s_t,
        "removed_autocorrelation": zeros,
        "alternate_removed_overlap": zeros,
    }
    expected_all = {
        "cross_correlation": np.zeros_like(default_cc),
        "alternate_overlap": zeros,
        "alternate_autocorrelation": zeros,
        "removed_autocorrelation": s_t,
        "alternate_removed_overlap": zeros,
    }
    for stats, expected in ((none_removed, expected_none), (all_removed, expected_all)):
        for key, value in expected.items():
            scale = max(np.abs(default_cc).max(), s_t.max())
            np.testing.assert_allclose(stats[key] / scale, value / scale, atol=1e-5)


def test_save_to_cache_requires_all_particles(particle_cache, tmp_path):
    """Caches can only be written from inputs processed for the whole stack."""
    cache = np.load(particle_cache)
    num = cache["particle_images"].shape[0]

    dataset = MosaicsDataset.model_construct()
    with pytest.raises(ValueError, match="No particle images"):
        dataset.save_to_cache(str(tmp_path / "empty.npz"))

    dataset = MosaicsDataset.model_construct(
        particle_stack=SimpleNamespace(num_particles=num + 1)
    )
    dataset.load_from_cache(particle_cache, torch.device("cpu"))
    with pytest.raises(ValueError, match="process_inputs"):
        dataset.save_to_cache(str(tmp_path / "partial.npz"))

    dataset.particle_stack = SimpleNamespace(num_particles=num)
    dataset.save_to_cache(str(tmp_path / "full.npz"))
    resaved = np.load(tmp_path / "full.npz")
    for key in ("rotation_matrices", "projective_filters", "particle_images"):
        np.testing.assert_array_equal(resaved[key], cache[key])


def test_bivariate_model_monte_carlo(setup):
    """Proposition 1: (c, c*) mean a G' theta and covariance sigma^2 Gram(t, t*)."""
    s = setup
    gen = torch.Generator().manual_seed(1)
    t, ts, h = (
        x[0].double() for x in (_proj(s), _proj(s, s["kept"]), _proj(s, s["removed"]))
    )
    a, gamma, sigma, n = 2.0, 0.3, 1.5, 40_000
    noise = sigma * torch.randn(n, BOX, BOX, generator=gen, dtype=torch.float64)
    images = a * (ts + gamma * h) + noise
    c = (images * t).sum((-2, -1))
    c_star = (images * ts).sum((-2, -1))

    dot = lambda u, v: float((u * v).sum())  # noqa: E731
    mean = a * np.array(
        [dot(t, ts) + gamma * dot(t, h), dot(ts, ts) + gamma * dot(ts, h)]
    )
    cov = sigma**2 * np.array([[dot(t, t), dot(t, ts)], [dot(t, ts), dot(ts, ts)]])
    sample = torch.stack([c, c_star]).numpy()

    se = np.sqrt(np.diag(cov) / n)
    assert np.all(np.abs(sample.mean(1) - mean) < 5 * se)
    np.testing.assert_allclose(np.cov(sample), cov, rtol=0.05)


def _alt(n, with_gram: bool):
    extra = {}
    if with_gram:
        extra = {
            "alternate_autocorrelation": np.full(n, 3.0),
            "removed_autocorrelation": np.full(n, 0.5),
            "alternate_removed_overlap": np.full(n, -0.25),
        }
    return AlternateTemplateResult(
        cross_correlation=np.arange(n, dtype=float),
        chain_ids=["A"],
        residue_ids=[1],
        removed_atom_indices=np.array([0]),
        sim_removed_atoms_only=True,
        alternate_overlap=np.full(n, 2.75),
        scattering_potential_full_length=1.0,
        scattering_potential_alternate=0.1,
        **extra,
    )


def _result(with_gram: bool):
    n = 4
    return MosaicsResult(
        default_cross_correlation=np.arange(n, dtype=float) + 1,
        default_overlap=np.full(n, 3.0),
        template_iterator_config={},
        sim_removed_atoms_only=True,
        alternate_template_results=[_alt(n, with_gram), _alt(n, with_gram)],
    )


def test_result_round_trip_and_backwards_compatibility():
    new = MosaicsResult.model_validate_json(_result(True).model_dump_json())
    arrays = new.bivariate_arrays()
    assert arrays["s_h"].shape == (4, 2)
    np.testing.assert_allclose(arrays["y"], 1.0)
    s_t_from_parts = arrays["s_ts"] + 2 * arrays["s_tsh"] + arrays["s_h"]
    np.testing.assert_allclose(
        np.broadcast_to(arrays["s_t"][:, None], (4, 2)), s_t_from_parts
    )

    old_json = _result(False).model_dump_json(
        exclude={
            "alternate_template_results": {
                "__all__": {
                    "alternate_autocorrelation",
                    "removed_autocorrelation",
                    "alternate_removed_overlap",
                }
            }
        }
    )
    old = MosaicsResult.model_validate_json(old_json)
    with pytest.raises(ValueError, match="Gram entries"):
        old.bivariate_arrays()


def _small_simulator(**kwargs) -> MosaicsSimulator:
    # Uniform B-factors as in typical 2DTM templates (the PDB B-factors are large
    # enough that the default stencil would warn about truncation)
    return MosaicsSimulator(
        pdb_filepath=str(PDB_PATH),
        pixel_spacing=1.0,
        volume_shape=(64, 64, 64),
        b_factor_scaling=0.0,
        additional_b_factor=60.0,
        simulator_config={"voltage": 300.0, "upsampling": 2, "apply_dqe": False},
        **kwargs,
    )


def test_simulator_stencil_radius():
    """The stencil radius is configured on the MOSAICS simulator."""
    simulator = _small_simulator()
    assert simulator.stencil_radius == DEFAULT_STENCIL_RADIUS == 3.5
    assert TemplateProjector.from_simulator(simulator, "cpu").stencil_radius == 3.5

    simulator = _small_simulator(stencil_radius=5.0)
    assert TemplateProjector.from_simulator(simulator, "cpu").stencil_radius == 5.0
    assert simulator.model_dump()["stencil_radius"] == 5.0

    with pytest.raises(ValidationError):
        _small_simulator(stencil_radius=0.0)


def test_small_stencil_warns(setup):
    s = setup
    with pytest.warns(UserWarning, match="stencil_radius"):
        TemplateProjector(
            s["positions"],
            s["atom_ids"],
            s["b_factors"],
            PIXEL,
            (BOX, BOX),
            stencil_radius=1.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_projections_match_eager(setup):
    """torch.compile of the potential calculation does not change the results."""
    s = setup
    kwargs = {
        "atom_positions_zyx": s["positions"],
        "atom_ids": s["atom_ids"],
        "atom_b_factors": s["b_factors"],
        "pixel_size": PIXEL,
        "template_shape": (BOX, BOX),
        "device": "cuda",
    }
    filters = s["filters"].cuda()
    eager = TemplateProjector(**kwargs, compile_potential=False)
    compiled = TemplateProjector(**kwargs)  # compiled by default on CUDA
    for atoms in (None, s["removed"]):
        a = eager.fourier_slices(s["rotmats"], filters, atoms)
        b = compiled.fourier_slices(s["rotmats"], filters, atoms)
        scale = a.abs().max()
        torch.testing.assert_close(b / scale, a / scale, rtol=0, atol=1e-5)


@pytest.fixture(scope="module")
def particle_cache(tmp_path_factory):
    """Cached (pre-processed) synthetic particles for end-to-end runs."""
    gen = torch.Generator().manual_seed(3)
    num = 7
    path = tmp_path_factory.mktemp("cache") / "particles.npz"
    np.savez(
        path,
        rotation_matrices=roma.random_rotmat(num).float().numpy(),
        projective_filters=(0.5 + torch.rand(num, 64, 33, generator=gen)).numpy(),
        particle_images=torch.randn(num, 64, 64, generator=gen).numpy(),
    )
    return str(path)


@pytest.mark.parametrize(
    "iterator_config",
    [
        {"type": "residue", "num_residues_removed": 40, "residue_increment": 40},
        {"type": "random", "num_atoms_removed": 25, "num_alternate_structures": 3},
    ],
    ids=["residue", "random"],
)
def test_run_mosaics_particle_chunks(particle_cache, iterator_config):
    """Processing particles in chunks gives the same results as all at once."""
    iterator = instantiate_template_iterator(
        {
            **iterator_config,
            "residue_types": ["amino_acid"],
            "structure_df": mmdf.read(str(PDB_PATH)),
        }
    )
    manager = MosaicsManager.model_construct(
        simulator=_small_simulator(),
        template_iterator=iterator,
        dataset=MosaicsDataset.model_construct(),
        sim_removed_atoms_only=True,
        particle_positions_are_centered=True,
    )

    results = []
    for chunk_size in (None, 3):
        np.random.seed(0)  # same random alternates for both runs
        results.append(
            manager.run_mosaics(
                "cpu",
                batch_size=2,
                cache_path=particle_cache,
                particle_chunk_size=chunk_size,
            )
        )
    full, chunked = (r.bivariate_arrays() for r in results)
    assert full["c"].shape == (7,)
    assert full["c_star"].shape[1] == len(results[0].alternate_template_results) > 1
    for key, value in full.items():
        # float32 sums over 64^2 pixels; tolerance relative to each array's scale
        atol = 1e-5 * np.abs(value).max()
        np.testing.assert_allclose(chunked[key], value, rtol=1e-5, atol=atol)
