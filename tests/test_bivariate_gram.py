"""Tests for the template Gram entries used by the bivariate (c, c*) model."""

import numpy as np
import pytest
import roma
import torch
from leopard_em.utils.fourier_slice import volume_to_rfft_fourier_slice

from mosaics.cross_correlation_core import (
    alternate_template_statistics,
    cross_correlate_particle_stack,
    project_filtered_templates,
)
from mosaics.mosaics_result import AlternateTemplateResult, MosaicsResult

BOX = 32


def _blobs(gen: torch.Generator, n: int, scale: float = 1.0) -> torch.Tensor:
    """Volume made of n random Gaussian blobs (a smooth stand-in for atoms)."""
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(BOX, dtype=torch.float64)] * 3, indexing="ij"),
        dim=-1,
    )
    centers = BOX / 2 + 5 * torch.randn(n, 3, generator=gen, dtype=torch.float64)
    d2 = ((grid[None] - centers[:, None, None, None]) ** 2).sum(-1)
    return (scale * torch.exp(-d2 / 4.0).sum(0)).float()


@pytest.fixture(scope="module")
def setup():
    gen = torch.Generator().manual_seed(0)
    alternate = _blobs(gen, 40)
    removed = _blobs(gen, 6)
    default = alternate + removed
    num = 8
    rotmats = roma.random_rotmat(num).float()
    filters = 0.5 + torch.rand(num, BOX, BOX // 2 + 1, generator=gen)
    images = torch.randn(num, BOX, BOX, generator=gen)
    return {
        "dft": {
            name: volume_to_rfft_fourier_slice(v)
            for name, v in [("t", default), ("ts", alternate), ("h", removed)]
        },
        "rotmats": rotmats,
        "filters": filters,
        "images": images,
    }


def _proj(s, name):
    return project_filtered_templates(s["dft"][name], s["rotmats"], s["filters"])


def test_projections_match_correlation_templates(setup):
    """<I, t> from the projections equals the MOSAICS cross-correlation."""
    s = setup
    t = _proj(s, "t")
    cc, _ = cross_correlate_particle_stack(
        s["images"], s["dft"]["t"], s["rotmats"], s["filters"], batch_size=3
    )
    direct = (s["images"] * t).sum((-2, -1))
    torch.testing.assert_close(cc, direct, rtol=1e-5, atol=1e-4)


def test_projection_linearity(setup):
    """t = t* + h holds after projection and filtering."""
    s = setup
    t, ts, h = (_proj(s, k) for k in ("t", "ts", "h"))
    scale = t.abs().max()
    torch.testing.assert_close(t / scale, (ts + h) / scale, rtol=0, atol=1e-5)


def test_alternate_statistics(setup):
    """Gram entries agree with brute-force inner products and the identities."""
    s = setup
    t, ts, h = (_proj(s, k) for k in ("t", "ts", "h"))
    out = alternate_template_statistics(
        s["images"], t, s["dft"]["ts"], s["dft"]["h"], s["rotmats"], s["filters"], 3
    )
    ip = lambda a, b: (a * b).sum((-2, -1))  # noqa: E731
    expected = {
        "cross_correlation": ip(s["images"], ts),
        "alternate_overlap": ip(t, ts),
        "alternate_autocorrelation": ip(ts, ts),
        "removed_autocorrelation": ip(h, h),
        "alternate_removed_overlap": ip(ts, h),
    }
    for key, value in expected.items():
        torch.testing.assert_close(out[key], value, rtol=1e-4, atol=1e-3)

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


def test_bivariate_model_monte_carlo(setup):
    """Proposition 1: (c, c*) mean a G' theta and covariance sigma^2 Gram(t, t*)."""
    s = setup
    gen = torch.Generator().manual_seed(1)
    t, ts, h = (_proj(s, k)[0].double() for k in ("t", "ts", "h"))
    a, gamma, sigma, n = 2.0, 0.3, 1.5, 40_000
    noise = sigma * torch.randn(n, BOX, BOX, generator=gen, dtype=torch.float64)
    images = a * (ts + gamma * h) + noise
    c = (images * t).sum((-2, -1))
    c_star = (images * ts).sum((-2, -1))

    ip = lambda u, v: float((u * v).sum())  # noqa: E731
    mean = a * np.array([ip(t, ts) + gamma * ip(t, h), ip(ts, ts) + gamma * ip(ts, h)])
    cov = sigma**2 * np.array([[ip(t, t), ip(t, ts)], [ip(t, ts), ip(ts, ts)]])
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
