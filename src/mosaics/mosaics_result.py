"""Class for storing the results from a MOSAICS run."""

from typing import Annotated, Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


def nd_array_before_validator(v: Any) -> np.ndarray:
    """Validator to ensure the value is a numpy array."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def nd_array_serializer(v: np.ndarray) -> list:
    """Serializer to convert numpy array to list."""
    return v.tolist()  # type: ignore


NDArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_before_validator),
    PlainSerializer(nd_array_serializer, return_type=list),
]


class AlternateTemplateResult(BaseModel):
    """Class for storing the result for a single alternate template run in MOSAICS.

    Attributes
    ----------
    cross_correlation : np.ndarray
        The cross-correlation values for the alternate (truncated) model for each
        particle in the dataset.
    chain_ids : list[Optional[str]]
        The chain IDs for each residue in the model. This list is index aligned with
        'residue_ids'.
    residue_ids : list[Optional[int]]
        The residue IDs for each residue in the model. This list is index aligned with
        'chain_ids'.
    removed_atom_indices : list[int]
        The atom indices that were removed from the full model to create the alternate
        (truncated) model.
    sim_removed_atoms_only : bool
        Whether only the removed residues were used for the cross-correlation
        calculation. Default is False which means the entire model (minus the removed
        atoms from the corresponding residues) were used in the calculation. If True,
        only the removed atoms were used in the calculation.
    alternate_overlap : np.ndarray
        The overlap <t, t*> between the alternate model projections t* and the default
        model projections t, per particle.
    alternate_autocorrelation : np.ndarray | None
        The alternate template autocorrelation s_{t*} = ||t*||^2, per particle.
    removed_autocorrelation : np.ndarray | None
        The removed structure autocorrelation s_h = ||h||^2, per particle, where
        h = t - t* is the filtered projection of the removed atoms.
    alternate_removed_overlap : np.ndarray | None
        The overlap s_{t*h} = <t*, h> between the alternate template and the removed
        structure, per particle. Together with the two autocorrelations, this fully
        describes the template Gram matrix of the bivariate model of (c, c*).
        The three Gram fields are None for results computed before they were tracked.
    scattering_potential_full_length : float
        The total scattering potential of the full-length model.
    scattering_potential_alternate : float
        The total scattering potential of the alternate (truncated) model. If
        'sim_removed_atoms_only' is True, this will be the scattering potential
        of the removed atoms only. Otherwise, it will be the scattering potential of the
        full model minus the removed atoms.

    Methods
    -------
    expected_correlation_decrease() -> float
        Returns the expected relative decrease in cross-correlation due to the change
        in scattering potential.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cross_correlation: NDArray
    chain_ids: list[Optional[str]]  # index aligned with residue_ids
    residue_ids: list[Optional[int]]  # index aligned with chain_ids
    removed_atom_indices: NDArray
    sim_removed_atoms_only: bool
    alternate_overlap: NDArray
    scattering_potential_full_length: float
    scattering_potential_alternate: float
    alternate_autocorrelation: Optional[NDArray] = None
    removed_autocorrelation: Optional[NDArray] = None
    alternate_removed_overlap: Optional[NDArray] = None

    def expected_correlation_decrease(self) -> float:
        """Relative decrease in cross-corr due to change in scattering potential."""
        if self.sim_removed_atoms_only:
            delta = self.scattering_potential_alternate
        else:
            delta = (
                self.scattering_potential_full_length
                - self.scattering_potential_alternate
            )
        return delta / self.scattering_potential_full_length


class MosaicsResult(BaseModel):
    """Class for storing the results from a MOSAICS run.

    Attributes
    ----------
    default_cross_correlation : np.ndarray
        The default (non-truncated model) cross-correlation values.
    default_overlap : np.ndarray
        The default template autocorrelation s_t = ||t||^2 for each particle, where t
        is the filtered default template projection at the particle's pose.
    template_iterator_config : dict[str, Any]
        The configuration of the template iterator used for the alternate models.
    sim_removed_atoms_only : bool
        Whether only the removed residues were used for the cross-correlation
        calculation.
    alternate_template_results : list[AlternateTemplateResult]
        The results for each of the alternate template runs.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    default_cross_correlation: NDArray
    default_overlap: NDArray
    template_iterator_config: dict[str, Any]
    sim_removed_atoms_only: bool
    alternate_template_results: list[AlternateTemplateResult]

    @property
    def num_particles(self) -> int:
        """Returns the number of particles in the dataset."""
        return len(self.default_cross_correlation)

    @property
    def num_alternate_models(self) -> int:
        """Returns the number of alternate (truncated) models used."""
        return len(self.alternate_template_results)

    def to_df(self, extra_columns: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """Containerizes the experiment results into a Pandas DataFrame.

        The returned DataFrame will have following columns:
        - particle_id: The particle ID, defaults to ['part_0', 'part_1', ...]
        - default_cc: The default cross-correlation value
        - default_overlap: The default template autocorrelation s_t = ||t||^2
        - alt_cc_0: The cross-correlation value for the first alternate model
        - alt_cc_1: The cross-correlation value for the second alternate model
        ...

        The 'extra_columns' parameter can be used to add additional columns to the
        DataFrame.

        Parameters
        ----------
        extra_columns : dict[str, Any], optional
            The extra columns to add to the DataFrame. Default is None and no extra
            columns will be added.

        Returns
        -------
            pd.DataFrame: The DataFrame containing the held data.
        """
        if extra_columns is None:
            extra_columns = {}

        # Figure out the dimensions of the data
        num_parts = self.num_particles

        # Create the base dataframe with particle IDs and default cross-correlations
        particle_id = [f"part_{i}" for i in range(num_parts)]
        default_cc = self.default_cross_correlation
        df = pd.DataFrame(
            {"particle_id": particle_id, **extra_columns, "default_cc": default_cc}
        )

        # Add the alternate cross-correlations values to the dataframe
        for i, alt_result in enumerate(self.alternate_template_results):
            df[f"alt_cc_{i}"] = alt_result.cross_correlation

        return df

    def expected_correlation_decreases(self) -> np.ndarray:
        """Returns the expected corr decrease for each alternate model."""
        res = []
        for alt_result in self.alternate_template_results:
            res.append(alt_result.expected_correlation_decrease())

        return np.array(res)

    def as_ndarrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the default and alternate cross-correlation values as numpy arrays.

        Returns
        -------
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The tuple containing
            the default and alternate cross-correlation values as numpy arrays.
            - The first array is the default cross-correlation values
            - The second array is the alternate cross-correlation values per particle
            - The third array is the default template autocorrelation s_t = ||t||^2
            - The fourth array is the overlap <t, t*> between the default template
              and each of the alternate templates.
        """
        default_cc = np.array(self.default_cross_correlation)
        default_overlap = np.array(self.default_overlap)
        alt_cc = np.array(
            [
                alt_result.cross_correlation
                for alt_result in self.alternate_template_results
            ]
        ).T
        alt_overlap = np.array(
            [
                alt_result.alternate_overlap
                for alt_result in self.alternate_template_results
            ]
        ).T
        return default_cc, alt_cc, default_overlap, alt_overlap

    def bivariate_arrays(self) -> dict[str, np.ndarray]:
        """Per-particle correlations and template Gram entries for the bivariate model.

        Notation follows the bivariate derivation: t is the default template, t* the
        alternate template and h = t - t* the removed structure, all filtered
        projections at each particle's pose.

        Returns
        -------
        dict[str, np.ndarray]
            - ``c``: default correlations <I, t>, shape (N,)
            - ``s_t``: ||t||^2, shape (N,)
            - ``c_star``: alternate correlations <I, t*>, shape (N, J)
            - ``y``: observed change c - c* = <I, h>, shape (N, J)
            - ``t_dot_ts``: <t, t*>, shape (N, J)
            - ``s_ts``: ||t*||^2, shape (N, J)
            - ``s_h``: ||h||^2, shape (N, J)
            - ``s_tsh``: <t*, h>, shape (N, J)

            where N is the number of particles and J the number of alternate models.

        Raises
        ------
        ValueError
            If the result predates tracking of the Gram entries.
        """
        alts = self.alternate_template_results
        if any(alt.removed_autocorrelation is None for alt in alts):
            raise ValueError(
                "This result does not contain the template Gram entries. Re-run "
                "MOSAICS with a version that tracks them."
            )

        def stack(name: str) -> np.ndarray:
            return np.stack([np.asarray(getattr(alt, name)) for alt in alts], axis=1)

        c = np.asarray(self.default_cross_correlation)
        c_star = stack("cross_correlation")
        return {
            "c": c,
            "s_t": np.asarray(self.default_overlap),
            "c_star": c_star,
            "y": c[:, None] - c_star,
            "t_dot_ts": stack("alternate_overlap"),
            "s_ts": stack("alternate_autocorrelation"),
            "s_h": stack("removed_autocorrelation"),
            "s_tsh": stack("alternate_removed_overlap"),
        }

    def mosaics_scores(self) -> np.ndarray:
        """Calculates the MOSAICS scores for each particle and alternate model.

        The MOSAICS score is calculated as the ratio between the alternate model
        cross-correlations (adjusted by mass) and the default model cross-correlations.

        Returns
        -------
            np.ndarray: The MOSAICS scores of shape
            (num_particles, num_alternate_models).
        """
        default_cc, alt_cc, default_overlap, alt_overlap = self.as_ndarrays()
        adjustment_factors = self.expected_correlation_decreases()

        scores = (alt_cc / adjustment_factors) / default_cc[:, None]

        return scores
