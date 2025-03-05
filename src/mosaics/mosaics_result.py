"""Class for storing the results from a MOSAICS run."""

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class MosaicsResult(BaseModel):
    """Class for storing the results from a MOSAICS run.

    Attributes
    ----------
    default_cross_correlation : np.ndarray
        The default (non-truncated model) cross-correlation values.
    alternate_cross_correlations : np.ndarray
        The cross-correlation values for the alternate (truncated) models.
    alternate_chain_residue_metadata : dict[str, list[tuple[str, int]]]
        The metadata for the alternate chain residues. Keys will correspond to the
        alternate model index, and the values will be a list of tuples containing the
        (chain, residue_id) pairs that were removed from that alternate model.
    sim_removed_atoms_only : bool
        Whether only the removed residues were used for the cross-correlation
        calculation. Default is False which means the entire model (minus the removed
        atoms from the corresponding residues) were used in the calculation. If True,
        only the removed atoms were used in the calculation.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    default_cross_correlation: np.ndarray
    alternate_cross_correlations: np.ndarray
    alternate_chain_residue_metadata: Optional[dict[str, list[tuple[str, int]]]] = None
    sim_removed_atoms_only: bool = False

    def to_df(self, extra_columns: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """Containerizes the held data into DataFrame.

        The returned DataFrame will have following columns:
        - particle_id: The particle ID, defaults to ['part_0', 'part_1', ...]
        - default_cc: The default cross-correlation value
        - alt_cc_0: The cross-correlation value for the first alternate model
        - alt_cc_1: The cross-correlation value for the second alternate model
        ...

        The 'extra_columns parameter can be used to add additional columns to the
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
        num_parts = self.default_cross_correlation.shape[0]
        num_alts = self.alternate_cross_correlations.shape[0]

        particle_id = [f"part_{i}" for i in range(num_parts)]
        default_cc = self.default_cross_correlation
        df = pd.DataFrame(
            {"particle_id": particle_id, **extra_columns, "default_cc": default_cc}
        )

        # Add the alternate cross-correlations values to the dataframe
        for i in range(num_alts):
            df[f"alt_cc_{i}"] = self.alternate_cross_correlations[i]

        return df

    def export_to_json(
        self,
        json_path: str,
        csv_path: str,
        extra_columns: Optional[dict[str, Any]] = None,
    ) -> None:
        """Export the MosaicsResult to JSON (metadata) and CSV (tabular data) files.

        Parameters
        ----------
        json_path : str
            The path to the JSON file to save the metadata.
        csv_path : str
            The path to the CSV file to save the tabular data.
        extra_columns : dict[str, Any], optional
            The extra columns to add to the DataFrame. Default is None and no extra
            columns will be added. Passed to the `to_df` method.
        """
        df = self.to_df(extra_columns)
        df.to_csv(csv_path, index=False)

        json_data = {
            "csv_path": csv_path,
            "sim_removed_atoms_only": self.sim_removed_atoms_only,
            "alternate_chain_residue_metadata": self.alternate_chain_residue_metadata,
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f)
