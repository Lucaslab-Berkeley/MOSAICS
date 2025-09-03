"""Utility functions for the MOSAICS package."""

import pandas as pd


def parse_out_coordinates_result(filename: str) -> pd.DataFrame:
    """Convert _cis_TEM formatted out_coordinates.txt to pandas DataFrame.

    Parse the columns of the make_template_result out_coordinates.txt file
    and place all the columns into a pandas DataFrame. First row defines the
    column names, separated by whitespace, with the subsequent rows in the file
    being the data.

    Arguments:
        (str) filename: The path to the out_coordinates.txt file to parse

    Returns
    -------
        (pd.DataFrame) df: The DataFrame containing the parsed data
    """
    # Get the column names from the first comment line
    with open(filename) as f:
        first_line = f.readline()
    # First character is a comment
    column_names = first_line.strip().split()[1:]

    coord_df = pd.read_csv(filename, sep=r"\s+", skiprows=1, names=column_names)

    return coord_df
