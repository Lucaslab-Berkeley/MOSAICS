def parse_single_ctffind5_result(result_file_path: str) -> dict:
    """Parses a single ctffind5 result file into a dictionary assuming ordering
    of the fields is consistent. The names of the actual fields are in above
    comments.

    Arguments:
        (str) result_file_path: Path to the ctffind5 result file.

    Returns:
        (dict) Dictionary of parsed fields.
    """
    with open(result_file_path, "r") as f:
        lines = f.readlines()
        result = lines[-1]

    result = result.split()  # Split along spaces
    result = [float(x) for x in result]

    return {
        "ctffind5.micrograph_number": result[0],
        "ctffind5.defocus_1": result[1],
        "ctffind5.defocus_2": result[2],
        "ctffind5.astigmatism_azimuth": result[3],
        "ctffind5.additional_phase_shift": result[4],
        "ctffind5.fit_cross_correlation": result[5],
        "ctffind5.fit_spacing": result[6],
        "ctffind5.estimated_tilt_axis_angle": result[7],
        "ctffind5.estimated_tilt_angle": result[8],
        "ctffind5.estimated_sample_thickness": result[9],
    }
