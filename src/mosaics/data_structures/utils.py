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

    result_split = result.split()  # Split along spaces
    result_float = [float(x) for x in result_split]

    return {
        "ctffind5.micrograph_number": result_float[0],
        "ctffind5.defocus_1": result_float[1],
        "ctffind5.defocus_2": result_float[2],
        "ctffind5.astigmatism_azimuth": result_float[3],
        "ctffind5.additional_phase_shift": result_float[4],
        "ctffind5.fit_cross_correlation": result_float[5],
        "ctffind5.fit_spacing": result_float[6],
        "ctffind5.estimated_tilt_axis_angle": result_float[7],
        "ctffind5.estimated_tilt_angle": result_float[8],
        "ctffind5.estimated_sample_thickness": result_float[9],
    }
