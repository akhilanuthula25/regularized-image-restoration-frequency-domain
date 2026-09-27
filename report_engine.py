
import json
import numpy as np


# ============================================================
# NUMPY / PYTHON TYPE CONVERSION
# ============================================================

def convert_numpy_types(obj):

    if isinstance(
        obj,
        np.ndarray
    ):
        return obj.tolist()

    if isinstance(
        obj,
        np.integer
    ):
        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):
        return float(obj)

    if isinstance(
        obj,
        np.bool_
    ):
        return bool(obj)

    if isinstance(
        obj,
        dict
    ):
        return {
            str(key): convert_numpy_types(value)
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        list
    ):
        return [
            convert_numpy_types(item)
            for item in obj
        ]

    if isinstance(
        obj,
        tuple
    ):
        return [
            convert_numpy_types(item)
            for item in obj
        ]

    return obj


# ============================================================
# CREATE RESTORATION REPORT
# ============================================================

def create_restoration_report(
    image_analysis=None,
    degradation_analysis=None,
    restoration_metrics=None,
    quality_analysis=None,
    selected_method=None,
    selected_parameter=None
):

    report = {

        "project": {
            "name":
                "Regularized Image Restoration "
                "in the Frequency Domain",

            "version":
                "Advanced Prototype"
        },

        "restoration": {

            "method":
                selected_method,

            "parameter":
                selected_parameter
        },

        "image_analysis":
            image_analysis or {},

        "degradation_analysis":
            degradation_analysis or {},

        "restoration_metrics":
            restoration_metrics or {},

        "quality_analysis":
            quality_analysis or {}
    }

    return convert_numpy_types(
        report
    )


# ============================================================
# REPORT -> JSON
# ============================================================

def report_to_json(
    report,
    indent=4
):

    clean_report = convert_numpy_types(
        report
    )

    return json.dumps(
        clean_report,
        indent=indent
    )
