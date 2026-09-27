
import numpy as np


def _safe_float(value, default=0.0):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return float(default)


def explain_degradation(
    degradation,
    confidence=None,
    reason=None
):
    """
    Explain the detected degradation type.
    """

    degradation = str(
        degradation
    ).strip().title()

    confidence = _safe_float(
        confidence,
        0.0
    )

    explanations = {

        "Gaussian":
            "The image shows predominantly isotropic "
            "blur characteristics, which are consistent "
            "with Gaussian-type smoothing.",

        "Motion":
            "Directional frequency and edge characteristics "
            "suggest directional motion blur.",

        "Defocus":
            "The frequency and edge characteristics are "
            "consistent with optical defocus.",

        "Unknown":
            "The degradation characteristics are not "
            "strong enough to confidently assign a specific "
            "degradation category."
    }

    explanation = explanations.get(
        degradation,
        "The detected degradation does not match "
        "a predefined degradation category."
    )

    return {
        "degradation": degradation,
        "confidence": confidence,
        "reason": reason if reason else explanation,
        "explanation": explanation
    }


def explain_method_selection(
    method,
    degradation=None,
    noise_level=None
):
    """
    Explain why a restoration method is appropriate.
    """

    method_key = str(
        method
    ).lower().strip()

    noise_level = _safe_float(
        noise_level,
        0.0
    )

    explanations = {

        "inverse":
            "Inverse filtering directly compensates for "
            "the estimated frequency response. It is most "
            "appropriate when the degradation model is "
            "reliable and noise amplification is limited.",

        "wiener":
            "Wiener filtering provides a balance between "
            "deblurring and noise suppression, making it "
            "appropriate when degradation is accompanied "
            "by noticeable noise.",

        "tikhonov":
            "Tikhonov regularization stabilizes the inverse "
            "problem by penalizing unstable solutions. It "
            "is useful when direct inversion is sensitive "
            "to noise or missing frequencies.",

        "cls":
            "Constrained Least Squares introduces a "
            "smoothness constraint while solving the "
            "inverse problem. It is useful when preserving "
            "a stable image structure is important."
    }

    explanation = explanations.get(
        method_key,
        "The selected restoration method was chosen "
        "by the intelligent restoration pipeline."
    )

    return {
        "method": str(method),
        "noise_level": noise_level,
        "explanation": explanation
    }


def explain_parameter_selection(
    parameter,
    parameter_mode="Manual",
    method=None
):
    """
    Explain how the restoration parameter was selected.
    """

    parameter = _safe_float(
        parameter,
        0.0
    )

    mode = str(
        parameter_mode
    ).strip()

    method_text = (
        str(method)
        if method is not None
        else "selected method"
    )

    if mode.lower() == "manual":

        explanation = (
            f"The {method_text} parameter was manually "
            f"specified as {parameter:.6g}."
        )

    elif "l-curve" in mode.lower():

        explanation = (
            f"The parameter {parameter:.6g} was selected "
            f"using L-curve analysis. The selected point "
            f"represents a balance between data fidelity "
            f"and regularization."
        )

    elif "benchmark" in mode.lower():

        explanation = (
            f"The parameter {parameter:.6g} was selected "
            f"using ground-truth benchmark optimization."
        )

    else:

        explanation = (
            f"The {method_text} parameter was automatically "
            f"selected as {parameter:.6g}."
        )

    return {
        "parameter": parameter,
        "selection_mode": mode,
        "explanation": explanation
    }


def explain_quality(
    health_score=None,
    restoration_confidence=None,
    ringing_indicator=None,
    smoothing_indicator=None,
    warnings=None
):
    """
    Explain restoration quality and possible artifacts.
    """

    health = _safe_float(
        health_score,
        0.0
    )

    confidence = _safe_float(
        restoration_confidence,
        0.0
    )

    ringing = _safe_float(
        ringing_indicator,
        0.0
    )

    smoothing = _safe_float(
        smoothing_indicator,
        0.0
    )

    if health >= 80:
        quality_level = "High"
    elif health >= 60:
        quality_level = "Moderate"
    elif health >= 40:
        quality_level = "Low"
    else:
        quality_level = "Very Low"

    artifact_notes = []

    if ringing >= 0.60:
        artifact_notes.append(
            "Possible ringing artifacts detected."
        )

    if smoothing >= 0.60:
        artifact_notes.append(
            "Possible excessive smoothing detected."
        )

    if not artifact_notes:
        artifact_notes.append(
            "No strong artifact warning was detected."
        )

    if warnings is None:
        warnings = []

    return {
        "quality_level": quality_level,
        "health_score": health,
        "restoration_confidence": confidence,
        "artifact_notes": artifact_notes,
        "warnings": warnings
    }


def generate_explanation(
    degradation="Unknown",
    detection_confidence=0.0,
    detection_reason=None,
    method="Unknown",
    noise_level=0.0,
    parameter=0.0,
    parameter_mode="Manual",
    health_score=0.0,
    restoration_confidence=0.0,
    ringing_indicator=0.0,
    smoothing_indicator=0.0,
    warnings=None
):
    """
    Generate the complete explainability report.
    """

    degradation_info = explain_degradation(
        degradation,
        detection_confidence,
        detection_reason
    )

    method_info = explain_method_selection(
        method,
        degradation,
        noise_level
    )

    parameter_info = explain_parameter_selection(
        parameter,
        parameter_mode,
        method
    )

    quality_info = explain_quality(
        health_score,
        restoration_confidence,
        ringing_indicator,
        smoothing_indicator,
        warnings
    )

    summary = (
        f"The system detected {degradation_info['degradation']} "
        f"degradation with {degradation_info['confidence']:.1%} "
        f"confidence. It selected {method} restoration. "
        f"The restoration parameter was determined using "
        f"{parameter_mode}. The resulting restoration quality "
        f"was classified as {quality_info['quality_level']}."
    )

    return {
        "degradation_analysis": degradation_info,
        "method_selection": method_info,
        "parameter_selection": parameter_info,
        "quality_analysis": quality_info,
        "summary": summary
    }


def explanation_to_text(explanation):
    """
    Convert explanation dictionary into a readable report.
    """

    degradation = explanation[
        "degradation_analysis"
    ]

    method = explanation[
        "method_selection"
    ]

    parameter = explanation[
        "parameter_selection"
    ]

    quality = explanation[
        "quality_analysis"
    ]

    lines = [

        "======================================",
        "      EXPLAINABLE RESTORATION REPORT",
        "======================================",

        "",

        "DEGRADATION ANALYSIS",
        f"Detected degradation: "
        f"{degradation['degradation']}",

        f"Detection confidence: "
        f"{degradation['confidence']:.2%}",

        f"Reason: "
        f"{degradation['reason']}",

        "",

        "METHOD SELECTION",
        f"Selected method: "
        f"{method['method']}",

        f"Reason: "
        f"{method['explanation']}",

        "",

        "PARAMETER SELECTION",
        f"Parameter: "
        f"{parameter['parameter']:.6g}",

        f"Mode: "
        f"{parameter['selection_mode']}",

        f"Explanation: "
        f"{parameter['explanation']}",

        "",

        "QUALITY ANALYSIS",
        f"Quality level: "
        f"{quality['quality_level']}",

        f"Health score: "
        f"{quality['health_score']:.2f}",

        f"Restoration confidence: "
        f"{quality['restoration_confidence']:.2f}",

        "",

        "ARTIFACT ANALYSIS"
    ]

    for note in quality["artifact_notes"]:
        lines.append(
            f"- {note}"
        )

    if quality["warnings"]:

        lines.append("")
        lines.append("WARNINGS")

        for warning in quality["warnings"]:
            lines.append(
                f"- {warning}"
            )

    lines.extend([
        "",
        "FINAL EXPLANATION",
        explanation["summary"],
        "======================================"
    ])

    return "\n".join(lines)
