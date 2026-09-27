
import streamlit as st
import numpy as np
import cv2
import io
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from PIL import Image

from restoration_core import (
    psf_to_otf,
    create_laplacian_otf,
    restore_image
)

from degradation_engine import (
    create_psf,
    get_frequency_response,
    analyze_psf
)

from degradation_detector import (
    detect_degradation
)

from parameter_optimizer import (
    optimize_restoration_parameter
)

from lcurve_optimizer import (
    optimize_lcurve
)

from analysis_engine import (
    analyze_image,
    calculate_degradation_severity,
    recommend_method,
    get_edge_map,
    calculate_edge_preservation,
    calculate_ringing_indicator,
    calculate_smoothing_indicator,
    calculate_restoration_confidence,
    calculate_health_score,
    generate_quality_warnings
)

from report_engine import (
    create_restoration_report,
    report_to_json
)

from explainability_engine import (
    generate_explanation,
    explanation_to_text
)

from utils import (
    normalize_image,
    convert_to_grayscale,
    resize_image,
    calculate_metrics
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Frequency Restoration AI",
    page_icon="🔬",
    layout="wide"
)


# =========================
# ADVANCED DASHBOARD STYLE
# =========================

st.markdown("""
<style>

.kpi-card {
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.08);
    min-height: 120px;
    margin-bottom: 10px;
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 8px;
}

.kpi-subtitle {
    font-size: 0.78rem;
    opacity: 0.65;
    margin-top: 5px;
}

.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1.2rem;
    margin-bottom: 0.7rem;
}

</style>
""", unsafe_allow_html=True)




# ============================================================
# CUSTOM HEADER
# ============================================================

st.title(
    "🔬 Regularized Image Restoration "
    "in the Frequency Domain"
)

st.caption(
    "Advanced AI-assisted frequency-domain "
    "image restoration and quality analysis"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Restoration Configuration")


uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff"
    ]
)


# ============================================================
# DEGRADATION MODEL
# ============================================================

st.sidebar.subheader(
    "1️⃣ Degradation Model"
)

degradation_model = st.sidebar.selectbox(
    "Select degradation",
    [
        "Automatic / Estimated",
        "Gaussian Blur",
        "Motion Blur",
        "Defocus Blur"
    ]
)


# ============================================================
# DEGRADATION PARAMETERS
# ============================================================

kernel_size = st.sidebar.slider(
    "PSF Kernel Size",
    min_value=5,
    max_value=31,
    value=15,
    step=2
)


if degradation_model == "Gaussian Blur":

    sigma = st.sidebar.slider(
        "Gaussian Sigma",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

else:

    sigma = 3.0


if degradation_model == "Motion Blur":

    angle = st.sidebar.slider(
        "Motion Angle",
        min_value=0,
        max_value=180,
        value=45,
        step=5
    )

else:

    angle = 45


# ============================================================
# RESTORATION METHOD
# ============================================================

st.sidebar.subheader(
    "2️⃣ Restoration Method"
)

restoration_mode = st.sidebar.radio(
    "Method selection",
    [
        "Automatic",
        "Manual"
    ]
)


if restoration_mode == "Manual":

    restoration_method = st.sidebar.selectbox(
        "Restoration method",
        [
            "Tikhonov",
            "Wiener",
            "CLS",
            "Inverse"
        ]
    )

else:

    restoration_method = None


# ============================================================
# PARAMETER
# ============================================================

st.sidebar.subheader(
    "3️⃣ Restoration Parameter"
)

parameter_mode = st.sidebar.radio(
    "Parameter selection",
    [
        "Manual",
        "Benchmark Optimization",
        "Deployment / L-Curve"
    ]
)

if parameter_mode == "Manual":

    parameter = st.sidebar.number_input(
        "Parameter",
        min_value=0.000001,
        max_value=1.0,
        value=0.001,
        step=0.001,
        format="%.6f"
    )

else:

    parameter = 0.001


# ============================================================
# PROCESS BUTTON
# ============================================================

process_button = st.sidebar.button(
    "🚀 Restore Image",
    use_container_width=True
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload an image from the sidebar "
        "to begin restoration."
    )

    st.markdown(
        """
        ### System Pipeline

        **Input Image**
        ↓  
        **Degradation Analysis**
        ↓  
        **PSF Generation**
        ↓  
        **FFT / OTF Analysis**
        ↓  
        **Restoration Method**
        ↓  
        **Quality Diagnostics**
        ↓  
        **Explainable Restoration Report**
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    uploaded_file
).convert("RGB")

image_rgb = np.array(image)

gray = convert_to_grayscale(
    image_rgb
)

gray = resize_image(
    gray,
    (512, 512)
)

gray = normalize_image(
    gray
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

image_analysis = analyze_image(
    gray
)

severity = calculate_degradation_severity(
    image_analysis
)


# ============================================================
# DEGRADATION ESTIMATION
# ============================================================

if degradation_model == "Automatic / Estimated":

    # --------------------------------------------------------
    # Intelligent frequency + spatial detection
    # --------------------------------------------------------

    detection_result = detect_degradation(
        gray
    )

    detection_profile = detection_result[
        "profile"
    ]

    detection_classification = (
        detection_result["classification"]
    )

    estimated_model = (
        detection_classification[
            "degradation"
        ]
    )

    detection_confidence = (
        detection_classification[
            "confidence"
        ]
    )

    detection_reason = (
        detection_classification[
            "reason"
        ]
    )

    actual_degradation_model = (
        estimated_model
    )

else:

    detection_result = None

    detection_profile = {}

    detection_confidence = 1.0

    detection_reason = (
        "Degradation model selected manually."
    )

    actual_degradation_model = (
        degradation_model
        .replace(" Blur", "")
    )


# ============================================================
# CREATE PSF
# ============================================================

if actual_degradation_model == "Gaussian":

    psf = create_psf(
        "gaussian",
        size=kernel_size,
        sigma=sigma
    )

elif actual_degradation_model == "Motion":

    psf = create_psf(
        "motion",
        size=kernel_size,
        angle=angle
    )

elif actual_degradation_model == "Defocus":

    psf = create_psf(
        "defocus",
        size=kernel_size
    )

else:

    psf = create_psf(
        "gaussian",
        size=kernel_size,
        sigma=sigma
    )


# ============================================================
# OTF
# ============================================================

otf = psf_to_otf(
    psf,
    gray.shape
)

laplacian_otf = (
    create_laplacian_otf(
        gray.shape
    )
)


# ============================================================
# AUTOMATIC RESTORATION METHOD
# ============================================================

recommendation = recommend_method(
    image_analysis,
    severity
)

if restoration_mode == "Automatic":

    selected_method = recommendation[
        "method"
    ]

else:

    selected_method = (
        restoration_method.lower()
    )


# ------------------------------------------------------------
# L-Curve safety constraint
# ------------------------------------------------------------
# The current L-curve implementation uses the Laplacian
# regularization formulation, so it is intended for
# regularized methods such as Tikhonov and CLS.

lcurve_method_adjustment = None

if parameter_mode == "Deployment / L-Curve":

    if selected_method not in ["tikhonov", "cls"]:

        original_method = selected_method

        selected_method = "tikhonov"

        lcurve_method_adjustment = (
            f"L-Curve optimization requires a regularized "
            f"restoration method. The requested method "
            f"'{original_method.upper()}' was replaced with "
            f"TIKHONOV for deployment-safe parameter selection."
        )


# ============================================================
# RESTORE
# ============================================================

if process_button:

    optimization_result = None

    if lcurve_method_adjustment is not None:

        st.warning(
            lcurve_method_adjustment
        )

    # --------------------------------------------------------
    # Manual
    # --------------------------------------------------------

    if parameter_mode == "Manual":

        with st.spinner(
            "Running frequency-domain restoration..."
        ):

            restored = restore_image(
                gray,
                otf,
                method=selected_method,
                parameter=parameter,
                laplacian_otf=laplacian_otf
            )


    # --------------------------------------------------------
    # Benchmark optimization
    # --------------------------------------------------------

    elif parameter_mode == "Benchmark Optimization":

        with st.spinner(
            "Running benchmark parameter search..."
        ):

            optimization_result = (
                optimize_restoration_parameter(
                    degraded_image=gray,
                    otf=otf,
                    method=selected_method,
                    restore_function=restore_image,
                    laplacian_otf=laplacian_otf
                )
            )

            parameter = (
                optimization_result[
                    "best_parameter"
                ]
            )

            restored = (
                optimization_result[
                    "best_image"
                ]
            )


    # --------------------------------------------------------
    # Deployment / L-Curve
    # --------------------------------------------------------

    else:

        with st.spinner(
            "Finding deployment-safe parameter "
            "using L-curve analysis..."
        ):

            optimization_result = (
                optimize_lcurve(
                    degraded_image=gray,
                    otf=otf,
                    restore_function=restore_image,
                    method=selected_method,
                    laplacian_otf=laplacian_otf
                )
            )

            parameter = (
                optimization_result[
                    "best_parameter"
                ]
            )

            restored = (
                optimization_result[
                    "best_image"
                ]
            )


    # ========================================================
    # AUTOMATIC OPTIMIZATION RESULT
    # ========================================================

    if optimization_result is not None:

        st.subheader(
            "🤖 Automatic Parameter Optimization"
        )

        opt1, opt2, opt3 = st.columns(3)

        opt1.metric(
            "Optimization Mode",
            parameter_mode
        )

        opt2.metric(
            "Optimal Parameter",
            f"{parameter:.6f}"
        )

        if parameter_mode == "Deployment / L-Curve":

            opt3.metric(
                "Candidates Tested",
                len(
                    optimization_result[
                        "parameters"
                    ]
                )
            )

            st.info(
                "Parameter selected using "
                "ground-truth-free L-curve analysis."
            )

        else:

            opt3.metric(
                "Candidates Tested",
                len(
                    optimization_result[
                        "results"
                    ]
                )
            )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        gray,
        restored
    )


    # ========================================================
    # QUALITY ANALYSIS
    # ========================================================

    edge_preservation = (
        calculate_edge_preservation(
            gray,
            restored
        )
    )

    ringing = (
        calculate_ringing_indicator(
            restored
        )
    )

    smoothing = (
        calculate_smoothing_indicator(
            gray,
            restored
        )
    )

    confidence = (
        calculate_restoration_confidence(
            psnr=metrics["PSNR"],
            ssim=metrics["SSIM"],
            edge_preservation=edge_preservation,
            ringing=ringing
        )
    )

    health_score = (
        calculate_health_score(
            confidence,
            edge_preservation,
            ringing,
            smoothing
        )
    )

    warnings = (
        generate_quality_warnings(
            ringing,
            smoothing,
            edge_preservation,
            confidence
        )
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    st.success(
        "✅ Restoration completed successfully"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "PSNR",
        f"{metrics['PSNR']:.2f} dB"
    )

    col2.metric(
        "SSIM",
        f"{metrics['SSIM']:.4f}"
    )

    col3.metric(
        "RMSE",
        f"{metrics['RMSE']:.5f}"
    )

    col4.metric(
        "Health Score",
        f"{health_score:.1f}/100"
    )


    # ========================================================
    # IMAGE COMPARISON
    # ========================================================

    st.subheader(
        "🖼️ Restoration Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            gray,
            caption="Input / Degraded Image",
            use_container_width=True
        )

    with col2:

        st.image(
            restored,
            caption=(
                f"Restored — "
                f"{selected_method.upper()}"
            ),
            use_container_width=True
        )


    # ========================================================
    # SYSTEM ANALYSIS
    # ========================================================

    st.subheader(
        "🧠 Intelligent Analysis"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Noise Level",
        f"{image_analysis['noise_level']:.5f}"
    )

    c2.metric(
        "HF Energy",
        f"{image_analysis['high_frequency_energy_ratio']:.4f}"
    )

    c3.metric(
        "Severity",
        f"{severity * 100:.1f}%"
    )

    c4.metric(
        "Confidence",
        f"{confidence * 100:.1f}%"
    )


    st.write(
        f"**Detected / Selected Degradation:** "
        f"{actual_degradation_model.title()}"
    )

    if degradation_model == "Automatic / Estimated":

        d1, d2 = st.columns(2)

        with d1:

            st.metric(
                "Detection Confidence",
                f"{detection_confidence * 100:.1f}%"
            )

        with d2:

            st.write(
                "**Detection Reason**"
            )

            st.info(
                detection_reason
            )

    st.write(
        f"**Selected Restoration Method:** "
        f"{selected_method.upper()}"
    )


    # ========================================================
    # PSF VISUALIZATION
    # ========================================================

    st.subheader(
        "🔬 Point Spread Function"
    )

    psf_col1, psf_col2 = st.columns(2)

    with psf_col1:

        fig_psf, ax_psf = plt.subplots(
            figsize=(5, 4)
        )

        ax_psf.imshow(
            psf,
            cmap="viridis"
        )

        ax_psf.set_title(
            f"{actual_degradation_model.title()} PSF"
        )

        ax_psf.set_xlabel("X")
        ax_psf.set_ylabel("Y")

        st.pyplot(
            fig_psf,
            clear_figure=True
        )


    with psf_col2:

        response = get_frequency_response(
            psf
        )

        fig_freq, ax_freq = plt.subplots(
            figsize=(5, 4)
        )

        ax_freq.imshow(
            np.log1p(
                np.abs(response)
            ),
            cmap="magma"
        )

        ax_freq.set_title(
            "Frequency Response"
        )

        ax_freq.set_xlabel(
            "Frequency X"
        )

        ax_freq.set_ylabel(
            "Frequency Y"
        )

        st.pyplot(
            fig_freq,
            clear_figure=True
        )


    # ========================================================
    # L-CURVE VISUALIZATION
    # ========================================================

    if (
        optimization_result is not None
        and parameter_mode == "Deployment / L-Curve"
    ):

        st.subheader(
            "📈 L-Curve Parameter Selection"
        )

        parameters = np.array(
            optimization_result[
                "parameters"
            ]
        )

        residuals = np.array(
            optimization_result[
                "residuals"
            ]
        )

        regularizations = np.array(
            optimization_result[
                "regularizations"
            ]
        )

        corner_index = (
            optimization_result[
                "corner_index"
            ]
        )

        fig_lcurve, ax_lcurve = plt.subplots(
            figsize=(7, 5)
        )

        ax_lcurve.plot(
            np.log10(residuals + 1e-12),
            np.log10(regularizations + 1e-12),
            marker="o"
        )

        ax_lcurve.scatter(
            np.log10(
                residuals[corner_index]
                + 1e-12
            ),
            np.log10(
                regularizations[corner_index]
                + 1e-12
            ),
            s=120
        )

        ax_lcurve.set_xlabel(
            "log10(Data Fidelity Residual)"
        )

        ax_lcurve.set_ylabel(
            "log10(Regularization Norm)"
        )

        ax_lcurve.set_title(
            "L-Curve and Selected Corner"
        )

        ax_lcurve.grid(
            True,
            alpha=0.3
        )

        st.pyplot(
            fig_lcurve,
            clear_figure=True
        )

        st.caption(
            "The selected corner represents the "
            "trade-off between data fidelity and "
            "regularization strength."
        )


    # ========================================================
    # QUALITY DIAGNOSTICS
    # ========================================================

    st.subheader(
        "📊 Restoration Quality Diagnostics"
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Edge Preservation",
        f"{edge_preservation * 100:.1f}%"
    )

    q2.metric(
        "Ringing Indicator",
        f"{ringing * 100:.1f}%"
    )

    q3.metric(
        "Smoothing Indicator",
        f"{smoothing * 100:.1f}%"
    )

    q4.metric(
        "Confidence",
        f"{confidence * 100:.1f}%"
    )


    # ========================================================
    # HEALTH SCORE
    # ========================================================

    st.subheader(
        "❤️ Restoration Health"
    )

    st.progress(
        int(
            np.clip(
                health_score,
                0,
                100
            )
        )
    )

    if health_score >= 80:

        st.success(
            "High restoration quality."
        )

    elif health_score >= 60:

        st.warning(
            "Moderate restoration quality."
        )

    else:

        st.error(
            "Low restoration quality. "
            "Consider changing the method "
            "or regularization parameter."
        )


    # ========================================================
    # WARNINGS
    # ========================================================

    st.subheader(
        "⚠️ Quality Warnings"
    )

    for warning in warnings:

        st.write(
            f"• {warning}"
        )


    # ========================================================
    
    # =========================
    # EXECUTIVE KPI DASHBOARD
    # =========================

    st.markdown(
    '<div class="section-header">📊 Restoration Overview</div>',
    unsafe_allow_html=True
    )

    # Detection confidence
    _kpi_confidence = "N/A"

    if "detection_result" in locals() and detection_result:
        try:
            _kpi_confidence = (
                f"{float(detection_result['classification']['confidence']) * 100:.1f}%"
            )
        except Exception:
            _kpi_confidence = "Available"

    # Optimal parameter
    _kpi_parameter = "N/A"

    for _param_name in [
        "optimal_parameter",
        "selected_parameter",
        "optimized_parameter",
        "manual_parameter"
    ]:
        if _param_name in locals():
            try:
                _kpi_parameter = f"{float(locals()[_param_name]):.6g}"
                break
            except Exception:
                pass

    # Selected restoration method
    _kpi_method = "N/A"
    if "selected_method" in locals() and selected_method:
        _kpi_method = str(selected_method).upper()

    # Health score
    _kpi_health = "N/A"
    if "health_score" in locals():
        try:
            _kpi_health = f"{float(health_score):.1f}/100"
        except Exception:
            pass

    # KPI cards
    _kpi_html = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0 20px 0;">
        <div class="kpi-card">
            <div class="kpi-title">Detection Confidence</div>
            <div class="kpi-value">{_kpi_confidence}</div>
            <div class="kpi-subtitle">AI degradation analysis</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Optimal Parameter</div>
            <div class="kpi-value">{_kpi_parameter}</div>
            <div class="kpi-subtitle">Selected restoration strength</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Method</div>
            <div class="kpi-value">{_kpi_method}</div>
            <div class="kpi-subtitle">Recommended restoration</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Health Score</div>
            <div class="kpi-value">{_kpi_health}</div>
            <div class="kpi-subtitle">Restoration quality</div>
        </div>
    </div>
    """

    st.markdown(_kpi_html, unsafe_allow_html=True)
    st.caption(
        f"Detection / recommendation confidence: {_kpi_confidence}"
    )

    # DIFFERENCE MAP
    # ========================================================

    st.subheader(
        "🗺️ Restoration Difference Map"
    )

    difference = np.abs(
        gray - restored
    )

    st.image(
        difference,
        caption="Absolute Difference Map",
        use_container_width=True
    )


    # ========================================================
    # EXPLAINABLE RESTORATION ENGINE
    # ========================================================

    st.subheader(
        "💡 Explainable Restoration"
    )

    # --------------------------------------------------------
    # Generate structured explanation
    # --------------------------------------------------------

    explanation_result = generate_explanation(

        degradation=actual_degradation_model,

        detection_confidence=detection_confidence,

        detection_reason=detection_reason,

        method=selected_method,

        noise_level=image_analysis[
            "noise_level"
        ],

        parameter=parameter,

        parameter_mode=parameter_mode,

        health_score=health_score,

        restoration_confidence=confidence,

        ringing_indicator=ringing,

        smoothing_indicator=smoothing,

        warnings=warnings
    )

    # --------------------------------------------------------
    # Explanation summary
    # --------------------------------------------------------

    st.info(
        explanation_result["summary"]
    )

    # --------------------------------------------------------
    # Explanation cards
    # --------------------------------------------------------

    ex1, ex2 = st.columns(2)

    with ex1:

        st.markdown(
            "### 🔍 Why this degradation?"
        )

        st.write(
            explanation_result[
                "degradation_analysis"
            ]["explanation"]
        )

        st.caption(
            f"Detection confidence: "
            f"{detection_confidence * 100:.1f}%"
        )

    with ex2:

        st.markdown(
            "### 🧠 Why this method?"
        )

        st.write(
            explanation_result[
                "method_selection"
            ]["explanation"]
        )

    ex3, ex4 = st.columns(2)

    with ex3:

        st.markdown(
            "### 🎯 Why this parameter?"
        )

        st.write(
            explanation_result[
                "parameter_selection"
            ]["explanation"]
        )

    with ex4:

        st.markdown(
            "### 📊 Quality interpretation"
        )

        st.write(
            f"Quality level: "
            f"**{explanation_result['quality_analysis']['quality_level']}**"
        )

        st.write(
            f"Health score: "
            f"**{health_score:.1f}/100**"
        )

        st.write(
            f"Restoration confidence: "
            f"**{confidence * 100:.1f}%**"
        )

    # --------------------------------------------------------
    # Artifact explanation
    # --------------------------------------------------------

    st.markdown(
        "### ⚠️ Artifact Analysis"
    )

    for note in explanation_result[
        "quality_analysis"
    ]["artifact_notes"]:

        st.write(
            f"• {note}"
        )

    # --------------------------------------------------------
    # Full technical explanation
    # --------------------------------------------------------

    with st.expander(
        "📋 View Full Technical Explanation"
    ):

        st.code(
            explanation_to_text(
                explanation_result
            ),
            language="text"
        )


    # ========================================================
    # REPORT
    # ========================================================

    quality_analysis = {

        "edge_preservation":
            edge_preservation,

        "ringing_indicator":
            ringing,

        "smoothing_indicator":
            smoothing,

        "confidence":
            confidence,

        "health_score":
            health_score,

        "warnings":
            warnings,

        "parameter_optimization":
            optimization_result
            if optimization_result is not None
            else None,

        "explainability":
            explanation_result
    }


    degradation_analysis = {

        "model":
            actual_degradation_model,

        "kernel_size":
            kernel_size,

        "sigma":
            sigma,

        "angle":
            angle,

        "psf_analysis":
            analyze_psf(psf),

        "intelligent_detection":
            detection_result,

        "detection_confidence":
            detection_confidence,

        "detection_reason":
            detection_reason
    }


    report = create_restoration_report(

        image_analysis=image_analysis,

        degradation_analysis=
            degradation_analysis,

        restoration_metrics=
            metrics,

        quality_analysis=
            quality_analysis,

        selected_method=
            selected_method,

        selected_parameter=
            parameter
    )


    # ========================================================
    # DOWNLOAD RESTORED IMAGE
    # ========================================================

    restored_uint8 = (
        np.clip(
            restored,
            0,
            1
        ) * 255
    ).astype(np.uint8)


    restored_buffer = io.BytesIO()

    Image.fromarray(
        restored_uint8
    ).save(
        restored_buffer,
        format="PNG"
    )


    st.download_button(
        label="⬇️ Download Restored Image",
        data=restored_buffer.getvalue(),
        file_name="restored_image.png",
        mime="image/png"
    )


    # ========================================================
    # DOWNLOAD JSON REPORT
    # ========================================================

    json_report = report_to_json(
        report
    )

    st.download_button(
        label="📄 Download Restoration Report",
        data=json_report,
        file_name="restoration_report.json",
        mime="application/json"
    )


else:

    st.info(
        "Configure the degradation and restoration "
        "settings in the sidebar, then click "
        "**Restore Image**."
    )
