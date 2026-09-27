# Regularized Image Restoration in the Frequency Domain

An advanced frequency-domain image restoration system built with Python, NumPy, OpenCV, SciPy, scikit-image, Plotly, and Streamlit.

The system analyzes degraded images, models image degradation using Point Spread Functions (PSFs), performs frequency-domain restoration using multiple regularization methods, automatically selects deployment-safe parameters using L-Curve analysis, evaluates restoration quality, and provides explainable restoration decisions.

---

## Key Features

### Frequency-Domain Restoration
- Fourier Transform based image processing
- PSF to OTF conversion
- Inverse filtering
- Wiener filtering
- Tikhonov regularization
- Constrained Least Squares (CLS) restoration

### Intelligent Degradation Analysis
- Automatic degradation detection
- Blur/noise analysis
- Frequency-domain characteristics
- Degradation severity estimation
- Method recommendation

### Automatic Parameter Selection
- Manual parameter selection
- Benchmark optimization
- L-Curve based deployment parameter selection
- Regularization candidate evaluation

### Restoration Quality Analysis
- Edge preservation
- Ringing detection
- Smoothing analysis
- Restoration confidence
- Restoration health score
- Quality warnings
- Difference/error visualization

### Explainable AI
The system explains:
- Why a degradation was detected
- Why a restoration method was selected
- Why a parameter was selected
- What the final restoration quality means

### Reporting
- Structured restoration report
- JSON report generation
- Restored image download
- Analysis and quality information

### Interactive Web Application
A Streamlit dashboard provides:
- Image upload
- Degradation configuration
- Restoration configuration
- Intelligent analysis
- Visualization
- Quality analysis
- Explainability
- Downloadable results

---

## System Architecture

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
Degradation Modeling / Detection
     |
     +----------------------+
     |                      |
     v                      v
 PSF Analysis        Intelligent Detection
     |                      |
     +----------+-----------+
                |
                v
        Method Recommendation
                |
                v
       Parameter Selection
        /             \
       /               \
   Manual          L-Curve
       \               /
        \             /
         v           v
       Frequency-Domain
          Restoration
                |
                v
        Quality Analysis
                |
       +--------+--------+
       |        |        |
       v        v        v
     Quality   XAI    Difference
     Metrics          Analysis
       |        |        |
       +--------+--------+
                |
                v
          Final Report
```

---

## Restoration Methods

### 1. Inverse Filtering
Inverse filtering attempts to recover the original image by compensating for the degradation transfer function.

### 2. Wiener Filtering
Wiener filtering balances inverse restoration with noise suppression.

### 3. Tikhonov Regularization
Tikhonov regularization introduces a regularization term to stabilize the inverse problem.

### 4. Constrained Least Squares
CLS restoration uses a regularization constraint to control unstable high-frequency reconstruction.

---

## Parameter Optimization

### Benchmark Optimization
Ground-truth based optimization can be used when a clean reference image is available.

Metrics include:
- PSNR
- SSIM
- MSE
- RMSE
- Edge preservation

### Deployment / L-Curve Optimization
The L-Curve approach selects a regularization parameter without requiring a clean ground-truth image.

This makes it suitable for deployment scenarios where the original clean image is unavailable.

---

## Intelligent Restoration Pipeline

```text
Image Analysis
      |
      v
Degradation Severity
      |
      v
Method Recommendation
      |
      v
Parameter Selection
      |
      v
Restoration
      |
      v
Quality Analysis
      |
      v
Explainable Result
```

---

## Project Structure

```text
frequency_restoration_project/
|
+-- app.py
+-- restoration_core.py
+-- degradation_engine.py
+-- degradation_detector.py
+-- parameter_optimizer.py
+-- benchmark_optimizer.py
+-- lcurve_optimizer.py
+-- benchmark_engine.py
+-- analysis_engine.py
+-- explainability_engine.py
+-- report_engine.py
+-- utils.py
|
+-- requirements.txt
|
+-- outputs/
|
+-- sample_images/
```

---

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd frequency_restoration_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## Typical Workflow

1. Upload an image.
2. Select or automatically estimate the degradation.
3. Configure the restoration method.
4. Select manual or automatic parameter optimization.
5. Run restoration.
6. Review restoration metrics.
7. Inspect degradation and frequency analysis.
8. Review quality warnings.
9. Read the explainable restoration analysis.
10. Download the restored image and JSON report.

---

## Technology Stack

- Python
- NumPy
- OpenCV
- SciPy
- scikit-image
- Matplotlib
- Plotly
- Streamlit
- Pillow

---

## Validation

The project has been validated for:
- Project structure
- Python syntax
- Core restoration algorithms
- PSF and OTF processing
- Intelligent degradation detection
- Method recommendation
- L-Curve parameter optimization
- Explainability engine
- Report generation
- End-to-end restoration pipeline

The complete end-to-end pipeline has been tested successfully.

---

## Important Evaluation Note

When a clean ground-truth image is unavailable, input-to-restored comparison metrics should be treated as diagnostic indicators rather than ground-truth restoration accuracy.

Ground-truth benchmark evaluation should be performed separately using a known clean reference image.

---

## Future Scope

- Deep-learning based restoration
- Learned degradation estimation
- Blind deconvolution
- Real-time video restoration
- GPU acceleration
- Batch image restoration
- Advanced restoration benchmarking
- Cloud deployment
- Model-based parameter learning
- Additional frequency-domain regularizers

---

## Project Status

**Status: Advanced prototype / research-oriented application**

The current implementation combines classical frequency-domain restoration, intelligent degradation analysis, automatic regularization selection, quality assessment, explainability, and an interactive Streamlit interface.