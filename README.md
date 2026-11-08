# NFC Characterization Workbench (Demo Prototype)
### *A Proof-of-Concept for Automated Semiconductor Test Data Analysis & Technical RAG*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nfc-characterization.streamlit.app/)
&nbsp;
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Wassim-Hamra/NFC-Characterization-Demo)

> **Live Demo:** **[https://nfc-characterization.streamlit.app/](https://nfc-characterization.streamlit.app/)**

> **Disclaimer:** This is an independent student engineering prototype built with 100% synthetic data. It is not affiliated with NXP Semiconductors, and no confidential or proprietary industrial measurements were used. All measurements, lot IDs, chip IDs, and specifications are generated for demonstration purposes only.

---

## Why I Built This

Following my recent interview for a 6-month internship in **Secure Transaction & Identification Product Engineering**, I wanted to take the initiative and explore the challenges discussed during our conversation:

1. **How to efficiently clean and prepare large-scale IC characterization data using Polars** instead of slower traditional tools.
2. **How to statistically detect abnormal lot behavior**, particularly **multimodal distributions** (which often signal process splits, mixed die revisions, or test equipment offsets).
3. **How to enforce deterministic specification validation** (pass/fail must always be decided by strict code, never guessed by an AI).
4. **How to connect technical documentation to data analysis using RAG**, allowing an engineer to ask questions about an anomaly and immediately receive an answer backed by technical references.

Rather than just talking about these ideas in theory, I wanted to build a working prototype to show how I think about solving these problems in practice.

---

## What’s Inside

The application is structured around a complete characterization workflow:

### 1. 🏠 Overview & Methodology (`app.py`)
- **Key metrics at a glance:** Total tests (5,000 measurements across 5 lots), tested chips, production lots, and overall yield pass rate.
- **Lot pass rate chart:** Instantly spots lots with yield drops (with a 99% threshold line).
- **Automated engineering findings:** Automatically flags detected anomalies (e.g. systematic shifts and bimodal distributions).
- **Engineering reference tab:** Explains the characterization workflow, the math behind multimodal detection, why data cleaning matters, and a quick 90-second demo walkthrough.

### 2. 🔎 Data Explorer (`pages/data_explorer.py`)
- Interactive, filterable table by lot, parameter, test site, temperature (0°C, 25°C, 85°C), and supply voltage (3.0V, 3.3V, 3.6V).
- **Polars Data Preparation Pipeline:** Runs automated checks for missing columns, type casting, duplicate rows, null values, and extreme outliers (>10σ).
- Full before-and-after audit accounting so no data point is silently dropped without an explanation.
- Filtered CSV export.

### 3. 📊 Statistical Analysis (`pages/statistical_analysis.py`)
- Full descriptive statistics: Mean, Median, Std Dev, Min, Max, 5th/25th/75th/95th percentiles, Skewness, Kurtosis, CV%, and Process Capability ($C_p, C_{pk}$).
- **Interactive charts:**
  - Distribution histograms with specification limits (LSL, USL, Nominal).
  - Cross-lot box plots to compare lot-to-lot spread and median shifts.
  - Time-series trend scatter with rolling averages.
  - Correlation heatmap between parameters and operating corners.
- **Multimodal Distribution Detection (GMM):** Fits 1- and 2-component Gaussian Mixture Models using scikit-learn. Compares Bayesian Information Criterion (BIC) scores to detect split populations, displays component bell curves, and generates a data-driven engineering insight explaining the pattern.

### 4. Technical Diagnostics & Root Cause Analysis (`pages/ai_investigation.py`)
- A technical investigation assistant built with **Retrieval-Augmented Generation (RAG)** over 8 synthetic technical guides (NFC standards, load modulation, receiver sensitivity, test procedures, failure analysis).
- Pure Python **BM25 retrieval engine** (starts up in seconds with zero heavy downloads).
- **Google Gemini API integration:** Synthesizes clear technical explanations strictly grounded in the retrieved documentation, with document citations.
- Context-aware panel: Select a production lot (e.g., `LOT-D`), and the assistant uses the lot's detected anomalies as additional context for the investigation.
- If no API key is provided, it gracefully operates in **retrieval-only mode**, showing the relevant technical passages directly without fabricating responses.

---

## Pre-wired Test Scenarios

The synthetic dataset includes 3 realistic scenarios to demonstrate different engineering situations:

| Scenario | Lot | What happens |
|---|---|---|
| **A. Normal Reference** | `LOT-A`, `LOT-B`, `LOT-E` | Clean Gaussian distributions within specification limits. Reference baseline. |
| **B. Systematic Shift** | `LOT-C` | Upward shift in `receiver_sensitivity_dbm` resulting in ~21% out-of-spec measurements (triggers a **FAIL** audit). |
| **C. Bimodal Distribution** | `LOT-D` | Two distinct sub-populations in `load_modulation_mv` ($\mu_1 \approx 145\text{ mV}$, $\mu_2 \approx 185\text{ mV}$). Detected by GMM with $\Delta\text{BIC} = 104.3$. |

---

## Tech Stack

- **Language:** Python 3.11+
- **Data Engine:** Polars (fast vector operations)
- **Statistics & ML:** NumPy, SciPy, scikit-learn (`GaussianMixture`)
- **Visuals:** Streamlit, Plotly (dark NXP-inspired theme with Orange `#FCA600`, Blue `#0EAFE0`, Green `#69CA00`)
- **AI / RAG:** BM25 local index + Google GenAI SDK (Gemini 3.5 Flash)

---

## Quick Start (Run Locally)

```bash
# 1. Clone the repository
git clone https://github.com/Wassim-Hamra/NFC-Characterization-Demo.git
cd NFC-Characterization-Demo

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add your Gemini API key for AI generation
cp .env.example .env
# Open .env and add: GOOGLE_API_KEY=your_key_here

# 5. Launch the application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## How to Test the Demo in 90 Seconds

1. **Overview Dashboard:** Check the top KPI cards and the lot pass rate chart. Notice that `LOT-C` shows a degraded pass rate, and the finding cards flag both `LOT-C` (shift) and `LOT-D` (bimodality).
2. **Data Explorer:** Use the filters to inspect temperature corners, and click the **Data Preparation** tab to see the Polars cleaning audit report.
3. **Statistical Analysis:** Select `load_modulation_mv` and open the **Multimodal** tab. Select `LOT-D` to see the two Gaussian components fitted by the GMM with the $\Delta\text{BIC}$ score and the automated insight text.
4. **AI Investigation:** Select `LOT-D` in the context panel, then click the pre-filled question: *"What could cause a bimodal distribution in load modulation?"* to see the retrieved technical references and the grounded Gemini explanation.

---

*Author: Wassim Hamra — Final-year Engineering Student*  
*Contact: wassimhamraa@gmail.com*
