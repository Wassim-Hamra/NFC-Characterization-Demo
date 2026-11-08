# NFC Characterization Workbench (NFC-CW)
### *Semiconductor Characterization & Intelligence*

> **Disclaimer:** This is an independent educational prototype built with entirely synthetic data.
> It is not affiliated with NXP Semiconductors and does not use any confidential industrial
> measurements or specifications. All chip IDs, lot IDs, and measurement values are artificially
> generated for demonstration and educational purposes only.

---

## Overview

**NFC Characterization Workbench** is an interactive engineering analytics platform demonstrating
how software, statistical analysis, and AI can support semiconductor product engineers in:

- Loading and preparing characterization test data
- Analysing measurement distributions and detecting statistical anomalies
- Validating measurements against engineering specification limits
- Investigating potential process issues using semantic document retrieval and optional LLM assistance

The project was built as a personal engineering prototype inspired by the challenges of automated
NFC IC characterization, industrial test-data analysis, and specification compliance.

---

## Application Structure

| Page | Description |
|---|---|
| 🏠 **Overview & Methodology** (`app.py`) | KPI cards, lot pass rates, automated findings, engineering workflow, architecture diagrams, and 90s demo script |
| 🔎 **Data Explorer** (`pages/data_explorer.py`) | Interactive table, filters, CSV download, Polars cleaning pipeline audit |
| 📈 **Statistical Analysis** (`pages/statistical_analysis.py`) | Histograms, box plots, trends, correlation, GMM bimodal distribution decomposition |
| 🤖 **AI Investigation** (`pages/ai_investigation.py`) | BM25 retrieval + Gemini 3.5 LLM grounded technical answers with document citations |

---

## Synthetic Dataset Scenarios

| Lot | Scenario | Detail |
|---|---|---|
| **LOT-A** | ✅ Normal | Reference baseline, all within spec |
| **LOT-B** | ✅ Normal | Slight voltage corner sensitivity |
| **LOT-C** | ⚠️ Systematic shift | `receiver_sensitivity_dbm` shifted +3 dBm → out-of-spec |
| **LOT-D** | 🔴 Bimodal | `load_modulation_mv` has two distinct populations |
| **LOT-E** | ✅ Normal | Slightly wider spread |

---

## Technology Stack

- **Python 3.11+** / **Streamlit** — application framework
- **Polars** — primary data processing
- **NumPy / SciPy** — statistical calculations
- **Plotly** — interactive charts
- **scikit-learn** — Gaussian Mixture Models for bimodal detection
- **BM25 (pure Python)** — document retrieval (no model downloads)
- **Google Gemini API** — optional LLM for grounded answers (free tier)

---

## Installation

```bash
# 1. Clone or unzip the project
cd path/to/chip

# 2. Create and activate virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Gemini API key
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_key_here
```

---

## Running the Application

```bash
# Activate venv first (see above), then:
streamlit run app.py
```

The app opens at **http://localhost:8501**

The synthetic dataset is generated automatically on first launch and saved to
`data/synthetic_measurements.csv`.

---

## Example Scenarios to Demonstrate

### 1. Specification Failure (LOT-C)
1. Navigate to **Specification Validation**
2. The lot × parameter heatmap shows LOT-C → Receiver Sensitivity in red
3. Scroll down to see the failing measurement records with exact values

### 2. Bimodal Distribution (LOT-D)
1. Navigate to **Statistical Analysis**
2. Select `load_modulation_mv` as parameter
3. Open the **Multimodal** tab, select LOT-D
4. The GMM detects two populations and reports a strong ΔBIC signal

### 3. AI Investigation
1. Navigate to **AI Investigation**
2. Select LOT-D in the context panel
3. Ask: *"What could cause a bimodal load modulation distribution?"*
4. Retrieved passages from `load_modulation.txt` and `failure_analysis.txt` are shown

---

## Limitations and Assumptions

- Dataset is entirely synthetic — patterns are engineered to demonstrate specific scenarios
- BM25 retrieval quality degrades at large document scale (suitable for the included 8-document set)
- Gemini API requires a valid key; without it, the app is fully functional in retrieval-only mode
- The app does not persist user sessions between restarts
- No authentication, multi-user support, or deployment infrastructure

---

## Project Structure

```
chip/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── .env.example
├── data/
│   ├── specifications.json     # Illustrative spec limits
│   ├── synthetic_measurements.csv  (auto-generated)
│   └── sample_documents/       # 8 synthetic NFC tech docs
├── src/
│   ├── data_generator.py       # Synthetic dataset creation
│   ├── data_processor.py       # Polars cleaning pipeline
│   ├── statistics_engine.py    # Descriptive stats
│   ├── multimodal_analysis.py  # GMM bimodal detection
│   ├── specification_validator.py  # Deterministic pass/fail
│   ├── rag_engine.py           # BM25 + Gemini integration
│   └── insight_generator.py    # Rule-based findings
├── pages/
│   ├── overview.py
│   ├── data_explorer.py
│   ├── statistical_analysis.py
│   ├── specification_validation.py
│   ├── ai_investigation.py
│   └── methodology.py
└── assets/
    ├── styles.css              # NXP-themed dark stylesheet
    └── nxp_logo.svg
```
