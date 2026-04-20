# Appendix P: User Manual

## Page 1: Title Page

# Guardian: Tri-Modal Fraud Detection System

**User Manual**  
COMP386 Capstone Project  
Winter 2026

Prepared by:

- Arsenii Buriak
- Sherwayne Allen
- Luis Mateo Sanchez Loaiza
- Ivan Yu

---

## Page 2: Introduction

### Purpose of the System

Guardian is a tri-modal fraud detection prototype designed to help identify suspicious transactions by combining three independent signal sources: Numerical, NLP, and Voice. The system supports fraud analysis by producing fraud scores, decisions, confidence levels, and explanations.

### Target Users

This system is intended for:

- instructors and evaluators reviewing the capstone prototype
- semi-technical users testing fraud scenarios through the dashboard
- developers or team members validating API behavior
- analysts interested in explainable fraud scoring

### Key Features

- tri-modal fraud analysis using Numerical, NLP, and Voice signals
- adaptive fusion layer with full, partial, and fallback modes
- confidence-aware decisions: approve, review, or block
- explainable outputs including top factors and modality contributions
- audit log view inside the dashboard
- standalone NLP and Voice APIs with Swagger documentation

---

## Page 3: System Requirements

### Hardware

- modern laptop or desktop
- minimum 8 GB RAM recommended
- 16 GB RAM preferred if running training workflows
- GPU is optional; CPU works for serving and demo use

### Software

- Python 3.12 recommended
- Node.js 18+ recommended for the dashboard
- `pip` for Python dependency installation
- browser such as Chrome, Edge, or Firefox
- optional: Conda or `venv` for environment management

### Internet Requirements

- internet is required the first time the NLP model is loaded from Hugging Face
- internet is required for the Voice transcription feature because it uses ElevenLabs
- internet is not strictly required for local dashboard simulation once dependencies are installed, if no hosted resources are needed

## Installation / Setup Instructions

### 1. Clone the project

```bash
git clone <your-repo-url>
cd Guardian---Tri-Modal_Fraud_Detection
```

### 2. Create and activate a Python environment

Option A: `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Option B: Conda

```bash
conda create -n guardian python=3.12 -y
conda activate guardian
```

### 3. Install Python dependencies

For the main Python services:

```bash
pip install -r requirements.txt
```

If you only want the NLP service:

```bash
pip install -r NPL/requirements.txt
```

If you only want the Voice service:

```bash
pip install -r Voice/requirements.txt
```

### 4. Install frontend dependencies

```bash
cd guardian-dashboard
npm install
cd ..
```

### 5. Configure environment variables

Voice transcription requires:

- `ELEVENLABS_API_KEY`

Example:

```bash
export ELEVENLABS_API_KEY=your_key_here
```

Optional NLP model variables:

- `NLP_MODEL_NAME`
- `NLP_MODEL_REVISION`
- `NLP_MODEL_DIR`
- `NLP_THRESHOLD`
- `NLP_DEVICE`
- `NLP_CACHE_DIR`

### 6. Run backend and frontend

Recommended local setup:

- Fusion API on port `8080`
- NLP API on port `8001`
- Voice API on port `8000`
- Dashboard on port `3000`

On Windows, a one-click launcher is included:

- `start_guardian.bat`

Run Fusion:

```bash
python3 -m uvicorn fusion.api:app --reload --port 8080
```

Run NLP:

```bash
python3 -m uvicorn NPL.api.api:app --reload --port 8001
```

Run Voice:

```bash
python3 -m uvicorn Voice.api:app --reload --port 8000
```

Run Dashboard:

```bash
cd guardian-dashboard
npm run dev
```

Notes:

- the dashboard proxy is configured to call `/api` through `localhost:8080`
- on Windows, `start_guardian.bat` opens Fusion, NLP, Voice, and Frontend automatically
- the first run may take longer because the NLP and Voice model artifacts are downloaded from Hugging Face and cached locally
- the dashboard’s live analysis screen currently uses simulated stream outputs for demo interaction
- the Fusion API itself is real and can be tested through Swagger or direct requests

## Login / Access Instructions

- no user account is required
- no login screen exists in the current prototype
- no default credentials are needed

## System Navigation Overview

### Dashboard

The main dashboard is the entry point for analysis. It allows the user to:

- enter transaction details
- choose the channel
- enable or disable NLP and Voice streams depending on the scenario
- run fraud analysis
- inspect scores and explanations

### Menu Options

Left sidebar options:

- `Dashboard`: main fraud analysis screen
- `Audit Log`: previously analyzed transactions and explanations

### Key Buttons

- `Analyze`: runs the current transaction scenario
- `Clear`: clears current transaction inputs and results
- stream toggles: enable or disable NLP and Voice where allowed
- sample buttons in NLP and Voice: preload example suspicious or legitimate inputs

## Core Functionalities

### Task 1: Start the system

**Purpose**  
Start the backend services and the dashboard so the prototype can be used.

**Steps**

1. Activate the Python environment.
2. Install dependencies if not already installed.
3. Start Fusion on port `8080`.
4. Start NLP on port `8001`.
5. Start Voice on port `8000`.
6. Start the dashboard in `guardian-dashboard`.

**Expected Outcome**

- dashboard opens at `http://localhost:3000`
- Fusion health endpoint is reachable
- Swagger pages for APIs are available

**Alternative (Windows)**  
Double-click `start_guardian.bat` from the project root to launch all major services automatically.

### Task 2: Analyze a transaction in the dashboard

**Purpose**  
Evaluate a sample transaction and view the predicted fraud outcome.

**Steps**

1. Open the dashboard.
2. Enter `Transaction ID`.
3. Enter `Amount`.
4. Select a `Channel`.
5. Enter `Country`.
6. Select `Merchant Category`.
7. Enter `Merchant Text`.
8. Optionally enter a `Transaction narrative`.
9. Optionally select a voice sample if the chosen channel allows Voice.
10. Click `Analyze`.

**Expected Outcome**

- a fraud score is displayed
- a decision appears: `approve`, `review`, or `block`
- confidence, risk level, ensemble mode, and explanation are shown

### Task 3: View the audit log

**Purpose**  
Review previous analysis results and explanations.

**Steps**

1. Open the dashboard.
2. Click `Audit Log`.
3. Use filters such as `all`, `approve`, `review`, or `block`.
4. Click a record to expand details.

**Expected Outcome**

- stored analysis results are listed
- expanded rows show modality scores, top risk factors, explanation, and metadata

### Task 4: Use the NLP API

**Purpose**  
Submit transaction text to the NLP model and receive fraud scoring.

**Steps**

1. Start the NLP API.
2. Open Swagger at `http://127.0.0.1:8001/docs`.
3. Open `POST /api/v1/nlp/score`.
4. Paste a request body.
5. Click `Execute`.

**Expected Outcome**

- the API returns `score_nlp`
- response includes `status`, `model_version`, and signal details

### Task 5: Use the Voice API

**Purpose**  
Score an audio file as spoof or bonafide, or transcribe it.

**Steps**

1. Start the Voice API.
2. Open Swagger for the selected port.
3. Use `POST /api/v1/voice/score` to upload audio.
4. Optionally use `POST /api/v1/voice/transcribe`.

**Expected Outcome**

- `/score` returns spoof/bonafide probabilities
- `/transcribe` returns text transcription metadata

## AI Features

### Numerical Fraud Detection

**What it does**

- evaluates transaction patterns like amount anomaly, geography, and merchant category risk

**What input the user must provide**

- amount
- country
- merchant category
- channel

**What output means**

- higher fraud score means stronger suspicion
- contributes to the fusion decision

### NLP Fraud Detection

**What it does**

- analyzes transaction text for semantic fraud cues, typosquatting, and inconsistencies

**What input the user must provide**

- merchant text
- optional narrative or other text fields

**What output means**

- `score_nlp` is the fraud probability from the NLP stream
- `predicted_fraud` indicates threshold-based classification
- `status=degraded` means the model used fallback behavior or had weak input

### Voice Fraud Detection

**What it does**

- estimates whether uploaded or selected voice audio appears spoofed or genuine
- optionally transcribes speech using ElevenLabs

**What input the user must provide**

- audio file

**What output means**

- high `spoof_prob` means likely synthetic or fake audio
- transcription output shows recognized speech and metadata

### Fusion Layer

**What it does**

- combines Numerical, NLP, and Voice outputs into one final fraud decision

**What input the user must provide**

- transaction payload and available modalities

**What output means**

- `fraud_score`: final combined risk
- `decision`: `approve`, `review`, or `block`
- `confidence`: trust level in the decision
- `ensemble_mode`: `full`, `partial`, or `fallback`

**Important note**  
Predictions may not be 100% accurate. The system is a prototype and should be treated as decision support, not an autonomous authority.

## AI Feature Usage

### Example: Dashboard Transaction Analysis

**Task**

1. Open the dashboard.
2. Fill in the transaction form.
3. Enable or disable optional streams.
4. Click `Analyze`.

**Result**

- the transaction is evaluated
- the fraud score, decision, and explanation are displayed

### Example: NLP API Classification

**Input**

- upload or submit transaction text payload

**Output**

- classification-style fraud score with related signal values

## API Endpoints

### Fusion API

- `GET /api/v1/fusion/health`
- `GET /api/v1/fusion/modalities`
- `POST /api/v1/fusion/evaluate`
- `POST /api/v1/fusion/batch`
- `GET /api/v1/audit`
- `GET /api/v1/audit/{request_id}`

### NLP API

- `GET /health`
- `POST /api/v1/nlp/score`

### Voice API

- `POST /api/v1/voice/score`
- `POST /api/v1/voice/transcribe`

## Error Handling & Troubleshooting

### Dashboard shows API offline

- ensure Fusion backend is running on port `8080`
- confirm dashboard dev server is running on port `3000`

### Swagger page does not open

- check whether the correct API service is running
- verify the port used in the startup command
- remember that NLP uses `8001` and Voice uses `8000`

### NLP API starts but returns degraded mode

- hosted Hugging Face model may not be reachable
- local fallback checkpoint may be missing
- the service may be using heuristic fallback

### Voice transcription fails

- ensure `ELEVENLABS_API_KEY` is set
- ensure internet access is available

### Module import or dependency errors

- reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Frontend loads but analysis is not fully connected

- the dashboard currently uses simulated stream scoring for the live analysis demo
- Fusion API health is real, but analysis display is primarily demo oriented

## Security & Privacy Notes

- do not upload real confidential customer data unless storage and access controls are in place
- voice files may contain sensitive biometric or identifying information
- API keys such as ElevenLabs credentials must not be shared publicly
- do not commit local model artifacts or private credentials to Git
- hosted models and logs should be access-controlled where appropriate

## FAQs

### Why is my prediction slow?

- first-time model loading from Hugging Face may take longer
- first-time Voice model download may also take time
- transcription depends on external API latency

### Why does NLP show degraded status?

- the hosted model may not have loaded
- local fallback may be missing
- the input text may be too short or incomplete

### What file formats are supported for voice?

- audio uploads are accepted through the Voice API as supported by the preprocessing stack
- common examples include WAV and FLAC

### Do I need all three modalities to use the system?

- no
- the Fusion layer supports full, partial, and fallback modes

### Why does the dashboard still work even if some services are down?

- the dashboard includes simulated analysis behavior for the demo
- some backend health checks are real, but analysis display is not fully wired to every live service

## Support / Contact Information

- GitHub repository: add your repository URL here
- NLP Stream: Luis Mateo Sanchez Loaiza
- Voice Stream: Arsenii Buriak
- Fusion Layer: Sherwayne Allen
- Numerical Stream: Ivan Yu
- Instructor / course contact: add instructor email if required
