# 🦠 ScrubDx AI – Scrub Typhus Prediction System

Industry-grade ML-powered clinical decision support for **Orientia tsutsugamushi** (Scrub Typhus) detection.

---

## 📁 Project Structure

```
scrub_typhus_project/
├── backend/
│   └── api.py              # FastAPI + Pydantic v2 backend
├── frontend/
│   └── index.html          # 4-page Material-style UI
├── ml/
│   ├── train_model.py      # Dataset generation + ML pipeline training
│   ├── model_pipeline.pkl  # Trained sklearn pipeline
│   └── model_metrics.json  # Accuracy, ROC-AUC, feature importance
├── data/
│   └── scrub_typhus_20k.csv  # 20,000-row clinically-grounded dataset
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (first time only)
```bash
python ml/train_model.py
```
Expected: Accuracy ~89.7%, ROC-AUC ~0.963

### 3. Start the FastAPI backend
```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```
API Docs: http://localhost:8000/api/docs

### 4. Open the frontend
```bash
# Simply open in browser:
open frontend/index.html
# or serve it:
python -m http.server 3000 --directory frontend
```

---

## 🏗 Architecture

```
[Browser / Clinician]
        ↓ POST /api/predict
[FastAPI + Pydantic v2]
  • 35+ clinical validators
  • Cross-field validation (jaundice ↔ bilirubin)
        ↓
[RandomForest Pipeline]
  • StandardScaler → RF(300 trees, max_depth=12)
  • Trained on 20K synthetic clinical cases
        ↓
[XAI Engine]
  • Gini Feature Importance (top 12)
  • Clinical annotations per feature
        ↓
[Treatment Engine]
  • WHO/IDSA-aligned doxycycline/azithromycin recs
  • Severity-based escalation
        ↓
[JSON Response → UI]
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 89.65% |
| ROC-AUC | 0.9634 |
| CV ROC-AUC | 0.965 ± 0.003 |
| F1 (Scrub Typhus) | 0.91 |
| Recall (Scrub Typhus) | 0.90 |
| Precision (Scrub Typhus) | 0.93 |

---

## 🔑 Key Features

### Clinical Feature Set (37 features)
| Category | Features |
|----------|----------|
| Demographic | age, gender, rural_background, occupation_risk |
| Environmental | monsoon_season, altitude, rainfall, humidity, temp, outdoor_exposure |
| Symptoms | fever_days, fever_temp, headache, myalgia, cough, nausea, abdo_pain, breathlessness, chills, altered_sensorium, jaundice, lymphadenopathy, rash, **eschar**, upper_eyelid_edema |
| Laboratory | WBC, platelets, hemoglobin, AST, ALT, blood_urea, creatinine, CRP, sodium, bilirubin |
| Diagnostics | IgM ELISA, Weil-Felix |

### Why These Features?
- **Eschar** (importance: 15.2%): Pathognomonic, present in 13–18% of Indian cases
- **Fever duration** (8.9%): >5 days significantly increases probability
- **Platelet count** (8.1%): Thrombocytopenia is key lab marker
- **AST** (7.3%): Anicteric hepatitis characteristic of rickettsial disease
- **Blood urea** (6.9%): Renal involvement in 87% of Himalayan cases (Pathania 2019)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root / service info |
| GET | `/health` | Health check |
| GET | `/api/docs` | Swagger UI |
| POST | `/api/predict` | **Main prediction endpoint** |
| GET | `/api/model-info` | Model metrics + feature importance |
| GET | `/api/dataset-stats` | Dataset summary statistics |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t scrubdx-api .

# Run
docker run -p 8000:8000 scrubdx-api
```

---

## 🌐 Deployment Options

| Component | Platform |
|-----------|----------|
| FastAPI Backend | **Render** / Railway / AWS EC2 |
| Frontend | **Vercel** / Netlify / GitHub Pages |
| Full Stack | **Railway** / Heroku |

### Deploy FastAPI on Render:
1. Push repo to GitHub
2. New Web Service → connect repo
3. Build: `pip install -r requirements.txt && python ml/train_model.py`
4. Start: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`

---

## 📚 Clinical References

1. Pathania et al. 2019 – Scrub typhus in sub-Himalayan India (n=54)
2. Huang et al. 2025 – BPNN prediction model, Ganzhou City
3. WHO 2023 – Scrub typhus treatment guidelines
4. IDSA – Rickettsial disease management guidelines
5. Pathania 2019 – Upper eyelid edema in 74% of Himalayan cases

---

## ⚕️ Disclaimer

This system is for **clinical decision support only**. It does not replace physician judgment.
In endemic areas, empirical doxycycline should be initiated without waiting for laboratory confirmation.

---

## 👨‍💻 Interview Talking Points

> **"Why FastAPI?"** – High-performance async framework with native Pydantic integration for strong type validation, auto-generated OpenAPI docs, and production-ready ASGI server support.

> **"Why Pydantic v2?"** – Strict clinical validators prevent garbage inputs reaching the model. Cross-field validators (e.g., jaundice ↔ bilirubin) ensure clinical consistency.

> **"Why Random Forest?"** – Optimal for tabular medical data: handles non-linear relationships, robust to noise, provides native feature importance for XAI, and requires minimal hyperparameter tuning.

> **"What is the XAI approach?"** – Gini impurity-based feature importance from 300 trees, ranked and annotated with clinical context, making predictions interpretable to clinicians.
