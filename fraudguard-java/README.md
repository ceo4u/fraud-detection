# 🛡️ FraudGuard Enterprise (Spring Boot + FastAPI)

FraudGuard is an enterprise-grade healthcare fraud intelligence system featuring a **Java Spring Boot 3.2** dashboard frontend coupled with a **Python FastAPI** machine learning microservice.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────┐
│  Java Spring Boot Enterprise Frontend (:8080)│
│  - Layered Architecture (Controller/Service) │
│  - Thymeleaf Server-Side Rendering (Dark UI) │
│  - Resilient RestTemplate API Client         │
│  - Plotly.js Visualizations                  │
└──────────────────────┬───────────────────────┘
                       │ HTTP REST (/api/*)
                       ▼
┌──────────────────────────────────────────────┐
│  Python FastAPI ML Backend Service (:8000)   │
│  - Zero Retraining: Reuses verified model.pkl│
│  - Sub-millisecond XGBoost Inferences        │
│  - SHAP Behavioral Explainability Vectors    │
│  - Real-time Pydantic Request Validation     │
└──────────────────────┬───────────────────────┘
                       │ Reads (Read-Only)
                       ▼
┌──────────────────────────────────────────────┐
│  models/ Directory (Existing Assets)         │
│  - model.pkl (PR-AUC 0.78, Recall 0.89)      │
│  - features.parquet (5,410 Providers)        │
│  - test_preds.parquet (1,082 Test Providers) │
│  - shap_values.npy & demo_provider.json      │
└──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Running Both Services)

### Option 1: Native Execution

#### Step 1 — Start the Python FastAPI Backend (:8000)
```bash
cd fraudguard-java/backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
*Health check:* `curl http://localhost:8000/health`

#### Step 2 — Start the Java Spring Boot Frontend (:8080)
```bash
cd fraudguard-java/frontend
mvn clean package -DskipTests
java -jar target/fraudguard-1.0.jar
```
*(Or use `mvn spring-boot:run`)*

#### Step 3 — Open Dashboard
Open your browser to: **[http://localhost:8080](http://localhost:8080)**

---

### Option 2: Docker Compose
```bash
cd fraudguard-java
docker-compose up --build
```

---

## 📊 Endpoints & Verification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health probe and model status |
| `GET` | `/providers?limit=50&sort=desc` | Real test provider predictions |
| `GET` | `/providers/count` | Total providers (`1,082`) and flagged count (`181`) |
| `POST`| `/predict` | Live scoring of raw provider feature dictionary |
| `GET` | `/explain/{providerId}` | SHAP waterfall attribution breakdown |
| `GET` | `/metrics` | Production evaluation metrics (PR-AUC: 0.78, Recall: 0.89) |
| `GET` | `/demo-provider` | Curated deterministic test provider (`PRV52340`) |
| `GET` | `/plots/{name}` | Evaluation curves and EDA plots |

---

## ⚖️ Disclaimer
FraudGuard flags healthcare providers for compliance and audit investigation; it does not constitute definitive proof of fraud. Model trained on historical CMS Medicare claims data.
