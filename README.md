# Heart Disease Risk Prediction — End-to-End MLOps

Predict presence/absence of heart disease from patient health data and serve it
as a containerised, monitored API. (BITS Pilani AIMLCZG523 — Assignment 01)

> This is a reference scaffold. **Run every step yourself**, capture **your own**
> metrics/screenshots, and **personalise** the EDA, features, tuning and docs so
> the submission reflects your individual work (as the academic-integrity note
> requires).

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data (Task 1)

The UCI archive file `data/processed.cleveland.data` is included. Build the clean CSV:

```bash
python src/data_download.py --source local --in data/processed.cleveland.data --out data/heart_raw.csv
# or fetch fresh:  python src/data_download.py --source uci --out data/heart_raw.csv
```

Generate EDA figures:

```bash
python src/eda.py        # writes reports/figures/{histograms,correlation_heatmap,class_balance}.png
```

## 3. Train + experiment tracking (Tasks 2, 3, 4)

```bash
python src/train.py      # trains LogReg + RandomForest, logs to MLflow, saves models/model.joblib
mlflow ui                # open http://127.0.0.1:5000 to view runs  (screenshot this)
```

## 4. Tests (Task 5)

```bash
pytest -q
flake8 src app tests --max-line-length=100
```

## 5. Serve the API locally (Task 6)

```bash
uvicorn app.main:app --reload          # http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

## 6. Docker (Task 6)

```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

## 7. Kubernetes (Task 7)

```bash
# push image first, then set it in k8s/deployment.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods,svc        # screenshot this
```
(Minikube: `minikube service heart-api-svc --url`)

## 8. Monitoring (Task 8)

```bash
docker compose up --build   # API :8000, Prometheus :9090, Grafana :3000
```
API exposes `/metrics`; Prometheus scrapes it; build a Grafana panel (screenshot).

## CI/CD (Task 5)
`.github/workflows/ci.yml` runs lint -> tests -> training on every push and
uploads the model + reports as artifacts. It fails on any lint/test/train error.

---

## Report skeleton (Task 9 — fill with YOUR results)
1. Setup / install instructions
2. EDA findings (your plots + what they show — class balance, correlations)
3. Modelling choices + tuning (your CV + test metrics table)
4. Experiment-tracking summary (MLflow screenshot)
5. Architecture diagram (data -> train -> MLflow -> model -> Docker -> K8s -> Prometheus/Grafana)
6. CI/CD + deployment screenshots
7. Link to your GitHub repo + deployed URL/access instructions
