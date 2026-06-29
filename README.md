# Lab 04 Starter — Monitoring & Production Deployment (Iris)

Deploys the service with Docker Compose, instruments it with Prometheus, visualises it in Grafana, drives traffic with a small Python script, and ships offline Responsible-AI reports (drift, explainability, fairness).

## Project structure
```
lab04/
├── src/                       # data.py (load_raw), features.py
├── api/main.py                # FastAPI + Prometheus /metrics
├── train.py                   # Pipeline(StandardScaler, RandomForest)
├── Dockerfile                 # python:3.11-slim, bakes model
├── docker-compose.yml         # api + prometheus + grafana
├── prometheus/prometheus.yml
├── grafana/provisioning/      # datasource + dashboard
├── loadtest/generate_traffic.py   # Python traffic generator
└── monitoring/
    ├── drift_report.py        # Evidently drift demo (HTML + console summary)
    ├── explain.py             # SHAP (Pipeline-aware)
    └── fairness.py            # group metrics (synthetic attribute)
```

## 1. Bring up the stack
```bash
docker compose up --build
```
 API        : http://localhost:8000 
 Prometheus : http://localhost:9090  
 Grafana    : http://localhost:3000  (admin / admin) 

## 2. Simulate the traffic
```bash
pip install -r requirements-dev.txt
python loadtest/generate_traffic.py --url http://localhost:8000 --requests 500 --rate 20
# prints throughput + latency; watch the same traffic on the Grafana dashboard
```

## 3. Responsible-AI reports (offline)
```bash
python monitoring/drift_report.py   # Evidently
python monitoring/explain.py        # SHAP 
python monitoring/fairness.py       # prints per-group metrics
```
> The fairness attribute is **synthetic** (Iris has no sensitive attribute).
