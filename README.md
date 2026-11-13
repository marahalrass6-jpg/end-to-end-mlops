# End-to-End MLOps Project

A production-style machine-learning service that covers the full model lifecycle:

1. Reproducible training and evaluation.
2. Experiment tracking with MLflow.
3. Versioned model artifact and metadata.
4. FastAPI prediction service.
5. Docker packaging.
6. Automated tests and GitHub Actions CI.
7. Health, readiness, and model metadata endpoints.

The demo uses scikit-learn's built-in breast-cancer classification dataset so the project can train without a separate data download. Replace `src/data.py` with a governed data connector for a real deployment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/train.py
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

## Docker

```bash
docker compose up --build
```

API endpoints:

- `GET /health` — process health.
- `GET /ready` — confirms the model is loaded.
- `GET /metadata` — model version, metrics, and feature schema.
- `POST /predict` — returns class, probabilities, and model version.

Example request:

```bash
curl -X POST http://localhost:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d @examples/predict.json
```

## MLflow

Training logs parameters, metrics, and the serialized model to the local SQLite backend `mlflow.db`. Start the local UI with:

```bash
mlflow ui --backend-store-uri ./mlflow.db
```

The application itself loads the controlled artifact under `artifacts/`, not an arbitrary latest run. In production, promote artifacts through a registry and deployment approval process.

## Verified run results

Using the built-in scikit-learn breast-cancer dataset and `random_state=42`, the Random Forest pipeline achieved `95.6140%` accuracy, `F1=0.9655`, and `ROC-AUC=0.9932`. Training completed in approximately `8.5 seconds` in the validation environment, created a `654,770-byte` model artifact, and recorded MLflow run `fa05b044687543818741eeb5f6c70a55` in a `1,032,192-byte` SQLite tracking database.

## Project structure

```text
app/                 FastAPI service
src/                 data, training, and artifact utilities
scripts/train.py     reproducible training entry point
tests/               unit and API tests
artifacts/           generated model and metadata
Dockerfile           container image
compose.yaml         local service definition
.github/workflows/   CI pipeline
```

## Production hardening checklist

- Replace the built-in demo data with a versioned data contract.
- Add authentication, rate limits, request logging, and PII controls.
- Add drift and data-quality monitoring.
- Use a model registry with approval gates and rollback.
- Pin dependencies with a lockfile and scan images for vulnerabilities.
- Use time-based validation where the business problem is temporal.
