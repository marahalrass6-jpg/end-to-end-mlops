# Architecture

```text
Dataset -> train.py -> MLflow tracking + artifacts/model.joblib
                                      |
                                      v
                              FastAPI /predict
                                      |
                            Docker / Compose / CI
```

The service has explicit readiness semantics: `/health` confirms the process is alive, while `/ready` confirms that the approved artifact and metadata are loaded. This separation supports container orchestration and safer rollouts.
