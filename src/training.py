from __future__ import annotations
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = load_breast_cancer().feature_names.tolist()

def train(output_dir: str | Path = 'artifacts', seed: int = 42):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    model = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1))])
    mlflow.set_tracking_uri('sqlite:///' + str((Path('mlflow.db')).resolve()))
    mlflow.set_experiment('end-to-end-mlops-demo')
    with mlflow.start_run() as run:
        mlflow.log_params({'model':'random_forest','n_estimators':200,'max_depth':10,'seed':seed})
        model.fit(X_train, y_train); pred=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
        metrics={'accuracy':float(accuracy_score(y_test,pred)),'f1':float(f1_score(y_test,pred)),'roc_auc':float(roc_auc_score(y_test,proba))}
        mlflow.log_metrics(metrics); mlflow.sklearn.log_model(model, 'model', serialization_format='pickle')
        metadata={'model_name':'breast-cancer-random-forest','version':'1.0.0','mlflow_run_id':run.info.run_id,'features':FEATURE_NAMES,'metrics':metrics,'positive_class':'malignant','input_shape':len(FEATURE_NAMES)}
    joblib.dump(model, out/'model.joblib'); (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata
