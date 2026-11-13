from fastapi.testclient import TestClient
from app.main import app

def test_health():
    assert TestClient(app).get('/health').status_code == 200

def test_request_validation():
    response=TestClient(app).post('/predict',json={'features':[1,2]})
    assert response.status_code == 422
