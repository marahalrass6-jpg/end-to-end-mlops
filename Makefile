install:
	python -m pip install -r requirements.txt
train:
	python scripts/train.py
api:
	uvicorn app.main:app --reload
test:
	pytest -q
