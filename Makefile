.PHONY: smoke compile validate mock-backend gateway benchmark-mock clean-pycache

smoke:
	python tests/smoke_test.py

compile:
	python -m compileall training registry serving evaluation benchmark tests scripts

validate:
	python -m scripts.validate_project

mock-backend:
	python -m serving.mock_vllm --host 0.0.0.0 --port 8001

gateway:
	uvicorn serving.gateway:app --host 0.0.0.0 --port 8000

benchmark-mock:
	python -m benchmark.loadgen --config configs/benchmark.yaml --url http://localhost:8000/v1/task/chat --concurrency 2 --duration 10 --output benchmark/results/mock_c2.jsonl
	python -m benchmark.analyze_results benchmark/results/mock_c2.jsonl

clean-pycache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

