.PHONY: setup setup-backend setup-frontend test lint run-backend run-frontend dev docker

# Setup all dependencies
setup: setup-backend setup-frontend

# Setup Python backend environment
setup-backend:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r backend/requirements.txt

# Setup frontend environment
setup-frontend:
	cd frontend && npm install

# Run unit tests only (excludes integration tests)
test:
	.venv/bin/pytest backend/tests -v --ignore=backend/tests/integration

# Run integration tests that use real APIs (consumes API quota)
test\:integration:
	.venv/bin/pytest backend/tests/integration -v -s

# Lint backend code
lint:
	.venv/bin/ruff check backend || true
	cd frontend && npm run lint

# Run backend server
run-backend:
	cd backend && ../.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run frontend dev server
run-frontend:
	cd frontend && npm run dev

# Run both services (requires separate terminals)
dev:
	@echo "Run 'make run-backend' in one terminal and 'make run-frontend' in another"

# Build and run with Docker
docker:
	docker-compose up --build

# Run evaluation suite
eval:
	.venv/bin/python backend/eval/run_eval.py
