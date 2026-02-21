.PHONY: setup setup-backend setup-frontend test lint run-backend run-frontend dev docker eval

# ------------------------------------------------------------------ #
# Setup
# ------------------------------------------------------------------ #

# Install all dependencies (backend + frontend)
setup: setup-backend setup-frontend

# Set up Python virtual environment and install backend dependencies
setup-backend:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r backend/requirements.txt

# Install frontend Node dependencies
setup-frontend:
	cd frontend && npm install

# ------------------------------------------------------------------ #
# Development
# ------------------------------------------------------------------ #

# Run backend API server with hot-reload
run-backend:
	cd backend && ../.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run frontend dev server
run-frontend:
	cd frontend && npm run dev

# Run both services concurrently (requires two terminals or a process manager)
dev:
	@echo "Starting backend and frontend in parallel..."
	@make -j2 run-backend run-frontend

# ------------------------------------------------------------------ #
# Testing
# ------------------------------------------------------------------ #

# Run unit tests only (excludes integration tests)
test:
	.venv/bin/pytest backend/tests -v --ignore=backend/tests/integration

# Run integration tests that use real APIs (consumes API quota)
test\:integration:
	.venv/bin/pytest backend/tests/integration -v -s

# ------------------------------------------------------------------ #
# Quality
# ------------------------------------------------------------------ #

# Lint backend and frontend
lint:
	.venv/bin/ruff check backend || true
	cd frontend && npm run lint

# ------------------------------------------------------------------ #
# Docker
# ------------------------------------------------------------------ #

# Build Docker images and start all services
docker:
	docker-compose up --build

# Build images without starting services
docker\:build:
	docker-compose build

# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #

# Run the evaluation suite against the live pipeline
eval:
	.venv/bin/python backend/eval/run_eval.py
