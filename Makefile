# Mission Tasking Service — developer Makefile.
#
# Targets are thin wrappers around `uv`, `docker compose`, and `kubectl` so
# `make <thing>` is the single discoverable surface. `make help` lists them.

COMPOSE := docker compose -f deploy/docker/docker-compose.yml
SAMPLE_COMMAND := Patrol the yard perimeter at 60 meters with EO.
SAMPLE_AREA := yard-simple
SAMPLE_DRONE := long-endurance-quad

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN{FS=":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# --- local dev ---------------------------------------------------------------

.PHONY: install
install: ## uv sync (creates .venv if missing)
	uv sync --extra dev

.PHONY: lint
lint: ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

.PHONY: fmt
fmt: ## auto-fix lint + format in place
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: typecheck
typecheck: ## mypy (strict)
	uv run mypy app

.PHONY: test
test: ## pytest -q
	uv run pytest -q

.PHONY: test-v
test-v: ## pytest -v
	uv run pytest -v

# --- one-command demo --------------------------------------------------------

.PHONY: demo
demo: up seed compile-sample ## bring up stack, seed data, compile a sample mission, then print where to look
	@echo ""
	@echo "✔ Demo ready."
	@echo "  UI:         http://localhost:8000"
	@echo "  Grafana:    http://localhost:3000  (admin / admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Docs:       open docs/REVIEWER_GUIDE.md"

.PHONY: up
up: ## docker compose up (MTS + Postgres + Redis + OTel + Prometheus + Grafana)
	$(COMPOSE) up --build -d
	@echo "waiting for /healthz..."
	@for i in $$(seq 1 30); do \
		curl -fs http://localhost:8000/healthz > /dev/null && break; \
		sleep 2; \
	done

.PHONY: down
down: ## docker compose down (preserves Postgres volume)
	$(COMPOSE) down

.PHONY: clean
clean: ## docker compose down + remove the Postgres volume
	$(COMPOSE) down -v

.PHONY: logs
logs: ## tail mts logs
	$(COMPOSE) logs -f mts

.PHONY: seed
seed: ## load data/areas + data/drones into Postgres
	$(COMPOSE) exec mts python scripts/seed_db.py

.PHONY: compile-sample
compile-sample: ## POST a sample command to /v1/missions:compile
	@curl -s http://localhost:8000/v1/missions:compile \
		-H 'content-type: application/json' \
		-d '{"command":"$(SAMPLE_COMMAND)","area_id":"$(SAMPLE_AREA)","drone_state":{"drone_profile_id":"$(SAMPLE_DRONE)","battery_pct":100}}' \
		| python -m json.tool

# --- evals -------------------------------------------------------------------

.PHONY: evals
evals: ## run the LangSmith eval suite against the local service
	uv run python evals/run_evals.py

# --- kubernetes (kind) -------------------------------------------------------

.PHONY: kind-up
kind-up: ## create kind cluster + load image + apply postgres-redis + helm install
	kind create cluster --name mts || true
	docker tag docker-mts:latest mts:0.1.0
	kind load docker-image mts:0.1.0 --name mts
	kubectl apply -f deploy/k8s/postgres-redis.yaml
	kubectl wait --for=condition=ready pod -l app=postgres --timeout=120s
	helm upgrade --install mts deploy/helm/mts/ -f deploy/helm/mts/values-kind.yaml \
		--set-string secrets.API_KEY="$$(grep '^API_KEY=' .env | cut -d= -f2-)" \
		--set-string secrets.LANGSMITH_API_KEY="$$(grep '^LANGSMITH_API_KEY=' .env | cut -d= -f2-)"

.PHONY: kind-down
kind-down: ## delete the kind cluster
	kind delete cluster --name mts
