# `uv` is expected on PATH. Override for a project-local bootstrap:
#   make UV=.venv-bootstrap/bin/uv test
UV ?= uv

# Keep uv's caches inside the project so they stay sandbox-writable and are
# removed by `make clean`. Adopters running `uv` directly get uv's defaults.
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python

# Matplotlib writes a font cache on first import and warns loudly if it cannot.
export MPLCONFIGDIR := $(CURDIR)/.mplcache

.PHONY: setup test lint dev demo demo-live numbers audit smoke clean

setup:
	$(UV) sync

test:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests
	$(UV) run pytest

lint:
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

# Inner dev loop: 3 personas, one noise level, one tag. Needs LAB_ANTHROPIC_API_KEY.
dev:
	$(UV) run lab run --config lab.dev.yaml

# Report from committed fixture artifacts. No API key, no network, $0.
demo:
	$(UV) run lab report --from-fixtures --tag v1
	$(UV) run lab diff --from-fixtures --baseline v1 --candidate v2 || true

# The real pipeline end to end. Needs LAB_ANTHROPIC_API_KEY.
demo-live:
	$(UV) run lab run --config lab.yaml

# One request per role against the real API, to validate params before a full run.
smoke:
	$(UV) run lab smoke

numbers:
	$(UV) run lab run --config lab.yaml
	$(UV) run python scripts/regen_numbers.py

audit:
	$(UV) run lab audit --sample 20

clean:
	rm -rf runs .cache .uv-cache .pytest_cache .ruff_cache
