.PHONY: install test lint run clean

install:
	uv venv && uv pip install -e ".[dev]"

test:
	uv run pytest tests -v

lint:
	uv run ruff check src tests scripts

run:
	uv run python scripts/run_ablation.py --outdir results

clean:
	rm -rf results .pytest_cache .ruff_cache __pycache__
