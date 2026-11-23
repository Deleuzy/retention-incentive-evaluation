# Full pipeline. About 2 minutes on an 8 GB M1 laptop.
.PHONY: all causal pilot r report site test lint check

all: causal pilot r report

causal:   ## LaLonde / NSW estimates, 500 bootstrap resamples
	uv run python -m rie.causal --reps 500

pilot:    ## power calculation and SIMULATED pilot
	uv run python -m rie.pilot

r:        ## R cross-check (needs R with jsonlite; foreign and survival ship with R)
	Rscript r/crosscheck.R

report:   ## regenerate every document that contains numbers
	uv run python -m rie.report

site:
	uv run python scripts/build_site.py --out site

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	@command -v actionlint >/dev/null && actionlint || echo "actionlint not installed, skipped"

check: lint test
	uv run python -m rie.report --check
