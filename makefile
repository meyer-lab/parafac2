.PHONY: clean test pyright

test: .venv
	uv run pytest -s -v -x --durations=0

.venv: pyproject.toml
	uv sync --dev

testprofile:
	uv run python3 -m cProfile -o profile -m pytest -s -x -v
	uv run gprof2dot -f pstats --node-thres=1.0 profile | dot -Tsvg -o profile.svg

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=parafac2 --cov-report xml:coverage.xml

pyright: .venv
	uv run pyright parafac2
	
clean:
	rm -rf output profile profile.svg

