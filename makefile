.PHONY: clean test pyright

test: .venv
	rye run pytest -s -v -x --durations=0

.venv: pyproject.toml
	rye sync

testprofile:
	rye run python3 -m cProfile -o profile -m pytest -s -x -v
	rye run gprof2dot -f pstats --node-thres=1.0 profile | dot -Tsvg -o profile.svg

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=parafac2 --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright parafac2
	
clean:
	rm -rf output profile profile.svg

