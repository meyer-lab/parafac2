
test: .venv
	rye run pytest -s -v -x

.venv: pyproject.toml
	rye sync

# testprofile:
# 	poetry run python3 -m cProfile -o profile -m pytest -s -x -v
# 	gprof2dot -f pstats --node-thres=1.0 profile | dot -Tsvg -o profile.svg

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=sccp --cov-report xml:coverage.xml

clean:
	rm -rf output profile profile.svg

pyright: .venv
	rye run pyright sccp