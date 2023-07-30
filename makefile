SHELL := /bin/bash

.PHONY: clean test

test:
	poetry run pytest -s -x -v --full-trace

coverage.xml:
	poetry run pytest -s -x -v --cov=parafac2 --cov-report=xml

clean:
	rm -rf profile profile.svg

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports parafac2
