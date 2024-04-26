
test:
	poetry run pytest -s -x -v

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -x -v
	gprof2dot -f pstats --node-thres=1.0 profile | dot -Tsvg -o profile.svg

clean:
	rm profile.svg profile

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports --check-untyped-defs parafac2
