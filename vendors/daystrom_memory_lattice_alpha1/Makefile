PYTHON ?= python3

bench-small:
	$(PYTHON) bench/bench_dml_vs_rag.py --corpus-size 50 --queries 5 --output bench/results-small.csv

bench-large:
	$(PYTHON) bench/bench_dml_vs_rag.py --corpus-size 250 --queries 25 --output bench/results-large.csv
