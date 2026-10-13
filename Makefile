# Convenience targets. Every target is a thin wrapper around a Python script,
# so the same steps work without make (see REPRODUCING.md).

PYTHON ?= python

.PHONY: install test reproduce download corpora mwus indexes benchmarks experiments report clean-data

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

download:
	$(PYTHON) pipeline/step1_download.py

corpora:
	$(PYTHON) pipeline/step2_prepare_corpora.py

mwus:
	$(PYTHON) pipeline/step3_extract_mwus.py

indexes:
	$(PYTHON) pipeline/step4_build_indexes.py

benchmarks:
	$(PYTHON) pipeline/step5_prepare_benchmarks.py

experiments:
	$(PYTHON) pipeline/step6_run_experiments.py

report:
	$(PYTHON) pipeline/step7_write_report.py

reproduce: download corpora mwus indexes benchmarks experiments report

clean-data:
	rm -rf data
