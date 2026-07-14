.PHONY: install test lint validate train quantize deploy
install:      ## install runtime + dev deps
	pip install -e ".[dev]"
test:         ## run unit tests with coverage
	pytest --cov=calfnest --cov-report=term-missing
lint:         ## static checks
	ruff check .
validate:     ## synthetic-data statistical validation (C3)
	python -m models.validate_synthetic
train:        ## train Sentinel + baselines and run the lead-time benchmark
	python -m models.train
quantize:     ## int8-quantise the risk head and check the <256 KB edge budget
	python -m models.quantize
serve:        ## run the model REST API (needs the serve extra)
	uvicorn models.serve:app --reload
deploy:       ## signed OTA rollout to the edge fleet (placeholder)
	@echo "Sign + publish firmware/model OTA bundle"
