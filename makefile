CONFIG ?= config.yml

.PHONY: train_single run_solver clean_reanalysis help



clean_neural_output:
	@PREVCOUNT=$$(ls model_output/neural_operator | wc -l); \
	rm -rf model_output/neural_operator/*; \
	CURRCOUNT=$$(ls model_output/neural_operator | wc -l); \
	echo "Removed $$(($$PREVCOUNT - $$CURRCOUNT)) files"


run_solver:
	python -m src.entries.run_solver $(CONFIG)

batch_simulation:
	python -m src.entries.batch_simulation $(CONFIG)

train_single:
	python src/neural_operator/train_singlestep.py $(CONFIG)

download_era5:
	python src/entries/download_era5.py $(CONFIG)

inference:
	python src/entries/inference.py $(CONFIG)
	