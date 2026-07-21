clean_reanalysis:
	rm -rf reanalysis_data/*
	@echo "reanalysis_data now contains : $$(ls reanalysis_data | wc -l) files"