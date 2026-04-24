.PHONY: download-data eval-open-models

download-data:
	bash scripts/download_eval_data.sh

eval-open-models:
	bash scripts/evaluate_open_models.sh
