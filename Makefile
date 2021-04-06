freeze_package:
	poetry export --without-hashes -f requirements.txt --output requirements.txt
run_genia:
	python run/run_ner.py run/configs/genia_config.json
run_genia_mrc:
	python run/run_ner.py run/configs/genia_mrc_config.json
run_twlife:
	python run/run_ner.py run/configs/twlife_config.json
run_twlife_mrc:
	python run/run_ner.py run/configs/twlife_mrc_config.json