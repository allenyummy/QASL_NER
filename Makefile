freeze_package:
	poetry export --without-hashes -f requirements.txt --output requirements.txt
runr:
	python run/run_ner.py run/configs/config.json