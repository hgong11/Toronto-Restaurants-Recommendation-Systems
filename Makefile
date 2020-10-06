ROOT=$(shell pwd)

run:
	python3 -u ${ROOT}/app.py

download:
	python3 -u ${ROOT}/tools/data_downloader.py