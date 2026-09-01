.PHONY: test compile

test:
	python -m pytest -q

compile:
	python -m compileall -q divid
