SHELL = /bin/bash
target: prerequiites
	recipe

.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Cleaning (here style is prerequisite)
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
    pytest -m "not training"
    cd tests && great_expectations checkpoint run project.csv
    cd tests && great_expectations checkpoint run tag
    cd tests && great_expectations checkpoint run label

# Makefile
.PHONY: dvc
dvc:
    dvc add data/projects.csv
    dvc add data/tags.csv
    dvc add data/labeled_projects.csv
    dvc push

