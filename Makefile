.PHONY: clean requirements help

.DEFAULT_GOAL:= requirements

## Install Python Dependencies
requirements: 
	conda env create --file environment.yaml

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
