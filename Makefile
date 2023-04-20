install:
	poetry add pre-commit
	git init
	pre-commit install
	git add .
	git commit -m "ADD: init repo"
