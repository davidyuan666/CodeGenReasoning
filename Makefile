commit:
	git add .
	git commit -m "update"
	git push origin main


install:
	pdm install .

run:
	pdm run run.py
