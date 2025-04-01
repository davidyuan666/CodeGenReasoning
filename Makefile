commit:
	git add .
	git commit -m "update"
	git push origin main


install:
	pdm install .


cot:
	pdm run python utils/cot_util.py

start:
	pdm run python main.py
