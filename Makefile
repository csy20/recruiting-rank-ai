.PHONY: install test lint clean docker-build docker-run docker-compose-up

install:
	pip install -r requirements.txt
	pip install pyyaml pytest

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m py_compile rank.py serve.py config.py models.py
	python3 -m py_compile features/extractor.py
	python3 -m py_compile scoring/ranker.py scoring/explainer.py scoring/skill_graph.py scoring/jd_parser.py
	python3 -m py_compile utils/nlp_utils.py

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf data/*.npz data/*.csv
	rm -f submission.csv
	rm -f *.json
	rm -rf build dist *.egg-info

docker-build:
	docker build -t recruiting-rank-ai .

docker-run:
	docker run -p 8000:8000 recruiting-rank-ai

docker-compose-up:
	docker-compose up --build
