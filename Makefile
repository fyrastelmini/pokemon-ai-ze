# basic makefile for python

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

dataset:
	if [ ! -d "sprites" ]; then \
		git clone https://github.com/PokeAPI/sprites.git; \
	fi
	python -c 'from dataset import generate_dataset; generate_dataset("./sprites/sprites/pokemon/", save=True)'

encoder:
	python encoder.py "config/encoder.yml"