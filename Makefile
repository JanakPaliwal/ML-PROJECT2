# Makefile for Short-Term Mandi Price Forecasting

.PHONY: run train dashboard clean help

help:
	@echo "Available commands:"
	@echo "  make run        : Train models and print benchmark evaluation report"
	@echo "  make dashboard  : Launch interactive web dashboard at http://localhost:8080"
	@echo "  make train      : Re-run ML training pipeline from scratch"
	@echo "  make clean      : Remove generated caches and temporary files"

run:
	python main.py

dashboard:
	python run_demo.py

train:
	python src/train_pipeline.py

clean:
	python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]"
