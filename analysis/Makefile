.PHONY: setup analyze visualize run clean

# Default parameters
SLOW_MO ?= 4
PLAYER ?= Shohei_Ohtani

setup:
	uv sync
	uv run scripts/setup_models.py --config models.yml

analyze:
	cd .. && uv run --project analysis pipelines/analyze_video.py --slow_mo $(SLOW_MO) --player $(PLAYER)

visualize:
	cd .. && uv run --project analysis pipelines/visualize_video.py --player $(PLAYER)

run: analyze visualize

clean:
	rm -rf ../data/interim/$(PLAYER)/*
	rm -rf ../data/processed/$(PLAYER)/*
