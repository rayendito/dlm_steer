#!/usr/bin/env bash

./.venv/bin/python run.py \
    --method timpa_steer \
    --steer add \
    --dataset imdb \
    --temperature 0.1 \
    --margin 0.05
