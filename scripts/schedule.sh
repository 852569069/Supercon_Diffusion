#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python train.py trainer.max_epochs=50000 trainer.min_epochs=50000 trainer.devices=4 data.batch_size=200


