#!/bin/bash
# E2 and E4 at the reduced seed counts, then the report.
cd "$(dirname "$0")/../.."
python3 experiments/open_space/campaign.py --exp E2 --seeds 201-212 --workers 2 --out experiments/open_space/out
python3 experiments/open_space/campaign.py --exp E4 --seeds 201-210 --workers 2 --out experiments/open_space/out
python3 experiments/open_space/report.py --out experiments/open_space/out
