#!/bin/bash
# Full open-space campaign, sequential experiments, resumable.
cd "$(dirname "$0")/../.."
W=${WORKERS:-2}
python3 experiments/open_space/campaign.py --exp E1 --seeds 201-230 --workers $W --out experiments/open_space/out
python3 experiments/open_space/campaign.py --exp E3 --seeds 201-230 --workers $W --out experiments/open_space/out
python3 experiments/open_space/campaign.py --exp E2 --seeds 201-220 --workers $W --out experiments/open_space/out
python3 experiments/open_space/campaign.py --exp E4 --seeds 201-215 --workers $W --out experiments/open_space/out
python3 experiments/open_space/report.py --out experiments/open_space/out
