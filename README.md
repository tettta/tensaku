# Tensaku

Config-first, plugin-ready toolkit for Japanese Short Answer Scoring (SAS) with Confidence, HITL, and Active Learning.

## Quick start
```bash
pip install -e .
tensaku -c configs/exp_al_hitl.yaml --help

nohup python scripts/run_al_grid.py \
  --config configs/exp_standard.yaml \
  --root . \
  > logs/master.log 2>&1 & \
  tail -f logs/master.log
