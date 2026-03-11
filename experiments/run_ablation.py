"""
Ablation Experiment Runner

Orchestrates the seed-ratio ablation study (Research Question 2).
For each combination of (dataset, seed_ratio, seed_strategy, run_id),
runs Phase 2 seed sampling → Phase 3 GPT translation → LaBSE quality eval.

Usage:
    python experiments/run_ablation.py --config experiments/configs/ablation.yaml
    python experiments/run_ablation.py --config experiments/configs/ablation.yaml \\
        --dataset spider --ratio 0.05 --strategy stratified --run 0

Output structure:
    results/ablation/
    └── {dataset}/
        └── {strategy}/
            └── ratio_{ratio:.2f}/
                └── run_{run_id}/
                    ├── seed.json          # sampled seed subset
                    ├── translations.json  # GPT-translated samples
                    └── eval.json          # LaBSE scores + summary stats
"""

# TODO: Implement after Phase 2 and Phase 3 scripts are complete.
