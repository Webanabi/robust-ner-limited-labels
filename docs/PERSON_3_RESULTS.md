# Person 3 DistilBERT results

All eight seed-42 runs and all 32 fixed-condition evaluations completed on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB). Each test cell reports strict entity-level micro F1 as a percentage, followed by inference seconds for 3,453 sentences in parentheses.

| Model | Train data | Selected λ | Clean | Lowercase | Typo-10 | Typo-20 | Train (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean-only | 10% | — | 80.84 (14.00) | 32.90 (13.06) | 77.24 (13.10) | 74.06 (13.68) | 125 |
| Aug | 10% | — | 82.30 (12.98) | 71.66 (13.59) | 79.35 (13.37) | 76.64 (13.70) | 176 |
| Ordinary consistency | 10% | 1.0 | 82.43 (12.59) | 72.79 (12.77) | 80.13 (13.10) | 76.30 (12.77) | 513 |
| Entity-balanced consistency | 10% | 1.0 | 82.99 (13.51) | 74.15 (13.39) | 79.87 (14.45) | 76.73 (13.56) | 524 |
| Clean-only | 100% | — | 87.88 (14.04) | 45.36 (16.09) | 83.63 (15.51) | 78.21 (15.73) | 884 |
| Aug | 100% | — | 88.56 (14.32) | 79.91 (13.69) | 86.91 (15.77) | 84.77 (15.78) | 1507 |
| Ordinary consistency | 100% | 1.0 | 88.84 (16.58) | 80.14 (15.99) | 87.09 (18.09) | 85.25 (18.12) | 4880 |
| Entity-balanced consistency | 100% | 0.5 | 88.00 (14.23) | 80.28 (13.92) | 86.42 (14.09) | 83.83 (14.08) | 4909 |

Each consistency run includes all three λ candidates in its training time. Checkpoint and λ selection uses the mean strict entity F1 on the three validation conditions. The table describes one seed.

## Reproduction and handoff

```bash
python -m unittest discover -s tests -v
python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt --seed 42
python -m scripts.run_person3 --hyperparameters configs/person3_distilbert.json
python -m scripts.summarize_person3
```

Use the approved local files described in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). See the [implementation notes](../src/models/distilbert/README.md) for model settings and CUDA verification.

Local artifacts for review by the project lead:

- `checkpoints/<run_id>/manifest.json` and `distilbert_run.json` contain training times and selection history.
- `results/metrics/<run_id>/test_summary.json` and each `test_<condition>/evaluation.json` contain test metrics.
- `results/predictions/<run_id>/test_<condition>_metadata.json` records inference times.
- `results/person3_training.log` records the training queue.

The run IDs use `<model_id>_<10pct|100pct>_seed42`. Share detailed artifacts through the approved private channel.
