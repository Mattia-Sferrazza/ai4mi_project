# Snellius preprocessing runbook

This workflow deliberately stops between experiment stages so preprocessing
and smoke-test outputs can be checked before more GPU jobs are submitted. The
six-patient test cohort is never evaluated during these decisions.

## Files

- `preprocess_full_segthor.sbatch`: extract, preprocess, validate, and render QC
- `train_preprocessing_experiment.sbatch`: one generic GPU training/evaluation job
- `submit_preprocessing_pair.sh`: submit two matched conditions for one or more seeds
- `submit_augmentation_screen.sh`: submit geometric, intensity, and combined augmentation
- `show_preprocessing_results.sh`: print completed summaries and HD95 results

## Experiment sequence

1. Run `preprocess_full_segthor.sbatch`.
2. Download and inspect the two QC PNGs.
3. Submit a one-epoch pair as a full-pipeline smoke test.
4. Submit HU-only versus CLAHE, seed 0, for 25 epochs.
5. Select the preprocessing variant using validation 3-D Dice, per-organ Dice,
   HD95, curves, and qualitative predictions.
6. Reuse the seed-0 no-augmentation baseline and submit geometric-only,
   intensity-only, and combined augmentation simultaneously for 25 epochs.
7. Only if a condition looks useful, repeat the winner and no augmentation
   with seeds 1 and 2.

## Pair commands

Smoke test:

```bash
bash snellius/submit_preprocessing_pair.sh \
  SEGTHOR_FULL none \
  SEGTHOR_FULL_CLAHE none \
  smoke_1epoch 0 1
```

CLAHE screen:

```bash
bash snellius/submit_preprocessing_pair.sh \
  SEGTHOR_FULL none \
  SEGTHOR_FULL_CLAHE none \
  clahe_screen_seed0 0 25
```

Augmentation screen, assuming HU-only was selected, consists of three jobs:

```bash
bash snellius/submit_augmentation_screen.sh \
  SEGTHOR_FULL augmentation_screen_seed0 0 25
```

If CLAHE was selected, replace `DATASET=SEGTHOR_FULL` with
`DATASET=SEGTHOR_FULL_CLAHE`.

Example confirmation with seeds 1 and 2:

```bash
bash snellius/submit_preprocessing_pair.sh \
  SEGTHOR_FULL none \
  SEGTHOR_FULL combined \
  augmentation_confirmation 1,2 25
```

Use a new run tag for every submission. Jobs in each pair are independent and
can run simultaneously.
