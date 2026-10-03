# SegTHOR preprocessing and augmentation plan

## Fixed data preparation

The final pipeline uses the professor-provided 40-patient labelled archive.
Patients 1-20 from the earlier corrected dataset are not appended because they
duplicate scans in the official archive.

Each patient is processed once with this deterministic transform:

1. Validate CT/GT shape, affine, LPS orientation, and labels 0-4.
2. Clip CT intensities to `[-1000, 1000]` HU and store them as uint8. The
   loader converts these values to `[0, 1]`.
3. Take a centred `384 x 384 mm` in-plane field of view. Stop if this would
   remove any labelled foreground voxel.
4. Resample to `384 x 384` pixels at approximately `1 x 1 mm`. Use bilinear
   interpolation with anti-aliasing for CT and nearest-neighbour interpolation
   for categorical masks.
5. Preserve the original slice count and z spacing. Save the effective 3-D
   spacing for volumetric Dice and HD95 evaluation.

The 1 mm target is close to the native in-plane spacing for 36/40 patients.
The old 1.5 mm target unnecessarily downsampled every scan.

`SEGTHOR_FULL` uses only the fixed HU window. `SEGTHOR_FULL_CLAHE` applies
deterministic slice-wise CLAHE after resizing (`kernel_size=32`,
`clip_limit=0.01`). The CLAHE dataset must have exactly the same geometry,
labels, patient split, and filenames as the primary dataset; only its CT pixel
values may differ.

## Fixed patient-level split

- 28 training patients, 6 validation patients, and 6 labelled test patients.
- Split seed is permanently fixed at `0`.
- Test patients are sampled only from Patients 21-40. Patients 1-20 are
  ineligible because preliminary experiments and visual inspection exposed
  them before this split was designed.
- Validation is stratified: four patients from Patients 1-20 and two from the
  remaining Patients 21-40 are selected after the test set is reserved.
- The test set is not used to select preprocessing, augmentation, checkpoints,
  epochs, or architectures.

`split.json` stores the direct `train`, `val`, and `test` patient lists. Each
patient is preprocessed only once in a shared image/label pool; the loader uses
the manifest to select the requested subset.

## Fixed training objective and reproducibility

Use unweighted Cross-Entropy + Dice loss (`ce_dice`) for the preprocessing
checks. The preliminary 20-patient experiment motivates this choice; this work
does not repeat the loss ablation.

The patient split always uses `split_seed=0`. Training seed `0` is used for the
initial screen. If a change appears useful, repeat only the relevant comparison
with training seeds `1` and `2` before claiming an improvement. A training seed
controls Python, NumPy, PyTorch/CUDA, initialization, data-loader shuffling,
workers, and online augmentation, but never changes patient membership.

Every run writes `run_config.json`, including split and training seeds,
augmentation mode, and software versions. Exact bitwise equality across GPU
types or software versions is not guaranteed even with deterministic settings.

## Minimal preprocessing experiments

Keep the reference model, loss, split, epoch budget, checkpoint rule, and
evaluation code identical in every comparison.

### 1. CLAHE check

With augmentation disabled and training seed 0, compare:

1. `SEGTHOR_FULL`: HU window only
2. `SEGTHOR_FULL_CLAHE`: HU window followed by CLAHE

If CLAHE improves validation performance, repeat both conditions with seeds 1
and 2. Keep CLAHE only if the benefit is consistent and it does not worsen
small-organ HD95 or visibly amplify noise.

### 2. Online augmentation check

Using the selected preprocessing variant, compare with seed 0:

1. `none`
2. `combined`

`combined` uses rotations up to 10 degrees, translations up to 5%, scaling
from 0.9 to 1.1, brightness shifts up to 0.05, contrast/gamma from 0.9 to 1.1,
and Gaussian noise sigma up to 0.02. Geometric and intensity groups each have
probability 0.5. Spatial parameters are shared by CT and mask, and mask
interpolation is nearest-neighbour. Augmentation is online and applied only to
training samples; validation and test are never augmented.

If augmentation improves validation performance, repeat `none` and `combined`
with seeds 1 and 2. The final deliverable is the selected deterministic
preprocessed dataset, `split.json`, metadata, validators, and optional online
augmentation module for all peer architectures.

## Model selection and reporting

Select checkpoints using mean foreground patient-level 3-D validation Dice.
Also inspect per-organ 3-D Dice, HD95 in millimetres, training curves, and a
small qualitative prediction sample. Do not inspect test predictions or test
metrics while choosing preprocessing. The team should evaluate the test set
only after the final modelling choices are locked.

## Local preparation commands

```bash
python -m preprocessing.slice_segthor \
  --source_dir data/segthor_train_full \
  --dest_dir data/SEGTHOR_FULL \
  --shape 384 384 \
  --target-spacing 1.0 1.0 \
  --window -1000 1000 \
  --validation-count 6 \
  --test-count 6 \
  --test-pool-start 21 \
  --split-seed 0 \
  -p -1

python -m preprocessing.validate_processed_segthor data/SEGTHOR_FULL

python -m preprocessing.preview_augmentations \
  --data-dir data/SEGTHOR_FULL \
  --output eda/preprocessing_figures/augmentation_preview.png

python -m preprocessing.slice_segthor \
  --source_dir data/segthor_train_full \
  --dest_dir data/SEGTHOR_FULL_CLAHE \
  --shape 384 384 \
  --target-spacing 1.0 1.0 \
  --window -1000 1000 \
  --clahe --clahe-kernel-size 32 --clahe-clip-limit 0.01 \
  --validation-count 6 --test-count 6 --test-pool-start 21 \
  --split-seed 0 -p -1

python -m preprocessing.validate_processed_segthor data/SEGTHOR_FULL_CLAHE

python -m preprocessing.validate_preprocessing_ablation \
  data/SEGTHOR_FULL data/SEGTHOR_FULL_CLAHE

python -m preprocessing.preview_preprocessing_ablation \
  --baseline data/SEGTHOR_FULL \
  --variant data/SEGTHOR_FULL_CLAHE \
  --output eda/preprocessing_figures/clahe_preview.png
```

Training commands and Slurm submission files are added only after these
outputs and previews have been checked.
