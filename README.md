# ML Assignment 3: MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR10

**Author:** Bighnaraj Mohapatra (M26CSA011) · **Split seed:** 26011 · **Report:** `report/main.pdf`

Every number in the report is produced by the commands below and written to `runs/`, `results/` and
`report/tables/` by scripts. No number is typed in by hand.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate a3-cifar10
# after the environment works, freeze the exact versions you used and commit the file:
conda env export --no-builds > environment.lock.yml
```

All experiments use one NVIDIA GPU. The code also runs on CPU, but slowly.

## 2. Reproduction (exact commands)

```bash
bash scripts/run_all.sh setup        # download CIFAR10, save split (splits/split_seed26011.npz), print parameter counts
bash scripts/run_all.sh q1a_tune     # Q1(a)/Q2(a): LR grid {3e-4,1e-3,3e-3} per model on validation, seed 26011
#   -> commit PREREGISTRATION.md BEFORE the next line
bash scripts/run_all.sh q1a_final    # MLP-B and CNN-A, seeds 26011-26013, selected LR
bash scripts/run_all.sh q2b          # Q2(b): fixed pixel permutation (train+val+test), same hyper-parameters
bash scripts/run_all.sh q2c          # Q2(c): grayscale (train+val+test), same hyper-parameters
bash scripts/run_all.sh q1b          # Q1(b): 40% wrong labels + 0% and 100% controls, analysis, test at t* and final
bash scripts/run_all.sh q3a          # Q3(a): AEs with d = 32, 256, 2048; selection by validation MSE
bash scripts/run_all.sh test         # official test set, once, for all selected classifiers + Q2(c) analysis
bash scripts/run_all.sh q3b          # Q3(b): sigma scan on val, denoising AE, frozen-CNN comparison on test
bash scripts/run_all.sh q3c          # Q3(c): frozen-encoder linear / kNN probes (+ saves features for Q4)
bash scripts/run_all.sh q4           # Q4: SVM sweep, difficult pairs, clustering
bash scripts/run_all.sh report       # figures/ and report/tables/ regenerated from logs
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The defaults are 60 classifier epochs, 50 AE epochs, 200 Q1(b) epochs and seeds 26011–26013.
Override them with environment variables, for example `EPOCHS=40 SEEDS="26011" bash scripts/run_all.sh q2b`.
If you change any of them, state it in the report.

## 3. Protocol (what is fixed, and where)

| Item | Value | Where |
|---|---|---|
| Split | stratified 45,000 / 5,000 from the 50k training images, `random_state=26011` | `src/common.py`, `splits/` |
| Normalisation | per-channel mean/std of the (transformed) 45k training images | saved in each `runs/<tag>/config.json` |
| Augmentation | none (crops/flips are a spatial prior and would confound Q1/Q2) | `src/train_classifier.py` |
| Optimiser | AdamW, wd 5e-4, batch 128, cosine schedule | `src/train_classifier.py` |
| Tuning budget | same LR grid for every compared model, validation accuracy, seed 26011 | `src/select_lr.py`, `configs/selected_lr.json` |
| Checkpoint rule | epoch with best validation accuracy (AEs: lowest validation MSE) | `best.pt` |
| Test set | touched only by `eval_test.py`, `q3_eval_ae.py`, `q3b_eval.py --mode test`, `q3c_probe.py`, `q4_*.py`, after selection | `results/test/` |
| Q2(b) transform | one fixed permutation of the 1,024 pixel positions, `default_rng(26011)`, same for all channels and splits | `apply_transform` |
| Q2(c) transform | Y = 0.299R + 0.587G + 0.114B replicated to 3 channels | `apply_transform` |
| Q1(b) noise | 40% of training labels, each replaced by a uniformly drawn *other* class, seed 26011 | `splits/noisy_labels_0.4.npz` |
| Q3(b) corruption | additive Gaussian, sigma = 0.2 on [0,1], clipped; fresh per batch in training, fixed seeds 26012 (val) / 26013 (test) | `src/train_ae.py` |
| Q4 representation | AE bottleneck z, d = 256 (`ae_c16`, Q3(a)), standardised on the subset | `src/q4_common.py` |
| Q4 subset | stratified 10,000 (1,000 per class) of the 45k, seed 26011, fixed before model selection | `splits/q4_train_subset.npy` |

## 4. Reuse map

| Component | Introduced | Reused in |
|---|---|---|
| Split, normalisation, training loop (`common.py`, `train_classifier.py`) | Q1(a) | Q1(b), Q2, Q3(b) |
| MLP-B, CNN-A and their checkpoints | Q1(a) | Q2(a) (same runs); CNN-A seed 26011 is the fixed classifier in Q3(b) |
| AE d = 32 / 256 / 2048 | Q3(a) | Q3(b) (clean-AE baseline), Q3(c), Q4 (d = 256 features) |
| Denoising AE | Q3(b) | Q3(c) |
| Selected SVM and its test confusion matrix | Q4(a) | Q4(b), Q4(c) |

## 5. Repository layout

```
src/            all code (one script per experiment; see the docstring at the top of each file)
scripts/        run_all.sh: exact reproduction commands
splits/         saved split, noisy labels, Q4 subset (committed)
configs/        selected_lr.json, selected_ae.json (written by the selection scripts, validation only)
runs/<tag>/     config.json, log.csv (per epoch), summary.json  [checkpoints *.pt are git-ignored]
results/        test results and analysis JSON/CSV
figures/        every figure, regenerated by the scripts
report/         main.tex, tables/ (auto-generated), references.bib
```

Checkpoints and features are not committed because of their size. Every one of them is regenerated by
the commands above.

## 6. Smoke test without CIFAR10

`A3_FAKE_DATA=1 EPOCHS=2 AE_EPOCHS=2 MEM_EPOCHS=3 SEEDS="26011" bash scripts/run_all.sh all` runs the
whole pipeline on a tiny synthetic dataset, to check the code only. Delete `runs/ results/ configs/ splits/*FAKE*`
afterwards: its numbers are meaningless.
