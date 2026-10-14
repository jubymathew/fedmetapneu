# FedMetaPneu

**Metaheuristic optimisation of privacy-preserving federated learning for pneumonia screening on chest radiographs.**

Reference implementation of the FedMetaPneu paper (J. Mathew, 2026). Four simulated hospital nodes train a hybrid
segmentation + foundation-encoder + cross-attention model with record-level differential privacy and secure
aggregation. A hybrid PSO–GA search tunes the federated settings, Harris hawks optimisation (HHO) selects a
compact feature subset at each site, and the server weights clients by calibration quality.

> **Research code only.** The federation is simulated from public retrospective datasets. Nothing here is a medical
> device or evidence of clinical readiness.

```
            ┌────────── PSO–GA search (fitness on validation splits only) ──────────┐
            │  continuous: lr, μ, dropout, σ, aggregation coefficients              │
            │  discrete:   attention heads, encoder depth, client sampling, epochs  │
            └───────────────────────────────┬───────────────────────────────────────┘
                                            ▼
 C1 CheXpert ─┐   local model: U-Net lung mask → shared encoder (raw + masked image)
 C2 MIMIC  ───┤   → cross-attention fusion → 1024-d features → classifier
 C3 RSNA   ───┤   DP-SGD: per-sample clip C, Gaussian noise σC, Poisson lots, RDP accountant
 C4 COVIDx ───┘   site-local normalisation (FedBN) · FedProx proximal term
                         │ masked updates (pairwise secure aggregation)
                         ▼
                 server: α_k ∝ size + inverse-ECE (Eq. 7) [+ performance]
                         │ after convergence: HHO masks per site → majority vote → prune → fine-tune
                         ▼
                 one global temperature (fitted federatedly) → internal / external evaluation
```

## What is implemented

| Paper | Code |
|---|---|
| Problem formulation, focal loss (Eq. 1) | `fedmetapneu/losses.py` |
| Hybrid client model, cross-attention fusion (Eq. 2) | `fedmetapneu/models/` |
| FedProx (Eq. 3), FedBN, FedAvg, FedNova, SCAFFOLD, Ditto | `fedmetapneu/fl/client.py`, `fl/server.py` |
| DP-SGD, Poisson subsampling, RDP accountant (Sec. 3.3) | `fedmetapneu/privacy/dp.py` |
| Secure aggregation with pairwise masks | `fedmetapneu/privacy/secure_agg.py` |
| Hybrid PSO–GA search and fitness (Eq. 4) | `fedmetapneu/optim/pso_ga.py`, `optim/space.py` |
| Binary HHO feature selection (Eqs. 5–6), majority vote | `fedmetapneu/optim/hho.py`, `pipeline.select_features` |
| Calibration-aware aggregation (Eq. 7), global temperature | `fedmetapneu/calibration.py`, `fl/server.py` |
| Partial participation, dropout, staleness (Sec. 4.6) | `fedmetapneu/fl/simulation.py` |
| AUROC/AUPRC/F1/ECE/Brier, Youden threshold, DeLong, bootstrap, Bonferroni | `fedmetapneu/metrics.py`, `compare.py` |
| Subgroup analysis by sex, age band, projection (Sec. 4.7) | `pipeline.subgroup_metrics` |
| Membership inference (confidence, loss, offline LiRA) | `fedmetapneu/attacks/mia.py` |
| Gradient inversion (Geiping et al.), PSNR/SSIM | `fedmetapneu/attacks/inversion.py` |
| Label harmonisation and patient-level splits (Sec. 4.2) | `fedmetapneu/data/harmonise.py` |
| Synthetic non-IID federation for tests | `fedmetapneu/data/synthetic.py` |

## Installation

```bash
git clone https://github.com/<your-account>/FedMetaPneu.git
cd FedMetaPneu
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -e ".[all]"
pytest -q                                                               # ~1 minute on CPU
```

Python ≥ 3.10, PyTorch ≥ 2.1, Opacus ≥ 1.4 (used for RDP accounting).

## Quick start (synthetic data, CPU)

```bash
bash scripts/run_smoke.sh
```

This runs every stage — privacy accounting, PSO–GA search, all baselines, FedMetaPneu with HHO, DeLong comparison
and the privacy audit — on a small synthetic federation with a tiny backbone. Results land in `runs/smoke/`.
The synthetic numbers only show that the code paths work.

## Reproducing the paper

### 1. Get the data

All datasets are public; each requires accepting its own licence.

| Node | Dataset | Expected layout under `data.roots.<name>` |
|---|---|---|
| C1 | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | `train.csv`, `valid.csv`, `train/patient*/…` |
| C2 | [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) | `mimic-cxr-2.0.0-{chexpert,metadata}.csv(.gz)`, `files/p*/…` |
| C3 | [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | `stage_2_train_labels.csv`, `stage_2_train_images/*.dcm` |
| C4 | [COVIDx (3-class release)](https://github.com/lindawangg/COVID-Net) | `train.txt`, `test.txt`, image folders |
| ext | [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | `PADCHEST_*labels*.csv`, `<ImageDir>/<ImageID>` |
| ext | [VinDr-CXR](https://physionet.org/content/vindr-cxr/) | `annotations/image_labels_test.csv`, `test/*.dicom` |

COVIDx: use a release with normal / pneumonia / COVID-19 labels. Releases that only label COVID positive/negative
put non-COVID pneumonia in the negative class; those negatives are dropped as unresolvable.
For MIMIC-CXR sex and age, point `data.loader_kwargs.mimic.patients_csv` to MIMIC-IV `patients.csv`.

Edit `data.roots` in `configs/default.yaml`, then:

```bash
fedmetapneu prepare-data --config configs/default.yaml
```

This writes `data/manifest.csv` (harmonised labels, patient-level 70/15/15 splits with a fixed seed),
`data/split_<dataset>_<split>.csv` record lists and `data/stats.json` (per-dataset normalisation).

### 2. Backbone and segmentation weights

* **CheXFound** (ViT-L/14): set `model.backbone_checkpoint` to the released weights. They are loaded into timm's
  `vit_large_patch14_dinov2` with prefix cleaning; the loader logs how many tensors matched — check that line.
  Any timm model also works: `model.backbone: timm:vit_base_patch16_224`.
* **Lung U-Net**: set `model.seg_checkpoint` to a pre-trained `LiteUNet` state dict (frozen during FL). Without it
  the U-Net is learned end-to-end as a soft attention mask, and a warning is printed.

### 3. Check the privacy budget first

```bash
fedmetapneu privacy --config configs/default.yaml --target-epsilon 3.8
```

prints, for each client, the RDP ε implied by `σ`, lot size, local epochs and rounds, and the `σ` needed for a target
ε. With very different client sizes the same `σ` gives very different ε per client. To fix the budget instead of
σ, set `privacy.target_epsilon`; the noise multiplier is then solved per client (and is no longer a search gene).

### 4. Run

```bash
fedmetapneu search  --config configs/default.yaml                   # one-time PSO–GA search
fedmetapneu train   --config configs/default.yaml --method all      # Table 3 methods
fedmetapneu compare --config configs/default.yaml                   # DeLong + Bonferroni table
fedmetapneu audit   --config configs/default.yaml --method fedmetapneu
```

or the whole protocol (three seeds, ablation, ε sweep, network conditions, audit, sensitivity):

```bash
bash scripts/reproduce_paper.sh
```

Override any setting with `--set key.path=value`, e.g. `--set fl.rounds=50 --set privacy.target_epsilon=5.0`.

### Method presets (`--method`)

`centralised`, `local`, `fedavg`, `fedprox`, `fedbn`, `fedavg_dp` (ε = 5), `scaffold`, `fednova`, `ditto`,
`fedmetapneu_nosearch`, `fedmetapneu`, and the cumulative ablation `abl_base` → `abl_cross` → `abl_search` →
`abl_hho` → `abl_dp` → `abl_full`. Presets are plain dotted overrides in `configs/default.yaml` → `methods`.

## Outputs

```
runs/<name>/
  search/best_params.yaml, search_log.csv, search_summary.json
  <method>/config.yaml        resolved configuration
  <method>/metrics.json       per-client, pooled internal, external, subgroup metrics; ε; traffic; HHO info
  <method>/history.json       per-round validation AUROC/ECE, bytes, ε, aggregation weights
  <method>/predictions/*.npz  raw logits + labels (inputs to DeLong tests)
  <method>/model.pt           global model, site-local states, temperature, kept feature indices
  <method>/audit/mia.json, inversion.json
  comparison.md / .csv / .json
```

## Implementation notes

* **Normalisation under DP.** Per-sample gradients are undefined for BatchNorm in training mode, so the model uses
  GroupNorm/LayerNorm. Site-local normalisation (FedBN) applies to these layers (`fl.local_norm_types`). All trainable
  parameters, including site-local ones, are trained with DP-SGD so that the guarantee covers every released update.
  For unseen sites the global model uses the size-weighted mean of the site-local parameters.
* **Proximal term** gradients are added after clipping and noise: they depend only on public parameters.
* **Accounting** composes every noisy step, including empty Poisson lots and HHO fine-tuning rounds, per client.
  As the paper states, the search fitness values, feature masks, reported validation ECEs and the temperature fit are
  outside the accountant.
* **Secure aggregation** is simulated with pairwise additive masks; dropout recovery via secret sharing is omitted.
  Aggregation weights are applied by clients before masking.
* **Temperature scaling** is fitted federatedly: clients return only summed validation NLL over a shared grid.
* **Search budget.** `search.max_evals` caps fitness evaluations (default 50, as in Section 5.8). With the full
  population sizes and no cap, one search would need thousands of short federated runs.
* **HHO** works on stored validation features with the trained classifier head (fitness Eq. 6); the vote keeps a
  feature when at least half the clients select it. Selected features are physically pruned from the fusion
  projection and classifier; `metrics.json → hho` reports the measured change in shared-parameter bytes.
* **Noise RNG** is seeded for reproducibility. A deployment should draw noise from a cryptographically secure source.
* **Shadow models** for LiRA are trained without DP on other clients' data (the curious-client view).

## Limitations

The federation is simulated in one administrative domain; network delay, governance and clinical label noise are
absent. Labels are an operational harmonisation across datasets with different evidence. The threat model excludes
a malicious server and server–client collusion. Calibration does not transfer to new sites without local
recalibration. No clinician reviewed any output.

## Citation

```bibtex
@article{mathew2026fedmetapneu,
  title  = {FedMetaPneu: Metaheuristic Optimization of Privacy-Preserving Federated Learning for Pneumonia
            Screening on Chest Radiographs},
  author = {Mathew, Juby},
  year   = {2026}
}
```

## Licence

MIT (see `LICENSE`). Dataset licences are separate and must be accepted individually.
