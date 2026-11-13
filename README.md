# AAE5103 Group Project — Departure-Delay Risk Screening

Code and data documentation for *"Screening Flights for Departure-Delay Risk from Schedule-Only Information:
A Capacity-Constrained Study at Three U.S. Hubs"*.

The project predicts whether a non-cancelled U.S. domestic flight will experience a **gate departure delay of at
least 15 minutes**, using only information that is available before departure. It then uses the predicted risk to
allocate a fixed number of operational readiness checks.

**Headline result (March 2025 hold-out, 76,878 flights):** random forest precision 0.281, recall 0.752, F1 0.409,
ROC-AUC 0.631; observed delay rate rises from 14.98% (Low band) to 34.00% (High band); at a 10% inspection budget,
risk-based selection covers 637 more delayed flights than schedule-order selection.

---

## Repository layout

```
.
├── data/A_handoff/        # data pipeline (member A)
│   ├── main.py            # entry point: --download, prepare, verify
│   ├── config.json        # airports, months, split dates
│   ├── scripts/           # download_bts.py, prepare_data.py, verify_data.py
│   └── docs/              # data dictionary, method notes
├── model/B_models/        # models and interface (member B)
│   ├── main.py            # `train` or `demo`
│   ├── src/               # features, pipelines, metrics, risk, predict
│   ├── app/web_demo.py    # local web interface (127.0.0.1:8765)
│   ├── scripts/           # train_models.py, predict_one.py
│   ├── models/            # fitted pipelines + risk_policy.json + selection.json
│   └── reports/           # metrics, figures, model comparison
└── operations/            # capacity scenario (member E)
    └── reproduce_operations.py
```

`models/` is committed so the demo can be started without retraining. Processed CSVs are **not** committed — see
*Data*.

---

## Data

Source: U.S. Bureau of Transportation Statistics, **Reporting Carrier On-Time Performance (1987–present)**,
monthly files for January, February and March 2025.

- Download page: <https://www.transtats.bts.gov/PREZIP/>
- Field definitions: <https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ>

The raw files are public domain and are **not redistributed here** (each monthly ZIP is roughly 30–60 MB). To
rebuild the modelling table:

```bash
cd data/A_handoff
python -m pip install -r requirements.txt
python main.py --download      # fetches the three official ZIPs and verifies SHA-256
python main.py                 # cleans, builds features, splits, verifies
```

The pipeline keeps flights departing from **ATL, DFW and ORD**, verifies each record against the source file,
reconciles the delay label with the reported delay in minutes, and writes `X_*.csv`, `y_*.csv`, `metadata_*.csv`
plus audit files.

Expected counts after a clean run: 1,645,503 raw records → 214,289 at the three airports → 5,118 cancelled →
209,171 labelled flights (46,691 delayed, 22.32%).

## Environment

Python 3.10+ recommended (tested on 3.9.6 and 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
cd model/B_models
python main.py train
```

Writes `models/`, `reports/metrics_{validation,test}.csv`, confusion matrices and ROC/PR curves. Model selection
and the decision threshold are fixed on the validation period (15–28 Feb); the test month (March) is evaluated
once.

## Run the demo

```bash
cd model/B_models
python main.py demo
# open http://127.0.0.1:8765
```

The interface accepts nine schedule fields only — airline, origin, destination, weekday, scheduled departure
time, distance and scheduled duration. Actual departure time, delay minutes, cancellation and diversion are never
inputs.

## Reproduce the operations scenario

```bash
cd operations
python reproduce_operations.py --a-zip /path/to/A_handoff.zip --b-zip /path/to/AAEwed.zip
```

Standard library only. Recomputes risk bands and the 5% / 10% / 20% capacity scenarios from the delivered
prediction files and writes them to `02_results/`.

---

## Method summary

| Step | Choice |
| --- | --- |
| Outcome | `DepDel15` = gate departure delay ≥ 15 minutes |
| Sample | Non-cancelled, labelled flights from ATL, DFW, ORD; Jan–Mar 2025 |
| Features | airline, origin, destination, weekday, scheduled hour, scheduled minute, weekend flag, distance, scheduled elapsed time |
| Split | chronological: 1 Jan–14 Feb (train), 15–28 Feb (validation), 1–31 Mar (test) |
| Models | logistic regression (standardised) and random forest (80 trees, depth 14, leaf 4) |
| Selection | threshold maximising delayed-class F1 on validation, then frozen; higher validation F1 wins |
| Metrics | precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, plus an all-on-time baseline |
| Decision layer | per airport × local date × scheduled hour capacity `K_g = floor(c · n_g)`; select top-`K_g` probabilities |

## Results

| Model (test) | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| All on-time baseline | 0.000 | 0.000 | 0.000 | – |
| Logistic regression | 0.280 | 0.755 | 0.409 | 0.624 |
| Random forest | 0.281 | 0.752 | 0.409 | 0.631 |

Risk bands (test): Low 14.98%, Medium 25.27%, High 34.00%.
Capacity scenario (10%): 6,912 slots; risk priority 2,146 delayed flights selected vs 1,509 under schedule order.

## Interpretation limits

- Three airports, one quarter, reporting carriers only — not a national or annual estimate.
- No weather, aircraft rotation, airspace or ground-resource data; the measured ceiling belongs to the
  schedule-only information set.
- BTS records are published retrospectively; the study does not preserve what was observable at a fixed lead time,
  so real-time deployability is not established.
- Scores are compressed and uncalibrated; they are ranks, not probabilities.
- Covering more delayed flights in a retrospective scenario is **not** evidence of preventing delays.

## Citation

If you reference this work:

> Members A–F (2026). *Screening Flights for Departure-Delay Risk from Schedule-Only Information: A
> Capacity-Constrained Study at Three U.S. Hubs.* AAE5103 Artificial Intelligence in Aviation Industry, The Hong
> Kong Polytechnic University.

Data: Bureau of Transportation Statistics. (2025). *Reporting Carrier On-Time Performance (1987–present)*
[Data set]. <https://www.transtats.bts.gov/PREZIP/>
