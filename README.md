# Customer Churn Prediction

A full-stack app that predicts whether a telecom customer will churn, using a model
trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(~7,000 customers). **Everything runs locally on a trained scikit-learn/XGBoost model —
no external API keys, no third-party services.**

- **Backend:** FastAPI (Python) serving a trained model + SHAP explainability
- **Frontend:** Plain HTML/CSS/JS (no framework)
- **Modeling:** Logistic Regression, Random Forest, and XGBoost, compared head-to-head

---

## Why churn prediction matters

Acquiring a new telecom customer typically costs **5-25x more** than retaining an
existing one, and even a modest reduction in churn compounds into large revenue
protection at scale (a telco with millions of subscribers and a 26% annual churn rate
is losing a large share of its base every year). A churn model's job is to flag
at-risk customers *before* they leave, early enough for a retention team to intervene
with an offer, a support call, or a plan change.

### The false negative / false positive tradeoff

This is a classic asymmetric-cost problem:

| | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actually stays** | Correct | **False positive** — wasted retention offer (discount, promo credit, outreach call) |
| **Actually churns** | **False negative** — customer leaves with no intervention, full revenue loss | Correct — retention team can act |

- A **false positive** costs the business a relatively small, controlled amount: an
  unnecessary discount or a support call to a happy customer.
- A **false negative** costs the business the customer's entire remaining lifetime
  value, plus the (higher) cost of acquiring a replacement customer.

Because false negatives are typically far more expensive than false positives, this
project treats **recall on the churn class** as a first-class metric alongside
precision and ROC-AUC, rather than optimizing for raw accuracy — a model that never
predicts churn can still look "accurate" on an imbalanced dataset while being
business-useless. See [Model comparison](#model-comparison-results) below for how
that traded off in practice, and the EDA notebook for the ~26.5% base churn rate that
makes accuracy alone a misleading metric here.

---

## Project structure

```
customerchurn/
├── data/                  # Place telco_churn.csv here (not committed)
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis
├── src/
│   ├── config.py          # Shared column/feature definitions
│   ├── preprocessing.py   # Cleaning, encoding, scaling, SMOTE
│   ├── train.py           # Trains + compares 3 models, saves the best
│   └── explain.py         # SHAP global + per-prediction explainability
├── backend/
│   ├── main.py            # FastAPI app: /predict, /model-info
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/                 # Generated: fitted pipeline, model, metrics (not committed)
└── requirements.txt         # Environment for notebook/training work
```

---

## How it works

1. **`src/preprocessing.py`** loads the raw CSV, fixes the known `TotalCharges`
   blank-string data issue (customers with `tenure == 0` haven't been billed yet, so
   blanks are filled with `0`), and builds a scikit-learn `ColumnTransformer` that
   median-imputes + scales numeric features and one-hot encodes categoricals. The
   fitted transformer is saved with `joblib` so training-time and request-time
   preprocessing can never drift apart.
2. **`src/train.py`** does a stratified 80/20 train/test split, fits the
   preprocessor on the training split only, then applies **SMOTE strictly to the
   transformed training data** (never the test data — that would leak information
   and produce falsely optimistic metrics). It trains Logistic Regression, Random
   Forest, and XGBoost, cross-validates each on the resampled training data, and
   evaluates all three on the untouched, realistically-imbalanced test set using
   precision, recall, F1, and ROC-AUC (plus a confusion matrix). The best model by
   test ROC-AUC is saved, along with a comparison table.
3. **`src/explain.py`** builds a SHAP explainer for the saved model and can render a
   global feature-importance summary plot, plus a per-prediction "top 3 contributing
   factors" function that the API calls live.
4. **`backend/main.py`** loads the saved preprocessor, model, and SHAP explainer
   **once at startup**, then serves `/predict` and `/model-info` over a CORS-enabled
   local API.
5. **`frontend/`** is a plain HTML form; `script.js` calls the API with `fetch()` and
   renders the prediction, probability, and top factors.

---

## Model comparison results

Results from a real run against the full 7,043-row Telco dataset (stratified 80/20
split, `random_state=42`, SMOTE applied to the training split only):

| Model | Precision | Recall | F1 | ROC-AUC | 5-fold CV ROC-AUC (resampled train) |
|---|---|---|---|---|---|
| Logistic Regression | 0.5025 | **0.7968** | 0.6163 | 0.8395 | 0.8526 ± 0.0028 |
| Random Forest | 0.5782 | 0.5829 | 0.5806 | 0.8192 | 0.9287 ± 0.0056 |
| **XGBoost (selected)** | 0.5736 | 0.6150 | 0.5935 | **0.8405** | 0.9367 ± 0.0044 |

The final model is selected automatically in `src/train.py` by **test-set ROC-AUC**
(overall ranking quality across thresholds). XGBoost edges out Logistic Regression
on ROC-AUC (0.8405 vs 0.8395) — close enough that either would be a defensible
choice — while both clearly beat Random Forest, which overfit the resampled
training data (CV ROC-AUC 0.93) without that gain transferring to the real test
distribution.

**Worth being honest about:** Logistic Regression actually has the highest
**recall** (0.80 vs XGBoost's 0.62) — it catches noticeably more of the customers
who actually churn, at the cost of more false positives (lower precision, 0.50 vs
0.57). Given the false-negative cost argument above, a team that weighs missed
churners heavily over wasted retention offers could reasonably override the
ROC-AUC-based selection and ship Logistic Regression instead — it's also the most
directly interpretable of the three via its coefficients. This project defaults to
the ROC-AUC winner (XGBoost) for its stronger overall ranking quality and its SHAP
explanations, but this tradeoff is a legitimate discussion point, not a settled
question — and it's exactly the kind of judgment call worth walking through in an
interview.

The SHAP top-3-factors output confirms the same signals seen in the EDA: for a
month-to-month, fiber-optic, electronic-check customer, the model correctly flags
`Contract = Month-to-month`, `InternetService = Fiber optic`, and
`PaymentMethod = Electronic check` as the top churn-risk drivers (77.7% predicted
probability); for a two-year-contract, long-tenure DSL customer, the same features
flip to the top churn-risk *reducers* (1.6% predicted probability).

`GET /model-info` serves this same table live to the frontend so it's always in sync
with whatever model is actually loaded. Re-run `python -m src.train` and these exact
numbers will shift slightly with the data split — they are not hardcoded anywhere
outside this README.

---

## Local setup & run instructions

### 0. Requirements
- Python 3.10+
- The dataset file at `data/telco_churn.csv` (IBM Telco Customer Churn, ~7,043 rows,
  21 columns — download from Kaggle or IBM's sample datasets and place it there)

### 1. Install dependencies

```bash
# From the project root
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt          # training/notebook environment
pip install -r backend/requirements.txt  # API environment
```

### 2. Train the model

```bash
python -m src.train      # trains + compares models, saves the best to models/
python -m src.explain    # generates the SHAP global summary plot (optional but recommended)
```

This populates `models/` with `preprocessor.joblib`, `best_model.joblib`,
`model_metadata.json`, and `model_comparison.csv` — all required by the backend.

### 3. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

### 4. Run the frontend

The frontend is static — just open it in a browser, or serve it locally:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://127.0.0.1:5500`. The page calls the backend at
`http://127.0.0.1:8000` (see `API_BASE_URL` in `script.js` — change it if you run the
backend on a different host/port).

### 5. (Optional) Explore the EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Deploying (Render)

This repo includes a `render.yaml` Blueprint that deploys two free-tier Render
services from a single connected GitHub repo:

- **`churn-backend`** — a Python web service running
  `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. It loads the
  `models/*.joblib` artifacts committed in this repo, so no retraining happens
  on deploy.
- **`churn-frontend`** — a static site serving `frontend/` as-is.

### Steps

1. Push this repo to GitHub (already done if you're reading this after asking
   Claude to deploy it).
2. In the [Render dashboard](https://dashboard.render.com), click **New +** →
   **Blueprint**, connect your GitHub account, and select this repo. Render
   will detect `render.yaml` and propose both services — click **Apply**.
3. Once `churn-backend` finishes deploying, copy its URL (something like
   `https://churn-backend.onrender.com` — Render appends a random suffix
   instead if that exact name is taken).
4. Update `API_BASE_URL` in `frontend/script.js` with that exact URL, commit,
   and push — Render auto-redeploys the static site on every push to the
   connected branch.

### Notes

- Render's free tier spins down an inactive web service after a period of no
  traffic; the first request after idling can take ~30-60s while it wakes up.
  This is normal, not a bug.
- CORS on the backend is currently wide open (`allow_origins=["*"]`) since
  this is a public demo with no auth or cookies. Tighten it to the frontend's
  exact origin in `backend/main.py` if you deploy this somewhere that matters.
- If you retrain locally (`python -m src.train`), commit the updated
  `models/*.joblib` files and push to redeploy the backend with the new model.

---

## No external services required

- No LLM or third-party API calls anywhere in this project.
- No API keys, `.env` secrets, or cloud credentials needed.
- The `/predict` endpoint runs a `joblib`-loaded scikit-learn/XGBoost model and a
  SHAP explainer entirely in-process.
- The frontend only talks to your own local backend.
