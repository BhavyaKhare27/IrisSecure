# IrisSecure

IrisSecure is a local iris recognition R&D prototype with privacy-aware upload handling. It combines a PyTorch iris embedding model with a FastAPI service, a Next.js dashboard, local SQLite persistence, and statistical threshold calibration.

The service stores normalized iris embeddings rather than uploaded images. A trained model checkpoint and the CASIA-Iris-Thousand dataset are required for training and image-based enrollment. The dashboard and API can start before those files are available.

## Stack

- **Python / PyTorch** — ResNet50 embeddings and ArcFace training
- **FastAPI** — REST API and OpenAPI documentation
- **Next.js / React / JavaScript** — dashboard and enrollment UI
- **Apache ECharts** — interactive FAR/FRR threshold chart
- **SQLite / SQLAlchemy** — local identity and analysis-job storage
- **FastAPI background tasks** — local threshold calibration without a separate worker or broker
- **SciPy / NumPy** — statistical analysis and operating-point optimization

The five-year paper analysis, project gap, proposed research question, protocol, and verified references are in [the literature review and R&D plan](docs/literature_review_and_project_plan.md).

## Quick start

Run the API and dashboard as two local processes. SQLite stores the local identity gallery and analysis jobs. Threshold calibration runs as a FastAPI background task. No separate database server, message broker, worker, or Docker installation is needed.

### API

Python 3.9–3.12:

```bash
python3 -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn api.main:app --reload
```

The API is available at <http://localhost:8000>; interactive docs are at <http://localhost:8000/docs>.

### Dashboard

Use Node.js 20.9 or newer:

```bash
cd web
npm install
npm run dev
```

Open <http://localhost:3000>. The dashboard calls the API at `http://localhost:8000`; override it with `NEXT_PUBLIC_API_URL` if needed.

## Model and dataset setup

The current subject-disjoint model is already trained and evaluated. Use the checkpoint in `models/runs/subject_disjoint_seed42/checkpoints/` for the demo; do not rerun training unless you intentionally want a new experiment. To reproduce training from a fresh dataset setup, place CASIA-Iris-Thousand under `data/CASIA-Iris-Thousand/`, keeping the subject and left/right-eye folder structure, then run:

```bash
python preprocess_dataset.py
python scripts/create_subject_disjoint_split.py
python scripts/preflight_training.py
python train.py --epochs 30
```

The split command creates the manifests once and refuses to overwrite an existing split directory. If `data/splits/subject_disjoint/` is already present, keep it and proceed to preflight instead of recreating it. The training script also refuses to overwrite existing run checkpoints; choose a new `training.run_name` for a new experiment.

The selected subject-disjoint checkpoint is `models/runs/subject_disjoint_seed42/checkpoints/best_model.pth` (best epoch 26). It is the API default; training will use a separate configured run directory. Image-based enrollment sends images to the local API for in-memory inference; the database stores the resulting 512-dimensional embedding, which remains sensitive biometric data.

To reproduce the subject-disjoint evaluation without retraining:

```bash
python evaluate.py --max_impostor 50000 --bootstrap_repetitions 500
```

`evaluate.py` loads the configured run checkpoint and subject-disjoint manifests. It selects EER and target-FAR operating thresholds on validation scores, freezes them, and reports test results alongside the fixed `0.75` baseline. It samples unique impostor image pairs and reports subject-cluster bootstrap intervals. Results are written to `outputs/evaluation/subject_disjoint_seed42/`. The evaluator refuses to run if checkpoint metadata does not match the current split manifests. A zero observed FAR is not evidence of zero population risk.

The subject-disjoint report is in `outputs/evaluation/subject_disjoint_seed42/evaluation_report.json`: validation EER was 0.7673% at threshold 0.373255; on 150 held-out people and 3,000 genuine / 50,000 sampled impostor pairs, test FAR was 0.906%, FRR 0.867%, TAR 99.133%, and ROC-AUC 0.999346. Fixed threshold 0.75 gave FAR 0.006% and TAR 73.333%, showing the FAR/FRR operating-point tradeoff. Before matching empirical score files are available, the dashboard uses a Gaussian approximation based on validation reference statistics and labels those values as estimates.

The evaluator writes validation calibration scores, test scores, plots, and protocol/results metadata in the run-specific evaluation directory. The API accepts empirical calibration only when the evaluation report fingerprint matches the active checkpoint; otherwise it labels its validation-aggregate fallback as an estimate. The checked-in aggregate summary is generated from the new model’s validation score distributions.

## API overview

- `GET /api/health` — service status
- `GET /api/dashboard` — model availability and reference statistics
- `GET /api/identities` — registered identity list
- `POST /api/identities/{user_id}` — persist a normalized 512-dimensional embedding
- `DELETE /api/identities/{user_id}` — remove a registered identity
- `POST /api/identify` — match an embedding against the gallery
- `POST /api/analytics/optimize` — queue EER optimization, or pass `?target_far=0.001` to minimize FRR under an FAR constraint
- `GET /api/jobs/{job_id}` — poll a job
- `GET /api/analytics/threshold` — selected threshold and FAR/FRR curve; optionally pass `?target_far=0.001`

The full interactive API reference is available at `/docs` while the API is running.

## Project layout

```text
api/                 FastAPI routes, SQLAlchemy models, local analytics jobs
web/                 Next.js dashboard and Apache ECharts visualizations
models/               ResNet feature extractor and ArcFace loss
utils/                Preprocessing, dataset, similarity and visualization helpers
database/              Reference score statistics and local SQLite database
docs/                  Five-year literature review and research protocol
train.py               Model training
preprocess_dataset.py  Dataset preparation
generate_embeddings.py Dataset embedding extraction
evaluate.py            Empirical biometric evaluation
```

Training, preprocessing, embedding generation, and empirical evaluation remain Python scripts because they are the project’s R&D/model pipeline. The React interface uses regular JavaScript in `.jsx` files. The former Gradio app, its sample-image gallery, and duplicate pickle-based registration/verification CLI are removed; those features now live in the FastAPI and Next.js application.

## Repository

```bash
git clone https://github.com/BhavyaKhare27/IrisSecure.git
cd IrisSecure
```

## Privacy and security note

Image enrollment sends image bytes to the local API for in-memory preprocessing and inference; the route does not save the uploaded image. The database stores the resulting embedding and identity metadata. Upload size, format, and dimensions are bounded, and local database/generated biometric files receive restrictive filesystem permissions. The prototype has no user authentication and does not encrypt embeddings at rest. It is for a local classroom/research demo only; do not expose the API or database to a public network or describe the system as production-secure, cancelable, or standards-compliant.
