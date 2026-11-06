# IrisSecure

IrisSecure is a privacy-focused iris recognition R&D project. It combines a PyTorch iris embedding model with a FastAPI service, a Next.js dashboard, local SQLite persistence, and statistical threshold calibration.

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

Download CASIA-Iris-Thousand and place it under `data/CASIA-Iris-Thousand/`, keeping the subject and left/right-eye folder structure. Then run:

```bash
python preprocess_dataset.py
python train.py --epochs 30
```

The trained checkpoint is saved to `models/checkpoints/best_model.pth`. Restart the API after adding the checkpoint. Image-based enrollment is then available from the dashboard; only the 512-dimensional embedding is saved.

To generate a dataset gallery and run full empirical evaluation:

```bash
python generate_embeddings.py --batch_size 64 --num_workers 0
python evaluate.py --split test --max_impostor 50000
```

Before empirical scores are available, the dashboard uses a Gaussian approximation based on the aggregate genuine/impostor means and standard deviations in `database/embedding_stats.json`. It labels these values as estimates.

After `evaluate.py` runs with its default output directory, it saves the genuine/impostor score distributions under `outputs/evaluation/`. The API uses these measured scores for empirical EER or target-FAR optimization; before that file exists, it clearly marks its Gaussian fallback as an estimate.

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

## Privacy note

Image enrollment sends image bytes to the API for in-memory preprocessing and inference. The route does not persist the uploaded image; it stores the resulting embedding in the database. Treat embeddings as sensitive biometric data and restrict access to the API and database in any real deployment.
