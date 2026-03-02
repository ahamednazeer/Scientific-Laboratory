# Scientific Laboratory AR Framework

AI-driven, web-based augmented reality framework for real-time scientific laboratory instrument recognition and instructional guidance.

## Stack
- Frontend: Next.js (TypeScript)
- Backend: FastAPI (Python)
- Database: SQLite
- AI assistant: Groq (OpenAI-compatible API)
- Detection pipeline: real EfficientDet inference + Kalman smoothing + context-aware NMS

## Repository Layout
- `backend/` FastAPI API, detector pipeline, SQLite models, seed data, metrics endpoints
- `frontend/` Next.js browser client with webcam AR overlays and guidance UI

## Features Implemented
- Browser-native webcam capture (no mobile app dependency)
- Image upload inference path for static laboratory photos
- Real-time detection API loop with AR box overlays
- Detection source selector in UI: `Model` (EfficientDet) or `AI` (Groq vision)
- UI shows actual detector used per request: `AI`, `Model`, `AI (No Match)`, `AI (Unavailable)`
- Instrument metadata overlays: name, description, operation steps, safety warnings
- Guidance source selector in UI: `AI` (Groq) or `Module` (local procedural guidance)
- Detection refinement:
  - Kalman filter smoothing for confidence/stability
  - context-aware NMS for overlap cleanup
- Quantitative metrics API:
  - latency
  - FPS estimate
  - mean confidence
  - confidence stability
- Groq-based instructional Q&A endpoint with local fallback mode

## Quick Start
### 1) Backend
```bash
cd /Users/syed.ahamed/skillup/Scientific-Laboratory/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 run.py
```

### 2) Frontend
```bash
cd /Users/syed.ahamed/skillup/Scientific-Laboratory/frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Docker (Multi-Stage)
You can run containers either with direct `docker run` commands or with Compose.

### 1) Build Backend Image
```bash
cd /Users/syed.ahamed/skillup/Scientific-Laboratory
docker build -f backend/Dockerfile -t scientific-lab-backend:latest backend
```

### 2) Run Backend Container
```bash
docker run --rm -p 8000:8000 \
  --env-file /Users/syed.ahamed/skillup/Scientific-Laboratory/backend/.env \
  -v /Users/syed.ahamed/skillup/Scientific-Laboratory/backend/models:/app/models \
  scientific-lab-backend:latest
```

### 3) Build Frontend Image
`NEXT_PUBLIC_API_BASE_URL` is compiled into the frontend bundle at build time.
```bash
cd /Users/syed.ahamed/skillup/Scientific-Laboratory
docker build -f frontend/Dockerfile -t scientific-lab-frontend:latest \
  --build-arg NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1 \
  frontend
```

### 4) Run Frontend Container
```bash
docker run --rm -p 3000:3000 scientific-lab-frontend:latest
```

### 5) Docker Compose (No Build)
`docker-compose.yml` uses prebuilt images only (`image:`) and does not contain `build:`.

```bash
cd /Users/syed.ahamed/skillup/Scientific-Laboratory
docker compose up -d
```

Stop:
```bash
docker compose down
```

If images are not present locally, pull/tag them first:
```bash
docker images | grep scientific-lab
```

## API Endpoints
- `GET /api/v1/health` (includes EfficientDet load status/detail)
- `GET /api/v1/instruments`
- `POST /api/v1/detection/predict`
- `POST /api/v1/guidance/ask`
- `GET /api/v1/metrics`

## EfficientDet Model Path
Default runtime mode is `mock` so the app works immediately without GPU/model downloads.

To switch to EfficientDet:
1. Fine-tune and export weights to:
   - `backend/models/efficientdet_lab_instruments.pth`
2. Ensure class index ordering is saved to:
   - `backend/models/class_labels.json`
   - class names should match instrument names in the catalog for metadata overlays.
3. Set in `backend/.env`:
   - `DETECTOR_MODE=efficientdet`
4. Install optional packages (`torch`, `torchvision`, `effdet`).
   - easiest: `pip install -r backend/requirements.efficientdet.txt`
5. Optional runtime tuning (in `backend/.env`):
   - `DETECTOR_ARCH=tf_efficientdet_d2`
   - `DETECTOR_DEVICE=auto|cuda|cpu|mps`
   - `DETECTOR_INPUT_SIZE=768`
   - `DETECTOR_CLASS_INDEX_BASE=1`

When `DETECTOR_MODE=efficientdet`, the API does not emit mock detections; if loading fails, `/api/v1/health` reports the failure reason.

To enable AI detection mode in the UI:
1. Set `GROQ_API_KEY` in `backend/.env`
2. Set/verify `GROQ_VISION_MODEL` (default: `meta-llama/llama-4-scout-17b-16e-instruct`)
3. Optional fallback list: `GROQ_VISION_FALLBACK_MODELS` (JSON array or comma-separated models)
4. Select `Detection Source = AI` in the frontend

Helper scripts:
- `python3 backend/scripts/recommend_pretrained.py`
- `python3 backend/scripts/train_efficientdet.py --dataset <path> --labels-file <labels.json> --output backend/models/efficientdet_lab_instruments.pth`

## Dataset Guidance (12,500 labeled images target)
Recommended dataset mix for robust lab recognition:
- controlled bench captures (institution-specific instruments)
- varied lighting/angles/occlusion conditions
- balanced class distribution across instrument categories
- train/val/test split with unseen lab scenes in test set

## Evaluation Mapping
This implementation exposes metrics needed for your evaluation section:
- detection accuracy proxy: confidence + instrument match review
- inference latency: `latency_ms`
- FPS performance: `fps_estimate`
- confidence stability: rolling `confidence_stability`

## Current Status
- Backend syntax check: pass
- Backend test (`health`): pass
- Frontend compile/lint: requires local dependency install (`npm install`) before running checks
