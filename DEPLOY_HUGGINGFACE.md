# Deploying DiaFlux to Hugging Face Spaces (Docker)

DiaFlux runs as a **single Docker container**: the Flask backend loads the trained
GradientBoosting model, exposes the `/api/*` endpoints, **and** serves the built
React frontend — all on one port. No separate Node server, no CORS setup.

The included `Dockerfile` does everything:
1. Builds the React app with Vite (`diaflux_frontend/` → static files).
2. Installs the Python backend (`scikit-learn==1.7.2`, Flask, gunicorn).
3. Copies the model artifacts (`models/*.pkl`) and the built frontend.
4. Serves it all with gunicorn on port **7860** (the port Spaces expects).

---

## Steps

### 1. Create the Space
1. Sign in at https://huggingface.co and click **New → Space**.
2. **Space SDK:** choose **Docker** → **Blank** template.
3. **Hardware:** `CPU basic` (free) is enough.
4. Give it a name (e.g. `diaflux`) and create it.

### 2. Add the code
The Space is a git repo. Push this project into it. From your machine:

```bash
# Clone the empty Space repo (replace <user>/<space-name>)
git clone https://huggingface.co/spaces/<user>/diaflux hf-diaflux
cd hf-diaflux

# Copy the project files into it (Dockerfile must be at the root)
#   Dockerfile, .dockerignore, backend/, diaflux_frontend/, models/
# (Skip venv/, node_modules/, notebooks/, data/ — not needed at runtime.)

git add .
git commit -m "Deploy DiaFlux (Flask + React in one container)"
git push
```

> Tip: the model files in `models/` are small, so plain git is fine (no Git LFS needed).

### 3. Configure the Space port
Open the Space's **README.md** (Hugging Face auto-creates it) and make sure the
YAML header at the top declares the Docker SDK and the app port:

```yaml
---
title: DiaFlux
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

### 4. Done
Hugging Face builds the `Dockerfile` and starts the container. Once the build
finishes, your app is live at:

```
https://<user>-diaflux.hf.space
```

The form, risk report, live simulator, and recommendations all run against the
real trained model.

---

## Verifying / troubleshooting
- **Health check:** visit `https://<user>-diaflux.hf.space/api/health` — it should
  report `model_loaded: true` and `GradientBoostingClassifier`.
- **Blank page:** make sure `app_port: 7860` is set in the Space README header.
- **`model not available`:** confirm `models/diabetes_model.pkl` and
  `models/scaler.pkl` were actually pushed to the Space (they must not be ignored).
- **Build pulls a different sklearn:** the model needs `scikit-learn==1.7.2`
  (already pinned in `backend/requirements.txt`); don't loosen that pin.

## Running the same container locally (optional)
```bash
docker build -t diaflux .
docker run -p 7860:7860 diaflux
# open http://localhost:7860
```
