# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------- #
# Stage 1 - build the React (Vite) frontend into static assets
# ---------------------------------------------------------------------------- #
FROM node:22-slim AS frontend
WORKDIR /app/diaflux_frontend

COPY diaflux_frontend/package.json diaflux_frontend/package-lock.json ./
RUN npm ci

COPY diaflux_frontend/ ./
# Produce the static SPA in diaflux_frontend/dist (vite build, no server bundle)
RUN npx vite build


# ---------------------------------------------------------------------------- #
# Stage 2 - Python ML backend that also serves the built frontend
# ---------------------------------------------------------------------------- #
FROM python:3.11-slim AS app
WORKDIR /app

# Install backend dependencies (scikit-learn pinned to the model's version)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt gunicorn

# App code, trained model artifacts, and the built frontend
COPY backend/ ./backend/
COPY models/ ./models/
COPY --from=frontend /app/diaflux_frontend/dist ./diaflux_frontend/dist

# Hugging Face Spaces expects the container to listen on 7860
ENV PORT=7860
ENV FRONTEND_DIST=/app/diaflux_frontend/dist
EXPOSE 7860

WORKDIR /app/backend
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
