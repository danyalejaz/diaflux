import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = 3000;

// URL of the Python ML backend (Flask) that serves the trained model.
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

app.use(express.json());

// --------------------------------------------------------------------------- //
// API proxy
// --------------------------------------------------------------------------- //
// All /api/* requests are forwarded to the Python ML backend so the frontend
// talks to the real trained GradientBoosting model instead of any client-side
// approximation. Keeping the same-origin /api prefix means the React code
// needs no changes and there are no CORS concerns in the browser.
app.use("/api", async (req, res) => {
  const targetUrl = `${BACKEND_URL}${req.originalUrl}`;
  try {
    const init: RequestInit = {
      method: req.method,
      headers: { "Content-Type": "application/json" },
    };

    if (req.method !== "GET" && req.method !== "HEAD") {
      init.body = JSON.stringify(req.body ?? {});
    }

    const backendRes = await fetch(targetUrl, init);
    const bodyText = await backendRes.text();

    res.status(backendRes.status);
    res.set(
      "Content-Type",
      backendRes.headers.get("content-type") || "application/json"
    );
    res.send(bodyText);
  } catch (err) {
    console.error(`Proxy error forwarding to ${targetUrl}:`, err);
    res.status(502).json({
      success: false,
      error:
        "ML backend is unreachable. Start the Python server (python backend/app.py) on port 5000.",
    });
  }
});

// --------------------------------------------------------------------------- //
// Frontend (Vite dev middleware in development, static build in production)
// --------------------------------------------------------------------------- //
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`DiaFlux frontend running on http://localhost:${PORT}`);
    console.log(`Proxying /api -> ${BACKEND_URL}`);
  });
}

startServer();
