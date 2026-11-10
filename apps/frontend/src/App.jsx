import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/tiff", "image/webp"];
const VIEWS = [
  ["overlay", "Overlay"],
  ["original", "Original"],
  ["mask", "Binary mask"],
  ["instances", "Instances"],
];

function Logo() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <i />
      <i />
      <span />
    </div>
  );
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14.5v3A2.5 2.5 0 0 0 7.5 20h9a2.5 2.5 0 0 0 2.5-2.5v-3" />
    </svg>
  );
}

function Status({ health, loading }) {
  const tone = loading ? "checking" : health?.ready ? "ready" : health?.status === "unreachable" ? "offline" : "setup";
  const label = loading ? "Checking API" : health?.ready ? "Model ready" : health?.status === "unreachable" ? "API offline" : "Setup required";
  return (
    <div className={`status-pill ${tone}`} title={health?.message || label}>
      <span className="status-dot" />
      {label}
    </div>
  );
}

function UploadPanel({ health, healthLoading, busy, onSelect }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const ready = Boolean(health?.ready) && !busy;

  const receive = (files) => {
    const file = files?.[0];
    if (file) onSelect(file);
  };

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="section-kicker"><span>01</span> Analyze an image</div>
      <h2 id="upload-title">Bring a tissue field into focus.</h2>
      <p className="section-copy">
        Upload an H&amp;E image. The U-Net predicts nuclear pixels; watershed then separates touching objects for an estimated count.
      </p>

      <button
        className={`dropzone ${dragging ? "dragging" : ""}`}
        type="button"
        disabled={!ready}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => { event.preventDefault(); setDragging(false); if (ready) receive(event.dataTransfer.files); }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.tif,.tiff,.webp"
          onChange={(event) => receive(event.target.files)}
          tabIndex="-1"
        />
        <span className="upload-icon"><ArrowIcon /></span>
        <strong>{busy ? "Running overlapping-patch inference…" : "Drop an H&E image here"}</strong>
        <span>{ready ? "or click to browse" : healthLoading ? "Checking model readiness" : "Connect the trained model to enable upload"}</span>
        <small>PNG, JPEG, TIFF or WebP · max 20 MB</small>
      </button>

      {!healthLoading && !health?.ready && (
        <div className="setup-note" role="status">
          <span>!</span>
          <div>
            <strong>{health?.status === "unreachable" ? "Backend connection needed" : "Model artifacts needed"}</strong>
            <p>{health?.message || "Start the API and add the trained checkpoint before inference."}</p>
          </div>
        </div>
      )}
    </section>
  );
}

function ResultPanel({ result, fileName, activeView, setActiveView, busy }) {
  if (busy) {
    return (
      <section className="result-card result-loading" aria-live="polite">
        <div className="scan-frame"><span /></div>
        <p>Reconstructing the full-resolution probability map</p>
        <small>Overlapping tiles are averaged before thresholding.</small>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="result-card result-empty">
        <div className="empty-visual" aria-hidden="true">
          <span className="nucleus one" />
          <span className="nucleus two" />
          <span className="nucleus three" />
          <span className="scan-line" />
        </div>
        <div>
          <span className="eyebrow">Inference canvas</span>
          <h2>Your result will appear here.</h2>
          <p>Compare the source, semantic mask, color overlay, and separated instance map without leaving the workspace.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="result-card result-ready" aria-live="polite">
      <div className="result-head">
        <div>
          <span className="eyebrow">Analysis complete</span>
          <h2>{fileName}</h2>
        </div>
        <span className="result-size">{result.width} × {result.height}px</span>
      </div>

      <div className="view-tabs" role="tablist" aria-label="Result view">
        {VIEWS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={activeView === key}
            onClick={() => setActiveView(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className={`image-stage view-${activeView}`}>
        <img src={result.visuals[activeView]} alt={`${activeView} result for ${fileName}`} />
        <span className="view-label">{VIEWS.find(([key]) => key === activeView)?.[1]}</span>
      </div>

      <div className="metrics-grid">
        <article><span>Predicted nuclei</span><strong>{result.predicted_count.toLocaleString()}</strong><small>watershed instances</small></article>
        <article><span>Nuclear area</span><strong>{result.nuclei_area_percent}%</strong><small>of image pixels</small></article>
        <article><span>Mask threshold</span><strong>{result.threshold.toFixed(2)}</strong><small>frozen on validation</small></article>
        <article><span>Processing time</span><strong>{result.inference_ms.toLocaleString()} ms</strong><small>server inference</small></article>
      </div>
    </section>
  );
}

function MethodStrip() {
  return (
    <section className="method-strip" aria-labelledby="method-title">
      <div>
        <span className="section-kicker"><span>02</span> Understand the method</span>
        <h2 id="method-title">Pixels first. Objects second.</h2>
      </div>
      <ol>
        <li><span>1</span><div><strong>Tile</strong><p>Split the field into overlapping 256px patches.</p></div></li>
        <li><span>2</span><div><strong>Segment</strong><p>Average U-Net probabilities across overlaps.</p></div></li>
        <li><span>3</span><div><strong>Separate</strong><p>Use distance peaks and watershed for touching nuclei.</p></div></li>
        <li><span>4</span><div><strong>Quantify</strong><p>Return the mask, instances, area, and count.</p></div></li>
      </ol>
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [fileName, setFileName] = useState("");
  const [activeView, setActiveView] = useState("overlay");

  const checkHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/health`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setHealth(await response.json());
    } catch (requestError) {
      setHealth({
        ready: false,
        status: "unreachable",
        message: `Could not reach ${API_URL}. Start the FastAPI service or set VITE_API_URL in Vercel.`,
      });
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => { checkHealth(); }, [checkHealth]);

  const handleFile = async (file) => {
    setError("");
    if (!ACCEPTED_TYPES.includes(file.type) && !/\.(tif|tiff)$/i.test(file.name)) {
      setError("Choose a PNG, JPEG, TIFF, or WebP image.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Choose an image smaller than 20 MB.");
      return;
    }

    setBusy(true);
    setFileName(file.name);
    setResult(null);
    const form = new FormData();
    form.append("image", file);
    try {
      const response = await fetch(`${API_URL}/api/segment`, { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Inference failed with status ${response.status}.`);
      setResult(payload);
      setActiveView("overlay");
    } catch (requestError) {
      setError(requestError.message || "Inference failed. Check the API and try again.");
    } finally {
      setBusy(false);
    }
  };

  const modelSummary = useMemo(() => {
    if (!health?.model) return "U-Net · 256px patches";
    return `U-Net · ${health.model.patch_size || 256}px patches${health.model.threshold != null ? ` · t=${health.model.threshold}` : ""}`;
  }, [health]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Nuclei Lens home">
          <Logo />
          <span><strong>Nuclei</strong> Lens</span>
        </a>
        <div className="header-meta">
          <span className="model-summary">{modelSummary}</span>
          <Status health={health} loading={healthLoading} />
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">Computational pathology · research demo</span>
            <h1>See every nucleus.<br /><em>Count what matters.</em></h1>
            <p>
              A transparent U-Net workflow for semantic nuclei segmentation and watershed-based instance counting in H&amp;E tissue images.
            </p>
            <div className="hero-badges"><span>MoNuSeg 2018</span><span>TensorFlow</span><span>Explainable pipeline</span></div>
          </div>
          <div className="hero-orbit" aria-hidden="true">
            <span className="orbit orbit-a" />
            <span className="orbit orbit-b" />
            <span className="cell cell-a" />
            <span className="cell cell-b" />
            <span className="cell cell-c" />
            <span className="crosshair horizontal" />
            <span className="crosshair vertical" />
            <small>H&amp;E / FIELD 001</small>
          </div>
        </section>

        {error && <div className="error-banner" role="alert"><strong>Couldn’t complete analysis.</strong><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss">×</button></div>}

        <section className="workspace">
          <UploadPanel health={health} healthLoading={healthLoading} busy={busy} onSelect={handleFile} />
          <ResultPanel result={result} fileName={fileName} activeView={activeView} setActiveView={setActiveView} busy={busy} />
        </section>

        <MethodStrip />

        <section className="disclaimer">
          <div className="disclaimer-icon">R</div>
          <div><strong>Research use only</strong><p>Nuclei Lens is an educational portfolio project and is not validated for diagnosis, treatment, or clinical decision-making.</p></div>
        </section>
      </main>

      <footer>
        <div className="brand compact"><Logo /><span><strong>Nuclei</strong> Lens</span></div>
        <p>From pixels to measurable cell instances.</p>
        <span>MoNuSeg · U-Net · Watershed</span>
      </footer>
    </div>
  );
}

