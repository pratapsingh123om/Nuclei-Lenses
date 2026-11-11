import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/tiff", "image/webp"];
const VIEW_LABELS = {
  typed_overlay: "Typed overlay", type_overlay: "Typed overlay", overlay: "Overlay", original: "Original",
  mask: "Binary mask", instances: "Instances", types: "Cell types", confidence: "Confidence",
  metastasis_overlay: "Probability overlay", metastasis_heatmap: "Probability heatmap", evidence_map: "Threshold map",
};
const VIEW_ORDER = ["metastasis_overlay", "metastasis_heatmap", "evidence_map", "typed_overlay", "type_overlay", "overlay", "original", "mask", "instances", "types", "confidence"];
const TYPE_META = {
  neoplastic: { label: "Neoplastic", color: "#ff7f73", description: "tumour-associated nuclear phenotype" },
  inflammatory: { label: "Inflammatory", color: "#70d6ff", description: "lymphoid and inflammatory phenotype" },
  connective: { label: "Connective", color: "#f4d27c", description: "stromal and connective phenotype" },
  dead: { label: "Dead", color: "#a487ff", description: "dead-cell nuclear phenotype" },
  non_neoplastic_epithelial: { label: "Non-neoplastic epithelial", color: "#4de4be", description: "non-neoplastic epithelial phenotype" },
};
const TYPE_ALIASES = {
  neoplastic: "neoplastic", inflammatory: "inflammatory", connective: "connective", connective_tissue: "connective",
  dead: "dead", epithelial: "non_neoplastic_epithelial", non_neoplastic: "non_neoplastic_epithelial",
  non_neoplastic_epithelial: "non_neoplastic_epithelial",
};

function normalizeTypeKey(value) {
  const key = String(value || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  return TYPE_ALIASES[key] || key;
}

function getTypeEntries(result) {
  const source = result?.type_counts || result?.class_counts || result?.counts_by_type || result?.cell_types;
  if (!source) return [];
  const rawEntries = Array.isArray(source)
    ? source.map((entry) => [entry.class_name || entry.name || entry.type, entry]) : Object.entries(source);
  const counts = new Map();
  rawEntries.forEach(([rawKey, rawValue]) => {
    const key = normalizeTypeKey(rawKey);
    if (!TYPE_META[key]) return;
    const count = Number(typeof rawValue === "object" ? rawValue.count ?? rawValue.value : rawValue);
    if (Number.isFinite(count) && count >= 0) counts.set(key, (counts.get(key) || 0) + count);
  });
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  return Object.keys(TYPE_META).filter((key) => counts.has(key)).map((key) => ({
    key, ...TYPE_META[key], count: counts.get(key), percent: total > 0 ? (counts.get(key) / total) * 100 : 0,
  }));
}

function getAvailableViews(result) {
  const visuals = result?.visuals || {};
  return VIEW_ORDER.filter((key) => !(key === "type_overlay" && visuals.typed_overlay) && Boolean(visuals[key]))
    .map((key) => [key, VIEW_LABELS[key] || key]);
}

function isTypedModel(health, result) {
  const classes = health?.model?.class_names || health?.model?.classes || [];
  const outputs = health?.model?.outputs || [];
  return getTypeEntries(result).length > 0 || classes.length >= 5 || outputs.includes("type")
    || Boolean(health?.model?.capabilities?.typed_nuclei)
    || String(health?.model?.instance_pipeline || "").toLowerCase().includes("hover");
}

function modelHealth(health, analysisMode) {
  if (health?.models?.[analysisMode]) return health.models[analysisMode];
  return analysisMode === "nuclei" ? health : null;
}

function resolveEndpoint(health, analysisMode, operatingPoint) {
  const selected = modelHealth(health, analysisMode);
  const fallback = analysisMode === "metastasis" ? "/api/metastasis/analyze" : "/api/segment";
  const configured = selected?.analysis_endpoint || health?.endpoints?.[analysisMode] || fallback;
  const endpoint = /^https?:\/\//i.test(configured) ? configured : `${API_URL}${configured.startsWith("/") ? "" : "/"}${configured}`;
  if (analysisMode !== "metastasis") return endpoint;
  const separator = endpoint.includes("?") ? "&" : "?";
  return `${endpoint}${separator}operating_point=${encodeURIComponent(operatingPoint)}`;
}

function formatNumber(value, fallback = "—") {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : fallback;
}

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

function DownloadIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0 4-4m-4 4-4-4M5 19h14" /></svg>;
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

function UploadPanel({ health, healthLoading, busy, onSelect, analysisMode, onModeChange, operatingPoint, setOperatingPoint }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const selectedHealth = modelHealth(health, analysisMode);
  const ready = Boolean(selectedHealth?.ready) && !busy;
  const typed = isTypedModel(health);

  const receive = (files) => {
    const file = files?.[0];
    if (file) onSelect(file);
  };

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="section-kicker"><span>01</span> Analyze an image</div>
      <h2 id="upload-title">Choose what to measure.</h2>
      <div className="analysis-switch" role="tablist" aria-label="Analysis model">
        <button type="button" role="tab" aria-selected={analysisMode === "nuclei"} onClick={() => onModeChange("nuclei")}>
          <span>Nuclei</span><small>segment · separate · count</small>
        </button>
        <button type="button" role="tab" aria-selected={analysisMode === "metastasis"} onClick={() => onModeChange("metastasis")}>
          <span>Metastatic pattern</span><small>classify patches · map evidence</small>
        </button>
      </div>
      <p className="section-copy">
        {analysisMode === "metastasis"
          ? "Upload a sentinel lymph-node H&E field. The model looks for tissue patterns associated with breast-cancer metastasis, shows where it found them, and explains uncertainty and next steps."
          : typed
          ? "Upload an H&E image. The model finds individual nuclei, separates touching cells, and estimates five nuclear phenotypes."
          : "Upload an H&E image. The U-Net predicts nuclear pixels; watershed then separates touching objects for an estimated count."}
      </p>

      {analysisMode === "metastasis" && (
        <div className="operating-switch" aria-label="Probability operating point">
          <button type="button" className={operatingPoint === "balanced" ? "active" : ""} onClick={() => setOperatingPoint("balanced")}>
            Balanced review <small>fewer false alarms</small>
          </button>
          <button type="button" className={operatingPoint === "sensitivity" ? "active" : ""} onClick={() => setOperatingPoint("sensitivity")}>
            Find more suspicious areas <small>fewer missed positive patches</small>
          </button>
        </div>
      )}

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
        <strong>{busy ? (analysisMode === "metastasis" ? "Classifying tissue patches…" : "Running overlapping-patch inference…") : "Drop an H&E image here"}</strong>
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
      {!healthLoading && health?.ready && !selectedHealth?.ready && (
        <div className="setup-note" role="status">
          <span>!</span>
          <div><strong>This analysis is not deployed yet</strong><p>{selectedHealth?.message || "Deploy its evaluated model bundle to enable this mode."}</p></div>
        </div>
      )}
    </section>
  );
}

function TypeComposition({ entries, fileName }) {
  if (!entries.length) return null;
  const total = entries.reduce((sum, entry) => sum + entry.count, 0);
  const neoplastic = entries.find((entry) => entry.key === "neoplastic")?.count || 0;
  const inflammatory = entries.find((entry) => entry.key === "inflammatory")?.count || 0;
  const ratio = inflammatory > 0 ? neoplastic / inflammatory : null;
  const downloadCsv = () => {
    const rows = ["class,count,percent", ...entries.map((entry) => `${entry.key},${entry.count},${entry.percent.toFixed(2)}`), `total,${total},100.00`];
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${fileName.replace(/\.[^.]+$/, "") || "nuclei-lens"}-cell-composition.csv`;
    document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  };
  return (
    <section className="composition-card" aria-labelledby="composition-title">
      <div className="composition-head">
        <div><span className="eyebrow">Estimated cell composition</span><h3 id="composition-title">Five phenotypes, one auditable field.</h3></div>
        <button className="download-button" type="button" onClick={downloadCsv}><DownloadIcon /> Download CSV</button>
      </div>
      <div className="composition-bar" aria-label="Cell type distribution">
        {entries.map((entry) => <span key={entry.key} style={{ width: `${entry.percent}%`, background: entry.color }} title={`${entry.label}: ${entry.percent.toFixed(1)}%`} />)}
      </div>
      <div className="type-grid">
        {entries.map((entry) => (
          <article key={entry.key} style={{ "--type-color": entry.color }}>
            <span className="type-swatch" /><div><small>{entry.label}</small><strong>{entry.count.toLocaleString()}</strong><p>{entry.percent.toFixed(1)}% · {entry.description}</p></div>
          </article>
        ))}
      </div>
      <div className="composition-notes">
        <div><span>Typed nuclei</span><strong>{total.toLocaleString()}</strong></div>
        <div><span>Neoplastic : inflammatory</span><strong>{ratio == null ? "Not available" : `${ratio.toFixed(2)} : 1`}</strong></div>
        <p>The ratio describes this image field only. It is not a diagnosis, tumour grade, or treatment indicator.</p>
      </div>
    </section>
  );
}

function MetastasisResult({ result, fileName, activeView, setActiveView }) {
  const views = getAvailableViews(result);
  const currentView = views.some(([key]) => key === activeView) ? activeView : views[0]?.[0];
  const benchmark = result?.benchmark || {};
  const thresholdName = result.operating_point === "sensitivity" ? "Sensitivity focused" : "Balanced";
  const probabilityPercent = (Number(result.mean_probability || 0) * 100).toFixed(1);
  const maxProbabilityPercent = (Number(result.max_probability || 0) * 100).toFixed(1);
  const finding = result.diagnostic_support || {
    finding_title: result.evidence_label || "Analysis complete",
    finding_summary: "Review the highlighted regions alongside the original tissue image.",
    evidence_strength: "moderate",
    uncertainty_percent: null,
    near_threshold_percent: null,
    next_step: "Review the highlighted regions and confirm the morphology with a qualified pathologist.",
    uncertainty_note: "Model performance can vary with stain, focus, magnification, and tissue preparation.",
  };

  return (
    <section className="result-card result-ready metastasis-result" aria-live="polite">
      <div className="result-head">
        <div><span className="eyebrow">AI-assisted preliminary finding</span><h2>{fileName}</h2></div>
        <div className="result-meta"><span className="typed-badge pcam-badge">PCam · EfficientNetV2</span><span className="result-size">{result.width} × {result.height}px</span></div>
      </div>

      <section className={`finding-card finding-${finding.finding_code || "review"}`} aria-labelledby="finding-title">
        <div className="finding-lead">
          <span className="finding-status">{finding.evidence_strength} evidence</span>
          <h3 id="finding-title">{finding.finding_title}</h3>
          <p>{finding.finding_summary}</p>
        </div>
        <div className="finding-guidance">
          <article><span>What this may indicate</span><p>The field contains—or does not contain—visual patterns similar to metastatic breast-cancer tissue represented in PCam.</p></article>
          <article><span>Uncertainty</span><p>{finding.uncertainty_percent == null ? "Not reported" : `${finding.uncertainty_percent}% average probability uncertainty`}{finding.near_threshold_percent == null ? "." : `; ${finding.near_threshold_percent}% of regions were borderline.`} {finding.uncertainty_note}</p></article>
          <article><span>Recommended next step</span><p>{finding.next_step}</p></article>
        </div>
      </section>

      <div className="view-tabs" role="tablist" aria-label="Metastatic-pattern result view">
        {views.map(([key, label]) => (
          <button key={key} type="button" role="tab" aria-selected={currentView === key} onClick={() => setActiveView(key)}>{label}</button>
        ))}
      </div>

      {currentView && <div className={`image-stage view-${currentView}`}><img src={result.visuals[currentView]} alt={`${VIEW_LABELS[currentView] || currentView} for ${fileName}`} /><span className="view-label">{VIEW_LABELS[currentView] || currentView}</span></div>}

      <div className="metrics-grid metastasis-metrics">
        <article><span>Flagged regions</span><strong>{formatNumber(result.positive_patch_count)} / {formatNumber(result.patch_count)}</strong><small>{formatNumber(result.positive_patch_percent)}% of analyzed regions</small></article>
        <article><span>Highlighted area</span><strong>{formatNumber(result.heatmap_area_percent)}%</strong><small>of this field crossed the review level</small></article>
        <article><span>Average evidence</span><strong>{probabilityPercent}%</strong><small>strongest region {maxProbabilityPercent}%</small></article>
        <article><span>Analysis time</span><strong>{formatNumber(result.inference_ms)} ms</strong><small>server processing</small></article>
      </div>

      <details className="model-details">
        <summary>Show model measurements and validation details</summary>
        <dl>
          <div><dt>Review mode</dt><dd>{thresholdName}</dd></div>
          <div><dt>Decision level</dt><dd>{Number(result.threshold).toFixed(3)}</dd></div>
          <div><dt>Held-out ROC-AUC</dt><dd>{benchmark.roc_auc ? Number(benchmark.roc_auc).toFixed(3) : "0.935"}</dd></div>
          <div><dt>Held-out PR-AUC</dt><dd>{benchmark.pr_auc ? Number(benchmark.pr_auc).toFixed(3) : "0.938"}</dd></div>
        </dl>
      </details>

      <section className="clinical-context" aria-labelledby="clinical-context-title">
        <div>
          <span className="eyebrow">Diagnostic context</span>
          <h3 id="clinical-context-title">Helps assess possible breast-cancer spread to sentinel lymph nodes</h3>
          <p>PCam was derived from H&amp;E lymph-node sections from breast-cancer cases. This model helps screen a tissue field for metastatic tumour patterns and directs attention to the regions most worth reviewing.</p>
        </div>
        <div className="context-columns">
          <article><strong>Useful result</strong><p>A preliminary positive, limited, or not-detected finding for this field, plus highlighted regions, evidence strength, uncertainty, and a review recommendation.</p></article>
          <article><strong>Required confirmation</strong><p>A pathologist must confirm the morphology on the complete case. This field alone cannot establish lymph-node status, stage the cancer, or choose treatment.</p></article>
        </div>
      </section>

      <section className="practical-card" aria-labelledby="metastasis-practical-title">
        <div className="practical-summary">
          <span className="eyebrow">Practical use</span>
          <h3 id="metastasis-practical-title">Prioritize regions for review, not patients for diagnosis.</h3>
          <p>This workflow turns a tissue field into a spatial probability map. It can help a researcher identify patches worth inspecting, compare model evidence across consistently prepared fields, and audit false-positive morphology.</p>
        </div>
        <div className="practical-use-grid">
          <article><span>01</span><div><strong>Localize evidence</strong><p>Use the overlay to see where high-probability tissue patterns occur.</p></div></article>
          <article><span>02</span><div><strong>Adjust the purpose</strong><p>Balanced mode limits false positives; sensitivity-focused mode reduces missed positive patches.</p></div></article>
          <article><span>03</span><div><strong>Verify visually</strong><p>Compare the heatmap with the original H&amp;E field and record the frozen model version.</p></div></article>
        </div>
        <div className="practical-caution"><strong>Patch-level research result.</strong><p>This output does not establish a cancer diagnosis, tumour stage or grade, survival, recurrence risk, molecular subtype, or treatment response.</p></div>
      </section>
    </section>
  );
}

function ResultPanel({ result, health, fileName, activeView, setActiveView, busy, analysisMode }) {
  if (busy) {
    return (
      <section className="result-card result-loading" aria-live="polite">
        <div className="scan-frame"><span /></div>
        <p>{analysisMode === "metastasis" ? "Mapping patch probabilities to the tissue field" : "Reconstructing the full-resolution prediction maps"}</p>
        <small>{analysisMode === "metastasis" ? "The calibrated PCam model is evaluating overlapping 96px patches." : "Nuclear, separation, and type outputs are averaged across overlapping tiles."}</small>
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
          <p>{analysisMode === "metastasis" ? "Compare the source field, probability heatmap, and frozen-threshold evidence map." : "Compare the source, mask, separated instances, and—when available—the five-class phenotype overlay."}</p>
        </div>
      </section>
    );
  }

  if (result.analysis_type === "metastasis_patch_map") {
    return <MetastasisResult result={result} fileName={fileName} activeView={activeView} setActiveView={setActiveView} />;
  }

  const views = getAvailableViews(result);
  const currentView = views.some(([key]) => key === activeView) ? activeView : views[0]?.[0];
  const typeEntries = getTypeEntries(result);
  const typed = isTypedModel(health, result);
  const countError = result?.benchmark?.count_relative_mae_percent ?? health?.model?.benchmark?.count_relative_mae_percent;

  return (
    <section className="result-card result-ready" aria-live="polite">
      <div className="result-head">
        <div>
          <span className="eyebrow">Analysis complete</span>
          <h2>{fileName}</h2>
        </div>
        <div className="result-meta">{typed && <span className="typed-badge">5-class model</span>}<span className="result-size">{result.width} × {result.height}px</span></div>
      </div>

      <div className="view-tabs" role="tablist" aria-label="Result view">
        {views.map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={currentView === key}
            onClick={() => setActiveView(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {currentView && <div className={`image-stage view-${currentView}`}><img src={result.visuals[currentView]} alt={`${VIEW_LABELS[currentView] || currentView} result for ${fileName}`} /><span className="view-label">{VIEW_LABELS[currentView] || currentView}</span></div>}

      <div className="metrics-grid">
        <article><span>Estimated nuclei</span><strong>{formatNumber(result.predicted_count)}</strong><small>separated nucleus-like objects</small></article>
        <article><span>Nuclear area fraction</span><strong>{formatNumber(result.nuclei_area_percent)}%</strong><small>of this field classified as nuclear</small></article>
        <article><span>{typed ? "Phenotypes reported" : "Field size"}</span><strong>{typed ? typeEntries.length || 5 : `${result.width} × ${result.height}`}</strong><small>{typed ? "PanNuke nuclear classes" : "pixels · compare at matched scale"}</small></article>
        <article><span>Processing time</span><strong>{formatNumber(result.inference_ms)} ms</strong><small>server inference</small></article>
      </div>

      <TypeComposition entries={typeEntries} fileName={fileName} />

      <section className="practical-card" aria-labelledby="practical-title">
        <div className="practical-summary">
          <span className="eyebrow">Practical meaning</span>
          <h3 id="practical-title">Turn this tissue field into a repeatable measurement.</h3>
          <p>
            In this {formatNumber(result.width)} × {formatNumber(result.height)} pixel field, the model estimates{" "}
            <strong>{formatNumber(result.predicted_count)} nucleus-like objects</strong>. Predicted nuclei occupy{" "}
            <strong>{formatNumber(result.nuclei_area_percent)}% of the field</strong>{typed ? ", with each separated object assigned one of five research phenotypes." : "."}
          </p>
        </div>

        <div className="practical-use-grid">
          <article>
            <span>01</span>
            <div><strong>Compare matched fields</strong><p>Track count, area, and composition across fields acquired at the same scale and staining conditions.</p></div>
          </article>
          <article>
            <span>02</span>
            <div><strong>Describe composition</strong><p>{typed ? "Measure predicted neoplastic, inflammatory, stromal, dead, and epithelial proportions." : "Use count and nuclear area as reproducible field-level measurements."}</p></div>
          </article>
          <article>
            <span>03</span>
            <div><strong>Check the evidence</strong><p>Switch between the source, mask, instances, and typed overlay to audit every reported measurement.</p></div>
          </article>
        </div>

        <div className="practical-caution">
          <strong>Interpret as an estimate.</strong>
          <p>
            {countError != null ? `The reported benchmark mean relative count error is ${Number(countError).toFixed(1)}%. ` : "Model performance varies by tissue, stain, image quality, and nucleus class. "}
            Predicted cell type is a visual phenotype estimate, not a diagnosis, cancer determination, grade, or treatment recommendation. Counts from different image scales are not directly comparable.
          </p>
        </div>
      </section>
    </section>
  );
}

function MethodStrip({ typed, analysisMode }) {
  if (analysisMode === "metastasis") {
    return (
      <section className="method-strip" aria-labelledby="method-title">
        <div><span className="section-kicker"><span>02</span> Understand the method</span><h2 id="method-title">Patch. Classify. Map. Review.</h2></div>
        <ol>
          <li><span>1</span><div><strong>Tile</strong><p>Sample the field as overlapping 96px H&amp;E patches without rescaling it.</p></div></li>
          <li><span>2</span><div><strong>Classify</strong><p>Estimate metastatic-tissue probability with the evaluated EfficientNetV2 model.</p></div></li>
          <li><span>3</span><div><strong>Calibrate</strong><p>Apply validation-frozen temperature scaling and operating thresholds.</p></div></li>
          <li><span>4</span><div><strong>Review</strong><p>Map the scores back to their image coordinates for visual audit.</p></div></li>
        </ol>
      </section>
    );
  }
  return (
    <section className="method-strip" aria-labelledby="method-title">
      <div>
        <span className="section-kicker"><span>02</span> Understand the method</span>
        <h2 id="method-title">Pixels. Objects. Phenotypes.</h2>
      </div>
      <ol>
        <li><span>1</span><div><strong>Tile</strong><p>Split the field into overlapping 256px patches.</p></div></li>
        <li><span>2</span><div><strong>Segment</strong><p>Average U-Net probabilities across overlaps.</p></div></li>
        <li><span>3</span><div><strong>Separate</strong><p>{typed ? "Use horizontal and vertical maps to divide touching nuclei." : "Use distance peaks and watershed for touching nuclei."}</p></div></li>
        <li><span>4</span><div><strong>{typed ? "Type" : "Compare"}</strong><p>{typed ? "Assign each instance a five-class nuclear phenotype and quantify composition." : "Compare count and area across matched tissue fields."}</p></div></li>
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
  const [analysisMode, setAnalysisMode] = useState("metastasis");
  const [operatingPoint, setOperatingPoint] = useState("balanced");

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
      const response = await fetch(resolveEndpoint(health, analysisMode, operatingPoint), { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Inference failed with status ${response.status}.`);
      setResult(payload);
      setActiveView(getAvailableViews(payload)[0]?.[0] || "overlay");
    } catch (requestError) {
      setError(requestError.message || "Inference failed. Check the API and try again.");
    } finally {
      setBusy(false);
    }
  };

  const changeAnalysisMode = (nextMode) => {
    setAnalysisMode(nextMode);
    setResult(null);
    setError("");
    setActiveView(nextMode === "metastasis" ? "metastasis_overlay" : "overlay");
  };

  const typed = isTypedModel(health, result);
  const selectedHealth = modelHealth(health, analysisMode);
  const modelSummary = useMemo(() => {
    if (!selectedHealth) return analysisMode === "metastasis" ? "PCam · 96px patches" : "U-Net · 256px patches";
    const architecture = selectedHealth.architecture || (analysisMode === "metastasis" ? "EfficientNetV2-B0" : isTypedModel(health) ? "HoVer U-Net" : "U-Net");
    const version = selectedHealth.version ? ` · ${selectedHealth.version}` : "";
    return `${architecture} · ${selectedHealth.patch_size || (analysisMode === "metastasis" ? 96 : 256)}px${version}`;
  }, [health, selectedHealth, analysisMode]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Nuclei Lens home">
          <Logo />
          <span><strong>Nuclei</strong> Lens</span>
        </a>
        <div className="header-meta">
          <span className="model-summary">{modelSummary}</span>
          <Status health={selectedHealth} loading={healthLoading} />
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">Computational pathology · research application</span>
            <h1>{analysisMode === "metastasis" ? <>Screen for metastasis.<br /><em>See what to review.</em></> : <>See every nucleus.<br /><em>Measure the field.</em></>}</h1>
            <p>
              {analysisMode === "metastasis"
                ? "Receive a preliminary finding for metastatic breast-cancer patterns in sentinel lymph-node H&E tissue, with highlighted regions, uncertainty, and clear next steps."
                : "Segment individual nuclei in H&E tissue, separate touching cells, and quantify field-level composition through an auditable visual workflow."}
            </p>
            <div className="hero-badges"><span>{analysisMode === "metastasis" ? "PCam · 327,680 patches" : typed ? "PanNuke · 19 tissues" : "MoNuSeg 2018"}</span><span>TensorFlow</span><span>{analysisMode === "metastasis" ? "0.935 test ROC-AUC" : typed ? "5 nuclear phenotypes" : "Instance counting"}</span></div>
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
          <UploadPanel health={health} healthLoading={healthLoading} busy={busy} onSelect={handleFile} analysisMode={analysisMode} onModeChange={changeAnalysisMode} operatingPoint={operatingPoint} setOperatingPoint={setOperatingPoint} />
          <ResultPanel result={result} health={health} fileName={fileName} activeView={activeView} setActiveView={setActiveView} busy={busy} analysisMode={analysisMode} />
        </section>

        <MethodStrip typed={typed} analysisMode={analysisMode} />

        <section className="disclaimer">
          <div className="disclaimer-icon">R</div>
          <div><strong>Research use only</strong><p>Nuclei Lens reports image-derived research measurements and is not validated for diagnosis, treatment, prognosis, or clinical decision-making.</p></div>
        </section>
      </main>

      <footer>
        <div className="brand compact"><Logo /><span><strong>Nuclei</strong> Lens</span></div>
        <p>From pixels to measurable cell instances.</p>
        <span>{analysisMode === "metastasis" ? "PCam · EfficientNetV2 · calibrated heatmaps" : typed ? "PanNuke · HoVer maps · 5 classes" : "MoNuSeg · U-Net · Watershed"}</span>
      </footer>
    </div>
  );
}

