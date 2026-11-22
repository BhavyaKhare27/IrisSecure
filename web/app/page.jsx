"use client";
import { useCallback, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
    const [activeTab, setActiveTab] = useState("verify");
    const [apiState, setApiState] = useState("loading");
    const [busy, setBusy] = useState(false);
    const [dashboard, setDashboard] = useState(null);
    const [identities, setIdentities] = useState([]);
    const [notice, setNotice] = useState(null);

    // Authenticate (Scanner) State
    const [probeFile, setProbeFile] = useState(null);
    const [probePreview, setProbePreview] = useState(null);
    const [verifyMode, setVerifyMode] = useState("identify");
    const [claimedId, setClaimedId] = useState("");
    const [threshold, setThreshold] = useState(0.75);
    const [verifyResult, setVerifyResult] = useState(null);

    // Enroll State
    const [enrollId, setEnrollId] = useState("");
    const [enrollEye, setEnrollEye] = useState("L");
    const [enrollFile, setEnrollFile] = useState(null);
    const [enrollPreview, setEnrollPreview] = useState(null);

    // Directory State
    const [searchQuery, setSearchQuery] = useState("");
    const [deletingId, setDeletingId] = useState("");

    const refresh = useCallback(async () => {
        try {
            const [health, stats, gallery] = await Promise.all([
                fetch(`${API}/api/health`),
                fetch(`${API}/api/dashboard`),
                fetch(`${API}/api/identities`),
            ]);
            if (!health.ok || !stats.ok || !gallery.ok) throw new Error("API unavailable");
            setDashboard(await stats.json());
            setIdentities(await gallery.json());
            setApiState("online");
        } catch {
            setApiState("offline");
        }
    }, []);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const handleProbeSelect = (file) => {
        if (!file) return;
        if (probePreview) URL.revokeObjectURL(probePreview);
        setProbeFile(file);
        setProbePreview(URL.createObjectURL(file));
        setVerifyResult(null);
    };

    const clearProbe = () => {
        if (probePreview) URL.revokeObjectURL(probePreview);
        setProbeFile(null);
        setProbePreview(null);
        setVerifyResult(null);
    };

    const handleEnrollFileSelect = (file) => {
        if (!file) return;
        if (enrollPreview) URL.revokeObjectURL(enrollPreview);
        setEnrollFile(file);
        setEnrollPreview(URL.createObjectURL(file));
    };

    const clearEnrollFile = () => {
        if (enrollPreview) URL.revokeObjectURL(enrollPreview);
        setEnrollFile(null);
        setEnrollPreview(null);
    };

    const runVerification = async (e) => {
        e.preventDefault();
        if (!probeFile) {
            setNotice({ type: "error", message: "Select or drop an iris image first." });
            return;
        }
        if (verifyMode === "verify" && !claimedId.trim()) {
            setNotice({ type: "error", message: "Choose an enrolled identity to verify against." });
            return;
        }

        setBusy(true);
        setNotice(null);
        setVerifyResult(null);

        try {
            const form = new FormData();
            form.append("image", probeFile);

            const extRes = await fetch(`${API}/api/recognition/embedding`, { method: "POST", body: form });
            const extData = await extRes.json();
            if (!extRes.ok) throw new Error(extData.detail ?? "Feature extraction failed");

            const endpoint = verifyMode === "identify"
                ? `${API}/api/identify?threshold=${threshold}`
                : `${API}/api/verify/${encodeURIComponent(claimedId.trim())}?threshold=${threshold}`;

            const recRes = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    embedding: extData.embedding,
                    model_fingerprint: extData.model_fingerprint,
                }),
            });
            const recData = await recRes.json();
            if (!recRes.ok) throw new Error(recData.detail ?? "Recognition failed");

            const isSuccess = verifyMode === "identify" ? recData.matched : recData.verified;
            const sim = recData.similarity;
            const margin = sim - threshold;

            setVerifyResult({
                success: isSuccess,
                title: isSuccess ? "ACCESS GRANTED" : "ACCESS DENIED",
                subtitle: isSuccess
                    ? `Matched Subject: ${recData.user_id}`
                    : verifyMode === "identify"
                        ? "No registered iris template matched above threshold"
                        : `Verification failed for claimed ID: ${claimedId}`,
                userId: recData.user_id ?? "None",
                similarity: sim,
                threshold: threshold,
                margin: margin,
            });
        } catch (err) {
            setNotice({ type: "error", message: err instanceof Error ? err.message : "Authentication failed." });
        } finally {
            setBusy(false);
        }
    };

    const runEnrollment = async (e) => {
        e.preventDefault();
        if (!enrollFile || !enrollId.trim()) {
            setNotice({ type: "error", message: "Provide an identity ID and an iris image." });
            return;
        }

        setBusy(true);
        setNotice(null);

        try {
            const form = new FormData();
            form.append("image", enrollFile);

            const extRes = await fetch(`${API}/api/recognition/embedding`, { method: "POST", body: form });
            const extData = await extRes.json();
            if (!extRes.ok) throw new Error(extData.detail ?? "Feature extraction failed");

            const compositeId = `${enrollId.trim()}_${enrollEye}`;
            const regRes = await fetch(
                `${API}/api/identities/${encodeURIComponent(compositeId)}?eye=${enrollEye}&person_id=${encodeURIComponent(enrollId.trim())}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        embedding: extData.embedding,
                        model_fingerprint: extData.model_fingerprint,
                    }),
                }
            );
            const regData = await regRes.json();
            if (!regRes.ok) throw new Error(regData.detail ?? "Enrollment failed");

            setNotice({
                type: "success",
                message: `Successfully enrolled ${regData.user_id}. The source photo was discarded.`,
            });
            setEnrollId("");
            clearEnrollFile();
            await refresh();
        } catch (err) {
            setNotice({ type: "error", message: err instanceof Error ? err.message : "Enrollment failed." });
        } finally {
            setBusy(false);
        }
    };

    const deleteSubject = async (identity) => {
        if (!window.confirm(`Delete ${identity.user_id} from the biometric gallery?`)) return;
        setDeletingId(identity.user_id);
        setNotice(null);
        try {
            const res = await fetch(`${API}/api/identities/${encodeURIComponent(identity.user_id)}`, {
                method: "DELETE",
            });
            if (!res.ok) throw new Error("Could not delete identity.");
            if (claimedId === identity.user_id) setClaimedId("");
            setNotice({ type: "info", message: `Deleted ${identity.user_id}.` });
            await refresh();
        } catch (err) {
            setNotice({ type: "error", message: err instanceof Error ? err.message : "Deletion failed." });
        } finally {
            setDeletingId("");
        }
    };

    const filteredIdentities = identities.filter((item) =>
        item.user_id.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="station-shell">
            {/* Top Navigation Bar */}
            <header className="top-navbar">
                <div className="brand-group">
                    <div className="brand-eye-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                            <circle cx="12" cy="12" r="3" />
                        </svg>
                    </div>
                    <div className="brand-title-wrap">
                        <span className="brand-name">IrisSecure</span>
                        <span className="station-badge">STATION</span>
                    </div>
                </div>

                <div className="top-actions">
                    <div className={`backend-status-pill ${apiState === "online" ? "online" : "offline"}`}>
                        <span className="status-dot" />
                        <span>
                            {apiState === "online"
                                ? dashboard?.model_available
                                    ? "Backend Online"
                                    : "Model Offline"
                                : "Disconnected"}
                        </span>
                    </div>

                    <button
                        className="refresh-circle-btn"
                        onClick={() => void refresh()}
                        title="Sync with backend"
                        aria-label="Refresh"
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                            <path d="M3 3v5h5" />
                            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                            <path d="M16 16h5v5" />
                        </svg>
                    </button>
                </div>
            </header>

            {/* Notification Toast */}
            {notice && (
                <div className={`toast-notice ${notice.type}`}>
                    <span>{notice.message}</span>
                    <button
                        style={{ background: "none", border: "none", color: "inherit", cursor: "pointer", fontSize: 16 }}
                        onClick={() => setNotice(null)}
                    >
                        ✕
                    </button>
                </div>
            )}

            {apiState === "offline" && (
                <div className="toast-notice error">
                    Backend offline at <code>{API}</code>. Ensure the FastAPI server is running.
                </div>
            )}

            {/* Main Station Body: Left Sidebar + Right Workspace */}
            <div className="station-body">
                {/* Left Navigation Sidebar */}
                <aside className="station-sidebar">
                    <button
                        className={`sidebar-tab-btn ${activeTab === "verify" ? "active" : ""}`}
                        onClick={() => setActiveTab("verify")}
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4" />
                            <path d="M14 13.12c0 2.38 0 6.38-1 8.88" />
                            <path d="M17.29 21.02c.12-.6.43-2.3.5-3.02" />
                            <path d="M2 12a10 10 0 0 1 18-6" />
                            <path d="M2 16h.01" />
                            <path d="M21.8 16c.2-2 .131-5.354 0-6" />
                            <path d="M5 19.5C5.5 18 6 15 6 12a6 6 0 0 1 .34-2" />
                            <path d="M8.65 22c.21-.66.45-1.32.57-2" />
                            <path d="M9 6.8a6 6 0 0 1 9 5.2v2" />
                        </svg>
                        <span>Authenticate</span>
                    </button>

                    <button
                        className={`sidebar-tab-btn ${activeTab === "enroll" ? "active" : ""}`}
                        onClick={() => setActiveTab("enroll")}
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M8 12h8" />
                            <path d="M12 8v8" />
                        </svg>
                        <span>Enroll Identity</span>
                    </button>

                    <button
                        className={`sidebar-tab-btn ${activeTab === "gallery" ? "active" : ""}`}
                        onClick={() => setActiveTab("gallery")}
                    >
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <ellipse cx="12" cy="5" rx="9" ry="3" />
                            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                            <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
                        </svg>
                        <span>Directory</span>
                        <span className="sidebar-badge">{identities.length}</span>
                    </button>
                </aside>

                {/* Right Workspace Content */}
                <main className="station-main">
                    {/* VIEW 1: AUTHENTICATE (BIOMETRIC SCANNER) */}
                    {activeTab === "verify" && (
                        <div>
                            <div className="view-heading-wrap">
                                <h1 className="view-title">Biometric Scanner</h1>
                                <p className="view-subtitle">
                                    Match an iris capture against the enrolled gallery or verify a specific subject claim.
                                </p>
                            </div>

                            <div className="terminal-card">
                                <form onSubmit={runVerification}>
                                    <div className="scanner-grid">
                                        {/* Left: Large Dropzone / Viewfinder */}
                                        <div className="dropzone-container">
                                            {!probePreview ? (
                                                <div
                                                    className="dropzone-box"
                                                    onDragOver={(e) => e.preventDefault()}
                                                    onDrop={(e) => {
                                                        e.preventDefault();
                                                        handleProbeSelect(e.dataTransfer.files?.[0]);
                                                    }}
                                                >
                                                    <div className="dropzone-squircle-icon">
                                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                                            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                                                            <circle cx="12" cy="12" r="3" />
                                                        </svg>
                                                    </div>
                                                    <div className="dropzone-main-text">Drop iris photo or click to browse</div>
                                                    <div className="dropzone-sub-text">Supports JPEG, PNG, WEBP, BMP</div>
                                                    <input
                                                        type="file"
                                                        accept="image/*"
                                                        onChange={(e) => handleProbeSelect(e.target.files?.[0])}
                                                    />
                                                </div>
                                            ) : (
                                                <div className="preview-viewport">
                                                    <img src={probePreview} alt="Probe" className="preview-image" />
                                                    <div className="reticle-overlay">
                                                        <div className="cyber-reticle">
                                                            <div className="reticle-core" />
                                                        </div>
                                                        <div className="laser-scan-sweep" />
                                                    </div>
                                                    <button type="button" className="change-image-btn" onClick={clearProbe}>
                                                        ✕ Change Image
                                                    </button>
                                                </div>
                                            )}
                                        </div>

                                        {/* Right: Controls Column */}
                                        <div className="controls-column">
                                            <div>
                                                <label className="field-label">Matching Mode</label>
                                                <div className="segmented-mode-toggle">
                                                    <button
                                                        type="button"
                                                        className={`mode-tab-btn ${verifyMode === "identify" ? "active" : ""}`}
                                                        onClick={() => setVerifyMode("identify")}
                                                    >
                                                        1:N Identify
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={`mode-tab-btn ${verifyMode === "verify" ? "active" : ""}`}
                                                        onClick={() => setVerifyMode("verify")}
                                                    >
                                                        1:1 Verify
                                                    </button>
                                                </div>
                                            </div>

                                            {verifyMode === "verify" && (
                                                <div>
                                                    <label className="field-label">Claimed Subject</label>
                                                    <select
                                                        className="select-box"
                                                        value={claimedId}
                                                        onChange={(e) => setClaimedId(e.target.value)}
                                                    >
                                                        <option value="">Select enrolled identity...</option>
                                                        {identities.map((item) => (
                                                            <option key={item.user_id} value={item.user_id}>
                                                                {item.user_id}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}

                                            <div>
                                                <label className="field-label">Security Sensitivity</label>
                                                <div className="sensitivity-card">
                                                    <div className="sensitivity-header">
                                                        <span>Cosine Cutoff (τ)</span>
                                                        <span className="sensitivity-value">{threshold.toFixed(2)}</span>
                                                    </div>
                                                    <input
                                                        type="range"
                                                        min="0.30"
                                                        max="0.95"
                                                        step="0.05"
                                                        value={threshold}
                                                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                                        className="range-slider"
                                                    />
                                                    <div className="sensitivity-marks">
                                                        <span>Permissive (0.30)</span>
                                                        <span>Standard (0.75)</span>
                                                        <span>Strict (0.95)</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <button
                                                type="submit"
                                                className="scan-cta-btn"
                                                disabled={busy || !probeFile || !dashboard?.model_available}
                                            >
                                                {busy ? (
                                                    <>
                                                        <span className="spin-loader" /> Analyzing Iris Texture...
                                                    </>
                                                ) : (
                                                    <>
                                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                                            <path d="M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4" />
                                                            <path d="M14 13.12c0 2.38 0 6.38-1 8.88" />
                                                            <path d="M17.29 21.02c.12-.6.43-2.3.5-3.02" />
                                                            <path d="M2 12a10 10 0 0 1 18-6" />
                                                            <path d="M2 16h.01" />
                                                            <path d="M21.8 16c.2-2 .131-5.354 0-6" />
                                                            <path d="M5 19.5C5.5 18 6 15 6 12a6 6 0 0 1 .34-2" />
                                                            <path d="M8.65 22c.21-.66.45-1.32.57-2" />
                                                            <path d="M9 6.8a6 6 0 0 1 9 5.2v2" />
                                                        </svg>
                                                        <span>Scan &amp; Authenticate</span>
                                                    </>
                                                )}
                                            </button>
                                        </div>
                                    </div>
                                </form>

                                {/* Result Output Card */}
                                {verifyResult && (
                                    <div className={`verification-result-panel ${verifyResult.success ? "granted" : "denied"}`}>
                                        <div className="result-headline-row">
                                            <div className="result-status-icon">{verifyResult.success ? "✓" : "✕"}</div>
                                            <div>
                                                <div className="result-status-title">{verifyResult.title}</div>
                                                <div className="result-status-desc">{verifyResult.subtitle}</div>
                                            </div>
                                        </div>

                                        <div className="result-metrics-grid">
                                            <div className="metric-cell">
                                                <span>SIMILARITY SCORE</span>
                                                <b>{verifyResult.similarity.toFixed(4)}</b>
                                            </div>
                                            <div className="metric-cell">
                                                <span>OPERATING THRESHOLD</span>
                                                <b>{verifyResult.threshold.toFixed(4)}</b>
                                            </div>
                                            <div className="metric-cell">
                                                <span>SCORE MARGIN</span>
                                                <b style={{ color: verifyResult.margin >= 0 ? "var(--emerald)" : "var(--rose)" }}>
                                                    {verifyResult.margin >= 0 ? "+" : ""}
                                                    {verifyResult.margin.toFixed(4)}
                                                </b>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* VIEW 2: ENROLL IDENTITY (Full-width two-column layout) */}
                    {activeTab === "enroll" && (
                        <div>
                            <div className="view-heading-wrap">
                                <h1 className="view-title">Enroll Identity</h1>
                                <p className="view-subtitle">
                                    Register a new biometric iris template into the secure gallery database.
                                </p>
                            </div>

                            <div className="terminal-card">
                                <form onSubmit={runEnrollment}>
                                    <div className="scanner-grid">
                                        {/* Left: Iris Template Image Dropzone / Preview */}
                                        <div className="dropzone-container">
                                            {!enrollPreview ? (
                                                <div
                                                    className="dropzone-box"
                                                    onDragOver={(e) => e.preventDefault()}
                                                    onDrop={(e) => {
                                                        e.preventDefault();
                                                        handleEnrollFileSelect(e.dataTransfer.files?.[0]);
                                                    }}
                                                >
                                                    <div className="dropzone-squircle-icon">
                                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                                            <circle cx="12" cy="12" r="10" />
                                                            <path d="M8 12h8" />
                                                            <path d="M12 8v8" />
                                                        </svg>
                                                    </div>
                                                    <div className="dropzone-main-text">Drop iris photo or click to browse</div>
                                                    <div className="dropzone-sub-text">Supports JPEG, PNG, WEBP, BMP</div>
                                                    <input
                                                        type="file"
                                                        accept="image/*"
                                                        onChange={(e) => handleEnrollFileSelect(e.target.files?.[0])}
                                                    />
                                                </div>
                                            ) : (
                                                <div className="preview-viewport">
                                                    <img src={enrollPreview} alt="Enrollment Probe" className="preview-image" />
                                                    <button type="button" className="change-image-btn" onClick={clearEnrollFile}>
                                                        ✕ Change Image
                                                    </button>
                                                </div>
                                            )}
                                        </div>

                                        {/* Right: Controls Column */}
                                        <div className="controls-column">
                                            <div>
                                                <label className="field-label">Subject Identifier</label>
                                                <input
                                                    type="text"
                                                    className="text-box"
                                                    placeholder="e.g. participant_042"
                                                    value={enrollId}
                                                    onChange={(e) => setEnrollId(e.target.value)}
                                                />
                                            </div>

                                            <div>
                                                <label className="field-label">Eye Designation</label>
                                                <div className="segmented-mode-toggle">
                                                    <button
                                                        type="button"
                                                        className={`mode-tab-btn ${enrollEye === "L" ? "active" : ""}`}
                                                        onClick={() => setEnrollEye("L")}
                                                    >
                                                        👁 Left Eye (L)
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className={`mode-tab-btn ${enrollEye === "R" ? "active" : ""}`}
                                                        onClick={() => setEnrollEye("R")}
                                                    >
                                                        👁 Right Eye (R)
                                                    </button>
                                                </div>
                                            </div>

                                            <div>
                                                <button
                                                    type="submit"
                                                    className="scan-cta-btn"
                                                    disabled={busy || !enrollFile || !enrollId.trim() || !dashboard?.model_available}
                                                >
                                                    {busy ? (
                                                        <>
                                                            <span className="spin-loader" /> Generating Neural Template...
                                                        </>
                                                    ) : (
                                                        "＋  Register Biometric Template"
                                                    )}
                                                </button>

                                                <p className="privacy-notice">
                                                    🔒 Source photos are discarded immediately after 512-D normalized embedding extraction.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        </div>
                    )}

                    {/* VIEW 3: DIRECTORY */}
                    {activeTab === "gallery" && (
                        <div>
                            <div className="view-heading-wrap">
                                <h1 className="view-title">Enrolled Directory</h1>
                                <p className="view-subtitle">
                                    Manage registered biometric templates stored in the active gallery.
                                </p>
                            </div>

                            <div className="terminal-card">
                                <div className="directory-filter-row">
                                    <input
                                        type="text"
                                        className="text-box"
                                        placeholder="Filter by subject ID..."
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                    />
                                </div>

                                {filteredIdentities.length > 0 ? (
                                    <div className="directory-cards-grid">
                                        {filteredIdentities.map((item) => (
                                            <div key={item.user_id} className="subject-card">
                                                <div className="subject-info">
                                                    <div className="subject-avatar">
                                                        {item.user_id.slice(0, 1).toUpperCase()}
                                                    </div>
                                                    <div>
                                                        <div className="subject-id">{item.user_id}</div>
                                                        <div className="subject-meta">
                                                            <span className="eye-tag-pill">{item.eye || "EYE"}</span>
                                                            <span>•</span>
                                                            <span>{item.num_samples ?? 1} sample</span>
                                                        </div>
                                                    </div>
                                                </div>
                                                <button
                                                    type="button"
                                                    className="delete-identity-btn"
                                                    disabled={Boolean(deletingId)}
                                                    onClick={() => deleteSubject(item)}
                                                    title={`Delete ${item.user_id}`}
                                                >
                                                    {deletingId === item.user_id ? "..." : "Delete"}
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="empty-gallery">
                                        <div className="empty-gallery-icon">⦾</div>
                                        <div style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: 16 }}>No Identities Found</div>
                                        <div style={{ fontSize: 13, marginTop: 4 }}>
                                            {searchQuery
                                                ? "No enrolled subjects match your search filter."
                                                : "No biometric templates are currently enrolled. Switch to the Enroll tab to register one."}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
