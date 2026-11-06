"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export default function Home() {
    const [dashboard, setDashboard] = useState(null);
    const [identities, setIdentities] = useState([]);
    const [calibration, setCalibration] = useState(null);
    const [apiState, setApiState] = useState("loading");
    const [busy, setBusy] = useState(false);
    const [notice, setNotice] = useState("");
    const [userId, setUserId] = useState("");
    const [eye, setEye] = useState("L");
    const [image, setImage] = useState(null);
    const [recognitionImage, setRecognitionImage] = useState(null);
    const [recognitionMode, setRecognitionMode] = useState("identify");
    const [claimedId, setClaimedId] = useState("");
    const [recognitionResult, setRecognitionResult] = useState("");
    const [recognitionStatus, setRecognitionStatus] = useState("");
    const [targetFar, setTargetFar] = useState("eer");
    const refresh = useCallback(async () => {
        try {
            const [health, stats, gallery, operatingPoint] = await Promise.all([
                fetch(`${API}/api/health`), fetch(`${API}/api/dashboard`),
                fetch(`${API}/api/identities`), fetch(`${API}/api/analytics/threshold`),
            ]);
            if (!health.ok || !stats.ok || !gallery.ok)
                throw new Error("API unavailable");
            setDashboard(await stats.json());
            setIdentities(await gallery.json());
            if (operatingPoint.ok)
                setCalibration(await operatingPoint.json());
            setApiState("online");
        }
        catch {
            setApiState("offline");
        }
    }, []);
    useEffect(() => { void refresh(); }, [refresh]);
    const chartOption = useMemo(() => ({
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", backgroundColor: "#111a2c", borderColor: "#263650", textStyle: { color: "#eaf0fc" } },
        legend: { top: 0, right: 4, textStyle: { color: "#8d9bb2" }, itemWidth: 8, itemHeight: 8 },
        grid: { left: 44, right: 18, top: 42, bottom: 34 },
        xAxis: { type: "value", min: -1, max: 1, name: "Cosine threshold", nameTextStyle: { color: "#71809a" }, axisLabel: { color: "#71809a" }, splitLine: { lineStyle: { color: "#1d2a40" } } },
        yAxis: { type: "log", min: 0.000001, max: 1, name: "Error rate", nameTextStyle: { color: "#71809a" }, axisLabel: { color: "#71809a", formatter: (value) => `${(value * 100).toPrecision(1)}%` }, splitLine: { lineStyle: { color: "#1d2a40" } } },
        series: calibration ? [
            { name: "False accept", type: "line", smooth: true, showSymbol: false, data: calibration.curve.map(p => [p.threshold, Math.max(p.far, 0.000001)]), lineStyle: { width: 2, color: "#ff7c7c" }, areaStyle: { color: "rgba(255,124,124,.06)" } },
            { name: "False reject", type: "line", smooth: true, showSymbol: false, data: calibration.curve.map(p => [p.threshold, Math.max(p.frr, 0.000001)]), lineStyle: { width: 2, color: "#66b6ff" } },
        ] : [],
    }), [calibration]);
    async function optimize() {
        setBusy(true);
        setNotice("");
        try {
            const query = targetFar === "eer" ? "" : `?target_far=${targetFar}`;
            const queued = await fetch(`${API}/api/analytics/optimize${query}`, { method: "POST" }).then(r => r.json());
            for (let attempt = 0; attempt < 30; attempt++) {
                await new Promise(resolve => setTimeout(resolve, 700));
                const job = await fetch(`${API}/api/jobs/${queued.job_id}`).then(r => r.json());
                if (job.status === "completed") {
                    setCalibration(job.result);
                    setNotice("Threshold calibration complete.");
                    break;
                }
                if (job.status === "failed")
                    throw new Error(job.error ?? "Calibration failed");
            }
        }
        catch (error) {
            setNotice(error instanceof Error ? error.message : "Could not run calibration.");
        }
        finally {
            setBusy(false);
        }
    }
    async function register(event) {
        event.preventDefault();
        if (!image || !userId.trim()) {
            setNotice("Choose an iris image and enter an identity ID.");
            return;
        }
        setBusy(true);
        setNotice("");
        try {
            const form = new FormData();
            form.append("image", image);
            const extracted = await fetch(`${API}/api/recognition/embedding`, { method: "POST", body: form });
            const output = await extracted.json();
            if (!extracted.ok)
                throw new Error(output.detail ?? "Embedding extraction failed");
            const response = await fetch(`${API}/api/identities/${encodeURIComponent(`${userId.trim()}_${eye}`)}?eye=${eye}&person_id=${encodeURIComponent(userId.trim())}`, {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ embedding: output.embedding }),
            });
            const result = await response.json();
            if (!response.ok)
                throw new Error(result.detail ?? "Registration failed");
            setNotice(`Registered ${result.user_id}. The uploaded image was not retained.`);
            setUserId("");
            setImage(null);
            await refresh();
        }
        catch (error) {
            setNotice(error instanceof Error ? error.message : "Registration failed.");
        }
        finally {
            setBusy(false);
        }
    }
    async function recognize(event) {
        event.preventDefault();
        if (!recognitionImage) {
            setRecognitionStatus("failure");
            setRecognitionResult("Choose an iris image first.");
            return;
        }
        if (recognitionMode === "verify" && !claimedId.trim()) {
            setRecognitionStatus("failure");
            setRecognitionResult("Enter the identity to verify.");
            return;
        }
        setBusy(true);
        setRecognitionStatus("");
        setRecognitionResult("");
        try {
            const form = new FormData();
            form.append("image", recognitionImage);
            const extracted = await fetch(`${API}/api/recognition/embedding`, { method: "POST", body: form });
            const output = await extracted.json();
            if (!extracted.ok)
                throw new Error(output.detail ?? "Embedding extraction failed");
            const endpoint = recognitionMode === "identify"
                ? `${API}/api/identify?threshold=0.75`
                : `${API}/api/verify/${encodeURIComponent(claimedId.trim())}?threshold=0.75`;
            const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ embedding: output.embedding }) });
            const result = await response.json();
            if (!response.ok)
                throw new Error(result.detail ?? "Recognition failed");
            const succeeded = recognitionMode === "identify" ? result.matched : result.verified;
            setRecognitionStatus(succeeded ? "success" : "failure");
            setRecognitionResult(recognitionMode === "identify"
                ? result.matched ? `Identified ${result.user_id} · similarity ${result.similarity.toFixed(4)}` : `No match above threshold · best similarity ${result.similarity.toFixed(4)}`
                : result.verified ? `Verified ${result.user_id} · similarity ${result.similarity.toFixed(4)}` : `Verification denied · similarity ${result.similarity.toFixed(4)}`);
        }
        catch (error) {
            setRecognitionStatus("failure");
            setRecognitionResult(error instanceof Error ? error.message : "Recognition failed.");
        }
        finally {
            setBusy(false);
        }
    }
    const ref = dashboard?.dataset_reference;
    return <main className="shell">
    <aside className="rail">
      <div className="brand-mark">I<span>·</span></div>
      <div className="rail-links"><a className="selected" title="Overview">◫</a><a href="#gallery" title="Gallery">◉</a><a href="#analytics" title="Analytics">⌁</a></div>
      <div className="rail-bottom">IS</div>
    </aside>
    <section className="workspace">
      <header className="topbar"><div className="crumb">IRISSECURE <span>/</span> BIOMETRIC INTELLIGENCE</div><div className="top-status"><i className={apiState === "online" ? "live" : ""}/>{apiState === "online" ? "API connected" : apiState === "offline" ? "API unavailable" : "Connecting"}<span className="avatar">BK</span></div></header>
      <div className="content">
        <div className="intro"><div><div className="eyebrow">SECURE ACCESS · R&D SYSTEM</div><h1>Good afternoon.</h1><p>Monitor your iris identity network and tune its verification boundary.</p></div><button className="quiet-button" onClick={() => void refresh()}>↻ <span>Refresh</span></button></div>
        {apiState === "offline" && <div className="banner">Could not reach the API at <code>{API}</code>. Start the backend and reload this page.</div>}
        {notice && <div className="banner notice">{notice}</div>}

        <div className="metrics">
          <article className="metric-card"><div className="metric-label">REGISTERED IDENTITIES <span className="metric-icon green">◉</span></div><div className="metric-value">{dashboard?.identity_count ?? "—"}</div><div className="metric-foot">Stored biometric templates</div></article>
          <article className="metric-card"><div className="metric-label">REFERENCE CLASSES <span className="metric-icon blue">▦</span></div><div className="metric-value">{ref?.classes?.toLocaleString() ?? "—"}</div><div className="metric-foot">CASIA-Iris-Thousand benchmark</div></article>
          <article className="metric-card"><div className="metric-label">EMBEDDING SIZE <span className="metric-icon violet">⌗</span></div><div className="metric-value">{ref?.embedding_dimension ?? 512}<small> dim</small></div><div className="metric-foot">L2-normalized feature vector</div></article>
          <article className="metric-card"><div className="metric-label">MODEL STATUS <span className={`status-pill ${dashboard?.model_available ? "ready" : "pending"}`}>{dashboard?.model_available ? "READY" : "NOT TRAINED"}</span></div><div className={`model-state ${dashboard?.model_available ? "ready-text" : "pending-text"}`}>{dashboard?.model_available ? "Online" : "Setup required"}</div><div className="metric-foot">{dashboard?.model_available ? "Inference endpoint enabled" : "Add a trained checkpoint to enable capture"}</div></article>
        </div>

        <div className="main-grid" id="analytics">
          <article className="panel chart-panel"><div className="panel-head"><div><div className="eyebrow">STATISTICAL OPERATING POINT</div><h2>Verification error curve</h2><p>{calibration?.data_source === "empirical" ? `Measured on the ${calibration.split ?? "saved"} evaluation split` : "Estimate from aggregate reference score statistics"}</p></div><div className="optimizer-controls"><select aria-label="Threshold optimization objective" value={targetFar} onChange={e => setTargetFar(e.target.value)}><option value="eer">Balance FAR and FRR</option><option value="0.01">FAR ≤ 1%</option><option value="0.001">FAR ≤ 0.1%</option><option value="0.0001">FAR ≤ 0.01%</option></select><button className="primary-button" onClick={() => void optimize()} disabled={busy || apiState !== "online"}>{busy ? <><span className="spinner"/> Calibrating</> : "◎  Optimize threshold"}</button></div></div>
            <div className="chart-wrap">{calibration ? <ReactECharts option={chartOption} notMerge style={{ height: 285, width: "100%" }}/> : <div className="chart-empty">Waiting for score summaries…</div>}</div>
            <div className="chart-foot"><span><i className="legend-dot red"/>False accept rate</span><span><i className="legend-dot blue"/>False reject rate</span><span className="chart-note">Aggregate-statistics estimate</span></div>
          </article>
          <article className="panel calibration-panel"><div className="eyebrow">SUGGESTED OPERATING POINT</div><h2>Threshold calibration</h2><p className="muted">{calibration?.objective === "minimize_frr_at_target_far" ? `Minimizes false rejects while satisfying the ${((calibration.target_far ?? 0) * 100).toPrecision(3)}% FAR target.` : "Selects the point where FAR and FRR are closest."}</p><div className="threshold">{calibration ? calibration.threshold.toFixed(3) : "—"}</div><div className="threshold-caption">COSINE SIMILARITY THRESHOLD</div><div className="rate-list"><div><span>{calibration?.data_source === "empirical" ? "Empirical FAR" : "Estimated FAR"}</span><b>{calibration ? `${(calibration.estimated_far * 100).toPrecision(3)}%` : "—"}</b></div><div><span>{calibration?.data_source === "empirical" ? "Empirical FRR" : "Estimated FRR"}</span><b>{calibration ? `${(calibration.estimated_frr * 100).toPrecision(3)}%` : "—"}</b></div><div><span>{calibration?.data_source === "empirical" ? "Empirical EER" : "Estimated EER"}</span><b>{calibration ? `${(calibration.estimated_eer * 100).toPrecision(3)}%` : "—"}</b></div></div><div className="assumption">ⓘ {calibration?.data_source === "empirical" ? `Computed from ${calibration.split ?? "evaluation"} score pairs. These metrics are split-specific.` : <>Gaussian estimate from reference aggregates. Run <code>evaluate.py</code> with the dataset to use empirical scores.</>}</div></article>
        </div>

        <div className="lower-grid" id="gallery">
          <article className="panel gallery-panel"><div className="panel-head compact"><div><div className="eyebrow">IDENTITY NETWORK</div><h2>Recent registrations</h2></div><span className="count-tag">{identities.length} shown</span></div><div className="table-wrap"><table><thead><tr><th>IDENTITY</th><th>EYE</th><th>SAMPLES</th><th>REGISTERED</th></tr></thead><tbody>{identities.length ? identities.slice(0, 7).map(row => <tr key={row.user_id}><td><span className="identity-icon">{row.user_id.slice(0, 1).toUpperCase()}</span>{row.user_id}</td><td><span className="eye-tag">{row.eye || "—"}</span></td><td>{row.num_samples}</td><td>{new Date(row.created_at).toLocaleDateString()}</td></tr>) : <tr><td colSpan={4} className="empty-cell">No identities registered yet.</td></tr>}</tbody></table></div></article>
          <article className="panel register-panel"><div className="eyebrow">ENROLLMENT</div><h2>Register an identity</h2><p className="muted">Images are processed in memory. Only the embedding is stored.</p><form onSubmit={register}><label>Identity ID<input value={userId} onChange={e => setUserId(e.target.value)} placeholder="e.g. participant-042"/></label><div className="form-row"><label>Eye<select value={eye} onChange={e => setEye(e.target.value)}><option value="L">Left eye</option><option value="R">Right eye</option></select></label><label className="file-label">Iris image<input type="file" accept="image/*" onChange={e => setImage(e.target.files?.[0] ?? null)}/></label></div><button className="primary-button full" disabled={busy || !dashboard?.model_available}>{busy ? "Working…" : "＋  Enroll identity"}</button>{!dashboard?.model_available && <div className="small-help">Training a model checkpoint is required before enrollment.</div>}</form></article>
        </div>
        <article className="panel recognition-panel"><div><div className="eyebrow">LIVE INFERENCE</div><h2>Identify or verify an iris</h2><p className="muted">Images are processed in memory. The source image is not saved.</p></div><form onSubmit={recognize} className="recognition-form"><label>Mode<select value={recognitionMode} onChange={e => setRecognitionMode(e.target.value)}><option value="identify">Identify · search gallery</option><option value="verify">Verify · 1:1 claim</option></select></label>{recognitionMode === "verify" && <label>Claimed identity<select value={claimedId} onChange={e => setClaimedId(e.target.value)}><option value="">Choose identity</option>{identities.map(identity => <option key={identity.user_id} value={identity.user_id}>{identity.user_id}</option>)}</select></label>}<label className="recognition-file">Iris image<input type="file" accept="image/*" onChange={e => setRecognitionImage(e.target.files?.[0] ?? null)}/></label><button className="primary-button" disabled={busy || !dashboard?.model_available}>{busy ? "Processing…" : "◉  Run recognition"}</button></form>{recognitionResult && <div className={`recognition-result ${recognitionStatus}`}>{recognitionResult}</div>}</article>
        <footer><span>IRISSECURE <b>2.0</b> · BIOMETRIC R&amp;D PLATFORM</span><span>FastAPI · Next.js · SQLite · ECharts</span></footer>
      </div>
    </section>
  </main>;
}
