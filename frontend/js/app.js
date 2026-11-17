/* DiabRetino AI — frontend controller */

// Live production backend (Render). The site pre-wires this on the deployed
// frontend host so it connects automatically; a user-saved URL still wins.
const PROD_BACKEND = 'https://diabretino-ai-1.onrender.com';
function isProdHost(host) {
    // Pre-wire on static-host deployments (Vercel / Netlify / GH Pages),
    // and never on localhost or on the Render host itself.
    if (host === 'localhost' || host === '127.0.0.1') return false;
    if (host.endsWith('.onrender.com')) return false;
    return host.endsWith('.vercel.app') || host.endsWith('.netlify.app');
}

const state = {
    api: localStorage.getItem('diabretino_api') || '',
    file: null,
    sample: null,
    result: null,
    theme: localStorage.getItem('diabretino_theme') || 'dark',
};

document.addEventListener('DOMContentLoaded', () => {
    applyTheme();
    // Backend URL precedence:
    //   1. ?api= query param (lets a deployment pre-wire the backend)
    //   2. saved value in localStorage
    //   3. localhost default only when actually on localhost (local dev / docker)
    const qApi = new URLSearchParams(location.search).get('api');
    if (qApi) {
        state.api = qApi.replace(/\/$/, '');
        localStorage.setItem('diabretino_api', state.api);
    } else if (!state.api && location.hostname === 'localhost') {
        state.api = 'http://localhost:8000';
    } else if (!state.api && isProdHost(location.hostname)) {
        // Production deployment: pre-wire the live Render backend so the site
        // connects automatically. Users can override it in API settings.
        state.api = PROD_BACKEND;
        localStorage.setItem('diabretino_api', state.api);
    }
    setupDropzone();
    setupClipboard();
    setupTabs();
    setupBatch();
    closeModalsOnBackdrop();
    checkHealth();
    loadSamples();
    loadMetrics();
});

function closeModalsOnBackdrop() {
    ['settingsModal', 'aboutModal'].forEach(id => {
        document.getElementById(id).addEventListener('click', e => {
            if (e.target.id === id) document.getElementById(id).hidden = true;
        });
    });
}

/* ---------- theme ---------- */
function applyTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
}
function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('diabretino_theme', state.theme);
    applyTheme();
}

/* ---------- settings ---------- */
function openSettings() {
    const m = document.getElementById('settingsModal');
    document.getElementById('apiUrl').value = state.api;
    m.hidden = false;
}
function closeSettings() { document.getElementById('settingsModal').hidden = true; }
function saveApi() {
    state.api = document.getElementById('apiUrl').value.trim().replace(/\/$/, '');
    localStorage.setItem('diabretino_api', state.api);
    closeSettings();
    checkHealth();
    loadSamples();
    toast('API URL saved');
}
async function testApi() {
    const url = (document.getElementById('apiUrl').value.trim().replace(/\/$/, ''));
    const msg = document.getElementById('apiTestMsg');
    msg.textContent = 'Testing…';
    try {
        const r = await fetch(url + '/api/health');
        msg.textContent = r.ok ? '✓ Connected' : '✕ ' + r.status;
        msg.style.color = r.ok ? '#10b981' : '#ef4444';
    } catch { msg.textContent = '✕ Unreachable'; msg.style.color = '#ef4444'; }
}

/* ---------- onboarding banner ---------- */
function hideBanner() {
    document.getElementById('onboardingBanner').hidden = true;
}
function showBanner() {
    document.getElementById('onboardingBanner').hidden = false;
}

/* ---------- health ---------- */
async function checkHealth() {
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');

    if (!state.api) {
        dot.className = 'status-dot err';
        txt.textContent = 'No API set';
        showBanner();
        return;
    }

    try {
        const r = await fetch(state.api + '/api/health');
        if (!r.ok) throw 0;
        const d = await r.json();
        dot.className = 'status-dot ok';
        txt.textContent = `Online · ${String(d.device).toUpperCase()}`;
        hideBanner();
    } catch {
        dot.className = 'status-dot err';
        txt.textContent = 'Connecting…';
        // Render services spin down; a 503 is a wake-up, not a failure.
        showBanner();
        setTimeout(checkHealth, 5000);
    }
}

/* ---------- samples ---------- */
async function loadSamples() {
    const grid = document.getElementById('sampleGrid');
    let samples = [];
    if (state.api) {
        try {
            const r = await fetch(state.api + '/api/samples');
            samples = (await r.json()).samples || [];
        } catch (e) { console.warn('samples fetch failed', e); }
    }
    if (!samples.length) {
        // local fallback (samples shipped in frontend/samples)
        samples = [
            { filename: 'stage_0_no_dr.png', stage_id: 0, label: 'Normal', color: 'var(--s0)', metadata: { name: 'No Diabetic Retinopathy', stage: 'Stage 0', badge_color: '#10b981', badge_bg: 'rgba(16,185,129,.15)' } },
            { filename: 'stage_1_mild.png', stage_id: 1, label: 'Mild', color: 'var(--s1)', metadata: { name: 'Mild NPDR', stage: 'Stage 1', badge_color: '#eab308', badge_bg: 'rgba(234,179,8,.15)' } },
            { filename: 'stage_2_moderate.png', stage_id: 2, label: 'Moderate', color: 'var(--s2)', metadata: { name: 'Moderate NPDR', stage: 'Stage 2', badge_color: '#f97316', badge_bg: 'rgba(249,115,22,.15)' } },
            { filename: 'stage_3_severe.png', stage_id: 3, label: 'Severe', color: 'var(--s3)', metadata: { name: 'Severe NPDR', stage: 'Stage 3', badge_color: '#ef4444', badge_bg: 'rgba(239,68,68,.15)' } },
            { filename: 'stage_4_proliferative.png', stage_id: 4, label: 'Proliferative', color: 'var(--s4)', metadata: { name: 'Proliferative DR', stage: 'Stage 4', badge_color: '#a855f7', badge_bg: 'rgba(168,85,247,.15)' } },
        ];
    }
    grid.innerHTML = '';
    samples.forEach(s => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.style.setProperty('--card-accent', s.metadata.badge_color);
        const imgSrc = s.thumbnail_base64
            ? s.thumbnail_base64
            : `samples/${s.filename}`;
        card.innerHTML = `
            <img src="${imgSrc}" alt="${s.label}">
            <div class="s-name">${s.metadata.stage}</div>
            <div class="s-stage">${s.label}</div>`;
        card.onclick = () => pickSample(s);
        grid.appendChild(card);
    });
}

function pickSample(s) {
    state.file = null;
    state.sample = s;
    showPreview(s.thumbnail_base64 || `samples/${s.filename}`);
    toast(`Loaded ${s.metadata.stage}: ${s.metadata.name}`);
    runAnalysis();
}

/* ---------- dropzone ---------- */
function setupDropzone() {
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('fileInput');
    document.getElementById('browseLink').onclick = (e) => { e.preventDefault(); input.click(); };
    dz.onclick = () => input.click();
    input.onchange = (e) => e.target.files[0] && handleFile(e.target.files[0]);
    ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
    dz.addEventListener('drop', e => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
    document.getElementById('clearImg').onclick = clearImage;
}
function setupClipboard() {
    window.addEventListener('paste', e => {
        for (const it of (e.clipboardData.items || [])) {
            if (it.type.startsWith('image')) {
                handleFile(it.getAsFile());
                toast('Pasted image from clipboard');
                break;
            }
        }
    });
}
function handleFile(file) {
    if (!file.type.startsWith('image/')) { toast('Please choose an image file', true); return; }
    state.file = file;
    state.sample = null;
    const reader = new FileReader();
    reader.onload = e => showPreview(e.target.result);
    reader.readAsDataURL(file);
}
function showPreview(src) {
    document.getElementById('previewImg').src = src;
    document.getElementById('dzPrompt').hidden = true;
    const p = document.getElementById('dzPreview'); p.hidden = false; p.style.display = 'flex';
}
function clearImage() {
    state.file = null; state.sample = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dzPreview').hidden = true;
    document.getElementById('dzPrompt').hidden = false;
}

/* ---------- tabs ---------- */
function setupTabs() {
    document.querySelectorAll('.vtab').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.vtab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const view = btn.dataset.view;
            ['Overlay', 'Compare', 'Heatmap'].forEach(v =>
                document.getElementById('view' + v).hidden = (v.toLowerCase() !== view));
        };
    });
    const op = document.getElementById('opacity');
    op.oninput = () => {
        document.getElementById('ovHeat').style.opacity = op.value / 100;
        document.getElementById('opacityVal').textContent = op.value + '%';
    };
}

/* ---------- run analysis ---------- */
document.getElementById('runBtn').onclick = runAnalysis;
async function runAnalysis() {
    if (!state.file && !state.sample) { toast('Upload an image or pick a sample', true); return; }
    if (!state.api) { openSettings(); toast('Set your backend API URL first', true); return; }

    const btn = document.getElementById('runBtn');
    btn.disabled = true;
    btn.querySelector('.btn-label').style.display = 'none';
    document.getElementById('runSpinner').hidden = false;

    const fd = new FormData();
    fd.append('model_name', document.getElementById('modelSelect').value);
    fd.append('apply_enhancement', document.getElementById('enhanceToggle').checked);
    if (state.file) fd.append('file', state.file);
    else fd.append('sample_filename', state.sample.filename);

    try {
        const r = await fetch(state.api + '/api/predict', { method: 'POST', body: fd });
        if (!r.ok) throw new Error((await r.json()).detail || 'Error ' + r.status);
        state.result = await r.json();
        render(state.result);
        toast('Analysis complete');
    } catch (err) {
        toast('Failed: ' + err.message, true);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-label').style.display = '';
        document.getElementById('runSpinner').hidden = true;
    }
}

/* ---------- render results ---------- */
function render(d) {
    document.getElementById('idleState').hidden = true;
    const res = document.getElementById('results'); res.hidden = false;

    const p = d.prediction, g = d.clinical_guidelines, v = d.visualizations;
    const stage = document.getElementById('stagePill');
    stage.textContent = p.stage;
    stage.style.background = p.badge_bg; stage.style.color = p.badge_color; stage.style.border = `1px solid ${p.badge_color}`;
    document.getElementById('urgencyPill').textContent = p.urgency;
    document.getElementById('diagTitle').textContent = p.class_name;
    document.getElementById('confValue').textContent = p.confidence_percentage + '%';
    document.getElementById('latencyTag').textContent = `${d.model_info.latency_ms}ms · ${d.model_info.architecture_display}`;

    // visualizers
    document.getElementById('ovBase').src = v.original_image;
    document.getElementById('ovHeat').src = v.heatmap;
    document.getElementById('cmpOrig').src = v.original_image;
    document.getElementById('cmpEnh').src = v.preprocessed_image;
    document.getElementById('cmpHeat').src = v.overlay || v.heatmap;
    document.getElementById('heatStandalone').src = v.heatmap;

    // probabilities
    const bars = document.getElementById('probBars');
    bars.innerHTML = '';
    d.probabilities_breakdown.forEach(item => {
        const el = document.createElement('div');
        el.className = 'prob-item';
        el.innerHTML = `
            <div class="prob-meta"><span>${item.name}</span><span style="color:${item.color}">${item.percentage}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="width:0;background:${item.color}"></div></div>`;
        bars.appendChild(el);
        setTimeout(() => { const f = el.querySelector('.prob-fill'); if (f) f.style.width = item.percentage + '%'; }, 60);
    });

    // report
    document.getElementById('icdrText').textContent = g.icdr_criteria;
    const fl = document.getElementById('findingsList'); fl.innerHTML = '';
    g.clinical_findings.forEach(f => { const li = document.createElement('li'); li.textContent = f; fl.appendChild(li); });
    document.getElementById('actionText').textContent = g.recommended_action;
    document.getElementById('laserWarn').hidden = !p.laser_required;

    // print + save
    document.getElementById('printBtn').onclick = () => window.print();
    document.getElementById('saveHeatBtn').onclick = saveHeat;
}
function saveHeat() {
    if (!state.result) return;
    const a = document.createElement('a');
    a.download = `diabretino_attention_${Date.now()}.png`;
    a.href = state.result.visualizations.heatmap;
    a.click();
}

/* ---------- batch ---------- */
function setupBatch() {
    document.getElementById('batchBrowse').onclick = () => document.getElementById('batchInput').click();
    document.getElementById('batchInput').onchange = async e => {
        const files = [...e.target.files];
        if (!files.length || !state.api) { if (!state.api) openSettings(); return; }
        toast(`Screening ${files.length} images…`);
        const fd = new FormData();
        files.forEach(f => fd.append('files', f));
        fd.append('model_name', document.getElementById('modelSelect').value);
        try {
            const r = await fetch(state.api + '/api/batch-predict', { method: 'POST', body: fd });
            if (!r.ok) throw new Error('Batch failed');
            renderBatch(await r.json());
        } catch (err) { toast('Batch error: ' + err.message, true); }
    };
}
function renderBatch(d) {
    const sum = document.getElementById('batchSummary');
    sum.hidden = false; sum.innerHTML = '';
    const colors = ['var(--s0)', 'var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
    const keys = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'];
    keys.forEach((k, i) => {
        const n = d.stage_distribution[k] || 0;
        sum.innerHTML += `<div class="sum-pill" style="border-left-color:${colors[i]}"><div class="sum-count">${n}</div><div class="sum-label">${k.replace('_DR','')}</div></div>`;
    });

    const wrap = document.getElementById('batchTableWrap'); wrap.hidden = false;
    const body = document.getElementById('batchBody'); body.innerHTML = '';
    d.results.forEach(item => {
        const tr = document.createElement('tr');
        if (item.error) {
            tr.innerHTML = `<td>—</td><td><strong>${item.filename}</strong></td><td colspan="3" style="color:#f87171">${item.error}</td>`;
        } else {
            const p = item.prediction;
            tr.innerHTML = `
                <td><img src="${item.thumbnail}" alt=""></td>
                <td>${item.filename}</td>
                <td><span style="color:${p.badge_color};font-weight:700">${p.stage}</span> ${p.class_name.replace(' Diabetic Retinopathy','').replace(' Non-Proliferative',' NPDR').replace('Proliferative Diabetic','Proliferative')}</td>
                <td><strong>${p.confidence_percentage}%</strong></td>
                <td class="small">${p.urgency}</td>`;
        }
        body.appendChild(tr);
    });
}

/* ---------- model performance ---------- */
async function loadMetrics() {
    const body = document.getElementById('perfBody');
    const loading = document.getElementById('perfLoading');
    if (!state.api) { loading.textContent = 'Set your API URL to load model metrics.'; return; }
    try {
        const r = await fetch(state.api + '/api/metrics');
        if (!r.ok) throw new Error('No metrics yet');
        const data = await r.json();
        renderPerf(data);
        renderAboutMetrics(data);
        if (loading) loading.remove();
    } catch (e) {
        loading.textContent = 'Metrics load once the backend is reachable (Render services wake on first request).';
    }
}

function renderAboutMetrics(data) {
    const host = document.getElementById('aboutMetrics');
    if (!host || !data || !data.models || !data.models.length) return;
    const m = data.models[0];
    host.innerHTML = `
        <div class="about-metric-row"><span class="am-k">Accuracy (held-out)</span><span class="am-v">${(m.accuracy*100).toFixed(1)}%</span></div>
        <div class="about-metric-row"><span class="am-k">Macro F1</span><span class="am-v">${(m.macro_f1*100).toFixed(1)}%</span></div>
        <div class="about-metric-row"><span class="am-k">Weighted F1</span><span class="am-v">${(m.weighted_f1*100).toFixed(1)}%</span></div>
        <p class="about-disc">Evaluated on a held-out 20% validation split of the APTOS 2019 dataset. Full per-class table and confusion matrix in the "Model performance" panel below.</p>`;
}

function renderPerf(data) {
    const body = document.getElementById('perfBody');
    const classes = data.class_names || [];
    const wrapper = document.createElement('div');

    // model cards (with an accuracy gauge on the flagship)
    const cards = document.createElement('div');
    cards.className = 'perf-models';
    (data.models || []).forEach((m, idx) => {
        const flag = m.model === 'hybrid_vit_cnn' ? ' <span class="flag">★ flagship</span>' : '';
        const isFlagship = m.model === 'hybrid_vit_cnn';
        const card = document.createElement('div');
        card.className = 'perf-card' + (isFlagship ? ' flagship' : '');
        card.innerHTML = `
            <div class="perf-name">${m.model}${flag}</div>
            ${isFlagship ? accuracyGauge(m.accuracy, m.macro_f1) : ''}
            <div class="perf-stat"><span class="k">Accuracy</span><span class="v">${(m.accuracy*100).toFixed(1)}%</span></div>
            <div class="perf-stat"><span class="k">Macro F1</span><span class="v">${(m.macro_f1*100).toFixed(1)}%</span></div>
            <div class="perf-stat"><span class="k">Weighted F1</span><span class="v">${(m.weighted_f1*100).toFixed(1)}%</span></div>
            <div class="perf-stat"><span class="k">Samples</span><span class="v">${m.n_samples}</span></div>
        `;
        cards.appendChild(card);
    });
    wrapper.appendChild(cards);

    // confusion matrix for the flagship (color-scaled)
    const flagship = (data.models || []).find(m => m.model === 'hybrid_vit_cnn') || (data.models||[])[0];
    if (flagship && flagship.confusion_matrix) {
        const cmBlock = document.createElement('div');
        cmBlock.className = 'cm-block';
        cmBlock.innerHTML = `<h4>Confusion matrix — ${flagship.model} (rows = actual, cols = predicted)</h4>`;
        const maxCell = Math.max(1, ...flagship.confusion_matrix.flat());
        const grid = document.createElement('div');
        grid.className = 'cm-grid';
        grid.appendChild(cell(''));
        classes.forEach(c => grid.appendChild(cell(c.replace('_',' '), true)));
        flagship.confusion_matrix.forEach((row, i) => {
            grid.appendChild(cell(classes[i].replace('_',' '), true));
            row.forEach((val, j) => {
                const c = cell(val === 0 ? '' : val, false, i === j ? 'diag' : 'off');
                // intensity scaled by value, green on the diagonal / red off it
                const t = val / maxCell;
                if (i === j) c.style.background = `rgba(16,185,129,${0.15 + 0.5*t})`;
                else if (val > 0) c.style.background = `rgba(239,68,68,${0.12 + 0.5*t})`;
                grid.appendChild(c);
            });
        });
        cmBlock.appendChild(grid);
        cmBlock.insertAdjacentHTML('beforeend',
            '<div class="cm-legend"><span class="lg lg-good"></span> correct &nbsp; <span class="lg lg-bad"></span> mislabelled</div>');
        wrapper.appendChild(cmBlock);
    }

    // class distribution
    if (data.class_distribution_pct) {
        const note = document.createElement('div');
        note.className = 'perf-note';
        note.innerHTML = 'Class distribution (train): ' +
            Object.entries(data.class_distribution_pct)
                .map(([k,v]) => `${k} ${v}%`).join(' · ');
        wrapper.appendChild(note);
    }

    body.innerHTML = '';
    body.appendChild(wrapper);
}

function accuracyGauge(acc, f1) {
    const pct = Math.round(acc * 100);
    // SVG donut: 260° sweep, value-colored arc
    const r = 30, c = 2 * Math.PI * r, arc = c * (acc) * 0.75;
    return `
        <div class="gauge" title="Accuracy / Macro F1">
            <svg viewBox="0 0 80 80" width="80" height="80">
                <circle cx="40" cy="40" r="${r}" fill="none" stroke="var(--panel)" stroke-width="8"/>
                <circle cx="40" cy="40" r="${r}" fill="none" stroke="var(--accent)" stroke-width="8"
                    stroke-linecap="round" stroke-dasharray="${arc} ${c}"
                    transform="rotate(135 40 40)"
                    style="transition:stroke-dasharray 1s ease"/>
                <text x="40" y="37" text-anchor="middle" class="gauge-num">${pct}%</text>
                <text x="40" y="52" text-anchor="middle" class="gauge-sub">acc</text>
            </svg>
        </div>`;
}

function cell(text, isHeader, cls) {
    const el = document.createElement('div');
    el.className = 'cm-cell' + (isHeader ? ' cm-hdr' : (cls ? ' ' + cls : ''));
    el.textContent = text;
    return el;
}

/* ---------- toast ---------- */
function toast(msg, isErr) {
    const c = document.getElementById('toasts');
    const t = document.createElement('div');
    t.className = 'toast' + (isErr ? ' error' : '');
    t.textContent = msg;
    c.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 3200);
}

/* wire modal buttons */
document.getElementById('testApiBtn').onclick = testApi;
document.getElementById('saveApiBtn').onclick = saveApi;
document.getElementById('settingsModal').onclick = e => { if (e.target.id === 'settingsModal') closeSettings(); };
