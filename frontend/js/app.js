/**
 * DiabRetino AI — Frontend Application Controller
 * Handles image upload, API communication, Grad-CAM blending,
 * interactive sample gallery, and ophthalmic report generation.
 */

// Application State
const state = {
    apiBaseUrl: localStorage.getItem('diabretino_api_url') || 'http://localhost:8000',
    currentFile: null,
    currentSampleFilename: null,
    currentResult: null,
    activeVisTab: 'overlay',
    currentTheme: localStorage.getItem('diabretino_theme') || 'dark',
    samples: []
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupDropZone();
    setupClipboardPaste();
    checkApiHealth();
    loadSampleGallery();
});

/* ==========================================================================
   Theme Management
   ========================================================================== */
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.currentTheme);
}

function toggleTheme() {
    state.currentTheme = state.currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.currentTheme);
    localStorage.setItem('diabretino_theme', state.currentTheme);
}

/* ==========================================================================
   API Communication & Health
   ========================================================================== */
async function checkApiHealth() {
    const badge = document.getElementById('apiStatusBadge');
    const text = document.getElementById('apiStatusText');
    const dot = badge ? badge.querySelector('.status-dot') : null;

    try {
        const response = await fetch(`${state.apiBaseUrl}/api/health`, { method: 'GET' });
        if (response.ok) {
            const data = await response.json();
            if (text) text.textContent = `Online (${data.device.toUpperCase()})`;
            if (dot) {
                dot.className = 'status-dot pulse-green';
            }
            return true;
        } else {
            throw new Error('API returned non-200');
        }
    } catch (err) {
        if (text) text.textContent = 'Backend Offline';
        if (dot) {
            dot.className = 'status-dot pulse-red';
        }
        return false;
    }
}

/* ==========================================================================
   1-Click Sample Gallery
   ========================================================================== */
async function loadSampleGallery() {
    const container = document.getElementById('samplesContainer');
    if (!container) return;

    try {
        const res = await fetch(`${state.apiBaseUrl}/api/samples`);
        if (res.ok) {
            const data = await res.json();
            state.samples = data.samples;
            renderSamples(state.samples);
            return;
        }
    } catch (e) {
        console.warn('Could not load samples from API, falling back to local paths:', e);
    }

    // Local fallback sample list
    const fallbackSamples = [
        { filename: 'stage_0_no_dr.jpg', stage_id: 0, label: 'Stage 0 - Normal Retina', color: '#10b981' },
        { filename: 'stage_1_mild_dr.jpg', stage_id: 1, label: 'Stage 1 - Mild NPDR', color: '#eab308' },
        { filename: 'stage_2_moderate_dr.jpg', stage_id: 2, label: 'Stage 2 - Moderate NPDR', color: '#f97316' },
        { filename: 'stage_3_severe_dr.jpg', stage_id: 3, label: 'Stage 3 - Severe NPDR', color: '#ef4444' },
        { filename: 'stage_4_proliferative_dr.jpg', stage_id: 4, label: 'Stage 4 - Proliferative DR', color: '#a855f7' }
    ];
    renderFallbackSamples(fallbackSamples);
}

function renderSamples(samples) {
    const container = document.getElementById('samplesContainer');
    container.innerHTML = '';

    samples.forEach(sample => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.style.setProperty('--card-accent', sample.metadata.badge_color);
        card.onclick = () => selectSample(sample);

        card.innerHTML = `
            <img class="sample-thumb" src="${sample.thumbnail_base64}" alt="${sample.label}">
            <div class="sample-stage-name">${sample.metadata.stage}</div>
            <div class="sample-stage-tier">${sample.metadata.name.replace('Diabetic Retinopathy', 'DR')}</div>
        `;
        container.appendChild(card);
    });
}

function renderFallbackSamples(samples) {
    const container = document.getElementById('samplesContainer');
    container.innerHTML = '';

    samples.forEach(sample => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.style.setProperty('--card-accent', sample.color);
        card.onclick = () => selectFallbackSample(sample);

        card.innerHTML = `
            <img class="sample-thumb" src="samples/${sample.filename}" alt="${sample.label}">
            <div class="sample-stage-name">Stage ${sample.stage_id}</div>
            <div class="sample-stage-tier">${sample.label.split(' - ')[1]}</div>
        `;
        container.appendChild(card);
    });
}

function selectSample(sample) {
    state.currentFile = null;
    state.currentSampleFilename = sample.filename;

    // Show preview in drop zone
    const previewWrapper = document.getElementById('previewWrapper');
    const previewImg = document.getElementById('previewImg');
    const dropPrompt = document.getElementById('dropPrompt');

    previewImg.src = sample.thumbnail_base64;
    previewWrapper.style.display = 'flex';
    dropPrompt.style.display = 'none';

    showToast(`Loaded ${sample.metadata.stage}: ${sample.metadata.name}`);
    runAnalysis();
}

function selectFallbackSample(sample) {
    state.currentFile = null;
    state.currentSampleFilename = sample.filename;

    const previewWrapper = document.getElementById('previewWrapper');
    const previewImg = document.getElementById('previewImg');
    const dropPrompt = document.getElementById('dropPrompt');

    previewImg.src = `samples/${sample.filename}`;
    previewWrapper.style.display = 'flex';
    dropPrompt.style.display = 'none';

    showToast(`Loaded Stage ${sample.stage_id} sample`);
    runAnalysis();
}

/* ==========================================================================
   Drag & Drop and Image Upload
   ========================================================================== */
function setupDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-active');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });
}

function setupClipboardPaste() {
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                const blob = item.getAsFile();
                handleSelectedFile(blob);
                showToast('Pasted retinal image from clipboard');
                break;
            }
        }
    });
}

function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload a valid image file (JPEG or PNG)', 'error');
        return;
    }

    state.currentFile = file;
    state.currentSampleFilename = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        const previewWrapper = document.getElementById('previewWrapper');
        const previewImg = document.getElementById('previewImg');
        const dropPrompt = document.getElementById('dropPrompt');

        previewImg.src = e.target.result;
        previewWrapper.style.display = 'flex';
        dropPrompt.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearSelectedImage(event) {
    if (event) event.stopPropagation();
    state.currentFile = null;
    state.currentSampleFilename = null;

    document.getElementById('fileInput').value = '';
    document.getElementById('previewWrapper').style.display = 'none';
    document.getElementById('dropPrompt').style.display = 'block';
    document.getElementById('previewImg').src = '';
}

/* ==========================================================================
   Diagnostic Inference Execution
   ========================================================================== */
async function runAnalysis() {
    if (!state.currentFile && !state.currentSampleFilename) {
        showToast('Please upload a retinal image or pick a sample first.', 'error');
        return;
    }

    const btn = document.getElementById('analyzeBtn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner');

    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';

    const modelName = document.getElementById('modelSelect').value;
    const applyEnhancement = document.getElementById('preprocessToggle').checked;

    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('apply_enhancement', applyEnhancement);

    if (state.currentFile) {
        formData.append('file', state.currentFile);
    } else if (state.currentSampleFilename) {
        formData.append('sample_filename', state.currentSampleFilename);
    }

    try {
        const response = await fetch(`${state.apiBaseUrl}/api/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Inference server error');
        }

        const data = await response.json();
        state.currentResult = data;
        displayResults(data);
        showToast('Analysis completed successfully');
    } catch (err) {
        console.error(err);
        showToast(`Analysis failed: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'flex';
        spinner.style.display = 'none';
    }
}

/* ==========================================================================
   Results Rendering & Visualizer Display
   ========================================================================== */
function displayResults(data) {
    const idleState = document.getElementById('idleState');
    const resultsContainer = document.getElementById('resultsContainer');

    idleState.style.display = 'none';
    resultsContainer.style.display = 'block';

    const pred = data.prediction;
    const guidelines = data.clinical_guidelines;
    const vis = data.visualizations;

    // Header & Badges
    const stagePill = document.getElementById('stagePill');
    stagePill.textContent = pred.stage;
    stagePill.style.backgroundColor = pred.badge_bg;
    stagePill.style.color = pred.badge_color;
    stagePill.style.border = `1px solid ${pred.badge_color}`;

    const urgencyPill = document.getElementById('urgencyPill');
    urgencyPill.textContent = pred.urgency;

    document.getElementById('diagnosisTitle').textContent = pred.class_name;
    document.getElementById('confidenceValue').textContent = `${pred.confidence_percentage}%`;
    document.getElementById('latencyTag').textContent = `${data.model_info.latency_ms} ms • ${data.model_info.architecture_display}`;

    // Visualizers Setup
    document.getElementById('visBaseImg').src = vis.original_image;
    document.getElementById('visHeatmapImg').src = vis.heatmap;
    document.getElementById('compOriginalImg').src = vis.original_image;
    document.getElementById('compPreprocessedImg').src = vis.preprocessed_image;
    document.getElementById('compHeatmapImg').src = vis.overlay || vis.heatmap;
    document.getElementById('visStandaloneHeatmap').src = vis.heatmap;

    // Probability Bars
    renderProbabilityBars(data.probabilities_breakdown);

    // Clinical Guidelines
    document.getElementById('icdrCriteriaText').textContent = guidelines.icdr_criteria;
    const findingsList = document.getElementById('findingsList');
    findingsList.innerHTML = '';
    guidelines.clinical_findings.forEach(f => {
        const li = document.createElement('li');
        li.textContent = f;
        findingsList.appendChild(li);
    });

    document.getElementById('actionText').textContent = guidelines.recommended_action;

    const laserWarning = document.getElementById('laserWarning');
    if (pred.laser_required) {
        laserWarning.style.display = 'block';
    } else {
        laserWarning.style.display = 'none';
    }
}

function renderProbabilityBars(breakdown) {
    const list = document.getElementById('probBarsList');
    list.innerHTML = '';

    breakdown.forEach(item => {
        const barItem = document.createElement('div');
        barItem.className = 'prob-bar-item';
        barItem.innerHTML = `
            <div class="prob-bar-meta">
                <span>${item.name} (${item.key})</span>
                <span style="color: ${item.color}">${item.percentage}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: 0%; background-color: ${item.color}"></div>
            </div>
        `;
        list.appendChild(barItem);

        // Animate bar fill
        setTimeout(() => {
            const fill = barItem.querySelector('.prob-bar-fill');
            if (fill) fill.style.width = `${item.percentage}%`;
        }, 50);
    });
}

/* ==========================================================================
   Visualizer Controls (Opacity & Tabs)
   ========================================================================== */
function adjustHeatmapOpacity(val) {
    const heatLayer = document.getElementById('visHeatmapImg');
    const opacityText = document.getElementById('opacityValueText');
    if (heatLayer) heatLayer.style.opacity = val / 100;
    if (opacityText) opacityText.textContent = `${val}%`;
}

function switchVisTab(mode) {
    state.activeVisTab = mode;
    const tabs = document.querySelectorAll('.vis-tab');
    tabs.forEach(t => t.classList.remove('active'));

    const overlayView = document.getElementById('visOverlayView');
    const splitView = document.getElementById('visSplitView');
    const heatmapView = document.getElementById('visHeatmapView');

    overlayView.style.display = 'none';
    splitView.style.display = 'none';
    heatmapView.style.display = 'none';

    if (mode === 'overlay') {
        overlayView.style.display = 'block';
        tabs[0].classList.add('active');
    } else if (mode === 'split') {
        splitView.style.display = 'block';
        tabs[1].classList.add('active');
    } else if (mode === 'heatmap') {
        heatmapView.style.display = 'block';
        tabs[2].classList.add('active');
    }
}

function downloadVisualization() {
    if (!state.currentResult) return;
    const link = document.createElement('a');
    link.download = `diabretino_overlay_${Date.now()}.jpg`;
    link.href = state.currentResult.visualizations.overlay || state.currentResult.visualizations.heatmap;
    link.click();
}

/* ==========================================================================
   Main View Navigation (Single vs Batch)
   ========================================================================== */
function switchMainTab(tab) {
    const tabSingle = document.getElementById('tabSingleView');
    const tabBatch = document.getElementById('tabBatchView');
    const singleSection = document.getElementById('singleScanSection');
    const batchSection = document.getElementById('batchScreeningSection');

    if (tab === 'single') {
        tabSingle.classList.add('active');
        tabBatch.classList.remove('active');
        singleSection.style.display = 'block';
        batchSection.style.display = 'none';
    } else {
        tabSingle.classList.remove('active');
        tabBatch.classList.add('active');
        singleSection.style.display = 'none';
        batchSection.style.display = 'block';
        initBatchFileInput();
    }
}

/* ==========================================================================
   Batch Screening Mode
   ========================================================================== */
function initBatchFileInput() {
    const input = document.getElementById('batchFileInput');
    if (!input || input.dataset.initialized) return;

    input.dataset.initialized = 'true';
    input.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        if (!files || files.length === 0) return;

        showToast(`Processing batch of ${files.length} retinal scans...`);
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));
        formData.append('model_name', document.getElementById('modelSelect').value);

        try {
            const res = await fetch(`${state.apiBaseUrl}/api/batch-predict`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error('Batch processing failed');
            const data = await res.json();
            renderBatchResults(data);
            showToast('Batch screening completed');
        } catch (err) {
            showToast(`Batch error: ${err.message}`, 'error');
        }
    });
}

function renderBatchResults(data) {
    const summaryGrid = document.getElementById('batchSummaryGrid');
    summaryGrid.style.display = 'grid';

    document.getElementById('countStage0').textContent = data.stage_distribution['No_DR'] || 0;
    document.getElementById('countStage1').textContent = data.stage_distribution['Mild'] || 0;
    document.getElementById('countStage2').textContent = data.stage_distribution['Moderate'] || 0;
    document.getElementById('countStage3').textContent = data.stage_distribution['Severe'] || 0;
    document.getElementById('countStage4').textContent = data.stage_distribution['Proliferate_DR'] || 0;

    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    data.results.forEach(item => {
        const tr = document.createElement('tr');
        if (item.error) {
            tr.innerHTML = `
                <td>-</td>
                <td><strong>${item.filename}</strong></td>
                <td colspan="4" class="text-danger">${item.error}</td>
            `;
        } else {
            const p = item.prediction;
            tr.innerHTML = `
                <td><img class="batch-thumb" src="${item.thumbnail}" alt="Thumb"></td>
                <td><strong>${item.filename}</strong></td>
                <td><span style="color: ${p.badge_color}; font-weight:700;">${p.stage}</span>: ${p.class_name}</td>
                <td><span class="badge" style="background:${p.badge_bg}; color:${p.badge_color}">${p.risk_level}</span></td>
                <td><strong>${p.confidence_percentage}%</strong></td>
                <td>${p.urgency}</td>
            `;
        }
        tbody.appendChild(tr);
    });
}

/* ==========================================================================
   Medical Report Generator Modal
   ========================================================================== */
function openReportModal() {
    if (!state.currentResult) return;
    const modal = document.getElementById('reportModal');
    const body = document.getElementById('reportModalBody');
    const pred = state.currentResult.prediction;
    const guidelines = state.currentResult.clinical_guidelines;
    const vis = state.currentResult.visualizations;

    const reportHtml = `
        <div style="font-family: inherit; color: var(--text-primary); line-height: 1.6;">
            <div style="display:flex; justify-content:space-between; border-bottom: 2px solid var(--border-color); padding-bottom: 1rem; margin-bottom: 1.2rem;">
                <div>
                    <h2 style="font-size: 1.3rem; font-weight: 800;">OPHTHALMIC EXAMINATION REPORT</h2>
                    <p style="font-size: 0.8rem; color: var(--text-secondary);">Diabetic Retinopathy Automated Screening & Severity Grading</p>
                </div>
                <div style="text-align: right; font-size: 0.8rem; color: var(--text-muted);">
                    <div><strong>Exam ID:</strong> DR-${Math.floor(100000 + Math.random() * 900000)}</div>
                    <div><strong>Date:</strong> ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 140px 1fr; gap: 1.2rem; align-items: center; margin-bottom: 1.2rem; background: rgba(0,0,0,0.15); padding: 0.8rem; border-radius: 8px;">
                <img src="${vis.overlay || vis.original_image}" style="width: 130px; height: 130px; border-radius: 8px; object-fit: cover;" alt="Retinal Scan">
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Primary AI Finding</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: ${pred.badge_color};">${pred.stage}: ${pred.class_name}</div>
                    <div style="font-size: 0.85rem; margin-top: 0.3rem;">
                        <strong>Confidence:</strong> ${pred.confidence_percentage}% &bull; 
                        <strong>Risk Tier:</strong> ${pred.risk_level} &bull; 
                        <strong>Follow-up:</strong> ${pred.urgency}
                    </div>
                </div>
            </div>

            <div style="margin-bottom: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 700; color: var(--accent-cyan); margin-bottom: 0.3rem;">ICDR Diagnostic Criteria</h4>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">${guidelines.icdr_criteria}</p>
            </div>

            <div style="margin-bottom: 1rem;">
                <h4 style="font-size: 0.85rem; font-weight: 700; color: var(--accent-cyan); margin-bottom: 0.3rem;">Identified Retinal Biomarkers</h4>
                <ul style="padding-left: 1.2rem; font-size: 0.82rem; color: var(--text-secondary);">
                    ${guidelines.clinical_findings.map(f => `<li>${f}</li>`).join('')}
                </ul>
            </div>

            <div style="background: rgba(255,255,255,0.03); border-left: 3px solid ${pred.badge_color}; padding: 0.8rem; border-radius: 6px; margin-bottom: 1.5rem;">
                <h4 style="font-size: 0.85rem; font-weight: 700; margin-bottom: 0.2rem;">Doctor Action Directive</h4>
                <p style="font-size: 0.85rem;">${guidelines.recommended_action}</p>
            </div>

            <div style="border-top: 1px dashed var(--border-color); padding-top: 1rem; display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                <div>Model: ${state.currentResult.model_info.architecture_display}</div>
                <div>Reviewing Ophthalmologist Signature: _________________________</div>
            </div>
        </div>
    `;

    body.innerHTML = reportHtml;
    modal.style.display = 'flex';
}

function closeReportModal() {
    document.getElementById('reportModal').style.display = 'none';
}

/* ==========================================================================
   API Settings Modal
   ========================================================================== */
function openSettingsModal() {
    const modal = document.getElementById('settingsModal');
    const input = document.getElementById('apiUrlInput');
    const swaggerLink = document.getElementById('swaggerLink');

    input.value = state.apiBaseUrl;
    swaggerLink.href = `${state.apiBaseUrl}/docs`;
    swaggerLink.textContent = `${state.apiBaseUrl}/docs`;

    modal.style.display = 'flex';
}

function closeSettingsModal() {
    document.getElementById('settingsModal').style.display = 'none';
}

async function testCustomApiConnection() {
    const input = document.getElementById('apiUrlInput');
    const statusText = document.getElementById('settingsTestStatus');
    const testUrl = input.value.trim().replace(/\/$/, '');

    statusText.textContent = 'Testing...';
    statusText.style.color = '#94a3b8';

    try {
        const res = await fetch(`${testUrl}/api/health`);
        if (res.ok) {
            statusText.textContent = '✓ Connected successfully!';
            statusText.style.color = '#10b981';
        } else {
            statusText.textContent = '✕ Server returned error';
            statusText.style.color = '#ef4444';
        }
    } catch (e) {
        statusText.textContent = '✕ Cannot reach server';
        statusText.style.color = '#ef4444';
    }
}

function saveSettings() {
    const input = document.getElementById('apiUrlInput');
    const newUrl = input.value.trim().replace(/\/$/, '');
    state.apiBaseUrl = newUrl;
    localStorage.setItem('diabretino_api_url', newUrl);
    closeSettingsModal();
    checkApiHealth();
    loadSampleGallery();
    showToast('Saved API configuration');
}

/* ==========================================================================
   Toast Notification Helper
   ========================================================================== */
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast';
    if (type === 'error') {
        toast.style.borderLeftColor = '#ef4444';
    }

    toast.innerHTML = `
        <span>${message}</span>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
