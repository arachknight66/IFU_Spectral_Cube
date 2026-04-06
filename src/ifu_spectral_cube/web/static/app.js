/**
 * JWST MIRI IFU Pipeline Dashboard — Frontend Logic
 * 
 * Handles: file uploads, pipeline execution, data fetching, 
 * Plotly interactive charts, tables, and lightbox gallery.
 */

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════
let spectraData = null;
let peaksData = [];
let matchesData = [];
let fitsData = [];
let summaryData = null;
let currentSpectrumView = 'processed';
let pollingInterval = null;

// Ionization class color map
const ION_COLORS = {
    low:            '#4FC3F7',
    intermediate:   '#81C784',
    high:           '#FFB74D',
    very_high:      '#EF5350',
    molecular:      '#CE93D8',
    neutral:        '#FFEE58',
    recombination:  '#80DEEA',
    absorption:     '#BDBDBD',
};

// Safe number parsing — handles null (from sanitized NaN)
function safeNum(v, fallback = 0) {
    const n = parseFloat(v);
    return isNaN(n) || v === null ? fallback : n;
}
function safeFmt(v, digits = 4, fallback = '—') {
    const n = parseFloat(v);
    return (isNaN(n) || v === null) ? fallback : n.toFixed(digits);
}
function safeExp(v, digits = 3, fallback = '—') {
    const n = parseFloat(v);
    return (isNaN(n) || v === null) ? fallback : n.toExponential(digits);
}

// ═══════════════════════════════════════════════════════════════════
// File upload
// ═══════════════════════════════════════════════════════════════════
const fileInput = document.getElementById('fileInput');
const fileUpload = document.getElementById('fileUpload');
const fileName = document.getElementById('fileName');

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileName.textContent = e.target.files[0].name;
    }
});

// Drag & drop
fileUpload.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUpload.classList.add('dragover');
});
fileUpload.addEventListener('dragleave', () => {
    fileUpload.classList.remove('dragover');
});
fileUpload.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUpload.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileName.textContent = e.dataTransfer.files[0].name;
    }
});

// ═══════════════════════════════════════════════════════════════════
// Pipeline execution
// ═══════════════════════════════════════════════════════════════════
async function runPipeline() {
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a FITS cube file first.');
        return;
    }

    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="spinner"></span> Running...';
    setStatus('running', 'Processing...');

    const formData = new FormData();
    formData.append('cube', file);

    // Collect parameters
    formData.append('extraction_mode', document.getElementById('extractionMode').value);
    formData.append('center_x', document.getElementById('centerX').value);
    formData.append('center_y', document.getElementById('centerY').value);
    formData.append('radius', document.getElementById('radius').value);
    formData.append('continuum_window', document.getElementById('continuumWindow').value);
    formData.append('iterative_continuum', document.getElementById('iterativeContinuum').checked);
    formData.append('defringe', document.getElementById('defringe').checked);
    formData.append('prominence_sigma', document.getElementById('prominenceSigma').value);
    formData.append('min_snr', document.getElementById('minSnr').value);
    formData.append('resolving_power', document.getElementById('resolvingPower').value);
    formData.append('use_cwt', document.getElementById('useCwt').checked);
    formData.append('min_confidence', document.getElementById('minConfidence').value);
    formData.append('try_voigt', document.getElementById('tryVoigt').checked);

    try {
        const resp = await fetch('/api/run', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Pipeline failed');
        }

        // Poll for completion
        startPolling();
    } catch (err) {
        setStatus('error', err.message);
        runBtn.disabled = false;
        runBtn.innerHTML = '🚀 Run Pipeline';
    }
}

function startPolling() {
    pollingInterval = setInterval(async () => {
        try {
            const resp = await fetch('/api/status');
            const status = await resp.json();

            setStatus(status.status, status.progress || '');

            if (status.status === 'complete') {
                clearInterval(pollingInterval);
                document.getElementById('runBtn').disabled = false;
                document.getElementById('runBtn').innerHTML = '🚀 Run Pipeline';
                await loadAllData();
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                document.getElementById('runBtn').disabled = false;
                document.getElementById('runBtn').innerHTML = '🚀 Run Pipeline';
                setStatus('error', status.error || 'Pipeline failed');
            }
        } catch (e) {
            console.error('Polling error:', e);
        }
    }, 1500);
}

// ═══════════════════════════════════════════════════════════════════
// Load existing results
// ═══════════════════════════════════════════════════════════════════
async function loadResults() {
    const path = document.getElementById('resultsPath').value;
    if (!path) return;

    setStatus('running', 'Loading results...');

    try {
        const resp = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error);

        setStatus('complete', 'Results loaded');
        await loadAllData();
    } catch (err) {
        setStatus('error', err.message);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Data loading
// ═══════════════════════════════════════════════════════════════════
async function loadAllData() {
    try {
        const [summaryResp, spectraResp, peaksResp, matchesResp, fitsResp] = await Promise.all([
            fetch('/api/summary'),
            fetch('/api/spectra'),
            fetch('/api/peaks'),
            fetch('/api/matches'),
            fetch('/api/fits'),
        ]);

        summaryData = await summaryResp.json();
        spectraData = await spectraResp.json();
        peaksData = await peaksResp.json();
        matchesData = await matchesResp.json();
        fitsData = await fitsResp.json();

        // Show results UI
        document.getElementById('welcomeScreen').classList.add('hidden');
        document.getElementById('resultsContent').classList.remove('hidden');

        renderStats();
        renderSpectrum();
        renderMatchesTable();
        renderPeaksTable();
        renderPlotGallery();
        renderTimings();
    } catch (err) {
        console.error('Failed to load data:', err);
        setStatus('error', 'Failed to load results');
    }
}

// ═══════════════════════════════════════════════════════════════════
// Status indicator
// ═══════════════════════════════════════════════════════════════════
function setStatus(status, text) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');

    indicator.className = `status-indicator status-indicator--${status}`;

    const labels = {
        idle: 'Idle',
        running: 'Processing...',
        complete: 'Complete',
        error: 'Error',
    };

    statusText.textContent = text || labels[status] || status;
}

// ═══════════════════════════════════════════════════════════════════
// Summary stat cards
// ═══════════════════════════════════════════════════════════════════
function renderStats() {
    if (!summaryData) return;

    const s = summaryData;
    const wlRange = s.wavelength_range_micron || [0, 0];
    const rel = s.reliability_counts || {};

    const cards = [
        { label: 'Spectral Channels', value: s.n_spectral_channels, detail: `λ: ${safeFmt(wlRange[0], 1)} – ${safeFmt(wlRange[1], 1)} µm` },
        { label: 'Spaxels Used', value: s.n_spaxels_used, detail: `Shape: ${(s.spatial_shape || []).join('×')}` },
        { label: 'Peaks Detected', value: s.n_peaks_detected, cls: 'stat-card--purple', detail: `σ_noise: ${safeExp(s.noise_sigma, 2)}` },
        { label: 'Lines Matched', value: s.n_lines_matched, cls: 'stat-card--green', detail: `${rel.secure || 0} secure · ${rel.probable || 0} probable` },
        { label: 'Gaussian Fits', value: s.n_gaussian_fits, detail: `Voigt: ${s.n_voigt_fits || 0}` },
        { label: 'Redshift (z)', value: safeFmt(s.estimated_redshift_fitted, 6, '0.000000'), cls: 'stat-card--orange', detail: `± ${safeFmt(s.redshift_scatter, 6, '—')}` },
    ];

    const grid = document.getElementById('statsGrid');
    grid.innerHTML = cards.map(c => `
        <div class="stat-card ${c.cls || ''} animate-in">
            <div class="stat-card__label">${c.label}</div>
            <div class="stat-card__value">${c.value}</div>
            <div class="stat-card__detail">${c.detail}</div>
        </div>
    `).join('');
}

// ═══════════════════════════════════════════════════════════════════
// Interactive spectrum (Plotly)
// ═══════════════════════════════════════════════════════════════════
function renderSpectrum() {
    if (!spectraData) return;
    plotSpectrumView(currentSpectrumView);
}

function switchSpectrumView(view, tabEl) {
    currentSpectrumView = view;
    document.querySelectorAll('#spectrumTabs .tab').forEach(t => t.classList.remove('tab--active'));
    tabEl.classList.add('tab--active');
    plotSpectrumView(view);
}

function plotSpectrumView(view) {
    if (!spectraData) return;

    const wl = spectraData.wavelength;
    const traces = [];
    const layout = getPlotlyLayout();

    if (view === 'processed') {
        traces.push({
            x: wl, y: spectraData.savgol,
            type: 'scatter', mode: 'lines',
            name: 'Savitzky-Golay',
            line: { color: '#58A6FF', width: 1.2 },
        });

        // Add peak markers
        if (peaksData.length > 0) {
            const matchedIndices = new Set(matchesData.map(m => m.peak_index));
            const unmatchedPeaks = peaksData.filter(p => !matchedIndices.has(p.index));
            const matchedPeaks = peaksData.filter(p => matchedIndices.has(p.index));

            if (unmatchedPeaks.length > 0) {
                traces.push({
                    x: unmatchedPeaks.map(p => parseFloat(p.wavelength_micron)),
                    y: unmatchedPeaks.map(p => parseFloat(p.flux)),
                    type: 'scatter', mode: 'markers',
                    name: 'Unidentified',
                    marker: { color: '#8B949E', size: 6, symbol: 'x' },
                    text: unmatchedPeaks.map(p => `SNR: ${parseFloat(p.snr).toFixed(1)}`),
                    hovertemplate: 'λ = %{x:.4f} µm<br>Flux = %{y:.3e}<br>%{text}<extra></extra>',
                });
            }

            // Matched peaks colored by ionization
            for (const m of matchesData) {
                const peak = peaksData.find(p => p.index == m.peak_index);
                if (!peak) continue;
                const color = ION_COLORS[m.ionization_class] || '#C9D1D9';
                traces.push({
                    x: [parseFloat(m.observed_wavelength_micron)],
                    y: [parseFloat(peak.flux)],
                    type: 'scatter', mode: 'markers+text',
                    name: m.line_name,
                    marker: { color, size: 8, symbol: 'diamond', line: { color: 'white', width: 0.5 } },
                    text: [m.line_name],
                    textposition: 'top center',
                    textfont: { size: 9, color },
                    hovertemplate: `<b>${m.line_name}</b><br>λ_obs = %{x:.4f} µm<br>Confidence: ${parseFloat(m.confidence).toFixed(2)}<br>Reliability: ${m.reliability}<extra></extra>`,
                    showlegend: false,
                });
            }
        }

        layout.title.text = 'Processed Spectrum with Identified Lines';

    } else if (view === 'raw') {
        traces.push({
            x: wl, y: spectraData.raw,
            type: 'scatter', mode: 'lines',
            name: 'Extracted Spectrum',
            line: { color: '#58A6FF', width: 0.8 },
        });
        traces.push({
            x: wl, y: spectraData.continuum,
            type: 'scatter', mode: 'lines',
            name: 'Continuum Model',
            line: { color: '#F78166', width: 1.5 },
        });
        layout.title.text = 'Extracted Spectrum and Continuum';

    } else if (view === 'comparison') {
        traces.push({
            x: wl, y: spectraData.continuum_subtracted,
            type: 'scatter', mode: 'lines',
            name: 'Continuum Subtracted',
            line: { color: '#8B949E', width: 0.6 },
            opacity: 0.5,
        });
        traces.push({
            x: wl, y: spectraData.savgol,
            type: 'scatter', mode: 'lines',
            name: 'Savitzky-Golay',
            line: { color: '#7EE787', width: 1.2 },
        });
        traces.push({
            x: wl, y: spectraData.gaussian,
            type: 'scatter', mode: 'lines',
            name: 'Gaussian',
            line: { color: '#D2A8FF', width: 1.2 },
        });
        layout.title.text = 'Smoothing Comparison: SG vs Gaussian';
    }

    Plotly.newPlot('spectrumPlot', traces, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
}

function getPlotlyLayout() {
    return {
        title: { text: '', font: { color: '#C9D1D9', size: 14 } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15,20,36,0.6)',
        font: { family: 'Inter, sans-serif', color: '#8B949E', size: 11 },
        xaxis: {
            title: 'Wavelength [µm]',
            gridcolor: 'rgba(48,54,61,0.5)',
            linecolor: '#30363D',
            zerolinecolor: '#30363D',
        },
        yaxis: {
            title: 'Flux',
            gridcolor: 'rgba(48,54,61,0.5)',
            linecolor: '#30363D',
            zerolinecolor: '#30363D',
        },
        legend: {
            bgcolor: 'rgba(21,27,48,0.8)',
            bordercolor: '#30363D',
            font: { size: 10 },
        },
        margin: { l: 65, r: 20, t: 40, b: 50 },
        hovermode: 'x unified',
        height: 420,
    };
}

// ═══════════════════════════════════════════════════════════════════
// Line matches table
// ═══════════════════════════════════════════════════════════════════
function renderMatchesTable() {
    const body = document.getElementById('matchesBody');
    document.getElementById('matchCount').textContent = `${matchesData.length} matches`;

    if (matchesData.length === 0) {
        body.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-tertiary); padding: 24px;">No lines matched</td></tr>';
        return;
    }

    body.innerHTML = matchesData.map(m => {
        const conf = parseFloat(m.confidence);
        const confColor = conf >= 0.75 ? '#7EE787' : conf >= 0.5 ? '#58A6FF' : conf >= 0.3 ? '#FFA657' : '#FF7B72';

        return `<tr>
            <td style="color: ${ION_COLORS[m.ionization_class] || '#C9D1D9'}; font-weight: 600;">${m.line_name}</td>
            <td>${parseFloat(m.observed_wavelength_micron).toFixed(4)}</td>
            <td>${parseFloat(m.rest_wavelength_micron).toFixed(4)}</td>
            <td>${parseFloat(m.delta_micron).toFixed(4)}</td>
            <td><span class="badge badge--ion-${m.ionization_class}">${m.ionization_class.replace('_', ' ')}</span></td>
            <td>
                ${conf.toFixed(2)}
                <span class="confidence-bar">
                    <span class="confidence-bar__fill" style="width: ${conf * 100}%; background: ${confColor};"></span>
                </span>
            </td>
            <td><span class="badge badge--${m.reliability}">${m.reliability}</span></td>
        </tr>`;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════════
// Peaks table
// ═══════════════════════════════════════════════════════════════════
function renderPeaksTable() {
    const body = document.getElementById('peaksBody');
    document.getElementById('peakCount').textContent = `${peaksData.length} peaks`;

    if (peaksData.length === 0) {
        body.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-tertiary); padding: 24px;">No peaks detected</td></tr>';
        return;
    }

    body.innerHTML = peaksData.map(p => `<tr>
        <td>${p.index}</td>
        <td>${parseFloat(p.wavelength_micron).toFixed(4)}</td>
        <td>${parseFloat(p.flux).toExponential(3)}</td>
        <td>${parseFloat(p.prominence).toExponential(3)}</td>
        <td>${parseFloat(p.width_samples).toFixed(1)}</td>
        <td style="color: ${parseFloat(p.snr) > 10 ? '#7EE787' : parseFloat(p.snr) > 5 ? '#FFA657' : '#FF7B72'}">${parseFloat(p.snr).toFixed(1)}</td>
        <td>${isNaN(parseFloat(p.local_snr)) ? '—' : parseFloat(p.local_snr).toFixed(1)}</td>
    </tr>`).join('');
}

// ═══════════════════════════════════════════════════════════════════
// Plot gallery
// ═══════════════════════════════════════════════════════════════════
function renderPlotGallery() {
    const gallery = document.getElementById('plotGallery');
    const plots = [
        { file: 'smoothing_comparison.png', title: 'Smoothing Comparison' },
        { file: 'peaks_and_matches.png', title: 'Peaks & Line Matches' },
        { file: 'gaussian_fit_panels.png', title: 'Gaussian Fit Panels' },
        { file: 'redshift_histogram.png', title: 'Redshift Consensus' },
        { file: 'noise_profile.png', title: 'Noise Profile' },
        { file: 'strongest_line_map.png', title: 'Strongest Line Map' },
        { file: 'line_diagnostic_grid.png', title: 'Line Diagnostic Grid' },
    ];

    gallery.innerHTML = '';

    for (const plot of plots) {
        const card = document.createElement('div');
        card.className = 'plot-card';
        card.onclick = () => openLightbox(`/api/plot/${plot.file}`);
        card.innerHTML = `
            <img class="plot-card__image" src="/api/plot/${plot.file}" alt="${plot.title}" 
                 onerror="this.parentElement.style.display='none'" loading="lazy">
            <div class="plot-card__title">${plot.title}</div>
        `;
        gallery.appendChild(card);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Timing
// ═══════════════════════════════════════════════════════════════════
function renderTimings() {
    if (!summaryData?.timings_seconds) return;

    const grid = document.getElementById('timingGrid');
    const timings = summaryData.timings_seconds;
    const total = Object.values(timings).reduce((a, b) => a + b, 0);

    const icons = {
        load: '📂', dq_mask: '🛡️', extraction: '🎯', continuum: '📏',
        defringe: '〰️', smoothing: '✨', peak_detection: '📈',
        identification: '🏷️', fitting: '📐', classification: '⭐',
        write_outputs: '💾', visualization: '🎨',
    };

    grid.innerHTML = Object.entries(timings).map(([name, time]) => `
        <div class="timing-item">
            <span class="timing-item__name">${icons[name] || '·'} ${name.replace(/_/g, ' ')}</span>
            <span class="timing-item__value">${time.toFixed(2)}s</span>
        </div>
    `).join('') + `
        <div class="timing-item" style="border: 1px solid var(--border-active);">
            <span class="timing-item__name" style="color: var(--text-primary); font-weight: 600;">⏱ Total</span>
            <span class="timing-item__value" style="color: var(--accent-green);">${total.toFixed(2)}s</span>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════════
// Lightbox
// ═══════════════════════════════════════════════════════════════════
function openLightbox(src) {
    document.getElementById('lightboxImage').src = src;
    document.getElementById('lightbox').classList.add('active');
}

function closeLightbox() {
    document.getElementById('lightbox').classList.remove('active');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox();
});
