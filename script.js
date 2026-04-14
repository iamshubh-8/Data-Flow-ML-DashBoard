// =====================================================
// Carbon Emission Predictor — Frontend Logic
// Backend URL: http://localhost:8000
// Dataset target: CarbonEmission (regression)
// =====================================================

// --- APP STATE ---
// These variables hold the current state of the app.
// We keep them at the top so they're easy to find.

let rows = [];          // Full dataset as array of objects (one per CSV row)
let cols = [];          // Column names
let numCols = [];       // Columns that have numeric values
let catCols = [];       // Columns that have text/categorical values

// TARGET is now dynamic — set by the user via dropdown after upload.
// We auto-suggest the most carbon-sounding column name, but the user
// can change it to work with ANY carbon emission related dataset.
let TARGET = '';

let currentFile  = null;  // The uploaded training CSV file
let predictFile  = null;  // The CSV uploaded for inference (Step 5)
let selectedFeatures = []; // Which columns the user wants to train on
let selectedModel   = 'random_forest'; // Which ML algorithm to use

// Chart.js instances — we store them so we can destroy before re-drawing
let scatterChartInst = null;
let residChartInst   = null;
let lcChartInst      = null;

// API base URL — our FastAPI backend running locally
const API = 'http://localhost:8000';


// =====================================================
// THEME TOGGLE
// We store the user's choice in localStorage so it
// persists across page refreshes.
// =====================================================

function initTheme() {
  const saved = localStorage.getItem('theme') || 'dark';
  applyTheme(saved);
}

function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    document.getElementById('themeIcon').textContent = '🌞';
  } else {
    document.documentElement.removeAttribute('data-theme');
    document.getElementById('themeIcon').textContent = '🌚';
  }
  localStorage.setItem('theme', theme);
}

function toggleTheme() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  applyTheme(isLight ? 'dark' : 'light');
}


// =====================================================
// TAB NAVIGATION
// We simply hide all .section elements and show the
// one matching the clicked tab. Also updates the
// active state on the nav tabs.
// =====================================================

function gotoTab(tab) {
  // Guard: don't let user skip ahead before data is loaded
  const guarded = ['explore', 'train', 'results', 'predict'];
  if (guarded.includes(tab) && rows.length === 0) {
    showError('Please upload a CSV dataset first.');
    return;
  }

  // Switch active nav tab
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');

  // Switch active section
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById(`sec-${tab}`).classList.add('active');

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateSplit(val) {
  // Called every time the slider moves — just update the label
  document.getElementById('splitLabel').textContent = `${val}% train / ${100 - val}% test`;
}


// =====================================================
// FILE UPLOAD — DRAG & DROP + CLICK
// We set up both the training upload zone and the
// prediction upload zone at the same time.
// =====================================================

function setupDropZones() {
  const zone  = document.getElementById('dropZone');
  const pzone = document.getElementById('predictZone');

  // Prevent browser from opening the file directly
  const noDefault = e => { e.preventDefault(); e.stopPropagation(); };
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
    [zone, pzone, document.body].forEach(el => el.addEventListener(ev, noDefault));
  });

  // Visual feedback when dragging over the zone
  ['dragenter', 'dragover'].forEach(ev => {
    zone.addEventListener(ev,  () => zone.classList.add('drag'));
    pzone.addEventListener(ev, () => pzone.classList.add('drag'));
  });
  ['dragleave', 'drop'].forEach(ev => {
    zone.addEventListener(ev,  () => zone.classList.remove('drag'));
    pzone.addEventListener(ev, () => pzone.classList.remove('drag'));
  });

  // Drop event — grab the file from the drag event
  zone.addEventListener('drop',  e => { if (e.dataTransfer.files[0]) loadTrainFile(e.dataTransfer.files[0]); });
  pzone.addEventListener('drop', e => { if (e.dataTransfer.files[0]) loadPredictFile(e.dataTransfer.files[0]); });

  // Click event — user clicked the zone, file picker fires
  document.getElementById('csvInput').addEventListener('change', e => {
    if (e.target.files[0]) loadTrainFile(e.target.files[0]);
    e.target.value = ''; // reset so the same file can be re-uploaded
  });
  document.getElementById('predictInput').addEventListener('change', e => {
    if (e.target.files[0]) loadPredictFile(e.target.files[0]);
    e.target.value = '';
  });
}


// =====================================================
// CSV PARSING
// We parse CSVs ourselves — no library needed.
// We handle quoted fields (e.g. "['Stove', 'Oven']").
// =====================================================

function parseCSV(text) {
  // Normalise line endings (Windows uses \r\n)
  const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
                    .trim().split('\n').filter(l => l.trim());

  if (!lines.length) return { rows: [], cols: [] };

  const headers = splitLine(lines[0]).map(h => h.replace(/^"|"$/g, '').trim());

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = splitLine(lines[i]);
    const obj  = {};
    headers.forEach((h, j) => {
      // Clean up the value — remove surrounding quotes
      obj[h] = (vals[j] ?? '').replace(/^"|"$/g, '').trim();
    });
    rows.push(obj);
  }
  return { rows, cols: headers };
}

// splitLine respects quoted commas (e.g. "['a', 'b']" should not split)
function splitLine(line) {
  const parts = [];
  let cur = '', inQ = false;
  for (const ch of line) {
    if (ch === '"') { inQ = !inQ; }
    else if (ch === ',' && !inQ) { parts.push(cur.trim()); cur = ''; }
    else { cur += ch; }
  }
  parts.push(cur.trim());
  return parts;
}


// =====================================================
// LOAD TRAINING FILE
// Called when user uploads the main CSV.
// We parse it, identify numeric vs categorical columns,
// then render the preview and exploration panels.
// =====================================================

function loadTrainFile(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) { showError('Please upload a .csv file.'); return; }
  currentFile = file;

  const reader = new FileReader();
  reader.onload = e => processData(e.target.result);
  reader.onerror = () => showError('Could not read file.');
  reader.readAsText(file);
}

function processData(text) {
  const parsed = parseCSV(text);
  if (!parsed.rows.length) { showError('CSV is empty or malformed.'); return; }

  rows = parsed.rows;
  cols = parsed.cols;

  // Decide if a column is numeric: >70% of sample values parse as a number
  numCols = cols.filter(c => isNumeric(c));
  catCols = cols.filter(c => !isNumeric(c));

  // Auto-detect the most likely target column by checking if the column name
  // contains common carbon/emission keywords (case-insensitive).
  // This works for: CarbonEmission, CO2_emissions, carbon_kg, emission_total, etc.
  const keywords = ['emission', 'carbon', 'co2', 'ghg', 'greenhouse', 'footprint'];
  const autoTarget = cols.find(c =>
    keywords.some(kw => c.toLowerCase().includes(kw))
  ) || cols[cols.length - 1]; // fallback: last column (common convention)

  TARGET = autoTarget;

  // Build the target column dropdown with all columns as options
  renderTargetDropdown();

  // Everything else that depends on TARGET
  refreshAfterTargetChange();

  document.getElementById('previewArea').classList.remove('hidden');
}

// Build the <select> dropdown for choosing the target column
function renderTargetDropdown() {
  const sel = document.getElementById('targetSel');
  sel.innerHTML = cols.map(c =>
    `<option value="${esc(c)}" ${c === TARGET ? 'selected' : ''}>${esc(c)}</option>`
  ).join('');
  updateTargetBadge();
}

// Called when user changes the dropdown — re-runs everything downstream
function onTargetChange() {
  TARGET = document.getElementById('targetSel').value;
  updateTargetBadge();
  refreshAfterTargetChange();
}

// Show whether the selected target is numeric (regression) or categorical (classification)
// This helps the user know what kind of task they're setting up.
function updateTargetBadge() {
  const badge = document.getElementById('targetTypeBadge');
  const hint  = document.getElementById('targetHint');
  if (numCols.includes(TARGET)) {
    badge.textContent = 'NUMERIC';
    badge.className   = 'col-badge num';
    hint.textContent  = '✓ Numeric target → Regression task. Model will predict a continuous value (e.g. kg of CO₂).';
    hint.style.color  = 'var(--green)';
  } else {
    badge.textContent = 'CATEGORICAL';
    badge.className   = 'col-badge cat';
    hint.textContent  = '⚠ Categorical target → Classification task. Model will predict a category (e.g. Low / Medium / High).';
    hint.style.color  = 'var(--amber)';
  }
}

// Re-render everything that depends on the chosen TARGET column
function refreshAfterTargetChange() {
  selectedFeatures = cols.filter(c => c !== TARGET);
  renderStats();
  renderPreviewTable();
  renderExplore();
  renderFeatureCheckboxes();
  renderModelCards();
  // Update hints that mention the target column name
  const fHint = document.getElementById('featureHint');
  if (fHint) fHint.textContent = `These columns will be used to predict "${TARGET}". Uncheck any you don't want.`;
  const pHint = document.getElementById('predictHint');
  if (pHint) pHint.innerHTML = `Upload a CSV with the same columns used during training. The model will append a <b>Predicted_${esc(TARGET)}</b> column.`;
}

// A column is numeric if >70% of its first 100 values parse as a finite number
function isNumeric(col) {
  const sample = rows.slice(0, 100).map(r => r[col]).filter(v => v !== '' && v != null);
  if (!sample.length) return false;
  const numParsed = sample.filter(v => !isNaN(parseFloat(v)) && isFinite(v)).length;
  return numParsed / sample.length > 0.7;
}


// =====================================================
// STEP 1 RENDERING — Basic Stats + Preview Table
// =====================================================

function renderStats() {
  const missing = rows.reduce((acc, r) => {
    return acc + Object.values(r).filter(v => v === '' || v === '?').length;
  }, 0);

  // Task type depends on whether target is numeric or categorical
  const taskType = numCols.includes(TARGET) ? 'Regression' : 'Classification';

  document.getElementById('basicStats').innerHTML = `
    <div class="stat-card">
      <div class="stat-label">Rows</div>
      <div class="stat-val">${rows.length.toLocaleString()}</div>
      <div class="stat-sub">Total samples</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Features</div>
      <div class="stat-val blue">${cols.filter(c => c !== TARGET).length}</div>
      <div class="stat-sub">${numCols.filter(c=>c!==TARGET).length} numeric · ${catCols.filter(c=>c!==TARGET).length} categorical</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Missing Values</div>
      <div class="stat-val ${missing > 0 ? 'amber' : 'green'}">${missing}</div>
      <div class="stat-sub">${missing > 0 ? 'Will be imputed' : 'Clean dataset!'}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Target Column</div>
      <div class="stat-val" style="font-size:13px;line-height:1.3">${esc(TARGET)}</div>
      <div class="stat-sub">${taskType} task</div>
    </div>
  `;
}

function renderPreviewTable() {
  // Show just the first 6 rows to verify the data loaded correctly
  const preview = rows.slice(0, 6);
  let html = `<table><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>`;
  preview.forEach(r => {
    html += `<tr>${cols.map(c => `<td>${esc(String(r[c] ?? '').slice(0, 25))}</td>`).join('')}</tr>`;
  });
  document.getElementById('previewTable').innerHTML = html + '</table>';
}


// =====================================================
// STEP 2 — EXPLORE
// Three visualisations built purely in HTML/CSS (no canvas):
// 1. CarbonEmission histogram (the target)
// 2. Correlation bars — Pearson r between each numeric
//    feature and CarbonEmission
// 3. Mini histograms for each numeric feature
// =====================================================

function renderExplore() {
  // Update card titles to reflect the actual chosen target column
  document.getElementById('targetDistTitle').textContent = `"${TARGET}" Distribution (Target Variable)`;
  document.getElementById('corrTitle').textContent       = `Correlation with "${TARGET}"`;
  renderTargetHist();
  renderCorrBars();
  renderFeatDists();
}

// --- Helper: get all numeric values of a column as floats ---
function getNumVals(col) {
  return rows.map(r => parseFloat(r[col])).filter(v => !isNaN(v) && isFinite(v));
}

// --- Statistical helpers ---
function mean(arr)   { return arr.reduce((a,b)=>a+b,0) / arr.length; }
function stdDev(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a,v)=>a+(v-m)**2, 0) / arr.length);
}

// Pearson correlation coefficient between two numeric columns
// Returns a value between -1 (strong negative) and +1 (strong positive)
function pearson(colA, colB) {
  const pairs = rows.map(r => [parseFloat(r[colA]), parseFloat(r[colB])])
                    .filter(([a,b]) => !isNaN(a) && !isNaN(b));
  if (pairs.length < 2) return 0;
  const xs = pairs.map(p=>p[0]), ys = pairs.map(p=>p[1]);
  const mx = mean(xs), my = mean(ys);
  const num = pairs.reduce((acc,[x,y])=> acc + (x-mx)*(y-my), 0);
  const den = Math.sqrt(pairs.reduce((a,[x])=>a+(x-mx)**2,0)) *
              Math.sqrt(pairs.reduce((a,[,y])=>a+(y-my)**2,0));
  return den === 0 ? 0 : Math.max(-1, Math.min(1, num/den));
}

// Build a simple CSS histogram (no canvas) — bins the values into 20 buckets
function buildHistHTML(vals, color = 'var(--green)') {
  if (!vals.length) return '<div class="muted" style="font-size:12px">No numeric data</div>';
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const BINS = 20;
  const counts = new Array(BINS).fill(0);
  vals.forEach(v => {
    const b = Math.min(BINS - 1, Math.floor(((v - mn) / (mx - mn + 1e-9)) * BINS));
    counts[b]++;
  });
  const maxC = Math.max(...counts, 1);
  const bars = counts.map(c =>
    `<div class="bar-col" style="height:${Math.max(3, (c/maxC)*100)}%;background:${color}"></div>`
  ).join('');
  return `<div class="bar-row">${bars}</div>
          <div class="mini-range"><span>${mn.toFixed(1)}</span><span>${mx.toFixed(1)}</span></div>`;
}

function renderTargetHist() {
  // If target is numeric → show a histogram with mean/std
  if (numCols.includes(TARGET)) {
    const vals = getNumVals(TARGET);
    if (!vals.length) { document.getElementById('targetDist').innerHTML = '<p class="muted">No numeric values found in target column.</p>'; return; }
    const m = mean(vals).toFixed(2);
    const s = stdDev(vals).toFixed(2);
    document.getElementById('targetDist').innerHTML = `
      <div style="display:flex;gap:24px;margin-bottom:12px;flex-wrap:wrap">
        <span style="font-size:13px;color:var(--text2)">Mean: <b style="color:var(--text1);font-family:var(--mono)">${m}</b></span>
        <span style="font-size:13px;color:var(--text2)">Std: <b style="color:var(--text1);font-family:var(--mono)">${s}</b></span>
        <span style="font-size:13px;color:var(--text2)">Samples: <b style="color:var(--text1);font-family:var(--mono)">${vals.length.toLocaleString()}</b></span>
      </div>
      ${buildHistHTML(vals, 'var(--green)')}
    `;
  } else {
    // Categorical target → show a bar chart of class frequencies
    const counts = {};
    rows.forEach(r => { const v = r[TARGET] || 'N/A'; counts[v] = (counts[v] || 0) + 1; });
    const entries = Object.entries(counts).sort((a,b) => b[1]-a[1]).slice(0, 12);
    const maxC = entries[0]?.[1] || 1;
    document.getElementById('targetDist').innerHTML =
      `<div style="font-size:12px;color:var(--text2);margin-bottom:12px">Categorical target — class distribution (${Object.keys(counts).length} unique classes)</div>` +
      entries.map(([label, count]) => `
        <div class="fi-row">
          <div class="fi-name">${esc(label)}</div>
          <div class="fi-bar-wrap">
            <div class="fi-bar" style="width:${(count/maxC*100).toFixed(1)}%;background:var(--blue)"></div>
          </div>
          <div class="fi-val">${count}</div>
        </div>
      `).join('');
  }
}

function renderCorrBars() {
  // For each numeric feature, compute Pearson r with CarbonEmission
  const feats = numCols.filter(c => c !== TARGET);
  if (!feats.length) { document.getElementById('corrBars').innerHTML = '<p class="muted">No numeric features found.</p>'; return; }

  const corrs = feats.map(f => ({ col: f, r: pearson(f, TARGET) }))
                     .sort((a,b) => Math.abs(b.r) - Math.abs(a.r));

  const html = corrs.map(({ col, r }) => {
    // Positive = blue, Negative = red, length = absolute value
    const barColor = r >= 0 ? 'var(--green)' : 'var(--red)';
    const barWidth = Math.abs(r) * 100;
    return `<div class="fi-row">
      <div class="fi-name">${esc(col)}</div>
      <div class="fi-bar-wrap">
        <div class="fi-bar" style="width:${barWidth.toFixed(1)}%;background:${barColor}"></div>
      </div>
      <div class="fi-val" style="color:${r>=0?'var(--green)':'var(--red)'}">${r.toFixed(3)}</div>
    </div>`;
  }).join('');

  document.getElementById('corrBars').innerHTML = html ||
    '<p class="muted">Not enough numeric data to compute correlations.</p>';
}

function renderFeatDists() {
  // Mini histogram for each numeric feature (excluding target)
  const feats = numCols.filter(c => c !== TARGET).slice(0, 12);
  document.getElementById('featDists').innerHTML = feats.map(col => {
    const vals = getNumVals(col);
    return `<div class="mini-hist">
      <div class="mini-hist-title">${esc(col)}</div>
      ${buildHistHTML(vals, 'var(--teal)')}
    </div>`;
  }).join('');
}


// =====================================================
// STEP 3 — FEATURE SELECTION + MODEL CARDS
// =====================================================

function renderFeatureCheckboxes() {
  // Build a checkbox for every column except CarbonEmission (our target)
  const feats = cols.filter(c => c !== TARGET);
  document.getElementById('featureList').innerHTML = feats.map((col, i) => {
    const isNum = numCols.includes(col);
    return `<label class="feat-check">
      <input type="checkbox" id="feat_${i}" checked
             onchange="toggleFeature('${escJs(col)}', this.checked)">
      ${esc(col)}
      <span class="col-badge ${isNum ? 'num' : 'cat'}">${isNum ? 'NUM' : 'CAT'}</span>
    </label>`;
  }).join('');
}

function toggleFeature(col, checked) {
  if (checked && !selectedFeatures.includes(col)) selectedFeatures.push(col);
  if (!checked) selectedFeatures = selectedFeatures.filter(c => c !== col);
}

// Three model choices — simpler than the original 5
const MODELS = [
  { id: 'random_forest',  name: 'Random Forest',     desc: 'Builds many decision trees and averages their predictions. Robust and rarely overfits.' },
  { id: 'gradient_boost', name: 'Gradient Boosting', desc: 'Trains trees sequentially, each correcting the last. Highest accuracy on tabular data.' },
  { id: 'linear_reg',     name: 'Ridge Regression',  desc: 'Fast regularised linear model. Great baseline to compare against tree methods.' },
];

function renderModelCards() {
  document.getElementById('modelCards').innerHTML = MODELS.map(m => `
    <div class="model-card ${m.id === selectedModel ? 'active' : ''}"
         onclick="selectModel('${m.id}')">
      <div class="model-name">${m.name}</div>
      <div class="model-desc">${m.desc}</div>
    </div>
  `).join('');
}

function selectModel(id) {
  selectedModel = id;
  renderModelCards(); // re-render to update the .active class
}


// =====================================================
// STEP 3 — TRAINING
// We send the CSV file + config to the Python backend
// via a FormData POST request. The backend returns
// metrics, predictions, and feature importances as JSON.
// =====================================================

function addLog(msg, type = 'info') {
  const log  = document.getElementById('trainLog');
  const line = document.createElement('div');
  line.className = `log-${type}`;
  line.innerHTML = `<span class="log-ts">[${new Date().toLocaleTimeString()}]</span>${esc(msg)}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight; // auto-scroll to bottom
}

function setProgress(pct) {
  document.getElementById('trainProgress').style.width = `${Math.min(100, pct)}%`;
}

async function startTraining() {
  if (!currentFile)             { showError('Please upload a dataset first.'); return; }
  if (!selectedFeatures.length) { showError('Select at least one feature.');   return; }

  const btn = document.getElementById('trainBtn');
  btn.disabled = true;
  document.getElementById('trainLog').innerHTML = '';
  setProgress(0);

  // Read split slider value and convert to test_size (0.0–1.0)
  const splitVal = parseInt(document.getElementById('splitSlider').value) || 80;
  const testSize = ((100 - splitVal) / 100).toFixed(2);

  addLog(`Starting pipeline — model: ${selectedModel}, features: ${selectedFeatures.length}, test: ${100-splitVal}%`);
  setProgress(10);

  // Build the form data to send to FastAPI
  const fd = new FormData();
  fd.append('file',       currentFile);
  fd.append('target_col', TARGET);
  fd.append('model_type', selectedModel);
  fd.append('features',   selectedFeatures.join(','));
  fd.append('test_size',  testSize);

  try {
    addLog('Sending to backend...', 'info');
    setProgress(25);

    const res = await fetch(`${API}/train`, { method: 'POST', body: fd });

    setProgress(75);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    setProgress(100);
    addLog(`Done! R² = ${data.r2?.toFixed(4)} | MAE = ${data.mae?.toFixed(2)} | RMSE = ${data.rmse?.toFixed(2)}`, 'ok');

    if (data.dropped_cols?.length) {
      addLog(`Dropped high-cardinality columns: ${data.dropped_cols.join(', ')}`, 'warn');
    }

    // Render the results tab and switch to it automatically
    renderResults(data);
    setTimeout(() => gotoTab('results'), 800);

  } catch (err) {
    let msg = err.message;
    // Give a helpful message if the backend isn't running
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      msg = 'Cannot reach backend. Make sure uvicorn is running: uvicorn main:app --reload';
    }
    addLog(`ERROR: ${msg}`, 'warn');
    showError(msg);
  } finally {
    btn.disabled = false;
  }
}


// =====================================================
// STEP 4 — RESULTS RENDERING
// Takes the JSON from the backend and draws:
//  • Metric cards (R², MAE, RMSE)
//  • Predicted vs Actual scatter chart
//  • Residuals histogram
//  • Feature importance bars
//  • Learning curve line chart
// =====================================================

function renderResults(data) {
  document.getElementById('noResults').classList.add('hidden');
  document.getElementById('evalArea').classList.remove('hidden');

  // Build metric cards — different metrics for regression vs classification
  let metricsHTML = '';
  if (data.task_type === 'regression') {
    const r2color = data.r2 > 0.7 ? 'green' : data.r2 > 0.4 ? 'amber' : '';
    metricsHTML = `
      <div class="stat-card">
        <div class="stat-label">R² Score</div>
        <div class="stat-val ${r2color}">${(data.r2 ?? 0).toFixed(4)}</div>
        <div class="stat-sub">${data.r2 > 0.7 ? 'Strong fit' : data.r2 > 0.4 ? 'Moderate fit' : 'Weak fit'}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">MAE</div>
        <div class="stat-val">${(data.mae ?? 0).toFixed(2)}</div>
        <div class="stat-sub">Mean Absolute Error</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">RMSE</div>
        <div class="stat-val">${(data.rmse ?? 0).toFixed(2)}</div>
        <div class="stat-sub">Root Mean Sq. Error</div>
      </div>
    `;
  } else {
    // Classification metrics
    const accColor = data.accuracy > 0.8 ? 'green' : data.accuracy > 0.6 ? 'amber' : '';
    metricsHTML = `
      <div class="stat-card">
        <div class="stat-label">Accuracy</div>
        <div class="stat-val ${accColor}">${((data.accuracy ?? 0) * 100).toFixed(2)}%</div>
        <div class="stat-sub">Overall correct predictions</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">F1 Score</div>
        <div class="stat-val">${(data.f1 ?? 0).toFixed(4)}</div>
        <div class="stat-sub">Weighted harmonic mean</div>
      </div>
    `;
  }

  // Append common cards
  metricsHTML += `
    <div class="stat-card">
      <div class="stat-label">Model</div>
      <div class="stat-val" style="font-size:13px;line-height:1.3">${esc(data.model_name ?? '')}</div>
      <div class="stat-sub">${data.task_type}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Test Samples</div>
      <div class="stat-val blue">${data.test_size ?? '—'}</div>
      <div class="stat-sub">Held out for evaluation</div>
    </div>
  `;

  document.getElementById('evalMetrics').innerHTML = metricsHTML;

  // Show scatter + residuals only for regression (they don't make sense for classification)
  const scatterCard = document.getElementById('scatterChart').closest('.card');
  const residCard   = document.getElementById('residChart').closest('.card');
  if (data.task_type === 'regression') {
    scatterCard.classList.remove('hidden');
    residCard.classList.remove('hidden');
    drawScatter(data.actuals || [], data.predictions || []);
    drawResiduals(data.actuals || [], data.predictions || []);
  } else {
    scatterCard.classList.add('hidden');
    residCard.classList.add('hidden');
  }

  drawFeatureImportance(data.feature_importance || []);
  drawLearningCurve(data.learning_curve);
}

// Predicted vs Actual scatter — perfect prediction = diagonal line
function drawScatter(actuals, preds) {
  if (scatterChartInst) scatterChartInst.destroy();
  scatterChartInst = new Chart(document.getElementById('scatterChart'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Predicted vs Actual',
        data: actuals.map((a, i) => ({ x: a, y: preds[i] })),
        backgroundColor: 'rgba(63,185,80,0.4)',
        pointRadius: 3,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Actual CO₂ (kg)' } },
        y: { title: { display: true, text: 'Predicted CO₂ (kg)' } },
      },
    },
  });
}

// Residuals = Predicted - Actual. A good model has residuals centred at 0.
function drawResiduals(actuals, preds) {
  if (residChartInst) residChartInst.destroy();
  const residuals = preds.map((p, i) => p - actuals[i]);
  const mn  = Math.min(...residuals);
  const mx  = Math.max(...residuals);
  const BINS = 20;
  const counts = new Array(BINS).fill(0);
  residuals.forEach(v => {
    const b = Math.min(BINS-1, Math.floor(((v-mn)/(mx-mn+1e-9))*BINS));
    counts[b]++;
  });
  const labels = counts.map((_, i) => (mn + ((mx-mn)/BINS)*i).toFixed(0));

  residChartInst = new Chart(document.getElementById('residChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Count', data: counts, backgroundColor: 'rgba(46,196,182,0.55)', borderRadius: 3 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Residual (kg CO₂)' } },
        y: { title: { display: true, text: 'Count' } },
      },
    },
  });
}

// Feature importance — horizontal bars showing which features the model relies on most
function drawFeatureImportance(fi) {
  const el = document.getElementById('featureImportance');
  if (!fi.length) { el.innerHTML = '<p class="muted">No feature importance data.</p>'; return; }
  const maxI = fi[0].importance || 1;
  el.innerHTML = fi.map(f => `
    <div class="fi-row">
      <div class="fi-name">${esc(f.feature)}</div>
      <div class="fi-bar-wrap">
        <div class="fi-bar" style="width:${(f.importance/maxI*100).toFixed(1)}%"></div>
      </div>
      <div class="fi-val">${f.importance.toFixed(4)}</div>
    </div>
  `).join('');
}

// Learning curve — does the model improve as we feed it more training data?
// Train score vs validation score across increasing data sizes.
function drawLearningCurve(lc) {
  const lcCard = document.getElementById('lcChart').closest('.card');

  if (!lc || !lc.sizes) {
    lcCard.classList.add('hidden');
    return;
  }

  lcCard.classList.remove('hidden');

  if (lcChartInst) lcChartInst.destroy();

  lcChartInst = new Chart(document.getElementById('lcChart'), {
    type: 'line',
    data: {
      labels: lc.sizes,
      datasets: [
        { label: 'Train', data: lc.train_mean },
        { label: 'Validation', data: lc.val_mean }
      ]
    }
  });
}


// =====================================================
// STEP 5 — INFERENCE (Predict new CSV)
// User uploads a CSV without the CarbonEmission column.
// We POST it to /predict_new and the backend returns
// the same CSV with a Predicted_CarbonEmission column.
// =====================================================

function loadPredictFile(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) { showError('Please upload a .csv file.'); return; }
  predictFile = file;
  document.getElementById('predictTitle').textContent = `✓ ${file.name}`;
  document.getElementById('runPredictBtn').disabled = false;
}

async function runPredict() {
  if (!predictFile) { showError('Please upload a CSV for prediction.'); return; }

  const btn = document.getElementById('runPredictBtn');
  btn.textContent = 'Processing...';
  btn.disabled = true;

  const fd = new FormData();
  fd.append('file', predictFile);

  try {
    const res = await fetch(`${API}/predict_new`, { method: 'POST', body: fd });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    // The response is a CSV file — trigger a browser download
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'carbon_predictions.csv'; a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 5000);

    btn.textContent = '✓ Downloaded!';
    btn.disabled = false;

  } catch (err) {
    let msg = err.message;
    if (msg.includes('Failed to fetch')) msg = 'Backend not running. Start uvicorn first.';
    showError(msg);
    btn.textContent = 'Predict & Download CSV';
    btn.disabled = false;
  }
}


// =====================================================
// UTILITIES
// =====================================================

// Show a red toast at the bottom-right for 5 seconds
let errTimer = null;
function showError(msg) {
  const el = document.getElementById('errorToast');
  el.textContent = msg;
  el.style.display = 'block';
  if (errTimer) clearTimeout(errTimer);
  errTimer = setTimeout(() => { el.style.display = 'none'; }, 5000);
}

// HTML-escape strings before injecting into the DOM (prevents XSS)
function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

// JS-escape for use inside onclick="..." attribute strings
function escJs(str) {
  return String(str).replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/"/g,'\\"');
}


// =====================================================
// BOOT — runs when the DOM is ready
// =====================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  setupDropZones();
});