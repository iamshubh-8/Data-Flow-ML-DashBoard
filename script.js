/* =============================================
   DataFlow Pro — script.js (Fixed & Complete)
   ============================================= */

let df = [];
let cols = [];
let numCols = [];
let catCols = [];
let targetCol = '';
let selectedFeatures = [];
let selectedModel = 'random_forest';
let currentFile = null;
let predictFile = null;
let pvChartInst = null;
let residChartInst = null;
let dataLoaded = false;

const MODELS = [
  { id: 'random_forest',  name: 'Random Forest',       desc: 'Ensemble of decision trees. Robust to outliers and non-linear data.',    badge: 'Recommended'  },
  { id: 'gradient_boost', name: 'Gradient Boosting',   desc: 'Sequential tree building. Often yields highest predictive accuracy.',     badge: 'High Accuracy' },
  { id: 'linear_reg',     name: 'Linear / Logistic',   desc: 'Regularized linear model. Fast, highly interpretable baseline.',          badge: 'Baseline'      },
];

/* ── Tab navigation ─────────────────────────── */
function gotoTab(tab) {
  // Guard: don't allow jumping ahead without data
  const guardedTabs = ['eda', 'feature', 'train', 'eval', 'predict'];
  if (guardedTabs.includes(tab) && !dataLoaded) {
    alert('Please upload a CSV file first on the Data Source tab.');
    return;
  }

  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  const activeTab = document.querySelector(`[data-tab="${tab}"]`);
  if (activeTab) activeTab.classList.add('active');

  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  const sec = document.getElementById(`sec-${tab}`);
  if (sec) sec.classList.add('active');

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateSplit(v) {
  document.getElementById('splitLabel').textContent = `${v}% / ${100 - v}%`;
  document.getElementById('trainBar').style.width = v + '%';
}

/* ── Drag-and-drop setup ────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
  const dropZone        = document.getElementById('dropZone');
  const predictDropZone = document.getElementById('predictDropZone');

  function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
    dropZone.addEventListener(ev, preventDefaults, false);
    predictDropZone.addEventListener(ev, preventDefaults, false);
    document.body.addEventListener(ev, preventDefaults, false);
  });

  ['dragenter', 'dragover'].forEach(ev => {
    dropZone.addEventListener(ev,        () => dropZone.classList.add('drag'),        false);
    predictDropZone.addEventListener(ev, () => predictDropZone.classList.add('drag'), false);
  });

  ['dragleave', 'drop'].forEach(ev => {
    dropZone.addEventListener(ev,        () => dropZone.classList.remove('drag'),        false);
    predictDropZone.addEventListener(ev, () => predictDropZone.classList.remove('drag'), false);
  });

  dropZone.addEventListener('drop', function (e) {
    const file = e.dataTransfer.files[0];
    if (file) handleMainUpload(file);
  });

  predictDropZone.addEventListener('drop', function (e) {
    const file = e.dataTransfer.files[0];
    if (file) handlePredictUpload(file);
  });

  document.getElementById('csvInput').addEventListener('change', function (e) {
    if (e.target.files[0]) handleMainUpload(e.target.files[0]);
    e.target.value = '';
  });

  document.getElementById('predictInput').addEventListener('change', function (e) {
    if (e.target.files[0]) handlePredictUpload(e.target.files[0]);
    e.target.value = '';
  });
});

/* ── File upload handlers ───────────────────── */
function handleMainUpload(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    alert('Please upload a valid CSV file.');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = ev => loadData(ev.target.result, file.name);
  reader.onerror = () => alert('Error reading file. Please try again.');
  reader.readAsText(file);
}

function handlePredictUpload(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) {
    alert('Please upload a valid CSV file.');
    return;
  }
  predictFile = file;
  document.getElementById('predictTitle').textContent = `✓ Ready: ${file.name}`;
  document.getElementById('runPredictBtn').disabled = false;
}

/* ── CSV Parsing ────────────────────────────── */
function parseCSV(text) {
  const cleaned = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
  const lines = cleaned.split('\n').filter(line => line.trim().length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };

  const headers = smartSplit(lines[0]).map(h => h.trim().replace(/^"|"$/g, ''));
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const vals = smartSplit(lines[i]);
    if (vals.length === 0) continue;
    const obj = {};
    headers.forEach((h, j) => {
      obj[h] = ((vals[j] ?? '') + '').trim().replace(/^"|"$/g, '');
    });
    rows.push(obj);
  }
  return { headers, rows };
}

function smartSplit(line) {
  const res = [];
  let cur = '';
  let inQ = false;
  for (const c of line) {
    if (c === '"')             { inQ = !inQ; }
    else if (c === ',' && !inQ){ res.push(cur); cur = ''; }
    else                       { cur += c; }
  }
  res.push(cur);
  return res;
}

function isNumeric(col) {
  const sample = df.slice(0, 100)
    .map(r => r[col])
    .filter(v => v !== '' && v !== undefined && v !== null);
  if (sample.length === 0) return false;
  return sample.filter(v => !isNaN(parseFloat(v)) && isFinite(v)).length > sample.length * 0.7;
}

/* ── Data loading ───────────────────────────── */
function loadData(text, fname) {
  try {
    const parsed = parseCSV(text);
    cols = parsed.headers;
    df   = parsed.rows;

    if (df.length === 0 || cols.length === 0) {
      alert('Error: The CSV file appears to be empty or malformed.');
      return;
    }

    numCols = cols.filter(c => isNumeric(c));
    catCols = cols.filter(c => !isNumeric(c));

    // Smart target detection
    if (numCols.includes('CarbonEmission')) {
      targetCol = 'CarbonEmission';
    } else if (numCols.length > 0) {
      targetCol = numCols[numCols.length - 1];
    } else {
      targetCol = cols[cols.length - 1];
    }

    selectedFeatures = cols.filter(c => c !== targetCol);
    dataLoaded = true;

    renderBasicStats(fname);
    renderPreviewTable();
    renderEDA();
    renderFeatureEngineering();
    renderModelCards();

    document.getElementById('previewArea').classList.remove('hidden');
    markTabDone('upload');

  } catch (err) {
    alert('Failed to parse CSV: ' + err.message);
    console.error(err);
  }
}

/* ── Stats & Preview ────────────────────────── */
function renderBasicStats(fname) {
  const el = document.getElementById('basicStats');
  const missing = df.reduce((a, r) =>
    a + Object.values(r).filter(v => v === '' || v === null || v === undefined).length, 0);
  const featureCount = cols.filter(c => c !== targetCol).length;

  el.innerHTML = `
    <div class="stat-card"><div class="stat-label">Total Records</div><div class="stat-val">${df.length.toLocaleString()}</div><div class="stat-sub">Rows in dataset</div></div>
    <div class="stat-card"><div class="stat-label">Features</div><div class="stat-val">${featureCount}</div><div class="stat-sub">${numCols.filter(c=>c!==targetCol).length} Numeric · ${catCols.filter(c=>c!==targetCol).length} Categorical</div></div>
    <div class="stat-card"><div class="stat-label">Missing Values</div><div class="stat-val">${missing}</div><div class="stat-sub">Empty cells detected</div></div>
    <div class="stat-card"><div class="stat-label">Target Variable</div><div class="stat-val" style="font-size:16px;line-height:30px;overflow:hidden;text-overflow:ellipsis">${escapeHtml(targetCol)}</div><div class="stat-sub">Auto-detected</div></div>
  `;
}

function renderPreviewTable() {
  const preview = df.slice(0, 5);
  let html = '<table><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr>';
  preview.forEach(r => {
    html += '<tr>' + cols.map(c => {
      const val = r[c] === undefined || r[c] === null ? '' : String(r[c]);
      return `<td>${escapeHtml(val.substring(0, 24))}</td>`;
    }).join('') + '</tr>';
  });
  html += '</table>';
  document.getElementById('previewTable').innerHTML = html;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* ── Math helpers ───────────────────────────── */
function getNumVals(col) {
  return df.map(r => parseFloat(r[col])).filter(v => !isNaN(v) && isFinite(v));
}

function mean(arr) {
  if (!arr || !arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
  if (!arr || !arr.length) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, v) => a + (v - m) ** 2, 0) / arr.length);
}

function median(arr) {
  if (!arr || !arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

/* ── EDA ────────────────────────────────────── */
function renderEDA() {
  const sel = document.getElementById('targetSel');
  sel.innerHTML = cols.map(c =>
    `<option value="${escapeHtml(c)}"${c === targetCol ? ' selected' : ''}>${escapeHtml(c)}</option>`
  ).join('');
  renderSummaryTable();
  renderTargetDist();
  renderCorrelation();
  renderFeatureDists();
}

function updateTarget() {
  const newTarget = document.getElementById('targetSel').value;
  if (newTarget !== targetCol) {
    // Add old target back to features, remove new target from features
    if (!selectedFeatures.includes(targetCol)) {
      selectedFeatures.push(targetCol);
    }
    targetCol = newTarget;
    selectedFeatures = selectedFeatures.filter(c => c !== targetCol);
    if (selectedFeatures.length === 0) {
      selectedFeatures = cols.filter(c => c !== targetCol);
    }
  }

  if (numCols.includes(targetCol)) {
    renderTargetDist();
  } else {
    document.getElementById('targetDist').innerHTML =
      '<div style="padding:10px;color:var(--text-secondary);font-size:13px;">Categorical target selected. Distribution not shown.</div>';
  }
  renderFeatureEngineering();
}

function renderTargetDist() {
  const vals = getNumVals(targetCol);
  if (!vals.length) {
    document.getElementById('targetDist').innerHTML =
      '<div style="padding:10px;color:var(--text-secondary);font-size:13px;">No numeric values for target.</div>';
    return;
  }
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const bins = 24, counts = new Array(bins).fill(0);
  vals.forEach(v => {
    const b = Math.min(bins - 1, Math.floor(((v - mn) / (mx - mn + 1e-9)) * bins));
    counts[b]++;
  });
  const maxC = Math.max(...counts, 1);
  const m = mean(vals), s = std(vals), med = median(vals);

  document.getElementById('targetDist').innerHTML = `
    <div style="font-size:13px;color:var(--text-secondary);display:flex;gap:24px;margin-bottom:16px;background:var(--bg-alt);padding:12px;border-radius:var(--radius-sm);flex-wrap:wrap">
      <span>Mean: <b style="color:var(--text-primary)">${m.toFixed(2)}</b></span>
      <span>Std Dev: <b style="color:var(--text-primary)">${s.toFixed(2)}</b></span>
      <span>Median: <b style="color:var(--text-primary)">${med.toFixed(2)}</b></span>
    </div>
    <div class="dist-bars">${counts.map(c =>
      `<div class="dist-bar" style="background:#4f46e5;opacity:${0.4 + (c/maxC)*0.6};height:${Math.max(4,(c/maxC)*100)}%"></div>`
    ).join('')}</div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-tertiary);margin-top:8px;font-family:var(--font-mono)">
      <span>${mn.toFixed(2)}</span><span>${mx.toFixed(2)}</span>
    </div>
  `;
}

function renderSummaryTable() {
  const displayCols = numCols.slice(0, 8);
  if (!displayCols.length) {
    document.getElementById('summaryTable').innerHTML =
      '<div style="padding:10px;color:var(--text-secondary);font-size:13px;">No numeric columns found.</div>';
    return;
  }
  let html = '<table><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Type</th></tr>';
  displayCols.forEach(c => {
    const vals = getNumVals(c);
    if (!vals.length) return;
    html += `<tr>
      <td><span style="font-family:var(--font-mono);font-size:12px">${escapeHtml(c)}</span></td>
      <td>${mean(vals).toFixed(2)}</td>
      <td>${std(vals).toFixed(2)}</td>
      <td>${Math.min(...vals).toFixed(2)}</td>
      <td>${Math.max(...vals).toFixed(2)}</td>
      <td><span class="badge badge-blue">Numeric</span></td>
    </tr>`;
  });
  html += '</table>';
  document.getElementById('summaryTable').innerHTML = '<div class="table-wrap">' + html + '</div>';
}

function pearsonColumns(colA, colB) {
  const x = [], y = [];
  for (const row of df) {
    const a = parseFloat(row[colA]), b = parseFloat(row[colB]);
    if (!isNaN(a) && !isNaN(b) && isFinite(a) && isFinite(b)) { x.push(a); y.push(b); }
  }
  if (x.length < 2) return 0;
  const mx = mean(x), my = mean(y);
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy; denX += dx * dx; denY += dy * dy;
  }
  const den = Math.sqrt(denX * denY);
  return den === 0 ? 0 : Math.max(-1, Math.min(1, num / den));
}

function renderCorrelation() {
  const corrCols = numCols.slice(0, 10);
  if (!corrCols.length) {
    document.getElementById('corrHeatmap').innerHTML =
      '<div style="padding:10px;color:var(--text-secondary);font-size:13px;">No numeric columns for correlation.</div>';
    return;
  }

  let html = `<div class="corr-grid" style="grid-template-columns:100px ${corrCols.map(() => '1fr').join(' ')}">`;
  html += '<div></div>' + corrCols.map(c =>
    `<div style="color:var(--text-tertiary);padding:4px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(c)}">${escapeHtml(c.substring(0,6))}</div>`
  ).join('');

  corrCols.forEach(rc => {
    html += `<div style="color:var(--text-secondary);text-align:right;padding-right:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;align-self:center" title="${escapeHtml(rc)}">${escapeHtml(rc.substring(0,12))}</div>`;
    corrCols.forEach(cc => {
      const r = rc === cc ? 1 : pearsonColumns(rc, cc);
      let bg = '#ffffff', textCol = '#111827';
      if (r > 0) {
        bg = `rgba(30,58,138,${Math.abs(r).toFixed(2)})`; if (r > 0.5) textCol = '#ffffff';
      } else if (r < 0) {
        bg = `rgba(127,29,29,${Math.abs(r).toFixed(2)})`; if (Math.abs(r) > 0.5) textCol = '#ffffff';
      }
      html += `<div class="hm-cell" style="background:${bg};color:${textCol}" title="${escapeHtml(rc)} vs ${escapeHtml(cc)}: ${r.toFixed(2)}">${r.toFixed(1)}</div>`;
    });
  });
  html += '</div>';
  html += `<div style="display:flex;gap:24px;margin-top:16px;font-size:12px;color:var(--text-secondary);flex-wrap:wrap">
    <span style="display:flex;align-items:center;gap:6px"><span style="width:12px;height:12px;background:rgba(30,58,138,0.8);border-radius:3px;display:inline-block"></span>Positive Correlation</span>
    <span style="display:flex;align-items:center;gap:6px"><span style="width:12px;height:12px;background:rgba(127,29,29,0.8);border-radius:3px;display:inline-block"></span>Negative Correlation</span>
  </div>`;
  document.getElementById('corrHeatmap').innerHTML = html;
}

function renderFeatureDists() {
  const el = document.getElementById('featDist');
  const displayCols = numCols.filter(c => c !== targetCol).slice(0, 8);
  if (!displayCols.length) {
    el.innerHTML = '<div style="padding:10px;color:var(--text-secondary);font-size:13px;">No numeric features for distributions.</div>';
    return;
  }
  el.innerHTML = displayCols.map(col => {
    const vals = getNumVals(col).slice(0, 500);
    if (!vals.length) return '';
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const bins = 15, counts = new Array(bins).fill(0);
    vals.forEach(v => {
      const b = Math.min(bins - 1, Math.floor(((v - mn) / (mx - mn + 1e-9)) * bins));
      counts[b]++;
    });
    const maxC = Math.max(...counts, 1);
    return `<div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px">
      <div style="font-size:12px;font-weight:500;color:var(--text-primary);margin-bottom:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(col)}</div>
      <div class="dist-bars" style="height:60px">${counts.map(c =>
        `<div class="dist-bar" style="height:${Math.max(4,(c/maxC)*100)}%;background:#0d9488;opacity:0.8"></div>`
      ).join('')}</div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-tertiary);margin-top:6px;font-family:var(--font-mono)">
        <span>${mn.toFixed(1)}</span><span>${mx.toFixed(1)}</span>
      </div>
    </div>`;
  }).join('');
}

/* ── Feature Engineering ────────────────────── */
function renderFeatureEngineering() {
  document.getElementById('featureOps').innerHTML = `
    <div style="display:flex;align-items:flex-start;gap:16px;padding:16px;background:var(--bg-surface);border-radius:var(--radius);border:1px solid var(--border)">
      <div style="margin-top:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg></div>
      <div style="flex:1"><b style="color:var(--text-primary);display:block;margin-bottom:4px">Sklearn Pipelines & One-Hot Encoding</b><span style="font-size:13px;color:var(--text-secondary)">Categorical string data is automatically converted to independent binary vectors. High-cardinality identifiers are dropped.</span></div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:16px;padding:16px;background:var(--bg-surface);border-radius:var(--radius);border:1px solid var(--border)">
      <div style="margin-top:2px"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg></div>
      <div style="flex:1"><b style="color:var(--text-primary);display:block;margin-bottom:4px">Robust Scaling & Imputation</b><span style="font-size:13px;color:var(--text-secondary)">Missing values imputed with median/mode. Numerical columns are Standard Scaled (Mean=0, Var=1).</span></div>
    </div>
  `;

  const featCols = cols.filter(c => c !== targetCol);
  let html = '';
  featCols.forEach((col, i) => {
    const checked = selectedFeatures.includes(col);
    const isCat = catCols.includes(col);
    html += `<div style="display:flex;align-items:center;gap:16px;padding:10px 14px;background:var(--bg-alt);border-radius:var(--radius-sm);border:1px solid var(--border)">
      <input type="checkbox" id="feat_${i}" ${checked ? 'checked' : ''} onchange="toggleFeature('${escapeForJs(col)}', this.checked)">
      <label for="feat_${i}" style="min-width:200px;font-size:13px;font-weight:500;cursor:pointer;font-family:var(--font-mono);flex:1">${escapeHtml(col)}</label>
      ${isCat ? '<span class="badge badge-default" style="font-size:10px">CAT</span>' : '<span class="badge badge-blue" style="font-size:10px">NUM</span>'}
    </div>`;
  });
  document.getElementById('featureList').innerHTML = html;

  document.getElementById('processedStats').innerHTML = `
    <div class="stat-card"><div class="stat-label">Training Pool</div><div class="stat-val">${Math.round(df.length * 0.8).toLocaleString()}</div><div class="stat-sub">80% of records</div></div>
    <div class="stat-card"><div class="stat-label">Testing Pool</div><div class="stat-val">${Math.round(df.length * 0.2).toLocaleString()}</div><div class="stat-sub">20% of records</div></div>
    <div class="stat-card"><div class="stat-label">Active Features</div><div class="stat-val" id="featCount">${selectedFeatures.length}</div><div class="stat-sub">Selected for training</div></div>
  `;
}

function escapeForJs(str) {
  return String(str).replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function toggleFeature(col, checked) {
  if (checked && !selectedFeatures.includes(col)) {
    selectedFeatures.push(col);
  } else if (!checked) {
    selectedFeatures = selectedFeatures.filter(c => c !== col);
  }
  const el = document.getElementById('featCount');
  if (el) el.textContent = selectedFeatures.length;
}

/* ── Model Cards ────────────────────────────── */
function renderModelCards() {
  document.getElementById('modelCards').innerHTML = MODELS.map(m => `
    <div class="model-card${m.id === selectedModel ? ' selected' : ''}" onclick="selectModel('${m.id}')">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <b style="font-size:15px;color:var(--text-primary)">${m.name}</b>
        <span class="check">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </span>
      </div>
      <div style="font-size:13px;color:var(--text-secondary);margin-bottom:16px;line-height:1.5">${m.desc}</div>
      <span class="badge badge-default">${m.badge}</span>
    </div>
  `).join('');
}

function selectModel(id) {
  selectedModel = id;
  renderModelCards();
}

/* ── Training helpers ───────────────────────── */
function markTabDone(tab) {
  const tabEl = document.querySelector(`[data-tab="${tab}"]`);
  if (!tabEl) return;
  tabEl.classList.add('done');
  const stepNum = tabEl.querySelector('.step-num');
  if (stepNum) {
    stepNum.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>';
  }
}

function addLog(msg, type = '') {
  const log = document.getElementById('trainingLog');
  const line = document.createElement('div');
  line.className = 'log-line ' + type;
  const timeStr = new Date().toLocaleTimeString();
  line.innerHTML = `<span style="color:var(--text-tertiary)">[${timeStr}]</span> ${escapeHtml(msg)}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

function setProgress(pct) {
  document.getElementById('trainProgress').style.width = Math.min(100, Math.max(0, pct)) + '%';
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/* ── TRAINING — main fix ────────────────────── */
async function startTraining() {
  if (!currentFile) {
    alert('Please upload a CSV dataset on the Data Source tab first.');
    return;
  }
  if (!selectedFeatures || selectedFeatures.length === 0) {
    alert('Please select at least one feature on the Engineering tab.');
    return;
  }

  const btn = document.getElementById('trainBtn');
  const complexity = document.getElementById('hyperSlider').value;

  btn.disabled = true;
  document.getElementById('trainStatus').textContent = 'Compiling Pipeline...';

  const log = document.getElementById('trainingLog');
  log.innerHTML = '';
  setProgress(0);

  addLog('Initializing execution environment...', 'info');
  setProgress(5);
  await sleep(200);

  addLog(`Target column: ${targetCol}`, 'info');
  addLog(`Features selected: ${selectedFeatures.length}`, 'info');
  addLog(`Model: ${selectedModel} | Complexity: ${complexity}`, 'info');
  setProgress(15);
  await sleep(200);

  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('target_col', targetCol);
  formData.append('model_type', selectedModel);
  formData.append('features', selectedFeatures.join(','));
  formData.append('complexity', complexity);

  try {
    addLog('Transmitting payload to Scikit-Learn engine...', 'info');
    setProgress(25);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2-min timeout

    const response = await fetch('http://localhost:8000/train', {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    setProgress(65);

    if (!response.ok) {
      let errMsg = `Server responded with status ${response.status}`;
      try {
        const errData = await response.json();
        errMsg = errData.error || errMsg;
      } catch (_) {
        try { errMsg = await response.text() || errMsg; } catch (_) {}
      }
      throw new Error(errMsg);
    }

    const results = await response.json();

    if (results.error) {
      throw new Error(results.error);
    }

    setProgress(85);
    addLog('Model training complete. Saving pipeline (.joblib)...', 'info');
    await sleep(200);

    setProgress(100);
    addLog(`✓ Task type: ${results.task_type.toUpperCase()}`, 'ok');

    if (results.dropped_cols && results.dropped_cols.length > 0) {
      addLog(`⚠ Dropped high-cardinality columns: ${results.dropped_cols.join(', ')}`, 'warn');
    }

    if (results.task_type === 'regression') {
      addLog(`✓ R² Score: ${Number(results.r2).toFixed(4)}  |  MAE: ${Number(results.mae).toFixed(4)}`, 'ok');
    } else {
      addLog(`✓ Accuracy: ${(results.accuracy * 100).toFixed(2)}%  |  F1: ${Number(results.f1).toFixed(4)}`, 'ok');
    }

    addLog('Pipeline complete. Navigate to Evaluation tab.', 'ok');

    btn.disabled = false;
    document.getElementById('trainStatus').textContent = 'Execution finished successfully';
    markTabDone('train');

    renderEvaluation(results);

    // Navigate AFTER a brief delay so user can see the success log
    setTimeout(() => gotoTab('eval'), 1200);

  } catch (error) {
    let userMsg = error.message;
    if (error.name === 'AbortError') {
      userMsg = 'Request timed out (2 minutes). Try with a smaller dataset or fewer features.';
    } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError') || error.message.includes('net::')) {
      userMsg = 'Cannot connect to backend. Make sure uvicorn is running: uvicorn main:app --reload';
    }

    addLog(`✗ ERROR: ${userMsg}`, 'warn');
    addLog('Ensure the FastAPI server is running: uvicorn main:app --reload', 'info');

    btn.disabled = false;
    document.getElementById('trainStatus').textContent = 'Pipeline Failed — Check Logs';
    setProgress(0);
    // IMPORTANT: Do NOT navigate away — stay on training tab so user sees the error
  }
}

/* ── Evaluation rendering ───────────────────── */
function renderEvaluation(res) {
  document.getElementById('noResults').classList.add('hidden');
  document.getElementById('evalResults').classList.remove('hidden');

  let metricsHtml = `
    <div class="stat-card">
      <div class="stat-label">Model Architecture</div>
      <div class="stat-val" style="font-size:14px;line-height:30px">${escapeHtml(res.model_name || '')}</div>
      <div class="stat-sub">Scikit-Learn Pipeline</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Test Samples</div>
      <div class="stat-val">${Number(res.test_size || 0).toLocaleString()}</div>
      <div class="stat-sub">Held-out evaluation set</div>
    </div>
  `;

  if (res.task_type === 'classification') {
    metricsHtml += `
      <div class="stat-card">
        <div class="stat-label">Accuracy Score</div>
        <div class="stat-val">${(res.accuracy * 100).toFixed(2)}%</div>
        <div class="stat-sub">Overall Correct</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">F1 Score</div>
        <div class="stat-val">${Number(res.f1).toFixed(4)}</div>
        <div class="stat-sub">Weighted Harmonic Mean</div>
      </div>
    `;
  } else {
    metricsHtml += `
      <div class="stat-card">
        <div class="stat-label">R² Score</div>
        <div class="stat-val" style="color:${res.r2 > 0.7 ? 'var(--success)' : 'var(--warning)'}">${Number(res.r2).toFixed(4)}</div>
        <div class="stat-sub">${res.r2 > 0.7 ? 'Strong predictive fit' : 'Moderate fit'}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Mean Abs Error</div>
        <div class="stat-val">${Number(res.mae).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
        <div class="stat-sub">Avg unit deviation</div>
      </div>
    `;
  }

  document.getElementById('evalMetrics').innerHTML = metricsHtml;

  // Predicted vs Actual chart
  if (pvChartInst) { pvChartInst.destroy(); pvChartInst = null; }

  const actuals     = res.actuals     || [];
  const predictions = res.predictions || [];

  pvChartInst = new Chart(document.getElementById('pvChart'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Predictions vs Actual',
        data: actuals.map((act, i) => ({ x: act, y: predictions[i] })),
        backgroundColor: 'rgba(79,70,229,0.6)',
        pointRadius: 4,
        pointHoverRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Actual' } },
        y: { title: { display: true, text: 'Predicted' } }
      }
    }
  });

  // Residuals chart
  let residuals;
  if (res.task_type === 'regression') {
    residuals = predictions.map((p, i) => p - actuals[i]);
  } else {
    residuals = predictions.map((p, i) => (p === actuals[i] ? 0 : 1));
  }

  const mn = Math.min(...residuals, 0);
  const mx = Math.max(...residuals, 0);
  const bins = 24, counts = new Array(bins).fill(0);
  residuals.forEach(v => {
    const b = Math.min(bins - 1, Math.floor(((v - mn) / (mx - mn + 1e-9)) * bins));
    counts[b]++;
  });

  if (residChartInst) { residChartInst.destroy(); residChartInst = null; }

  residChartInst = new Chart(document.getElementById('residChart'), {
    type: 'bar',
    data: {
      labels: counts.map((_, i) => (mn + ((mx - mn) / bins) * i).toFixed(1)),
      datasets: [{
        label: 'Frequency',
        data: counts,
        backgroundColor: '#0d9488',
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Residual' } },
        y: { title: { display: true, text: 'Count' } }
      }
    }
  });

  markTabDone('eval');
}

/* ── Inference ──────────────────────────────── */
async function runInference() {
  if (!predictFile) {
    alert('Please upload a CSV file for prediction first.');
    return;
  }

  const btn = document.getElementById('runPredictBtn');
  const originalText = btn.textContent;
  btn.textContent = 'Processing...';
  btn.disabled = true;

  const formData = new FormData();
  formData.append('file', predictFile);

  try {
    const response = await fetch('http://localhost:8000/predict_new', {
      method: 'POST',
      body: formData
    });

    const contentType = response.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
      const errData = await response.json();
      throw new Error(errData.error || `Server error ${response.status}`);
    }

    if (!response.ok) {
      throw new Error(`Failed to generate predictions (HTTP ${response.status})`);
    }

    const blob = await response.blob();
    const url  = window.URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.style.display = 'none';
    a.href          = url;
    a.download      = 'predictions_output.csv';
    document.body.appendChild(a);
    a.click();

    setTimeout(() => {
      window.URL.revokeObjectURL(url);
      if (a.parentNode) a.parentNode.removeChild(a);
    }, 1000);

    btn.textContent = '✓ Predictions Downloaded!';
    btn.disabled = false;
    markTabDone('predict');

  } catch (error) {
    let msg = error.message;
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      msg = 'Cannot connect to backend. Ensure uvicorn is running.';
    }
    alert(`Prediction error: ${msg}`);
    btn.textContent = originalText;
    btn.disabled = false;
  }
}
