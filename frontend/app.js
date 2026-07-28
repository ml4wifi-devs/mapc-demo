/* App wiring: catalog-driven forms, scenario preview, run lifecycle, WebSocket. */
'use strict';

(async () => {
  const $ = (id) => document.getElementById(id);
  const catalog = await (await fetch('/api/catalog')).json();
  const runs = new Map(); // run_id -> {label, el, stateEl}
  const pendingMsgs = new Map(); // run_id -> [msgs] arriving before the run is registered
  let runCounter = 0;

  // ---------- form rendering ----------

  function fieldHtml(p, value) {
    const val = value !== undefined ? value : p.default;
    const tip = (p.tooltip || '').replace(/"/g, '&quot;');
    const label = `<span>${p.label} <span class="info" data-tip="${tip}">ⓘ</span></span>`;
    if (p.type === 'select') {
      const opts = p.options.map(o => {
        const v = typeof o === 'object' ? o.value : o;
        const l = typeof o === 'object' ? o.label : o;
        return `<option value="${v}" ${String(v) === String(val) ? 'selected' : ''}>${l}</option>`;
      }).join('');
      return `<label class="field" data-name="${p.name}">${label}<select name="${p.name}">${opts}</select></label>`;
    }
    if (p.type === 'checkbox') {
      return `<label class="field inline" data-name="${p.name}">${label}<input type="checkbox" name="${p.name}" ${val ? 'checked' : ''}></label>`;
    }
    return `<label class="field" data-name="${p.name}">${label}` +
      `<input type="number" name="${p.name}" value="${val}" min="${p.min ?? ''}" max="${p.max ?? ''}" step="${p.step ?? 'any'}"></label>`;
  }

  function renderParams(container, params, grid = true) {
    container.innerHTML = grid ? `<div class="params-grid">${params.map(p => fieldHtml(p)).join('')}</div>`
                               : params.map(p => fieldHtml(p)).join('');
  }

  function setParams(container, values) {
    if (!values) return;
    container.querySelectorAll('input, select').forEach(el => {
      if (!(el.name in values)) return;
      if (el.type === 'checkbox') el.checked = !!values[el.name];
      else el.value = values[el.name];
    });
  }

  function readParams(container) {
    const out = {};
    container.querySelectorAll('input, select').forEach(el => {
      if (el.type === 'checkbox') out[el.name] = el.checked;
      else if (el.type === 'number') out[el.name] = el.value === '' ? null : +el.value;
      else out[el.name] = isNaN(+el.value) || el.value === '' ? el.value : +el.value;
    });
    return out;
  }

  // ---------- scenario tab ----------

  const scenarioSelect = $('scenario-select');
  // Custom first, the rest alphabetical.
  catalog.scenarios.sort((a, b) =>
    a.id === 'custom' ? -1 : b.id === 'custom' ? 1 : a.name.localeCompare(b.name));
  catalog.scenarios.forEach(s => {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = s.name;
    scenarioSelect.appendChild(o);
  });
  scenarioSelect.value = 'small_office';

  renderParams($('global-params'), catalog.globals);

  function currentScenarioEntry() {
    return catalog.scenarios.find(s => s.id === scenarioSelect.value);
  }

  function renderScenarioParams() {
    const entry = currentScenarioEntry();
    $('scenario-description').textContent = entry.description;
    renderParams($('scenario-params'), entry.params);
    $('btn-customize').style.display = entry.id === 'custom' ? 'none' : '';
    $('topo-help').style.display = entry.id === 'custom' ? '' : 'none';
    $('topo-mode-badge').textContent = entry.id === 'custom' ? 'custom — editable' : 'preview';
    $('topo-mode-badge').classList.toggle('custom', entry.id === 'custom');
  }

  function scenarioConfig() {
    return {
      id: scenarioSelect.value,
      params: readParams($('scenario-params')),
      custom: Topology.getCustom(),
      globals: readParams($('global-params')),
    };
  }

  let lastPreview = null;

  async function refreshPreview() {
    if (scenarioSelect.value === 'custom') {
      Topology.setEditable();
      return;
    }
    const res = await (await fetch('/api/scenario/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scenarioConfig()),
    })).json();
    if (res.error) {
      $('scenario-description').textContent = `⚠ ${res.error}`;
      return;
    }
    lastPreview = res;
    Topology.setPreview(res);
  }

  function onScenarioChanged() {
    // The comparison is only meaningful on one fixed scenario: any change
    // stops running simulations and resets the chart.
    stopAll();
    resetChart();
    refreshPreview();
  }

  scenarioSelect.addEventListener('change', () => { renderScenarioParams(); onScenarioChanged(); });
  $('scenario-params').addEventListener('change', onScenarioChanged);
  $('global-params').addEventListener('change', onScenarioChanged);

  $('btn-customize').addEventListener('click', () => {
    if (!lastPreview) return;
    Topology.fromPreview(lastPreview);
    scenarioSelect.value = 'custom';
    renderScenarioParams();
    stopAll();
    resetChart();
  });
  $('btn-undo').addEventListener('click', () => Topology.undo());
  $('btn-clear').addEventListener('click', () => Topology.clear());
  $('btn-wall-mode').addEventListener('click', () => {
    Topology.setWallMode(!Topology.isWallMode());
    $('btn-wall-mode').classList.toggle('active', Topology.isWallMode());
  });
  Topology.setOnChange(() => { stopAll(); resetChart(); });

  // ---------- method tab ----------

  const methodSelect = $('method-select');
  catalog.methods.forEach(m => {
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = m.name;
    methodSelect.appendChild(o);
  });

  function currentMethod() {
    return catalog.methods.find(m => m.id === methodSelect.value);
  }

  function renderAgentParams() {
    const m = currentMethod();
    const container = $('agent-params');
    if (!m.agent_defaults) { container.innerHTML = ''; return; }
    const agent = $('method-params').querySelector('[name=agent_type]')?.value || 'UCB';
    const defaults = m.agent_defaults[agent];
    const tips = m.agent_param_tooltips || {};
    const levelNames = { lvl1: 'Level 1 — AP selection', lvl2: 'Level 2 — station selection', lvl3: 'Level 3 — TX power' };
    container.innerHTML = m.levels.map(lvl => {
      const key = (m.id === 'fmab') ? 'flat' : lvl;
      const fields = Object.entries(defaults[key]).map(([name, def]) =>
        fieldHtml({ name: `${lvl}:${name}`, label: name, type: 'number', default: def, tooltip: tips[name] || '', step: 'any' })
      ).join('');
      const title = m.id === 'fmab' ? 'Agent parameters' : levelNames[lvl];
      return `<div class="level-block"><h4>${title}</h4><div class="params-grid">${fields}</div></div>`;
    }).join('');
  }

  function renderMethodParams() {
    const m = currentMethod();
    $('method-description').textContent = m.description;
    renderParams($('method-params'), m.params);
    renderAgentParams();
  }

  methodSelect.addEventListener('change', renderMethodParams);
  $('method-params').addEventListener('change', (e) => {
    if (e.target.name === 'agent_type') renderAgentParams();
  });

  function methodParams() {
    const params = readParams($('method-params'));
    const m = currentMethod();
    if (m.agent_defaults) {
      const byLevel = {};
      $('agent-params').querySelectorAll('input').forEach(el => {
        const [lvl, name] = el.name.split(':');
        (byLevel[lvl] = byLevel[lvl] || {})[name] = +el.value;
      });
      for (const [lvl, vals] of Object.entries(byLevel)) params[`params_${lvl}`] = vals;
    }
    return params;
  }

  // ---------- runs ----------

  function runLabel(m, params) {
    runCounter += 1;
    const variant = params.agent_type || (params.algorithm ? params.algorithm.toUpperCase() : '');
    const suffix = variant ? ` (${variant})` : '';
    return `${m.name}${suffix} #${runCounter}`;
  }

  function addRunItem(runId, label, color, meta, imported = false) {
    const div = document.createElement('div');
    div.className = 'run-item';
    div.innerHTML = `<span class="swatch" style="background:${color}"></span>` +
      `<span class="label" title="${label}">${label}</span>` +
      `<span class="state">${imported ? 'loaded' : 'queued'}</span>` +
      (imported ? '' : `<button title="Stop this run">stop</button>`);
    if (!imported) {
      div.querySelector('button').addEventListener('click', () => {
        fetch('/api/run/stop', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ run_id: runId }),
        });
      });
    }
    $('run-list').prepend(div);
    runs.set(runId, { label, el: div, stateEl: div.querySelector('.state'), ...meta });
    const pending = pendingMsgs.get(runId);
    if (pending) {
      pendingMsgs.delete(runId);
      pending.forEach(handleMsg);
    }
  }

  async function startRun() {
    const m = currentMethod();
    const params = methodParams();
    const label = runLabel(m, params);
    const res = await (await fetch('/api/run/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method: m.id, params, scenario: scenarioConfig() }),
    })).json();
    const color = Chart.addRun(res.run_id, label, m.kind);
    addRunItem(res.run_id, label, color, { method: m.id, method_name: m.name, kind: m.kind, params });
  }

  function stopAll() {
    fetch('/api/run/stop', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
  }

  function resetChart() {
    Chart.reset();
    runs.clear();
    pendingMsgs.clear();
    runCounter = 0;
    $('run-list').innerHTML = '';
  }

  $('btn-start').addEventListener('click', startRun);
  $('btn-stop').addEventListener('click', stopAll);
  $('btn-reset').addEventListener('click', resetChart);

  // ---------- save / load ----------
  // Everything needed to reproduce a comparison lives in the browser already:
  // the scenario form, the per-run method parameters, and the streamed points.

  const FILE_FORMAT = 'jaxwifi-demo-results';

  function snapshot() {
    return {
      format: FILE_FORMAT,
      version: 1,
      created: new Date().toISOString(),
      scenario: scenarioConfig(),
      runs: [...runs.entries()].map(([runId, r]) => ({
        label: r.label,
        method: r.method,
        method_name: r.method_name,
        kind: r.kind,
        state: r.stateEl.textContent,
        params: r.params,
        ...Chart.getSeries(runId),
      })).reverse(),  // chronological, the run list is newest-first
    };
  }

  function saveResults() {
    const data = snapshot();
    const stamp = data.created.replace(/[:.]/g, '-').slice(0, 19);
    const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `jaxwifi-${data.scenario.id}-${stamp}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // Restores the scenario (and optionally the recorded runs) from a snapshot.
  // Exposed on window so it can also be driven from the browser console.
  function applySnapshot(data, withRuns) {
    if (data.format !== FILE_FORMAT) throw new Error(`not a ${FILE_FORMAT} file`);
    stopAll();
    resetChart();

    const s = data.scenario || {};
    if (s.id) {
      scenarioSelect.value = s.id;
      renderScenarioParams();
      setParams($('scenario-params'), s.params);
      setParams($('global-params'), s.globals);
    }
    if (s.custom && s.custom.aps) Topology.setCustom(s.custom.aps, s.custom.walls || [], false);
    refreshPreview();

    // Preload the method panel with the first recorded run, so the loaded
    // configuration can be re-run as it was.
    const first = (data.runs || [])[0];
    if (first && catalog.methods.some(m => m.id === first.method)) {
      methodSelect.value = first.method;
      renderMethodParams();
      setParams($('method-params'), first.params);
      renderAgentParams();
      for (const [key, vals] of Object.entries(first.params || {})) {
        if (!key.startsWith('params_')) continue;
        const lvl = key.slice('params_'.length);
        for (const [name, v] of Object.entries(vals)) {
          const el = $('agent-params').querySelector(`[name="${lvl}:${name}"]`);
          if (el) el.value = v;
        }
      }
    }

    if (!withRuns) return 0;
    // Imported runs are inert (no backend run to stop), so they get negative ids.
    let importId = -1;
    for (const r of data.runs || []) {
      runCounter += 1;
      const color = Chart.loadRun(importId, r.label, r.kind, r);
      addRunItem(importId, r.label, color,
        { method: r.method, method_name: r.method_name, kind: r.kind, params: r.params }, true);
      importId -= 1;
    }
    return (data.runs || []).length;
  }
  window.applySnapshot = applySnapshot;

  $('btn-save').addEventListener('click', saveResults);
  $('btn-load').addEventListener('click', () => $('file-load').click());
  $('file-load').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';  // allow re-loading the same file
    try {
      const data = JSON.parse(await file.text());
      const n = (data.runs || []).length;
      const withRuns = n > 0 && confirm(`Loaded scenario from ${file.name}.\n\nAlso load its ${n} recorded run(s) into the chart?`);
      applySnapshot(data, withRuns);
    } catch (err) {
      alert(`Could not load ${file.name}: ${err.message}`);
    }
  });

  // ---------- EMA ----------

  $('ema-slider').addEventListener('input', (e) => {
    const v = +e.target.value;
    $('ema-value').textContent = v.toFixed(2);
    Chart.setEma(v);
  });

  // ---------- tabs ----------

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      $(`tab-${tab.dataset.tab}`).classList.add('active');
    });
  });

  // ---------- tooltips ----------
  // Custom tooltip layer: shows instantly on hover/click of ⓘ markers, is not
  // clipped by the scrolling config column, and works on touch.

  const tipEl = document.createElement('div');
  tipEl.id = 'tooltip';
  document.body.appendChild(tipEl);

  function showTip(target) {
    tipEl.textContent = target.dataset.tip;
    tipEl.style.display = 'block';
    const r = target.getBoundingClientRect();
    const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - tw / 2), innerWidth - tw - 8);
    let y = r.bottom + 8;
    if (y + th > innerHeight - 8) y = r.top - th - 8;
    tipEl.style.left = `${x}px`;
    tipEl.style.top = `${y}px`;
  }

  document.addEventListener('pointerover', (e) => {
    const t = e.target.closest('.info[data-tip]');
    if (t) showTip(t);
    else tipEl.style.display = 'none';
  });
  document.addEventListener('click', (e) => {
    const t = e.target.closest('.info[data-tip]');
    if (t) { e.preventDefault(); showTip(t); }
  }, true);

  // ---------- WebSocket ----------

  function connectWs() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      handleMsg(msg);
    };
    ws.onclose = () => setTimeout(connectWs, 1500);
    setInterval(() => { if (ws.readyState === 1) ws.send('ping'); }, 20_000);
  }

  function handleMsg(msg) {
      const run = runs.get(msg.run_id);
      if (!run) {
        // The run's first messages can beat the /api/run/start response;
        // stash them and replay once the run is registered.
        const q = pendingMsgs.get(msg.run_id) || [];
        if (q.length < 1000) q.push(msg);
        pendingMsgs.set(msg.run_id, q);
        return;
      }
      if (msg.type === 'points') Chart.addPoints(msg.run_id, msg.points);
      else if (msg.type === 'point') Chart.addPoints(msg.run_id, [msg]);
      else if (msg.type === 'hline') Chart.setHline(msg.run_id, msg.value, msg.ci_low, msg.ci_high, msg.config);
      else if (msg.type === 'status_msg') run.stateEl.textContent = msg.msg;
      else if (msg.type === 'status') {
        run.stateEl.textContent = msg.state === 'error' ? `error: ${msg.msg}` : msg.state;
        run.stateEl.className = `state ${msg.state}`;
      }
  }
  connectWs();

  // ---------- theme changes ----------

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    Chart.render();
    Topology.render();
  });

  // ---------- init ----------

  renderScenarioParams();
  renderMethodParams();
  Chart.render();
  refreshPreview();
})();
