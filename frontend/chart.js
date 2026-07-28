/* Plotly chart wrapper: multiple runs, EMA smoothing, horizontal lines with CI bands. */
'use strict';

const Chart = (() => {
  const el = () => document.getElementById('chart');
  const runs = new Map(); // run_id -> {label, color, kind, x:[], y:[], hline, ci}
  let emaAlpha = 0.9;

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const PALETTE_SIZE = 8;
  let colorCounter = 0;

  function nextColor() {
    const idx = (colorCounter % PALETTE_SIZE) + 1;
    colorCounter += 1;
    return cssVar(`--series-${idx}`);
  }

  function ema(values, alpha) {
    if (alpha <= 0 || values.length === 0) return values;
    const out = new Array(values.length);
    out[0] = values[0];
    for (let i = 1; i < values.length; i++) {
      out[i] = alpha * out[i - 1] + (1 - alpha) * values[i];
    }
    return out;
  }

  function fmtLinks(links) {
    if (!links) return '';
    if (!links.length) return '(no active links)';
    return links.map(l => `AP${l.ap}→STA${l.sta}: ${l.tx_power.toFixed(1)}dBm, MCS${l.mcs} (${l.rate.toFixed(1)}Mb/s)`).join('<br>');
  }

  function fmtConfig(config) {
    if (!config) return '';
    if (Array.isArray(config)) {
      // T/F-Optimal: time-shared list of {share, links}.
      return config.map(c => `<b>${(c.share * 100).toFixed(0)}% of time</b><br>${fmtLinks(c.links)}`).join('<br><br>');
    }
    return fmtLinks(config.links);
  }

  function hexToRgba(hex, a) {
    const v = hex.replace('#', '');
    const r = parseInt(v.slice(0, 2), 16), g = parseInt(v.slice(2, 4), 16), b = parseInt(v.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  function layout() {
    const text = cssVar('--text-primary');
    const secondary = cssVar('--text-secondary');
    const grid = cssVar('--grid');
    const surface = cssVar('--surface-1');
    let maxStep = 100;
    for (const r of runs.values()) {
      if (r.x.length) maxStep = Math.max(maxStep, r.x[r.x.length - 1]);
    }
    return {
      margin: { l: 64, r: 16, t: 8, b: 46 },
      paper_bgcolor: surface,
      plot_bgcolor: surface,
      font: { color: secondary, size: 12.5 },
      xaxis: {
        title: { text: 'Transmission opportunity (step)', font: { color: text, size: 13 } },
        gridcolor: grid, zerolinecolor: grid, ticks: 'outside', tickcolor: grid,
        rangemode: 'tozero',
      },
      yaxis: {
        title: { text: 'Effective data rate [Mb/s]', font: { color: text, size: 13 } },
        gridcolor: grid, zerolinecolor: grid, ticks: 'outside', tickcolor: grid,
        rangemode: 'tozero',
      },
      legend: { orientation: 'h', y: 1.12, x: 0, font: { color: text } },
      hovermode: 'x unified',
      shapes: [],
      annotations: [],
      datarevision: Date.now(),
    };
  }

  function traces() {
    const out = [];
    const lay = layout();

    for (const [id, r] of runs) {
      if (r.kind === 'hline') {
        if (r.hline === null) continue;
        // CI band via a full-width translucent rect, line via shape; dummy trace for the legend.
        if (r.ci && r.ci[1] > r.ci[0]) {
          lay.shapes.push({
            type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: r.ci[0], y1: r.ci[1],
            fillcolor: hexToRgba(r.color, 0.13), line: { width: 0 }, layer: 'below',
          });
        }
        lay.shapes.push({
          type: 'line', xref: 'paper', x0: 0, x1: 1, y0: r.hline, y1: r.hline,
          line: { color: r.color, width: 2, dash: 'dash' },
        });
        const hlineTip = fmtConfig(r.hlineConfig);
        out.push({
          x: [null], y: [null], mode: 'lines', name: `${r.label} (${r.hline.toFixed(1)} Mb/s)`,
          line: { color: r.color, width: 2, dash: 'dash' },
          ...(hlineTip
            ? { customdata: [hlineTip], hovertemplate: '%{customdata}<extra>' + r.label + '</extra>' }
            : { hoverinfo: 'skip' }),
        });
      } else {
        const smooth = emaAlpha > 0;
        if (smooth && r.y.length > 1) {
          out.push({
            x: r.x, y: r.y, mode: 'lines', name: `${r.label} (raw)`,
            line: { color: hexToRgba(r.color, 0.25), width: 1 },
            hoverinfo: 'skip', showlegend: false,
          });
        }
        out.push({
          x: r.x, y: smooth ? ema(r.y, emaAlpha) : r.y, mode: 'lines', name: r.label,
          line: { color: r.color, width: 2 },
          customdata: r.config.map(c => fmtLinks(c && c.links)),
          hovertemplate: '%{y:.1f} Mb/s<br>%{customdata}<extra>' + r.label + '</extra>',
        });
      }
    }
    return { traces: out, layout: lay };
  }

  let resizeObs = null;

  function render() {
    const { traces: t, layout: lay } = traces();
    Plotly.react(el(), t, lay, { responsive: true, displaylogo: false });
    if (!resizeObs) {
      // The chart fills a flexible panel; follow container resizes, not just window ones.
      resizeObs = new ResizeObserver(() => Plotly.Plots.resize(el()));
      resizeObs.observe(el());
    }
  }

  return {
    addRun(runId, label, kind) {
      runs.set(runId, { label, color: nextColor(), kind, x: [], y: [], config: [], hline: null, ci: null, hlineConfig: null });
      render();
      return runs.get(runId).color;
    },
    addPoints(runId, points) {
      const r = runs.get(runId);
      if (!r) return;
      for (const p of points) { r.x.push(p.step); r.y.push(p.thr); r.config.push(p.config || null); }
      render();
    },
    setHline(runId, value, ciLow, ciHigh, config) {
      const r = runs.get(runId);
      if (!r) return;
      r.hline = value;
      if (ciLow !== undefined && ciHigh !== undefined) r.ci = [ciLow, ciHigh];
      if (config !== undefined) r.hlineConfig = config;
      render();
    },
    setEma(alpha) { emaAlpha = alpha; render(); },
    getSeries(runId) {
      const r = runs.get(runId);
      return r ? {
        steps: r.x.slice(), data_rate: r.y.slice(), config: r.config.slice(),
        hline: r.hline, ci: r.ci, hline_config: r.hlineConfig,
      } : null;
    },
    loadRun(runId, label, kind, series) {
      const color = nextColor();
      runs.set(runId, {
        label, color, kind,
        x: (series.steps || []).slice(), y: (series.data_rate || []).slice(),
        config: (series.config || []).slice(),
        hline: series.hline ?? null, ci: series.ci ?? null, hlineConfig: series.hline_config ?? null,
      });
      render();
      return color;
    },
    reset() { runs.clear(); colorCounter = 0; render(); },
    hasRuns() { return runs.size > 0; },
    render,
  };
})();
