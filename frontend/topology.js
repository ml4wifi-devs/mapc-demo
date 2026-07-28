/* SVG topology editor: right-click places an AP, left-click places a STA for the
   selected AP; drag moves nodes, double-click deletes. In preview (read-only)
   mode it renders a server-side scenario. */
'use strict';

const Topology = (() => {
  const svg = document.getElementById('topology');
  const NS = 'http://www.w3.org/2000/svg';
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const bssColor = (i) => cssVar(`--series-${(i % 8) + 1}`);

  // Custom topology state: [{x, y, stas: [{x, y}]}], walls: [[x1,y1,x2,y2], ...]
  let aps = [];
  let walls = [];
  let selectedAp = -1;
  let editable = true;
  let wallMode = false;
  let preview = null; // {pos, associations, walls_pos}
  let onChange = () => {};
  let dragging = null;
  let drawingWall = null; // {x1, y1}

  // World<->screen transform.
  let view = { x0: -10, y0: -10, x1: 60, y1: 60 };

  function bounds() {
    const pts = [];
    if (preview) {
      preview.pos.forEach(p => pts.push(p));
      preview.walls_pos.forEach(w => { pts.push([w[0], w[1]]); pts.push([w[2], w[3]]); });
    } else {
      aps.forEach(ap => { pts.push([ap.x, ap.y]); ap.stas.forEach(s => pts.push([s.x, s.y])); });
      walls.forEach(w => { pts.push([w[0], w[1]]); pts.push([w[2], w[3]]); });
    }
    if (!pts.length) return { x0: -10, y0: -10, x1: 60, y1: 60 };
    let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
    let y0 = Math.min(...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1]));
    const pad = Math.max(5, (x1 - x0) * 0.15, (y1 - y0) * 0.15);
    return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
  }

  function fitView() {
    const b = bounds();
    const rect = svg.getBoundingClientRect();
    const aspect = rect.width / Math.max(rect.height, 1);
    let w = b.x1 - b.x0, h = b.y1 - b.y0;
    if (w / h > aspect) h = w / aspect; else w = h * aspect;
    const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    view = { x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2 };
  }

  const sx = (x) => (x - view.x0) / (view.x1 - view.x0) * svg.getBoundingClientRect().width;
  const sy = (y) => {
    const rect = svg.getBoundingClientRect();
    return rect.height - (y - view.y0) / (view.y1 - view.y0) * rect.height;
  };
  function toWorld(evt) {
    const rect = svg.getBoundingClientRect();
    const px = evt.clientX - rect.left, py = evt.clientY - rect.top;
    return {
      x: view.x0 + px / rect.width * (view.x1 - view.x0),
      y: view.y0 + (rect.height - py) / rect.height * (view.y1 - view.y0),
    };
  }

  function make(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  function niceStep(range) {
    const raw = range / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
    return 10 * mag;
  }

  function drawGrid() {
    const g = make('g', { class: 'grid' });
    const gridColor = cssVar('--grid');
    const textColor = cssVar('--text-muted');
    const step = niceStep(view.x1 - view.x0);
    for (let x = Math.ceil(view.x0 / step) * step; x <= view.x1; x += step) {
      g.appendChild(make('line', { x1: sx(x), y1: 0, x2: sx(x), y2: '100%', stroke: gridColor, 'stroke-width': 1 }));
      const t = make('text', { x: sx(x) + 3, y: svg.getBoundingClientRect().height - 5, fill: textColor, 'font-size': 10 });
      t.textContent = `${Math.round(x * 10) / 10}`;
      g.appendChild(t);
    }
    for (let y = Math.ceil(view.y0 / step) * step; y <= view.y1; y += step) {
      g.appendChild(make('line', { x1: 0, y1: sy(y), x2: '100%', y2: sy(y), stroke: gridColor, 'stroke-width': 1 }));
      const t = make('text', { x: 4, y: sy(y) - 3, fill: textColor, 'font-size': 10 });
      t.textContent = `${Math.round(y * 10) / 10} m`;
      g.appendChild(t);
    }
    svg.appendChild(g);
  }

  function drawApMarker(g, x, y, color, selected, big) {
    const r = big ? 8 : 7;
    if (selected) {
      g.appendChild(make('circle', { cx: x, cy: y, r: r + 6, fill: 'none', stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }));
    }
    g.appendChild(make('line', { x1: x - r, y1: y - r, x2: x + r, y2: y + r, stroke: color, 'stroke-width': 3.5, 'stroke-linecap': 'round' }));
    g.appendChild(make('line', { x1: x - r, y1: y + r, x2: x + r, y2: y - r, stroke: color, 'stroke-width': 3.5, 'stroke-linecap': 'round' }));
  }

  function render() {
    svg.innerHTML = '';
    fitView();
    drawGrid();
    const surface = cssVar('--surface-1');

    if (preview) {
      const apIds = Object.keys(preview.associations).map(Number);
      for (const w of preview.walls_pos) {
        svg.appendChild(make('line', {
          x1: sx(w[0]), y1: sy(w[1]), x2: sx(w[2]), y2: sy(w[3]),
          stroke: cssVar('--wall'), 'stroke-width': 4, 'stroke-linecap': 'round',
        }));
      }
      apIds.forEach((ap, i) => {
        const color = bssColor(i);
        const [ax, ay] = preview.pos[ap];
        for (const s of preview.associations[ap]) {
          const [sxp, syp] = preview.pos[s];
          svg.appendChild(make('line', { x1: sx(ax), y1: sy(ay), x2: sx(sxp), y2: sy(syp), stroke: color, 'stroke-width': 1, opacity: 0.5 }));
          svg.appendChild(make('circle', { cx: sx(sxp), cy: sy(syp), r: 4.5, fill: color, stroke: surface, 'stroke-width': 1.5 }));
        }
        drawApMarker(svg, sx(ax), sy(ay), color, false, false);
      });
      return;
    }

    walls.forEach((w, i) => {
      svg.appendChild(make('line', {
        x1: sx(w[0]), y1: sy(w[1]), x2: sx(w[2]), y2: sy(w[3]),
        stroke: cssVar('--wall'), 'stroke-width': 4, 'stroke-linecap': 'round',
      }));
      svg.appendChild(make('line', {
        x1: sx(w[0]), y1: sy(w[1]), x2: sx(w[2]), y2: sy(w[3]),
        stroke: 'transparent', 'stroke-width': 14, cursor: 'pointer', 'data-wall': i,
      }));
    });

    aps.forEach((ap, i) => {
      const color = bssColor(i);
      ap.stas.forEach((s, j) => {
        svg.appendChild(make('line', { x1: sx(ap.x), y1: sy(ap.y), x2: sx(s.x), y2: sy(s.y), stroke: color, 'stroke-width': 1, opacity: 0.5 }));
        const c = make('circle', {
          cx: sx(s.x), cy: sy(s.y), r: 6, fill: color, stroke: surface, 'stroke-width': 2, cursor: 'move',
          'data-ap': i, 'data-sta': j,
        });
        svg.appendChild(c);
      });
      const g = make('g', { cursor: 'move', 'data-ap': i, 'data-sta': -1 });
      drawApMarker(g, sx(ap.x), sy(ap.y), color, i === selectedAp, true);
      // Transparent hit area on top of the X marker.
      g.appendChild(make('circle', { cx: sx(ap.x), cy: sy(ap.y), r: 14, fill: 'transparent' }));
      svg.appendChild(g);
    });
  }

  function hit(evt) {
    let t = evt.target;
    while (t && t !== svg) {
      if (t.dataset && t.dataset.ap !== undefined) {
        return { ap: +t.dataset.ap, sta: +t.dataset.sta };
      }
      if (t.dataset && t.dataset.wall !== undefined) {
        return { wall: +t.dataset.wall };
      }
      t = t.parentNode;
    }
    return null;
  }

  svg.addEventListener('contextmenu', (evt) => {
    evt.preventDefault();
    if (!editable) return;
    const p = toWorld(evt);
    aps.push({ x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10, stas: [] });
    selectedAp = aps.length - 1;
    render();
    onChange();
  });

  // Native dblclick is unreliable here: selecting a node re-renders the SVG
  // between the two clicks, which resets the browser's double-click tracking.
  let lastTap = null; // {ap, sta, t}
  let suppressClick = false;

  function deleteNode(h) {
    if (h.wall !== undefined) {
      walls.splice(h.wall, 1);
    } else if (h.sta === -1) {
      aps.splice(h.ap, 1);
      if (selectedAp >= aps.length) selectedAp = aps.length - 1;
    } else {
      aps[h.ap].stas.splice(h.sta, 1);
    }
    render();
    onChange();
  }

  function sameHit(a, b) {
    if (a.wall !== undefined || b.wall !== undefined) return a.wall === b.wall;
    return a.ap === b.ap && a.sta === b.sta;
  }

  svg.addEventListener('pointerdown', (evt) => {
    if (evt.button !== 0 || !editable) return;
    const h = hit(evt);
    if (wallMode && !h) {
      const p = toWorld(evt);
      drawingWall = { x1: Math.round(p.x * 10) / 10, y1: Math.round(p.y * 10) / 10 };
      suppressClick = true;
      svg.setPointerCapture(evt.pointerId);
      return;
    }
    if (!h) { lastTap = null; return; }
    const now = performance.now();
    // Pointer capture retargets the trailing click to the svg itself, so the
    // click handler cannot tell it started on a node — suppress it here.
    suppressClick = true;
    if (lastTap && sameHit(lastTap, h) && now - lastTap.t < 400) {
      lastTap = null;
      deleteNode(h);
      return;
    }
    lastTap = { ...h, t: now };
    if (h.wall !== undefined) return; // walls: select-to-delete only, no drag-move
    dragging = { ...h, moved: false };
    if (h.sta === -1) { selectedAp = h.ap; render(); }
    svg.setPointerCapture(evt.pointerId);
  });

  svg.addEventListener('pointermove', (evt) => {
    const p = toWorld(evt);
    document.getElementById('topo-coords').textContent =
      editable ? `x: ${p.x.toFixed(1)} m, y: ${p.y.toFixed(1)} m` : '';
    if (drawingWall) {
      render();
      const x2 = Math.round(p.x * 10) / 10, y2 = Math.round(p.y * 10) / 10;
      svg.appendChild(make('line', {
        x1: sx(drawingWall.x1), y1: sy(drawingWall.y1), x2: sx(x2), y2: sy(y2),
        stroke: cssVar('--wall'), 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-dasharray': '6 4',
      }));
      return;
    }
    if (!dragging) return;
    dragging.moved = true;
    const target = dragging.sta === -1 ? aps[dragging.ap] : aps[dragging.ap].stas[dragging.sta];
    target.x = Math.round(p.x * 10) / 10;
    target.y = Math.round(p.y * 10) / 10;
    render();
  });

  svg.addEventListener('pointerup', (evt) => {
    if (drawingWall) {
      const p = toWorld(evt);
      const x2 = Math.round(p.x * 10) / 10, y2 = Math.round(p.y * 10) / 10;
      const len = Math.hypot(x2 - drawingWall.x1, y2 - drawingWall.y1);
      if (len >= 0.5) {
        walls.push([drawingWall.x1, drawingWall.y1, x2, y2]);
        onChange();
      }
      drawingWall = null;
      render();
      return;
    }
    if (dragging) {
      if (dragging.moved) onChange();
      dragging = null;
    }
  });

  svg.addEventListener('click', (evt) => {
    if (suppressClick) { suppressClick = false; return; }
    if (!editable || dragging || wallMode) return;
    if (hit(evt)) return; // click on node = select, handled in pointerdown
    if (!aps.length) return;
    const p = toWorld(evt);
    if (selectedAp < 0) selectedAp = aps.length - 1;
    aps[selectedAp].stas.push({ x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 });
    render();
    onChange();
  });

  window.addEventListener('resize', render);

  return {
    setPreview(p) { preview = p; editable = false; render(); },
    setEditable() { preview = null; editable = true; render(); },
    isEditable: () => editable,
    getCustom: () => ({ aps, walls }),
    setCustom(newAps, newWalls = [], select = true) {
      aps = newAps;
      walls = newWalls;
      selectedAp = select ? aps.length - 1 : -1;
      preview = null;
      editable = true;
      render();
    },
    fromPreview(p) {
      // Convert a server preview into an editable custom topology.
      const apIds = Object.keys(p.associations).map(Number);
      aps = apIds.map(ap => ({
        x: p.pos[ap][0], y: p.pos[ap][1],
        stas: p.associations[ap].map(s => ({ x: p.pos[s][0], y: p.pos[s][1] })),
      }));
      walls = p.walls_pos.map(w => [...w]);
      selectedAp = -1;
      preview = null;
      editable = true;
      render();
    },
    undo() {
      if (!editable || !aps.length) return;
      const ap = aps[selectedAp >= 0 ? selectedAp : aps.length - 1];
      if (ap && ap.stas.length) ap.stas.pop();
      else { aps.pop(); selectedAp = aps.length - 1; }
      render();
      onChange();
    },
    clear() { aps = []; walls = []; selectedAp = -1; render(); onChange(); },
    setWallMode(v) { wallMode = v; drawingWall = null; },
    isWallMode: () => wallMode,
    setOnChange(fn) { onChange = fn; },
    render,
  };
})();
