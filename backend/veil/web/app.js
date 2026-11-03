'use strict';
/* Veil editor. Plain JavaScript, no build step. All image work happens on the server; this page only
   shows pictures, collects selections, and sends edit requests. */

const $ = (s) => document.querySelector(s);
const SLIDERS = [['brightness', 'Brightness'], ['contrast', 'Contrast'], ['saturation', 'Saturation'],
                 ['warmth', 'Warmth'], ['sharpness', 'Sharpness']];
const ASPECTS = [['free', 'Free'], ['orig', 'Original'], ['1:1', '1:1'], ['4:5', '4:5'], ['5:4', '5:4'],
                 ['3:2', '3:2'], ['2:3', '2:3'], ['16:9', '16:9']];
const SEL_COLOR = 'rgba(196, 87, 126, 0.5)';

const S = {
  photo: null, tool: 'view', imgW: 0, imgH: 0, frameW: 0, frameH: 0,
  adj: {}, previewSeq: 0, previewTimer: 0, previewUrl: null,
  sel: { points: [], boxes: [], brush: null, tint: null, painted: false, erase: false, size: 28 },
  crop: null, aspect: 'free', drag: null,
  compare: false, split: 0.5, holding: false,
  zoom: 1, panX: 0, panY: 0, k0: 1, pointers: new Map(), gesture: null, panning: null, space: false, block: false,
};

// ---------------------------------------------------------------- helpers
async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    const err = new Error(typeof msg === 'string' ? msg : 'Something went wrong.');
    err.status = res.status;
    throw err;
  }
  return res;
}
const postJSON = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

let toastTimer = 0;
function toast(msg, bad = false) {
  const t = $('#toast');
  t.textContent = msg; t.classList.toggle('bad', bad); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 4000);
}
const fail = (e) => toast(e.message || 'Something went wrong.', true);

// ---------------------------------------------------------------- screens
const SCREENS = ['#screen-setup', '#screen-login', '#screen-notice', '#screen-app'];
function show(id) { SCREENS.forEach((s) => { $(s).hidden = s !== id; }); }

async function boot() {
  const st = await (await fetch('/api/status', { credentials: 'same-origin' })).json();
  if (st.setup_needed) return show('#screen-setup');
  if (!st.logged_in) return show('#screen-login');
  if (!st.adult_notice_acknowledged) return show('#screen-notice');
  show('#screen-app');
  await loadLibrary(true);
}

function wireForm(sel, url) {
  $(sel).addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target; const err = f.querySelector('.error'); err.textContent = '';
    try {
      await postJSON(url, { username: f.username.value, password: f.password.value });
      f.password.value = '';
      await boot();
    } catch (ex) { err.textContent = ex.message; }
  });
}
wireForm('#form-setup', '/api/auth/setup');
wireForm('#form-login', '/api/auth/login');
$('#btn-ack').addEventListener('click', async () => {
  try { await postJSON('/api/notice/ack', {}); await boot(); } catch (ex) { $('#screen-notice .error').textContent = ex.message; }
});
$('#btn-logout').addEventListener('click', async () => { await postJSON('/api/auth/logout', {}); S.photo = null; await boot(); });

// ---------------------------------------------------------------- library and upload
async function loadLibrary(openLatest = false) {
  const list = await (await api('/api/photos')).json();
  const grid = $('#library-grid'); grid.replaceChildren();
  for (const p of list) {
    const cell = document.createElement('div'); cell.className = 'thumb' + (S.photo && S.photo.id === p.id ? ' cur' : '');
    cell.tabIndex = 0; cell.setAttribute('role', 'button'); cell.setAttribute('aria-label', 'Open ' + p.name);
    const img = document.createElement('img'); img.alt = p.name;
    img.src = `/api/photos/${p.id}/image?size=240&v=${p.cursor}-${p.versions.length}`;
    const del = document.createElement('button'); del.className = 'del'; del.textContent = 'Delete';
    let armed = 0;
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!armed) { armed = setTimeout(() => { armed = 0; del.textContent = 'Delete'; }, 3000); del.textContent = 'Sure?'; return; }
      try { await api('/api/photos/' + p.id, { method: 'DELETE' }); if (S.photo && S.photo.id === p.id) clearPhoto(); await loadLibrary(); } catch (ex) { fail(ex); }
    });
    const open = () => { openPhoto(p.id); $('#library').hidden = true; };
    cell.addEventListener('click', open);
    cell.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    cell.append(img, del); grid.append(cell);
  }
  if (openLatest && list.length && !S.photo) openPhoto(list[0].id);
}

async function upload(file) {
  if (!file) return;
  const fd = new FormData(); fd.append('file', file);
  try {
    toast('Uploading…');
    const meta = await (await api('/api/photos', { method: 'POST', body: fd })).json();
    toast('Photo added.');
    await openPhoto(meta.id); await loadLibrary();
  } catch (ex) { fail(ex); }
}
const fileInput = $('#file-input');
['#btn-open', '#btn-open2'].forEach((s) => $(s).addEventListener('click', () => fileInput.click()));
fileInput.addEventListener('change', () => { upload(fileInput.files[0]); fileInput.value = ''; });
$('#btn-library').addEventListener('click', async () => { $('#library').hidden = false; await loadLibrary(); });
$('#btn-library-close').addEventListener('click', () => { $('#library').hidden = true; });
const stage = $('#stage');
stage.addEventListener('dragover', (e) => { e.preventDefault(); stage.classList.add('drop'); });
stage.addEventListener('dragleave', () => stage.classList.remove('drop'));
stage.addEventListener('drop', (e) => { e.preventDefault(); stage.classList.remove('drop'); upload(e.dataTransfer.files[0]); });

// ---------------------------------------------------------------- opening a photo, drawing it
function clearPhoto() { S.photo = null; $('#frame').hidden = true; $('#empty').hidden = false; refreshUI(); }

async function openPhoto(id) {
  try { S.photo = await (await api('/api/photos/' + id)).json(); } catch (ex) { return fail(ex); }
  resetSelection(); S.crop = null; resetSliders(); setTool('view'); S.compare = false; S.zoom = 1; S.panX = S.panY = 0;
  await refresh(true);
}

function setAfter(url) { $('#img-after').src = url; }
function actualUrl() { const p = S.photo; return `/api/photos/${p.id}/image?version=current&v=${p.cursor}-${p.versions.length}-${p.width}`; }

async function refresh(firstLoad = false) {
  const p = S.photo; if (!p) return;
  const sizeChanged = p.width !== S.imgW || p.height !== S.imgH;
  $('#empty').hidden = true; $('#frame').hidden = false;
  await new Promise((resolve) => {
    const img = $('#img-after');
    img.onload = () => { S.imgW = img.naturalWidth; S.imgH = img.naturalHeight; resolve(); };
    img.onerror = resolve;
    setAfter(actualUrl());
  });
  if (firstLoad || sizeChanged) { resetSelection(); S.crop = null; S.zoom = 1; S.panX = S.panY = 0; if (S.tool === 'crop') initCrop(); }
  $('#img-before').src = `/api/photos/${p.id}/image?version=original&v=${p.versions.length}-${p.cursor}`;
  layout(); refreshUI();
}

function layout() {
  if (!S.photo || !S.imgW) return;
  const availW = Math.max(120, stage.clientWidth - 24), availH = Math.max(120, stage.clientHeight - 24);
  S.k0 = Math.min(availW / S.imgW, availH / S.imgH, 3);   // screen pixels per photo pixel when fitted
  S.zoom = Math.min(maxZoom(), Math.max(1, S.zoom));
  const k = S.k0 * S.zoom;
  S.frameW = Math.round(S.imgW * k); S.frameH = Math.round(S.imgH * k);
  const f = $('#frame'); f.style.width = S.frameW + 'px'; f.style.height = S.frameH + 'px';
  const b = $('#img-before'); b.style.width = S.frameW + 'px'; b.style.height = S.frameH + 'px';
  const c = $('#overlay'); const q = Math.min(window.devicePixelRatio || 1, 4096 / Math.max(S.frameW, S.frameH));
  c.width = Math.max(1, Math.round(S.frameW * q)); c.height = Math.max(1, Math.round(S.frameH * q));
  applyView(); applyCompare(); draw(); updateZoomUI();
}

// ---------------------------------------------------------------- zoom and pan
const maxZoom = () => Math.max(1, 8 / S.k0);          // up to 8 screen pixels per photo pixel
function clampPan() {
  if (S.zoom <= 1.0001) { S.panX = S.panY = 0; return; }
  const mx = Math.max(0, (stage.clientWidth + S.frameW) / 2 - 80), my = Math.max(0, (stage.clientHeight + S.frameH) / 2 - 80);
  S.panX = Math.min(mx, Math.max(-mx, S.panX)); S.panY = Math.min(my, Math.max(-my, S.panY));
}
function applyView() {
  clampPan();
  $('#frame').style.transform = `translate(calc(-50% + ${S.panX}px), calc(-50% + ${S.panY}px))`;
  stage.classList.toggle('pannable', !!S.photo && (S.tool === 'view' || S.space));
}
function updateZoomUI() {
  $('#zoombar').hidden = !S.photo; $('#zoom-pct').textContent = Math.round(S.k0 * S.zoom * 100) + '%';
  $('#zoom-out').disabled = S.zoom <= 1.0001; $('#zoom-in').disabled = S.zoom >= maxZoom() - 0.0001;
}
/** Zoom to `z`, keeping the photo point under (cx, cy) (client coordinates) where it is. */
function zoomAt(z, cx, cy) {
  const r = stage.getBoundingClientRect(); const oldW = S.frameW, oldH = S.frameH, oldZ = S.zoom;
  z = Math.min(maxZoom(), Math.max(1, z)); if (z === oldZ) return;
  const px = cx - r.left - r.width / 2, py = cy - r.top - r.height / 2;      // cursor relative to the stage centre
  const u = (px - S.panX) / oldW, v = (py - S.panY) / oldH;                  // photo point under the cursor, 0..1 from the frame centre
  S.zoom = z; const newW = S.imgW * S.k0 * z, newH = S.imgH * S.k0 * z;
  S.panX = px - u * newW; S.panY = py - v * newH;
  layout();
}
function zoomCentre(z) { const r = stage.getBoundingClientRect(); zoomAt(z, r.left + r.width / 2, r.top + r.height / 2); }
$('#zoom-in').addEventListener('click', () => zoomCentre(S.zoom * 1.5));
$('#zoom-out').addEventListener('click', () => zoomCentre(S.zoom / 1.5));
$('#zoom-fit').addEventListener('click', () => { S.zoom = 1; S.panX = S.panY = 0; layout(); });
$('#zoom-100').addEventListener('click', () => zoomCentre(1 / S.k0));
stage.addEventListener('wheel', (e) => {
  if (!S.photo || $('#frame').hidden) return;
  e.preventDefault();
  const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  zoomAt(S.zoom * Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX, e.clientY);
}, { passive: false });
stage.addEventListener('dblclick', (e) => {
  if (S.tool !== 'view' || !S.photo || e.target.closest('#divider, #zoombar')) return;
  S.zoom > 1.01 ? (S.zoom = 1, S.panX = S.panY = 0, layout()) : zoomAt(1 / S.k0, e.clientX, e.clientY);
});

// one finger (View tool), middle button or Space+drag pans; two fingers pinch-zoom and pan, whatever the tool
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const gap = (a, b) => Math.hypot(a.x - b.x, a.y - b.y) || 1;
stage.addEventListener('pointerdown', (e) => {
  if (!S.photo || $('#frame').hidden || e.target.closest('#divider, #zoombar')) return;
  S.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (S.pointers.size === 2) {
    const [a, b] = [...S.pointers.values()]; S.gesture = { d: gap(a, b), m: mid(a, b) };
    S.drag = null; S.panning = null; e.stopPropagation(); return;
  }
  if (S.pointers.size > 2) { e.stopPropagation(); return; }
  if (S.tool === 'view' || e.button === 1 || S.space) {
    S.panning = { id: e.pointerId, x: e.clientX, y: e.clientY, px: S.panX, py: S.panY };
    stage.setPointerCapture(e.pointerId); stage.classList.add('panning'); e.preventDefault(); e.stopPropagation();
  }
}, true);
stage.addEventListener('pointermove', (e) => {
  if (!S.pointers.has(e.pointerId)) return;
  S.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (S.gesture && S.pointers.size === 2) {
    const [a, b] = [...S.pointers.values()]; const d = gap(a, b), m = mid(a, b);
    zoomAt(S.zoom * d / S.gesture.d, m.x, m.y);
    S.panX += m.x - S.gesture.m.x; S.panY += m.y - S.gesture.m.y; applyView();
    S.gesture = { d, m }; e.stopPropagation(); return;
  }
  if (S.panning && S.panning.id === e.pointerId) {
    S.panX = S.panning.px + e.clientX - S.panning.x; S.panY = S.panning.py + e.clientY - S.panning.y; applyView(); e.stopPropagation();
  }
}, true);
function pointerGone(e) {
  if (!S.pointers.delete(e.pointerId)) return;
  if (S.gesture && S.pointers.size < 2) { S.gesture = null; S.block = S.pointers.size > 0; }  // the leftover finger must not start a stroke
  if (S.pointers.size === 0) S.block = false;
  if (S.panning && S.panning.id === e.pointerId) { S.panning = null; stage.classList.remove('panning'); }
}
stage.addEventListener('pointerup', pointerGone, true);
stage.addEventListener('pointercancel', pointerGone, true);
window.addEventListener('resize', layout);
new ResizeObserver(layout).observe(stage);

function refreshUI() {
  const p = S.photo; const has = !!p;
  $('#btn-undo').disabled = !has || p.cursor === 0;
  $('#btn-redo').disabled = !has || p.cursor >= p.versions.length - 1;
  $('#btn-compare').disabled = !has || p.cursor === 0;
  $('#btn-compare').setAttribute('aria-pressed', String(S.compare));
  $('#btn-export').disabled = !has;
  $('#btn-straighten').disabled = !has || Number($('#straighten').value) === 0;
  const dirty = SLIDERS.some(([k]) => S.adj[k]);
  $('#btn-adjust-apply').disabled = !has || !dirty; $('#btn-adjust-reset').disabled = !has || !dirty;
  $('#adjust-scope').textContent = hasMask() ? '(selected area only)' : '(whole photo)';
  const h = $('#history'); h.replaceChildren();
  if (has) p.versions.forEach((v, i) => {
    const li = document.createElement('li'); li.textContent = v.label;
    if (i === p.cursor) li.className = 'cur'; else if (i > p.cursor) li.className = 'undone';
    li.addEventListener('click', () => step(() => postJSON(`/api/photos/${p.id}/goto`, { cursor: i })));
    h.append(li);
  });
}

// ---------------------------------------------------------------- history buttons
async function step(call) {
  try { S.photo = await (await call()).json(); resetSliders(); await refresh(); } catch (ex) { fail(ex); }
}
const undo = () => S.photo && S.photo.cursor > 0 && step(() => postJSON(`/api/photos/${S.photo.id}/undo`, {}));
const redo = () => S.photo && S.photo.cursor < S.photo.versions.length - 1 && step(() => postJSON(`/api/photos/${S.photo.id}/redo`, {}));
$('#btn-undo').addEventListener('click', undo);
$('#btn-redo').addEventListener('click', redo);
document.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName) && document.activeElement.type !== 'range';
  if (typing) return;
  if (e.key === ' ' && S.photo) { e.preventDefault(); if (!S.space) { S.space = true; applyView(); } return; }
  if (S.photo && !e.ctrlKey && !e.metaKey && !e.altKey) {
    if (e.key === '+' || e.key === '=') { zoomCentre(S.zoom * 1.5); return; }
    if (e.key === '-') { zoomCentre(S.zoom / 1.5); return; }
    if (e.key === '0') { S.zoom = 1; S.panX = S.panY = 0; layout(); return; }
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
  else if (e.key === '\\' && !e.repeat && S.photo && S.photo.cursor > 0) { S.holding = true; applyCompare(); }
});
document.addEventListener('keyup', (e) => {
  if (e.key === '\\' && S.holding) { S.holding = false; applyCompare(); }
  if (e.key === ' ' && S.space) { e.preventDefault(); S.space = false; applyView(); }   // preventDefault: Space must not click a focused button
});

// ---------------------------------------------------------------- before / after
function applyCompare() {
  const on = (S.compare || S.holding) && S.photo && S.photo.cursor > 0;
  const f = $('#frame'); f.classList.toggle('compare', !!on);
  const split = S.holding ? 1 : S.split;
  $('#before-clip').style.width = (split * 100) + '%';
  const d = $('#divider'); d.hidden = !(on && !S.holding); d.style.left = (split * 100) + '%';
}
$('#btn-compare').addEventListener('click', () => { S.compare = !S.compare; applyCompare(); refreshUI(); });
$('#divider').addEventListener('pointerdown', (e) => {
  const d = $('#divider'); d.setPointerCapture(e.pointerId);
  const move = (ev) => { const r = $('#frame').getBoundingClientRect(); S.split = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)); applyCompare(); };
  d.addEventListener('pointermove', move);
  d.addEventListener('pointerup', () => d.removeEventListener('pointermove', move), { once: true });
});

// ---------------------------------------------------------------- selection (point / brush / box)
function resetSelection() {
  const bw = Math.min(S.imgW || 1, 1024), bh = Math.max(1, Math.round(bw * (S.imgH || 1) / (S.imgW || 1)));
  const mk = () => { const c = document.createElement('canvas'); c.width = bw; c.height = bh; return c; };
  S.sel.points = []; S.sel.boxes = []; S.sel.brush = mk(); S.sel.tint = mk(); S.sel.painted = false;
  scheduleAdjustPreview(); draw();
}
const hasMask = () => S.sel.boxes.length > 0 || S.sel.painted;

function norm(e) {
  const r = $('#overlay').getBoundingClientRect();
  return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
}

function paintBrush(a, b) {
  const c = S.sel.brush; if (!c) return;
  const ctx = c.getContext('2d'); const radius = (S.sel.size / Math.max(1, S.frameW)) * c.width / 2;
  ctx.globalCompositeOperation = S.sel.erase ? 'destination-out' : 'source-over';
  ctx.strokeStyle = ctx.fillStyle = '#fff'; ctx.lineWidth = radius * 2; ctx.lineCap = ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(a.x * c.width, a.y * c.height); ctx.lineTo(b.x * c.width, b.y * c.height); ctx.stroke();
  if (!S.sel.erase) S.sel.painted = true;
}

function selectionMask() {
  if (!hasMask()) return null;
  const c = document.createElement('canvas'); c.width = S.sel.brush.width; c.height = S.sel.brush.height;
  const ctx = c.getContext('2d'); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(S.sel.brush, 0, 0); ctx.fillStyle = '#fff';
  for (const b of S.sel.boxes) ctx.fillRect(b.x0 * c.width, b.y0 * c.height, (b.x1 - b.x0) * c.width, (b.y1 - b.y0) * c.height);
  return c.toDataURL('image/png');
}

// ---------------------------------------------------------------- crop box
function currentRatio() {
  if (S.aspect === 'free') return 0;
  if (S.aspect === 'orig') return S.imgW / S.imgH;
  const [a, b] = S.aspect.split(':').map(Number); return a / b;
}
function initCrop() {
  const r = currentRatio(); let w = 0.9, h = 0.9;
  if (r) { const pw = 0.9 * S.imgW, ph = 0.9 * S.imgH; if (pw / ph > r) { h = 0.9; w = (ph * r) / S.imgW; } else { w = 0.9; h = pw / r / S.imgH; } }
  S.crop = { x0: (1 - w) / 2, y0: (1 - h) / 2, x1: (1 + w) / 2, y1: (1 + h) / 2 };
  draw();
}
function rectFrom(a, p, ratio) {
  const sx = p.x >= a.x ? 1 : -1, sy = p.y >= a.y ? 1 : -1;
  const maxW = (sx > 0 ? 1 - a.x : a.x) * S.imgW, maxH = (sy > 0 ? 1 - a.y : a.y) * S.imgH;
  let w = Math.min(Math.abs(p.x - a.x) * S.imgW, maxW), h = Math.min(Math.abs(p.y - a.y) * S.imgH, maxH);
  if (ratio) { if (w / Math.max(h, 1e-6) > ratio) w = h * ratio; else h = w / ratio; }
  const x2 = a.x + sx * w / S.imgW, y2 = a.y + sy * h / S.imgH;
  return { x0: Math.min(a.x, x2), y0: Math.min(a.y, y2), x1: Math.max(a.x, x2), y1: Math.max(a.y, y2) };
}
function cropHit(p) {
  const c = S.crop; if (!c) return null; const tol = 16;
  const px = (v, d) => Math.abs(v - d) * S.frameW < tol, py = (v, d) => Math.abs(v - d) * S.frameH < tol;
  for (const [cx, cy] of [['x0', 'y0'], ['x1', 'y0'], ['x0', 'y1'], ['x1', 'y1']])
    if (px(p.x, c[cx]) && py(p.y, c[cy])) return { mode: 'resize', anchor: { x: cx === 'x0' ? c.x1 : c.x0, y: cy === 'y0' ? c.y1 : c.y0 } };
  if (p.x > c.x0 && p.x < c.x1 && p.y > c.y0 && p.y < c.y1) return { mode: 'move', last: p };
  return { mode: 'new', anchor: p };
}

// ---------------------------------------------------------------- pointer handling on the overlay
const overlay = $('#overlay');
overlay.addEventListener('pointerdown', (e) => {
  if (S.tool === 'view' || !S.photo || S.gesture || S.block) return;
  overlay.setPointerCapture(e.pointerId); const p = norm(e);
  if (S.tool === 'point') { S.sel.points.push({ x: p.x, y: p.y }); draw(); return; }
  if (S.tool === 'brush') { S.drag = { last: p }; paintBrush(p, p); draw(); return; }
  if (S.tool === 'box') { S.drag = { anchor: p, rect: null }; return; }
  if (S.tool === 'crop') S.drag = cropHit(p);
});
overlay.addEventListener('pointermove', (e) => {
  const d = S.drag; if (!d || S.gesture) return; const p = norm(e);
  if (S.tool === 'brush') { paintBrush(d.last, p); d.last = p; }
  else if (S.tool === 'box') d.rect = rectFrom(d.anchor, p, 0);
  else if (S.tool === 'crop') {
    if (d.mode === 'move') {
      const c = S.crop; let dx = p.x - d.last.x, dy = p.y - d.last.y;
      dx = Math.max(-c.x0, Math.min(1 - c.x1, dx)); dy = Math.max(-c.y0, Math.min(1 - c.y1, dy));
      S.crop = { x0: c.x0 + dx, x1: c.x1 + dx, y0: c.y0 + dy, y1: c.y1 + dy }; d.last = p;
    } else S.crop = rectFrom(d.anchor, p, currentRatio());
  }
  draw();
});
function endDrag() {
  const d = S.drag; S.drag = null; if (!d) return;
  if (S.tool === 'box' && d.rect && (d.rect.x1 - d.rect.x0) * S.frameW > 4 && (d.rect.y1 - d.rect.y0) * S.frameH > 4) { S.sel.boxes.push(d.rect); }
  scheduleAdjustPreview(); refreshUI(); draw();
}
overlay.addEventListener('pointerup', endDrag);
overlay.addEventListener('pointercancel', endDrag);

function draw() {
  const c = $('#overlay'); if (!c.width) return;
  const ctx = c.getContext('2d');
  ctx.setTransform(c.width / S.frameW, 0, 0, c.height / S.frameH, 0, 0); ctx.clearRect(0, 0, S.frameW, S.frameH);
  const W = S.frameW, H = S.frameH;
  if (S.sel.painted && S.sel.brush) {
    const t = S.sel.tint, tctx = t.getContext('2d');
    tctx.globalCompositeOperation = 'source-over'; tctx.clearRect(0, 0, t.width, t.height); tctx.drawImage(S.sel.brush, 0, 0);
    tctx.globalCompositeOperation = 'source-in'; tctx.fillStyle = SEL_COLOR; tctx.fillRect(0, 0, t.width, t.height);
    ctx.drawImage(t, 0, 0, W, H);
  }
  const boxes = S.drag && S.tool === 'box' && S.drag.rect ? [...S.sel.boxes, S.drag.rect] : S.sel.boxes;
  ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
  for (const b of boxes) { ctx.fillStyle = SEL_COLOR; ctx.strokeStyle = '#fff'; ctx.fillRect(b.x0 * W, b.y0 * H, (b.x1 - b.x0) * W, (b.y1 - b.y0) * H); ctx.strokeRect(b.x0 * W, b.y0 * H, (b.x1 - b.x0) * W, (b.y1 - b.y0) * H); }
  ctx.setLineDash([]);
  S.sel.points.forEach((p, i) => {
    ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 9, 0, Math.PI * 2); ctx.fillStyle = '#c4577e'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i + 1), p.x * W, p.y * H + 0.5);
  });
  if (S.tool === 'crop' && S.crop) {
    const r = S.crop; ctx.fillStyle = 'rgba(0,0,0,.55)';
    ctx.fillRect(0, 0, W, r.y0 * H); ctx.fillRect(0, r.y1 * H, W, H - r.y1 * H);
    ctx.fillRect(0, r.y0 * H, r.x0 * W, (r.y1 - r.y0) * H); ctx.fillRect(r.x1 * W, r.y0 * H, W - r.x1 * W, (r.y1 - r.y0) * H);
    const x = r.x0 * W, y = r.y0 * H, w = (r.x1 - r.x0) * W, h = (r.y1 - r.y0) * H;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.beginPath();
    for (let i = 1; i < 3; i++) { ctx.moveTo(x + w * i / 3, y); ctx.lineTo(x + w * i / 3, y + h); ctx.moveTo(x, y + h * i / 3); ctx.lineTo(x + w, y + h * i / 3); }
    ctx.stroke(); ctx.fillStyle = '#fff';
    for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) ctx.fillRect(cx - 5, cy - 5, 10, 10);
  }
}

// ---------------------------------------------------------------- tools and their options
const TOOL_HELP = {
  view: 'Look around. Pick a tool on the left to select an area or crop.',
  point: 'Click to drop numbered points. Points are saved for the AI tools coming in a later phase.',
  brush: 'Paint over the area to select. The Adjust sliders below then change only that area.',
  box: 'Drag to draw a box around an area. Draw more than one if you like.',
  crop: 'Drag the corners to resize, drag inside to move.',
};
function setTool(t) {
  S.tool = t; document.querySelectorAll('.tools button').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  overlay.style.cursor = t === 'view' ? 'default' : 'crosshair'; applyView(); overlay.style.pointerEvents = t === 'view' ? 'none' : 'auto';
  if (t === 'crop' && S.photo) initCrop(); else S.crop = null;
  const box = $('#tool-options'); box.replaceChildren();
  const h = document.createElement('p'); h.className = 'hint'; h.textContent = TOOL_HELP[t]; box.append(h);
  if (t === 'brush') {
    box.append(optionSlider('Brush size', 4, 120, S.sel.size, (v) => { S.sel.size = v; }));
    const row = document.createElement('div'); row.className = 'row';
    for (const [label, erase] of [['Paint', false], ['Erase', true]]) {
      const b = document.createElement('button'); b.textContent = label; b.setAttribute('aria-pressed', String(S.sel.erase === erase));
      b.addEventListener('click', () => { S.sel.erase = erase; row.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); });
      row.append(b);
    }
    box.append(row);
  }
  if (['point', 'brush', 'box'].includes(t)) {
    const b = document.createElement('button'); b.textContent = 'Clear selection'; b.style.marginTop = '8px';
    b.addEventListener('click', () => { resetSelection(); refreshUI(); }); box.append(b);
  }
  if (t === 'crop') {
    const sel = document.createElement('select'); sel.setAttribute('aria-label', 'Crop shape');
    for (const [v, l] of ASPECTS) { const o = document.createElement('option'); o.value = v; o.textContent = l; sel.append(o); }
    sel.value = S.aspect; sel.addEventListener('change', () => { S.aspect = sel.value; initCrop(); });
    const row = document.createElement('div'); row.className = 'row';
    const apply = document.createElement('button'); apply.className = 'primary'; apply.textContent = 'Apply crop'; apply.addEventListener('click', applyCrop);
    const cancel = document.createElement('button'); cancel.textContent = 'Cancel'; cancel.addEventListener('click', () => setTool('view'));
    row.append(apply, cancel); box.append(sel, row);
  }
  draw(); refreshUI();
}
function optionSlider(label, min, max, value, onInput) {
  const wrap = document.createElement('label'); wrap.className = 'slider'; wrap.textContent = label + ' ';
  const out = document.createElement('output'); out.textContent = value;
  const input = document.createElement('input'); input.type = 'range'; input.min = min; input.max = max; input.value = value;
  input.addEventListener('input', () => { out.textContent = input.value; onInput(Number(input.value)); });
  wrap.append(out, input); return wrap;
}
document.querySelectorAll('.tools button').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));

// ---------------------------------------------------------------- edits
async function edit(op, params, withMask = false) {
  const p = S.photo; if (!p) return;
  const mask = withMask ? selectionMask() : null;
  await step(() => postJSON(`/api/photos/${p.id}/edit`, { op, params, mask }));
}
async function applyCrop() {
  const c = S.crop; if (!c) return;
  await edit('crop', { x0: Math.round(c.x0 * S.imgW), y0: Math.round(c.y0 * S.imgH), x1: Math.round(c.x1 * S.imgW), y1: Math.round(c.y1 * S.imgH) });
  setTool('view');
}
$('#btn-rot-left').addEventListener('click', () => edit('rotate90', { turns: -1 }));
$('#btn-rot-right').addEventListener('click', () => edit('rotate90', { turns: 1 }));
$('#btn-flip-h').addEventListener('click', () => edit('flip', { axis: 'horizontal' }));
$('#btn-flip-v').addEventListener('click', () => edit('flip', { axis: 'vertical' }));
const straight = $('#straighten');
straight.addEventListener('input', () => { $('#out-straighten').textContent = Number(straight.value).toFixed(1) + '°'; refreshUI(); });
$('#btn-straighten').addEventListener('click', async () => {
  await edit('straighten', { degrees: Number(straight.value) }); straight.value = 0; $('#out-straighten').textContent = '0.0°'; refreshUI();
});

// ---------------------------------------------------------------- adjust sliders with live preview
function buildSliders() {
  const box = $('#sliders');
  for (const [key, label] of SLIDERS) {
    const wrap = document.createElement('label'); wrap.className = 'slider'; wrap.textContent = label + ' ';
    const out = document.createElement('output'); out.textContent = '0';
    const input = document.createElement('input'); input.type = 'range'; input.min = -100; input.max = 100; input.value = 0; input.dataset.key = key;
    input.addEventListener('input', () => { S.adj[key] = Number(input.value); out.textContent = input.value; scheduleAdjustPreview(); refreshUI(); });
    input.addEventListener('dblclick', () => { input.value = 0; input.dispatchEvent(new Event('input')); });
    wrap.append(out, input); box.append(wrap);
  }
}
function resetSliders() {
  S.adj = {}; document.querySelectorAll('#sliders input').forEach((i) => { i.value = 0; i.parentElement.querySelector('output').textContent = '0'; });
  clearTimeout(S.previewTimer); S.previewSeq++; dropPreview();
}
function dropPreview() { if (S.previewUrl) { URL.revokeObjectURL(S.previewUrl); S.previewUrl = null; } }
function scheduleAdjustPreview() {
  if (!S.photo) return;
  clearTimeout(S.previewTimer);
  if (!SLIDERS.some(([k]) => S.adj[k])) { S.previewSeq++; dropPreview(); if (S.photo) setAfter(actualUrl()); return; }
  S.previewTimer = setTimeout(async () => {
    const seq = ++S.previewSeq;
    try {
      const res = await postJSON(`/api/photos/${S.photo.id}/preview`, { op: 'adjust', params: S.adj, mask: selectionMask() });
      const blob = await res.blob(); if (seq !== S.previewSeq) return;
      dropPreview(); S.previewUrl = URL.createObjectURL(blob); setAfter(S.previewUrl);
    } catch (ex) { fail(ex); }
  }, 130);
}
$('#btn-adjust-apply').addEventListener('click', () => edit('adjust', { ...S.adj }, true));
$('#btn-adjust-reset').addEventListener('click', () => { resetSliders(); setAfter(actualUrl()); refreshUI(); });

// ---------------------------------------------------------------- export
$('#btn-export').addEventListener('click', () => { $('#export-dialog').hidden = false; });
$('#btn-export-cancel').addEventListener('click', () => { $('#export-dialog').hidden = true; });
$('#export-quality').addEventListener('input', (e) => { $('#out-quality').textContent = e.target.value; });
$('#btn-export-go').addEventListener('click', async () => {
  const fmt = $('#export-format').value;
  try {
    const res = await postJSON(`/api/photos/${S.photo.id}/export`, { format: fmt, quality: Number($('#export-quality').value) });
    const blob = await res.blob(); const a = document.createElement('a');
    const base = S.photo.name.replace(/\.[^.]+$/, '') || 'photo';
    a.href = URL.createObjectURL(blob); a.download = `${base}-veil.${fmt === 'jpeg' ? 'jpg' : fmt}`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    $('#export-dialog').hidden = true; toast('Exported. Hidden details were removed from the file.');
  } catch (ex) { fail(ex); }
});

// ---------------------------------------------------------------- start
buildSliders(); setTool('view'); boot().catch(fail);
