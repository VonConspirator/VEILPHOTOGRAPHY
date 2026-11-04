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
  faces: { list: [], found: false }, faceAt: {},
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
    if (remote && (res.status === 401 || res.status === 403) && !path.startsWith('/api/pair')) boot().catch(() => {});
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
const SCREENS = ['#screen-setup', '#screen-login', '#screen-notice', '#screen-unpaired', '#screen-app'];
function show(id) { SCREENS.forEach((s) => { $(s).hidden = s !== id; }); }

let remote = false;           // true on a paired phone, false at the computer
let phoneWait = 0;

function showUnpaired(title, text, showSteps) {
  $('#unpaired-title').textContent = title; $('#unpaired-text').textContent = text;
  $('#unpaired-steps').hidden = !showSteps; show('#screen-unpaired');
}

async function boot() {
  clearTimeout(phoneWait);
  const res = await fetch('/api/status', { credentials: 'same-origin' });
  if (res.status === 403) { remote = true; return showUnpaired('Phone access is off', 'Phone access is switched off on your computer. Turn on "Allow phone access" in Veil\'s Phone window, then check again.', false); }
  const st = await res.json();
  remote = st.scope === 'remote';
  $('#btn-phone').hidden = remote;
  $('#btn-logout').textContent = remote ? 'Unpair this phone' : 'Log out';
  if (remote && !st.logged_in) {
    if (st.device_state === 'pending') {
      phoneWait = setTimeout(boot, 2500);
      return showUnpaired('Waiting for approval', 'On your computer, click Approve next to this phone in Veil\'s Phone window.', false);
    }
    const again = ['denied', 'revoked'].includes(st.device_state);
    return showUnpaired(again ? 'This phone was removed' : 'Pair this phone', again ? 'The computer no longer allows this phone. Pair it again:' : 'This phone is not paired with Veil yet.', true);
  }
  if (st.setup_needed) return show('#screen-setup');
  if (!st.logged_in) return show('#screen-login');
  if (!st.adult_notice_acknowledged) {
    if (remote) return showUnpaired('One step on the computer', 'Open Veil on your computer once and confirm the adult-use notice. Then check again here.', false);
    return show('#screen-notice');
  }
  show('#screen-app');
  if (!remote) pollPhoneBadge();
  loadAI(); loadVision();
  await loadLibrary(true);
}
$('#btn-unpaired-retry').addEventListener('click', () => boot().catch(fail));

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
let unpairArmed = 0;
$('#btn-logout').addEventListener('click', async () => {
  if (remote) {   // a phone un-pairs itself: it needs a second click, and must be approved again to come back
    const b = $('#btn-logout');
    if (!unpairArmed) { unpairArmed = setTimeout(() => { unpairArmed = 0; b.textContent = 'Unpair this phone'; }, 3000); b.textContent = 'Sure?'; return; }
    clearTimeout(unpairArmed); unpairArmed = 0;
    try { await postJSON('/api/pair/forget', {}); } catch (ex) { return fail(ex); }
    S.photo = null; return boot();
  }
  await postJSON('/api/auth/logout', {}); S.photo = null; await boot();
});

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
  resetSelection(); clearFaces(); S.faceAt = {}; S.crop = null; resetSliders(); setTool('view'); S.compare = false; S.zoom = 1; S.panX = S.panY = 0;
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
  $('#cmd-text').disabled = $('#cmd-send').disabled = !has;
  if (typeof renderCleanup === 'function') renderCleanup();
  if (typeof renderFaces === 'function') renderFaces();
  $('#btn-straighten').disabled = !has || Number($('#straighten').value) === 0;
  const dirty = SLIDERS.some(([k]) => S.adj[k]);
  $('#btn-adjust-apply').disabled = !has || !dirty; $('#btn-adjust-reset').disabled = !has || !dirty;
  $('#adjust-scope').textContent = hasMask() ? '(selected area only)' : '(whole photo)';
  const h = $('#history'); h.replaceChildren();
  if (has) p.versions.forEach((v, i) => {
    const li = document.createElement('li'); li.textContent = v.label;
    if (i === p.cursor) li.className = 'cur'; else if (i > p.cursor) li.className = 'undone';
    li.addEventListener('click', () => { step(() => postJSON(`/api/photos/${p.id}/goto`, { cursor: i }), 'nav'); });
    h.append(li);
  });
}

// ---------------------------------------------------------------- history buttons
// Found faces and ticks are remembered per version, so Undo and Redo bring them back. 'nav' = undo/redo/history,
// 'edit' = a change that keeps the framing (the boxes stay valid), 'geom' = crop/rotate/flip/straighten (they would not line up).
function snapFaces() { if (S.photo) S.faceAt[S.photo.cursor] = S.faces.found ? { found: true, list: S.faces.list.map((f) => ({ box: f.box, on: f.on })) } : null; }
async function step(call, how = 'edit') {
  const from = S.photo ? S.photo.cursor : 0;
  snapFaces();
  try { S.photo = await (await call()).json(); } catch (ex) { fail(ex); return false; }
  const to = S.photo.cursor, snap = S.faceAt[how === 'nav' ? to : from];
  if (how !== 'nav') for (const k of Object.keys(S.faceAt)) if (Number(k) > from) delete S.faceAt[k];    // the undone branch is gone
  if (how === 'geom') S.faces = { list: [], found: false };
  else S.faces = snap ? { found: true, list: snap.list.map((f) => ({ box: f.box, on: f.on })) } : { list: [], found: false };
  if (how !== 'nav') S.faceAt[to] = how === 'geom' ? null : snap || null;
  try { resetSliders(); renderFaces(); await refresh(); return true; } catch (ex) { fail(ex); return false; }
}
const undo = () => { if (S.photo && S.photo.cursor > 0) return step(() => postJSON(`/api/photos/${S.photo.id}/undo`, {}), 'nav'); };
const redo = () => { if (S.photo && S.photo.cursor < S.photo.versions.length - 1) return step(() => postJSON(`/api/photos/${S.photo.id}/redo`, {}), 'nav'); };
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
  if (S.tool === 'point') { S.sel.points.push({ x: p.x, y: p.y, label: (S.sel.outside || e.shiftKey) ? 0 : 1 }); draw(); smartSelect({ points: S.sel.points.map((q) => [q.x, q.y, q.label]) }); return; }
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
  if (S.tool === 'box' && d.rect && (d.rect.x1 - d.rect.x0) * S.frameW > 4 && (d.rect.y1 - d.rect.y0) * S.frameH > 4) {
    if (S.sel.snap) smartSelect({ box: [d.rect.x0, d.rect.y0, d.rect.x1, d.rect.y1] }); else S.sel.boxes.push(d.rect);
  }
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
    ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 9, 0, Math.PI * 2); ctx.fillStyle = p.label === 0 ? '#555' : '#c4577e'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i + 1), p.x * W, p.y * H + 0.5);
  });
  S.faces.list.forEach((f, i) => {
    const [x0, y0, x1, y1] = f.box; ctx.lineWidth = 2; ctx.setLineDash(f.on ? [] : [5, 4]);
    ctx.strokeStyle = f.on ? '#fff' : 'rgba(255,255,255,.55)'; ctx.fillStyle = f.on ? 'rgba(196,87,126,.22)' : 'rgba(0,0,0,.12)';
    ctx.fillRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H); ctx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H); ctx.setLineDash([]);
    ctx.fillStyle = f.on ? '#c4577e' : '#555'; ctx.fillRect(x0 * W, y0 * H - 18, 22, 18); ctx.fillStyle = '#fff'; ctx.font = 'bold 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), x0 * W + 11, y0 * H - 8.5);
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
  point: 'Click on the thing you want (a person, an object, a tattoo). Veil outlines it. Click more to refine; use Outside (or Shift+click) to cut away parts.',
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
  if (t === 'point') {
    const row = document.createElement('div'); row.className = 'row';
    for (const [label, outside] of [['Inside', false], ['Outside', true]]) {
      const b = document.createElement('button'); b.textContent = label; b.setAttribute('aria-pressed', String(!!S.sel.outside === outside));
      b.addEventListener('click', () => { S.sel.outside = outside; row.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); });
      row.append(b);
    }
    box.append(row);
  }
  if (t === 'box') {
    const lab = document.createElement('label'); lab.className = 'check';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = !!S.sel.snap; cb.addEventListener('change', () => { S.sel.snap = cb.checked; });
    lab.append(cb, ' Snap the box to the object (AI)'); box.append(lab);
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
  const p = S.photo; if (!p) return false;
  const mask = withMask ? selectionMask() : null;
  const geom = ['crop', 'rotate90', 'flip', 'straighten'].includes(op);
  return step(() => postJSON(`/api/photos/${p.id}/edit`, { op, params, mask }), geom ? 'geom' : 'edit');
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

// ---------------------------------------------------------------- phone pairing (owner, at the computer)
let phoneTimer = 0, qrTimer = 0, lastLink = '', deviceSig = '';
const deviceInfo = new Map();   // device id -> its grey info line, updated in place
const ago = (t) => { const s = Math.max(0, Math.round(Date.now() / 1000 - t)); return s < 60 ? 'just now' : s < 3600 ? Math.round(s / 60) + ' min ago' : s < 86400 ? Math.round(s / 3600) + ' h ago' : Math.round(s / 86400) + ' days ago'; };
const STATUS_TEXT = { pending: 'Waiting for your approval', approved: 'Paired', denied: 'Not approved', revoked: 'Expired' };

function phoneError(msg) { $('#phone-error').textContent = msg || ''; }
async function phoneCall(path, method = 'POST', body) {
  try {
    const opts = { method, credentials: 'same-origin' };
    if (body !== undefined) { opts.headers = { 'Content-Type': 'application/json' }; opts.body = JSON.stringify(body); }
    const res = await api(path, opts); phoneError(''); return await res.json();
  } catch (ex) { phoneError(ex.message); return null; }
}

async function phoneRefresh() {
  const d = await phoneCall('/api/phone', 'GET'); if (!d) return;
  const ts = d.tailscale, status = $('#ts-status');
  if (ts.running) {
    status.className = ts.listening ? 'ok' : 'warn';
    status.textContent = ts.listening ? `Tailscale is connected (${ts.name || ts.address}).`
      : 'Tailscale started after Veil. Close Veil and start it again, then phones can connect.';
  } else { status.className = 'warn'; status.textContent = ts.problem + ' Phones cannot connect until it is running.'; }
  $('#phone-enabled').checked = d.enabled;
  $('#phone-pair-area').hidden = !d.enabled;
  $('#btn-make-qr').disabled = !ts.running || !ts.listening;
  if (!d.enabled) { $('#qr-box').hidden = true; clearInterval(qrTimer); }
  const infoText = (dev) => `${STATUS_TEXT[dev.status] || dev.status} · ${dev.status === 'approved' ? 'last used ' + ago(dev.last_seen) : 'scanned ' + ago(dev.created)}`;
  const sig = d.devices.map((x) => x.id + ':' + x.status).join('|');
  const waiting = d.devices.filter((x) => x.status === 'pending').length;
  const badge = $('#phone-badge'); badge.textContent = waiting; badge.hidden = !waiting;
  if (sig === deviceSig) {            // nothing changed: do NOT rebuild the buttons (a click in progress would be lost)
    for (const dev of d.devices) { const el = deviceInfo.get(dev.id); if (el) el.textContent = infoText(dev); }
    return;
  }
  deviceSig = sig; deviceInfo.clear();
  const list = $('#device-list'); list.replaceChildren();
  if (!d.devices.length) { const li = document.createElement('li'); li.textContent = 'No phones yet.'; list.append(li); }
  for (const dev of d.devices) {
    const li = document.createElement('li');
    const meta = document.createElement('div'); meta.className = 'meta';
    const name = document.createElement('strong'); name.textContent = dev.name;           // phone-supplied text: textContent only
    const info = document.createElement('span');
    info.textContent = infoText(dev); deviceInfo.set(dev.id, info);
    meta.append(name, info);
    const acts = document.createElement('div'); acts.className = 'acts';
    const btn = (label, cls, fn) => { const b = document.createElement('button'); b.textContent = label; if (cls) b.className = cls; b.addEventListener('click', fn); acts.append(b); return b; };
    if (dev.status === 'pending') {
      btn('Approve', 'primary', async () => { await phoneCall(`/api/phone/devices/${dev.id}/approve`); phoneRefresh(); });
      btn('Deny', '', async () => { await phoneCall(`/api/phone/devices/${dev.id}/deny`); phoneRefresh(); });
    } else {
      let armed = 0; const rm = btn('Remove', '', async () => {
        if (!armed) { armed = setTimeout(() => { armed = 0; rm.textContent = 'Remove'; }, 3000); rm.textContent = 'Sure?'; return; }
        await phoneCall(`/api/phone/devices/${dev.id}`, 'DELETE'); phoneRefresh();
      });
    }
    li.append(meta, acts); list.append(li);
  }
}

function openPhone() { $('#phone-dialog').hidden = false; phoneError(''); deviceSig = '\0'; phoneRefresh(); clearInterval(phoneTimer); phoneTimer = setInterval(phoneRefresh, 2500); }
function closePhone() { $('#phone-dialog').hidden = true; clearInterval(phoneTimer); clearInterval(qrTimer); $('#qr-box').hidden = true; $('#qr-img').removeAttribute('src'); lastLink = ''; }
$('#btn-phone').addEventListener('click', openPhone);
$('#btn-phone-close').addEventListener('click', closePhone);
$('#phone-enabled').addEventListener('change', async (e) => { await phoneCall('/api/phone/enable', 'POST', { enabled: e.target.checked }); phoneRefresh(); });
$('#btn-make-qr').addEventListener('click', async () => {
  const d = await phoneCall('/api/phone/pair'); if (!d) return;
  lastLink = d.url; $('#qr-img').src = d.qr; $('#qr-box').hidden = false;
  let left = d.expires_in; clearInterval(qrTimer);
  const tick = () => {
    if (left <= 0) { clearInterval(qrTimer); $('#qr-box').hidden = true; $('#qr-img').removeAttribute('src'); lastLink = ''; phoneError('That QR code expired. Click Show QR code for a new one.'); return; }
    $('#qr-timer').textContent = `Works once, for ${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')} more.`; left -= 1;
  };
  tick(); qrTimer = setInterval(tick, 1000);
});
$('#btn-copy-link').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(lastLink); toast('Link copied. Open it on your phone within 5 minutes.'); } catch (_) { toast('Could not copy. Scan the QR code instead.', true); }
});
// a small number on the Phone button when a phone is waiting
let badgeTimer = 0;
function pollPhoneBadge() {
  clearInterval(badgeTimer);
  badgeTimer = setInterval(async () => {
    if (remote || $('#screen-app').hidden || !$('#phone-dialog').hidden) return;
    try { const d = await (await fetch('/api/phone', { credentials: 'same-origin' })).json(); const n = (d.devices || []).filter((x) => x.status === 'pending').length; $('#phone-badge').textContent = n; $('#phone-badge').hidden = !n; } catch (_) { /* ignore */ }
  }, 10000);
}

// ---------------------------------------------------------------- AI selection and cleanup tools (phase 5)
S.vision = null;
async function loadVision() {
  try { S.vision = await (await api('/api/vision')).json(); } catch (_) { S.vision = null; }
  renderCleanup();
}
function renderCleanup() {
  const v = S.vision, ok = !!v && v.ready;
  $('#cleanup-status').textContent = !v ? 'Could not check the editing models.'
    : ok ? 'Editing models are ready.' : (!v.libraries ? 'The editing libraries are not installed.' : 'Some editing models are not downloaded yet.') + (remote ? ' Set them up on the computer.' : ' Open Model, then Editing models.');
  document.querySelectorAll('#cleanup button').forEach((b) => { b.dataset.needs = '1'; b.disabled = !ok || !S.photo || aiBusy; });
  if (typeof renderFaces === 'function') renderFaces();
}
let aiBusy = false;
async function smartSelect(body) {
  const p = S.photo; if (!p || aiBusy) return;
  aiBusy = true; renderCleanup(); toast('Finding the outline…');
  try {
    const blob = await (await postJSON(`/api/photos/${p.id}/segment`, body)).blob();
    const bmp = await createImageBitmap(blob);
    const c = S.sel.brush, ctx = c.getContext('2d');
    const tmp = document.createElement('canvas'); tmp.width = c.width; tmp.height = c.height;
    const tctx = tmp.getContext('2d'); tctx.drawImage(bmp, 0, 0, c.width, c.height);
    const px = tctx.getImageData(0, 0, c.width, c.height), d = px.data;
    for (let i = 0; i < d.length; i += 4) { d[i + 3] = d[i]; d[i] = d[i + 1] = d[i + 2] = 255; }   // white where selected, clear elsewhere
    ctx.globalCompositeOperation = 'source-over'; ctx.clearRect(0, 0, c.width, c.height); ctx.putImageData(px, 0, 0);
    S.sel.painted = true; S.sel.boxes = [];
    toast('Selected. Click more to refine, or pick an action on the right.');
  } catch (ex) { S.sel.points.pop(); fail(ex); }
  finally { aiBusy = false; renderCleanup(); refreshUI(); draw(); }
}
async function aiEdit(op, params, useSelection) {
  if (aiBusy) return false;
  aiBusy = true; renderCleanup(); toast('Working… this can take a few seconds.');
  try {
    const ok = await edit(op, params, useSelection && hasMask());
    if (ok) { resetSelection(); setTool('view'); toast('Done. Undo takes it back.'); }
    return ok;
  } finally { aiBusy = false; renderCleanup(); }
}
const needArea = () => { if (!hasMask()) { toast('Select the area first: use the Point, Brush or Box tool.', true); return false; } return true; };
$('#btn-remove-object').addEventListener('click', () => needArea() && aiEdit('remove_object', {}, true));
$('#btn-remove-tattoo').addEventListener('click', () => needArea() && aiEdit('remove_tattoo', {}, true));
$('#btn-remove-bg').addEventListener('click', () => aiEdit('remove_background', {}, true));
$('#btn-blur-bg').addEventListener('click', () => aiEdit('blur_background', { amount: Number($('#blur-amount').value) }, true));
$('#btn-replace-bg').addEventListener('click', () => aiEdit('replace_background', { color: $('#bg-color').value }, true));
$('#blur-amount').addEventListener('input', () => { $('#out-blur').textContent = $('#blur-amount').value; });


// ---------------------------------------------------------------- faces and looks (phase 6)
function clearFaces() { S.faces = { list: [], found: false }; if (typeof renderFaces === 'function') renderFaces(); }
function renderFaces() {
  const v = S.vision, libs = !!v && v.faces_libs && v.models.filter((m) => ['yunet', 'face-landmarker'].includes(m.name)).every((m) => m.installed);
  const gen = !!v && v.generative_libs && v.models.some((m) => m.name === 'sdxl-inpaint' && m.installed);
  const has = !!S.photo, n = S.faces.list.filter((f) => f.on).length;
  $('#faces-status').textContent = !v ? '' : !libs ? 'The face models are not installed yet.' + (remote ? '' : ' Open Model, then Editing models.')
    : !S.faces.found ? 'Click Find faces, then tick the faces an edit should change.'
    : S.faces.list.length ? `${S.faces.list.length} found, ${n} ticked.` : 'No faces were found in this photo.';
  $('#btn-find-faces').disabled = !libs || !has || aiBusy;
  document.querySelectorAll('#faces .face-btn').forEach((b) => { b.disabled = !libs || !has || aiBusy || !n || (b.classList.contains('gen') && !gen); b.title = b.classList.contains('gen') && !gen ? 'Needs the optional Stable Diffusion XL download (Model, then Editing models).' : ''; });
  $('#btn-remove-jewelry').disabled = !(!!v && v.ready) || !has || aiBusy;
  const ul = $('#face-list'); ul.replaceChildren();
  S.faces.list.forEach((f, i) => {
    const li = document.createElement('li'), cb = document.createElement('input'), lab = document.createElement('label');
    cb.type = 'checkbox'; cb.checked = f.on; cb.id = 'face-' + i; lab.htmlFor = cb.id; lab.textContent = `Face ${i + 1}`;
    cb.addEventListener('change', () => { f.on = cb.checked; snapFaces(); renderFaces(); draw(); });
    li.append(cb, lab); ul.append(li);
  });
}
async function findFaces() {
  const p = S.photo; if (!p || aiBusy) return false;
  aiBusy = true; renderCleanup(); renderFaces(); toast('Looking for faces…');
  try {
    const d = await (await postJSON(`/api/photos/${p.id}/faces`, {})).json();
    S.faces = { found: true, list: d.faces.map((f) => ({ box: f.box, on: true })) }; snapFaces();
    toast(d.faces.length ? `Found ${d.faces.length} face${d.faces.length > 1 ? 's' : ''}.` : 'No faces found.', !d.faces.length);
    return d.faces.length > 0;
  } catch (ex) { fail(ex); return false; }
  finally { aiBusy = false; renderCleanup(); renderFaces(); draw(); }
}
const ticked = () => S.faces.list.filter((f) => f.on).map((f) => f.box);
async function faceEdit(op, params, useSelection = false) {
  if (!ticked().length) { toast('Tick at least one face first (Find faces).', true); return false; }
  if (aiBusy) return false;
  aiBusy = true; renderCleanup(); renderFaces(); toast('Working… generated edits can take up to a minute.');
  try {
    const ok = await edit(op, { ...params, faces: ticked() }, useSelection && hasMask());
    if (ok) { if (useSelection && hasMask()) { resetSelection(); setTool('view'); } toast('Done. Undo takes it back.'); }
    return ok;
  } finally { aiBusy = false; renderCleanup(); renderFaces(); draw(); }
}
$('#btn-find-faces').addEventListener('click', findFaces);
$('#hide-amount').addEventListener('input', () => { $('#out-hide').textContent = $('#hide-amount').value; });
$('#skin-amount').addEventListener('input', () => { $('#out-skin').textContent = $('#skin-amount').value; });
$('#lash-amount').addEventListener('input', () => { $('#out-lash').textContent = $('#lash-amount').value; });
$('#btn-blur-face').addEventListener('click', () => faceEdit('blur_face', { amount: Number($('#hide-amount').value) }));
$('#btn-pixelate-face').addEventListener('click', () => faceEdit('pixelate_face', { amount: Number($('#hide-amount').value) }));
$('#btn-generate-face').addEventListener('click', () => faceEdit('generate_face', {}));
$('#btn-skin-smooth').addEventListener('click', () => faceEdit('skin_smooth', { amount: Number($('#skin-amount').value) }));
$('#btn-teeth-whiten').addEventListener('click', () => faceEdit('teeth_whiten', {}));
$('#btn-eye-color').addEventListener('click', () => faceEdit('eye_color', { color: $('#eye-color').value }));
$('#btn-eyelash').addEventListener('click', () => faceEdit('eyelash_length', { amount: Number($('#lash-amount').value) }));
$('#btn-makeup-add').addEventListener('click', () => { const part = $('#makeup-part').value; faceEdit('makeup_add', part === 'face' ? { part } : { part, color: $('#makeup-color').value }); });
$('#btn-makeup-remove').addEventListener('click', () => faceEdit('makeup_remove', { part: 'all' }));
$('#btn-hair-color').addEventListener('click', () => faceEdit('hair_color', { color: $('#hair-color').value }, true));
$('#btn-hairstyle').addEventListener('click', () => { const style = $('#hair-style').value.trim(); if (!style) return toast('Describe the style first, for example: curly.', true); faceEdit('hairstyle', { style }, true); });
$('#btn-facial-hair').addEventListener('click', () => faceEdit('facial_hair', { mode: $('#fh-mode').value, style: $('#fh-style').value }));
$('#btn-remove-jewelry').addEventListener('click', () => needArea() && aiEdit('remove_jewelry', {}, true));

// ---------------------------------------------------------------- command bar: type a request, Veil asks where, then applies it
S.ai = null; S.pending = null;
let cloudArmed = false, cmdBusy = false;

async function loadAI() {
  try { S.ai = (await (await api('/api/ai')).json()).active; } catch (_) { S.ai = null; }
  renderAI();
}
function renderAI() {
  const a = S.ai, look = $('#cmd-look');
  $('#ai-chip').textContent = a ? a.model : 'not set';
  look.disabled = !a || !a.vision || (remote && !a.local);
  if (look.disabled) look.checked = false;
  const online = !!a && !a.local;
  $('#cmd-look').closest('label').classList.toggle('online', online);
  $('#cmd-look-label').textContent = !a ? 'Let the model see the photo'
    : !a.vision ? 'This model cannot see photos'
    : remote && online ? 'Photos can only be sent online from the computer'
    : online ? `Send the photo to ${a.name} (online)` : 'Let the model see the photo';
  disarmCloud();
}
function disarmCloud() { cloudArmed = false; $('#cmd-send').textContent = 'Go'; }
['input', 'change'].forEach((ev) => { $('#cmd-text').addEventListener(ev, disarmCloud); $('#cmd-look').addEventListener(ev, disarmCloud); });

function showReply(text, bad = false, buttons = []) {
  const box = $('#cmd-reply'); box.hidden = !text; box.classList.toggle('bad', bad);
  $('#cmd-reply-text').textContent = text || '';
  const row = $('#cmd-reply-btns'); row.replaceChildren();
  for (const [label, fn, primary] of buttons) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
    if (primary) b.className = 'primary';
    b.addEventListener('click', fn); row.append(b);
  }
}

function fullAdjust(params) { const p = {}; for (const [k] of SLIDERS) p[k] = Number(params[k]) || 0; return p; }

async function runPlan(plan) {
  S.pending = null;
  const note = plan.asked && plan.asked.photo_sent ? ` (the photo was sent to ${plan.asked.provider})` : '';
  if (plan.action === 'none') return showReply((plan.reply || 'I could not turn that into an edit.') + note);
  if (!plan.available) return showReply(`I understood: ${plan.label}. That tool is not built yet; it arrives in a later phase. Nothing was changed.`);
  const finish = (ok) => showReply(ok ? `Done: ${plan.label}. Undo (Ctrl+Z) takes it back.` + note : `Could not do: ${plan.label}. Nothing was changed.`, !ok);
  const done = async (op, params, withMask) => {
    const ok = await edit(op, params, withMask);
    if (ok && withMask) { resetSelection(); setTool('view'); }
    finish(ok);
  };
  const askArea = (op, params, allowWhole) => {            // "Where should it apply?"
    const startArea = (tool) => {
      setTool(tool); S.pending = plan;
      const apply = async () => { if (!hasMask()) return toast('Mark an area first.', true); S.pending = null; await done(op, params, true); };
      showReply('Mark the area, then click Apply here.', false, [['Apply here', apply, true], ['Cancel', () => { S.pending = null; resetSelection(); setTool('view'); showReply(''); }]]);
    };
    const btns = [];
    if (allowWhole) btns.push(['Whole photo', () => done(op, params, false), true]);
    btns.push(['Click it (AI)', () => startArea('point'), !allowWhole], ['Brush an area', () => startArea('brush')], ['Box an area', () => startArea('box')]);
    showReply((plan.reply ? plan.reply + ' ' : '') + 'Where should it apply?' + note, false, btns);
  };
  if (plan.action === 'rotate90' || plan.action === 'flip' || plan.action === 'straighten') return done(plan.action, plan.params, false);
  if (plan.action === 'crop') {
    S.aspect = plan.params.aspect || 'free'; setTool('crop');
    return showReply('Adjust the crop box, then click Apply crop on the right.' + note);
  }
  if (plan.action === 'adjust') {
    const params = fullAdjust(plan.params);
    if (hasMask()) return done('adjust', params, true);       // an area is already marked: use it
    return askArea('adjust', params, true);
  }
  if (plan.region === 'faces') {
    showReply('Looking for faces…');
    const found = await findFaces();
    if (!found) return showReply('I could not find a face to change. Nothing was changed.', true);
    const go = async () => { S.pending = null; const ok = await faceEdit(plan.action, plan.params, ['hair_color', 'hairstyle'].includes(plan.action)); finish(ok); };
    S.pending = plan;
    return showReply(`I found ${S.faces.list.length} face${S.faces.list.length > 1 ? 's' : ''}, numbered on the photo. Untick any you do not want changed (right panel), then click Apply.` + note, false, [['Apply', go, true], ['Cancel', () => { S.pending = null; clearFaces(); showReply(''); }]]);
  }
  if (plan.action === 'remove_tattoo' || plan.action === 'remove_object' || plan.action === 'remove_jewelry') {
    if (hasMask()) { aiBusy = true; renderCleanup(); try { return await done(plan.action, {}, true); } finally { aiBusy = false; renderCleanup(); } }
    return askArea(plan.action, {}, false);
  }
  if (['remove_background', 'blur_background', 'replace_background'].includes(plan.action)) {
    aiBusy = true; renderCleanup(); showReply('Working… this can take a few seconds.');
    try { return await done(plan.action, plan.params, true); } finally { aiBusy = false; renderCleanup(); }
  }
  showReply('I could not apply that.', true);
}

$('#cmd-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = $('#cmd-text').value.trim();
  if (!S.photo || cmdBusy || !text) return;
  const a = S.ai, look = $('#cmd-look').checked && !!a && a.vision, online = !!a && !a.local;
  if (look && online && !cloudArmed) {        // sending a photo online is confirmed every single time
    cloudArmed = true; $('#cmd-send').textContent = 'Confirm: send photo online';
    return showReply(`This sends the photo to ${a.name} over the internet. Click the button again to confirm, or untick the box to send only your words.`);
  }
  cmdBusy = true; $('#cmd-send').disabled = true; disarmCloud();
  showReply('Thinking… the first request after a while can take longer while the model loads.');
  try {
    const plan = await (await postJSON('/api/ai/command', { photo_id: S.photo.id, text, look_at_photo: look, confirm_cloud_photo: look && online })).json();
    $('#cmd-text').value = ''; if (online) $('#cmd-look').checked = false;
    await runPlan(plan);
  } catch (ex) { showReply(ex.message || 'Something went wrong.', true); }
  finally { cmdBusy = false; $('#cmd-send').disabled = !S.photo; }
});

// ---------------------------------------------------------------- language-model window (computer only)
let aiTimer = 0;
const aiError = (m) => { $('#ai-error').textContent = m || ''; };
function twoClick(btn, label, fn) {
  let armed = 0;
  btn.addEventListener('click', () => {
    if (!armed) { armed = setTimeout(() => { armed = 0; btn.textContent = label; }, 3000); btn.textContent = 'Sure?'; return; }
    clearTimeout(armed); armed = 0; btn.textContent = label; fn();
  });
}
async function aiRefresh() {
  try {
    const set = await (await api('/api/ai/settings')).json();
    const list = $('#ai-providers'); list.replaceChildren();
    for (const p of set.providers) {
      const li = document.createElement('li'); const meta = document.createElement('div'); meta.className = 'meta';
      const nm = document.createElement('strong'); nm.textContent = p.name + (p.id === set.active ? ' (in use)' : '');
      const sub = document.createElement('span'); sub.textContent = `${p.model} · ${p.local ? 'on this computer' : 'online'}${p.has_key ? ' · key saved' : ''}${p.vision ? ' · sees photos' : ''}`;
      meta.append(nm, sub);
      const acts = document.createElement('div'); acts.className = 'acts';
      if (p.id !== set.active) { const use = document.createElement('button'); use.textContent = 'Use'; use.addEventListener('click', () => aiPost('POST', '/api/ai/active', { id: p.id })); acts.append(use); }
      const ed = document.createElement('button'); ed.textContent = 'Edit'; ed.addEventListener('click', () => {
        const f = $('#ai-form'); f.kind.value = p.kind; f.name.value = p.name; f.id.value = p.id; f.base_url.value = p.base_url; f.model.value = p.model; f.vision.checked = !!p.vision; f.api_key.value = '';
      });
      const rm = document.createElement('button'); rm.textContent = 'Remove'; twoClick(rm, 'Remove', () => aiPost('DELETE', `/api/ai/providers/${encodeURIComponent(p.id)}`));
      acts.append(ed, rm); li.append(meta, acts); list.append(li);
    }
    const m = await (await api('/api/ai/models')).json();
    $('#ai-models-note').textContent = !m.reachable ? `Could not reach ${m.provider}. Is it running?` : m.can_download ? 'Suggested models by computer power. Download only when you click.' : 'The selected model is not Ollama on this computer, so nothing can be downloaded here.';
    const tl = $('#ai-tiers'); tl.replaceChildren();
    for (const t of m.tiers) {
      const li = document.createElement('li'); const meta = document.createElement('div'); meta.className = 'meta';
      const nm = document.createElement('strong'); nm.textContent = `${t.tier}: ${t.name}`;
      const sub = document.createElement('span'); sub.textContent = `${t.hardware} · ${t.size} · ${t.license} · ${t.note}`;
      meta.append(nm, sub); li.append(meta);
      const acts = document.createElement('div'); acts.className = 'acts';
      if (t.installed) { const ok = document.createElement('span'); ok.className = 'ok'; ok.textContent = 'Installed'; acts.append(ok); }
      else if (m.can_download) {
        const dl = document.createElement('button'); dl.textContent = 'Download'; dl.disabled = m.download.running;
        dl.addEventListener('click', () => aiPost('POST', '/api/ai/models/pull', { name: t.name }));
        acts.append(dl);
      }
      li.append(acts); tl.append(li);
    }
    await visionRefresh();
    const d = m.download, dl = $('#ai-dl'); dl.hidden = !(d.running || d.error);
    dl.textContent = d.error ? d.error : `Downloading ${d.name}: ${d.status}${d.total ? ` ${Math.round(100 * d.done / d.total)}%` : ''}`;
    if (!d.running) loadAI();
  } catch (ex) { aiError(ex.message); }
}
const fmtSize = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + ' GB' : Math.round(n / 1e6) + ' MB';
async function visionRefresh() {
  const v = await (await api('/api/vision')).json();
  $('#vision-note').textContent = v.ready ? `Ready on ${v.device === 'cuda' ? 'the graphics card' : 'the processor (slow)'}.` : (v.libraries ? 'Download the Recommended models to use the cleanup and background tools.' : 'The editing libraries are not installed (see the README).');
  const list = $('#vision-models'); list.replaceChildren();
  for (const m of v.models) {
    const li = document.createElement('li'); const meta = document.createElement('div'); meta.className = 'meta';
    const nm = document.createElement('strong'); nm.textContent = `${m.label}${m.required ? '' : ' (optional)'}`;
    const sub = document.createElement('span'); sub.textContent = `${m.use} · ${fmtSize(m.size)} · ${m.license}`;
    meta.append(nm, sub);
    if (m.extra_packages && m.libs_ok === false) { const x = document.createElement('span'); x.textContent = `Also needs: pip install ${m.extra_packages}`; meta.append(x); }
    li.append(meta);
    const acts = document.createElement('div'); acts.className = 'acts';
    if (m.installed) { const ok = document.createElement('span'); ok.className = 'ok'; ok.textContent = 'Installed'; acts.append(ok); }
    else { const b = document.createElement('button'); b.textContent = 'Download'; b.disabled = v.download.running; b.addEventListener('click', () => aiPost('POST', `/api/vision/models/${encodeURIComponent(m.name)}/download`)); acts.append(b); }
    li.append(acts); list.append(li);
  }
  const d = v.download, el = $('#vision-dl'); el.hidden = !(d.running || d.error);
  el.textContent = d.error ? d.error : `Downloading ${d.name}: ${d.total ? Math.round(100 * d.done / d.total) : 0}% (${d.file})`;
  if (!d.running) loadVision();
}
async function aiPost(method, url, body) {
  aiError('');
  try {
    await api(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    await aiRefresh(); await loadAI();
  } catch (ex) { aiError(ex.message); }
}
$('#btn-ai').addEventListener('click', () => {
  if (remote) return toast(S.ai ? `Commands are read by ${S.ai.name} (${S.ai.model}). Change it on the computer.` : 'No model is set up.');
  $('#ai-dialog').hidden = false; aiError(''); aiRefresh(); clearInterval(aiTimer); aiTimer = setInterval(aiRefresh, 2000);
});
$('#btn-ai-close').addEventListener('click', () => { $('#ai-dialog').hidden = true; clearInterval(aiTimer); });
$('#ai-form').addEventListener('submit', async (e) => {
  e.preventDefault(); const f = e.target;
  await aiPost('PUT', '/api/ai/providers', { id: f.id.value.trim(), name: f.name.value.trim(), kind: f.kind.value, base_url: f.base_url.value.trim(), model: f.model.value.trim(), vision: f.vision.checked, api_key: f.api_key.value });
  f.api_key.value = '';
});
$('#btn-ai-test').addEventListener('click', async () => {
  aiError(''); const b = $('#btn-ai-test'); b.disabled = true; b.textContent = 'Testing…';
  try { const r = await (await postJSON('/api/ai/test', {})).json(); toast(`${r.provider} answered (${r.model}).`); }
  catch (ex) { aiError(ex.message); }
  finally { b.disabled = false; b.textContent = 'Test the active model'; }
});

// ---------------------------------------------------------------- start
buildSliders(); setTool('view'); boot().catch(fail);
