'use strict';
/* Veil's first-launch tutorial. Plain JavaScript, no build step, no network calls, no inline styles or HTML strings:
   every piece of text goes in with textContent. It shows once (skipped or finished = never again by itself) and the
   Help button replays it. "Seen" is remembered in this browser only. */
(function () {
  const KEY = 'veil.tutorial.done';
  const LINKS = {
    computer: { label: 'Tailscale for Linux, Windows and Mac', url: 'https://tailscale.com/download' },
    linux: { label: 'Tailscale for Linux (install steps)', url: 'https://tailscale.com/download/linux' },
    ios: { label: 'Tailscale for iPhone and iPad (App Store)', url: 'https://apps.apple.com/app/tailscale/id1470499037' },
    android: { label: 'Tailscale for Android (Google Play)', url: 'https://play.google.com/store/apps/details?id=com.tailscale.ipn' },
    machines: { label: 'Your Tailscale devices (opens Tailscale\'s website)', url: 'https://login.tailscale.com/admin/machines' },
  };

  // who: 'pc' = at the computer, 'phone' = on a paired phone or tablet (a shorter run). helpOnly = left out of the first run.
  const STEPS = [
    { id: 'welcome', who: ['pc', 'phone'], title: 'Welcome to Veil', target: null,
      body: ['Veil edits photos on your own computer. Nothing leaves your computer unless you choose.',
             'This takes about two minutes. You can skip it now and replay it any time from Help.'],
      phone: { body: ['This phone is a window into Veil on your computer. Nothing leaves your computer unless you choose.',
                      'This short tour takes about a minute. You can skip it now and replay it any time from Help.'] } },
    { id: 'photos', who: ['pc', 'phone'], title: 'Your photos', target: '#btn-library',
      body: ['Click Photos, then Add a photo. Your photos live in Veil\'s private folder on your computer.',
             'Delete removes a photo for good.'] },
    { id: 'look', who: ['pc', 'phone'], title: 'Looking around', target: '.tools',
      body: ['View just looks. Scroll (or pinch on a touch screen) to zoom, drag to move around, and Fit shows the whole photo.'] },
    { id: 'select', who: ['pc', 'phone'], title: 'Selecting', target: '.tools',
      body: ['Point: click a thing to select it. Shift-click (or the Outside option) marks something to leave out.',
             'Brush: paint over an area; switch to Erase to take parts back. Box: drag a rectangle.',
             'A selection limits the next edit to that area.'] },
    { id: 'adjust', who: ['pc', 'phone'], title: 'Adjust', target: '#sliders',
      body: ['Brightness, contrast, saturation, warmth and sharpness change the whole photo, or only your selection.',
             'Apply keeps the change. Reset drops it.'] },
    { id: 'crop', who: ['pc', 'phone'], title: 'Crop, rotate and flip', target: '.tools',
      body: ['Crop with a shape you choose, straighten a tilted horizon, turn the photo 90 degrees, or flip it.'] },
    { id: 'cleanup', who: ['pc', 'phone'], title: 'Cleanup and background', target: '#cleanup',
      body: ['Select an object or a tattoo, then Remove object or Remove tattoo. Veil fills in the space from what is around it.',
             'Remove, blur or replace the background: Veil finds the person itself, or uses your selection.'] },
    { id: 'faces', who: ['pc', 'phone'], title: 'Faces and looks', target: '#faces',
      body: ['Find faces, tick the faces an edit should apply to, then use any of: hide a face (blur, pixelate or generated face), skin, teeth, eyes, makeup, hair colour or style, facial hair, remove jewelry.',
             'Veil never puts a face from another picture into yours.'] },
    { id: 'typed', who: ['pc', 'phone'], title: 'Typed commands', target: '#cmd-form',
      body: ['Type what you want, for example "make the skin a bit smoother", and press Go.',
             'A model on your computer reads your words. A photo is only shown to an online model if you tick the box, each time.'] },
    { id: 'undo', who: ['pc', 'phone'], title: 'Undo and compare', target: '#btn-undo',
      body: ['Undo and Redo, the History list, and Before / after let you try anything safely. Every edit is a new version; the original is never overwritten.'] },
    { id: 'export', who: ['pc', 'phone'], title: 'Export', target: '#btn-export',
      body: ['Choose the format and quality. Location and camera information are removed from the exported copy.',
             'You can add a watermark; it is only drawn on exports, never on your saved versions.'],
      phone: { body: ['Choose the format and quality. Location and camera information are removed from the exported copy.',
                      'You can switch the watermark on or off and choose one of the saved presets. Presets are made at the computer.'] } },
    { id: 'watermarks', who: ['pc'], title: 'Watermarks', target: '#btn-wm',
      body: ['Make presets: text or a logo, position, size and opacity. Phones can switch one on or off, but only the computer can make or change them.'] },
    { id: 'ts-pc', who: ['pc'], title: 'Phone and tablet, part 1: this computer', target: '#btn-phone',
      body: ['Your phone or tablet reaches Veil through Tailscale, a free private network between your own devices. Nothing is opened to the internet.',
             'On this computer: install Tailscale (Veil\'s installer can do it), run "sudo tailscale up" in a terminal, open the web address it prints, and sign in. If you have no account, that page creates a free one (Google, Microsoft, Apple or GitHub).',
             'Then click Connect.'],
      links: ['computer', 'linux', 'machines'],
      see: 'You should see: a "Connected" page in your browser, the terminal command finishing by itself, and this computer listed as Connected on your Tailscale devices page.' },
    { id: 'ts-phone', who: ['pc'], title: 'Phone and tablet, part 2: the phone or tablet', target: '#btn-phone',
      body: ['On the phone or tablet, install the Tailscale app. Open it and sign in with the SAME account you used on the computer.',
             'Switch the toggle at the top to On.'],
      links: ['ios', 'android'],
      see: 'You should see: the Tailscale app saying Connected, and your computer in its list of devices.' },
    { id: 'ts-pair', who: ['pc'], title: 'Phone and tablet, part 3: pair it', target: '#btn-phone',
      body: ['In Veil, click Phone, tick "Allow phone access", then click Show QR code.',
             'Point the phone or tablet camera at the code and tap the link. The phone says it is waiting.',
             'Back here, click Approve next to it in the list.'],
      see: 'You should see: a QR code with a 5-minute timer; then "Waiting for approval" on the phone; and after you click Approve, Veil\'s editor on the phone and the device listed under Phones. You can remove it there at any time.' },
    { id: 'worker', who: ['pc'], helpOnly: true, title: 'GPU worker (optional)', target: '#btn-worker',
      body: ['Another computer of yours with a strong graphics card can do the heavy jobs over Tailscale. Click GPU worker to pair one.',
             'If it is ever unreachable, Veil does the job on this computer and tells you. The README has the setup steps, including for a rented cloud GPU machine.'] },
    { id: 'done', who: ['pc', 'phone'], title: 'You\'re set', target: '#btn-help',
      body: ['You can replay this tour any time with Help.'] },
  ];

  function stepsFor(opts) {
    const who = opts.remote ? 'phone' : 'pc';
    return STEPS.filter((s) => s.who.includes(who) && !(s.helpOnly && !opts.replay)).map((s) => {
      const o = (opts.remote && s.phone) ? Object.assign({}, s, s.phone) : Object.assign({}, s);
      o.links = s.links || [];
      return o;
    });
  }

  function storage() {
    try { const t = '__veil_t'; window.localStorage.setItem(t, '1'); window.localStorage.removeItem(t); return window.localStorage; } catch (_) { return null; }
  }
  function isDone() { const s = storage(); try { return !s || s.getItem(KEY) === '1'; } catch (_) { return true; } }   // no storage: never nag
  function markDone() { const s = storage(); try { if (s) s.setItem(KEY, '1'); } catch (_) { /* optional */ } }

  if (typeof document === 'undefined') { module.exports = { STEPS, LINKS, stepsFor }; return; }

  // ------------------------------------------------------------ the card
  let card = null, steps = [], at = 0, lit = null, replay = false, restoreFocus = null;
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };

  function clearLit() { if (lit) { lit.classList.remove('tut-hl'); lit = null; } }

  function place(target) {
    card.classList.remove('tut-center');
    const narrow = window.matchMedia('(max-width: 700px)').matches;
    const r = target && target.getBoundingClientRect();
    if (narrow || !r || r.width === 0 || r.height === 0) { card.classList.toggle('tut-center', !narrow); card.style.left = card.style.top = ''; return; }
    const w = card.offsetWidth, h = card.offsetHeight, m = 12;
    let x = r.right + m, y = r.top;
    if (r.bottom < 70) { x = Math.min(Math.max(m, r.left), window.innerWidth - w - m); y = r.bottom + m; }       // a top-bar button: just below it
    else if (x + w > window.innerWidth - m) x = r.left - w - m;                 // no room on the right: try the left
    if (x < m && r.bottom >= 70) { x = Math.min(Math.max(m, r.left), window.innerWidth - w - m); y = r.bottom + m; }   // narrow gap: below it
    if (y + h > window.innerHeight - m) y = Math.max(m, window.innerHeight - h - m);
    card.style.left = Math.round(x) + 'px'; card.style.top = Math.round(y) + 'px';
  }

  function render() {
    const s = steps[at];
    clearLit();
    card.replaceChildren();
    card.append(el('p', 'tut-count', `Step ${at + 1} of ${steps.length}`), el('h2', '', s.title));
    card.querySelector('h2').id = 'tut-title';
    for (const p of s.body) card.append(el('p', '', p));
    if (s.links.length) {
      const ul = el('ul', 'tut-links');
      for (const id of s.links) {
        const l = LINKS[id], li = el('li'), a = el('a', '', l.label);
        a.href = l.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; li.append(a); ul.append(li);
      }
      card.append(ul);
    }
    if (s.see) card.append(el('p', 'tut-see', s.see));
    const nav = el('div', 'tut-nav');
    const back = el('button', '', 'Back'); back.disabled = at === 0; back.addEventListener('click', () => go(at - 1));
    const last = at === steps.length - 1;
    const next = el('button', 'primary', last ? 'Done' : 'Next'); next.addEventListener('click', () => (last ? finish() : go(at + 1)));
    const jump = el('select'); jump.setAttribute('aria-label', 'Jump to a step');
    steps.forEach((x, i) => { const o = el('option', '', `${i + 1}. ${x.title}`); o.value = String(i); jump.append(o); });
    jump.value = String(at); jump.addEventListener('change', () => go(Number(jump.value)));
    const skip = el('button', 'tut-skip', last ? 'Close' : 'Skip tutorial'); skip.addEventListener('click', finish);
    nav.append(back, next, skip);
    card.append(nav, jump);
    const t = s.target ? document.querySelector(s.target) : null;
    const visible = t && t.getClientRects().length > 0;
    if (visible) {
      t.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      t.classList.add('tut-hl'); lit = t;
    }
    place(visible ? t : null);
    next.focus({ preventScroll: true });
  }

  function go(i) { at = Math.max(0, Math.min(steps.length - 1, i)); render(); }

  function onKey(e) {
    if (!card) return;
    if (e.key === 'Escape') { e.preventDefault(); finish(); }
    else if (e.key === 'ArrowRight' && !['SELECT', 'INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); go(at + 1); }
    else if (e.key === 'ArrowLeft' && !['SELECT', 'INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); go(at - 1); }
  }
  const onResize = () => { if (card) render(); };

  function finish() {
    markDone();
    clearLit();
    if (card) card.remove();
    card = null;
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', onResize);
    if (restoreFocus && document.contains(restoreFocus)) restoreFocus.focus();
    restoreFocus = null;
  }

  function open(remote, isReplay) {
    if (card) finish();
    replay = !!isReplay;
    steps = stepsFor({ remote: !!remote, replay });
    at = 0;
    restoreFocus = document.activeElement;
    card = el('section', 'tut');
    card.setAttribute('role', 'dialog'); card.setAttribute('aria-labelledby', 'tut-title'); card.setAttribute('aria-modal', 'false');
    document.body.append(card);
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', onResize);
    render();
  }

  window.veilTutorial = {
    maybe(remote) { if (!isDone() && !card) open(remote, false); },       // the first time only
    replay(remote) { open(remote, true); },                              // the Help button
    steps: stepsFor,
  };
})();
