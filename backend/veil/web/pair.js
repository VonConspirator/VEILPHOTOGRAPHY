'use strict';
/* The page a phone opens after scanning the QR code. The one-time token is in the address after the #, which the
   browser never sends to the server by itself; it is removed from the address bar straight away. */
const $ = (s) => document.querySelector(s);
const say = (title, text, hint = '') => { $('#pair-title').textContent = title; $('#pair-text').textContent = text; $('#pair-hint').textContent = hint; };

function deviceName() {
  const ua = navigator.userAgent;
  if (/iPhone/.test(ua)) return 'iPhone';
  if (/iPad/.test(ua)) return 'iPad';
  if (/Android/.test(ua)) return 'Android phone';
  return 'Phone';
}

async function post(path, body) {
  const res = await fetch(path, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let data = {}; try { data = await res.json(); } catch (_) { /* not JSON */ }
  if (!res.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
}

async function waitForApproval() {
  say('Waiting for approval', 'On your computer, click Approve next to this phone in Veil’s Phone window.', 'This page continues by itself.');
  for (;;) {
    await new Promise((r) => setTimeout(r, 2000));
    let state = 'pending';
    try { state = (await (await fetch('/api/pair/state', { credentials: 'same-origin' })).json()).state; } catch (_) { continue; }
    if (state === 'approved') { say('Paired', 'Opening Veil…'); location.replace('/'); return; }
    if (state !== 'pending') { say('Not approved', 'The computer did not approve this phone. Make a new QR code on the computer and scan it again.'); return; }
  }
}

(async function start() {
  const token = location.hash.slice(1);
  history.replaceState(null, '', location.pathname);
  if (!token) return say('Scan the QR code', 'Open this page by scanning the QR code shown in Veil on your computer (Phone, then Show QR code).');
  try {
    await post('/api/pair/redeem', { token, name: deviceName() });
    await waitForApproval();
  } catch (e) {
    say('Could not pair', e.message, 'On your computer, click Show QR code to make a new one, then scan it.');
  }
})();
