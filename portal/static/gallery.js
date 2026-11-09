// Gallery module (portal/modules/gallery.py serves this file as /gallery/gallery.js). No inline script anywhere.
// - Gallery / compare pages (#cj-gallery): loads each conversion's marks (GET /gallery/<cid>/api/state), paints them on the
//   tiles and montage boxes, filters (mark / sample / grid / reason), and opens a full-size viewer on a click.
//   Viewer keys: left / right = previous / next, K keep, M maybe, R reject (again = clear), 0-5 stars, T reasons,
//   N note, Esc close. Marks are saved one at a time (POST /gallery/<cid>/api/event); the server stamps who and when.
//   The scalebar is drawn from the image's data-nm (nm per pixel of the image file).
// - Report page: the Print button. Text from the server only ever goes in through textContent.
(function () {
  'use strict';
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;

  var pr = document.querySelector('[data-print]');
  if (pr) pr.addEventListener('click', function () { window.print(); });

  var root = document.getElementById('cj-gallery');
  if (!root) return;
  var cids = (root.dataset.cids || '').split(' ').filter(Boolean);
  var TAGS = [];
  try { TAGS = JSON.parse(root.dataset.tags || '[]'); } catch (e) { TAGS = []; }
  var NICE = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  var S = {};                       // cid -> key -> {flag, stars, tags, note}
  var msg = document.getElementById('cj-gallery-msg');
  var sum = document.getElementById('cj-gallery-sum');
  var filt = {flag: '', sample: '', grid: '', tag: ''};
  var cur = null, view = null, noteTimer = null;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = String(text);
    return e;
  }
  function say(t) { if (msg) msg.textContent = t || ''; }
  function entry(cid, k) { var s = S[cid] || (S[cid] = {}); return s[k] || {flag: '', stars: 0, tags: [], note: ''}; }
  function each(sel, fn) { Array.prototype.forEach.call(root.querySelectorAll(sel), fn); }
  function sel2(cid, k) {
    var out = [];
    each('[data-cid="' + cid + '"][data-k]', function (e) { if (e.dataset.k === k) out.push(e); });
    return out;
  }

  // ---- painting
  var LET = {keep: 'K', maybe: 'M', reject: 'R'};
  function paint(cid, k) {
    var e = entry(cid, k);
    sel2(cid, k).forEach(function (n) {
      n.dataset.flag = e.flag; n.dataset.stars = String(e.stars);
      var b = n.querySelector('.cj-gallery-badge');
      if (b) b.textContent = (LET[e.flag] || '') + (e.stars ? (e.flag ? ' ' : '') + '★' + e.stars : '');
    });
  }
  function names(cid) {
    var out = {};
    each('.cj-gallery-tile[data-cid="' + cid + '"]', function (n) { out[n.dataset.k] = 1; });
    return out;
  }
  function counts(tiles, cid) {
    var c = {n: 0, keep: 0, maybe: 0, reject: 0};
    tiles.forEach(function (n) { c.n++; var f = entry(cid, n.dataset.k).flag; if (c[f] != null) c[f]++; });
    return c;
  }
  function line(c) {
    return c.keep + ' keep · ' + c.maybe + ' maybe · ' + c.reject + ' reject · ' + (c.n - c.keep - c.maybe - c.reject) + ' unmarked of ' + c.n;
  }
  function recount() {
    var all = {n: 0, keep: 0, maybe: 0, reject: 0};
    each('.cj-gallery-mont', function (m) {
      var tiles = Array.prototype.slice.call(m.querySelectorAll('.cj-gallery-tile')), c = counts(tiles, m.dataset.cid);
      var s = m.querySelector('.cj-gallery-counts'); if (s) s.textContent = line(c);
    });
    cids.forEach(function (cid) {
      var tiles = Array.prototype.slice.call(root.querySelectorAll('.cj-gallery-tile[data-cid="' + cid + '"]')), c = counts(tiles, cid);
      for (var k in all) all[k] += c[k];
    });
    if (sum) sum.textContent = line(all);
  }
  function paintAll() {
    cids.forEach(function (cid) { each('[data-cid="' + cid + '"][data-k]', function (n) { paint(cid, n.dataset.k); }); });
    recount(); applyFilters();
  }

  // ---- filters
  function match(n) {
    var e = entry(n.dataset.cid, n.dataset.k);
    if (filt.flag === 'none' ? e.flag !== '' : (filt.flag && e.flag !== filt.flag)) return false;
    if (filt.tag && e.tags.indexOf(filt.tag) < 0) return false;
    return true;
  }
  function applyFilters() {
    each('.cj-gallery-group', function (g) {
      var hide = (filt.grid && g.dataset.g !== filt.grid) || (filt.sample && g.dataset.s !== filt.sample);
      g.hidden = !!hide;
    });
    each('.cj-gallery-tile', function (n) { n.hidden = !match(n); });
    var narrowed = !!(filt.flag || filt.tag);
    each('.cj-gallery-mont', function (m) {
      var any = m.querySelector('.cj-gallery-tile:not([hidden])');
      m.hidden = narrowed && !any;
    });
    each('.cj-gallery-box', function (b) { b.classList.toggle('cj-gallery-dim', narrowed && !match(b)); });
    var rs = root.querySelector('[data-filter-reset]');
    if (rs) rs.hidden = !(filt.flag || filt.sample || filt.grid || filt.tag);
  }
  each('[data-filter]', function (s) {
    s.addEventListener('change', function () { filt[s.dataset.filter] = s.value; applyFilters(); });
  });
  var reset = root.querySelector('[data-filter-reset]');
  if (reset) reset.addEventListener('click', function () {
    filt = {flag: '', sample: '', grid: '', tag: ''};
    each('[data-filter]', function (s) { s.value = ''; });
    applyFilters();
  });

  // ---- saving
  function post(cid, k, f, v) {
    return fetch('/gallery/' + cid + '/api/event', {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: {'Content-Type': 'application/json', 'X-Requested-With': 'gallery'},
      body: JSON.stringify({k: k, f: f, v: v})
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) throw new Error(j.error || 'Not saved (' + r.status + ')');
        return j;
      });
    });
  }
  // Only the field that was changed is applied, confirmed or reverted: a slow save of one field never overwrites a
  // newer change to another (stars, a reason, the note) on the same image.
  function setField(cid, k, f, v) {
    var s = S[cid] || (S[cid] = {}), e = s[k];
    if (!e) e = s[k] = {flag: '', stars: 0, tags: [], note: ''};
    e[f] = v;
    paint(cid, k); recount(); applyFilters(); refresh();
  }
  function set(cid, k, f, v) {
    var prev = JSON.parse(JSON.stringify(entry(cid, k)[f]));
    setField(cid, k, f, v);
    say('');
    return post(cid, k, f, v).then(function (j) {
      if (j.entry && j.entry[f] !== undefined) setField(cid, k, f, j.entry[f]);
    }).catch(function (err) {
      setField(cid, k, f, prev);
      say(err.message || 'Not saved.');
    });
  }

  // ---- the viewer
  function buildView() {
    var v = el('div', 'cj-gallery-view'); v.hidden = true;
    v.setAttribute('role', 'dialog'); v.setAttribute('aria-modal', 'true'); v.setAttribute('aria-label', 'Full-size image');
    var stage = el('div', 'cj-gallery-vstage'), wrap = el('div', 'cj-gallery-vwrap');
    var img = el('img'); img.alt = '';
    var bar = el('div', 'cj-gallery-sbar'); bar.appendChild(el('i')); bar.appendChild(el('b'));
    wrap.appendChild(img); wrap.appendChild(bar); stage.appendChild(wrap);
    var side = el('div', 'cj-gallery-vside');
    var title = el('p', 'cj-gallery-vtitle'), pos = el('p', 'cj-mut');
    var flags = el('div', 'cj-gallery-vflags'); flags.setAttribute('role', 'group'); flags.setAttribute('aria-label', 'Mark');
    ['keep', 'maybe', 'reject'].forEach(function (f) {
      var b = el('button', 'cj-gallery-btn', f.charAt(0).toUpperCase() + f.slice(1) + ' (' + LET[f] + ')');
      b.type = 'button'; b.dataset.flag = f;
      b.addEventListener('click', function () { toggleFlag(f); });
      flags.appendChild(b);
    });
    var stars = el('div', 'cj-gallery-vstars'); stars.setAttribute('role', 'group'); stars.setAttribute('aria-label', 'Stars');
    for (var i = 0; i <= 5; i++) {
      (function (n) {
        var b = el('button', 'cj-gallery-btn', n === 0 ? '0' : '★' + n);
        b.type = 'button'; b.dataset.stars = String(n); b.setAttribute('aria-label', n + ' stars');
        b.addEventListener('click', function () { setStars(n); });
        stars.appendChild(b);
      })(i);
    }
    var tags = el('div', 'cj-gallery-vtags'); tags.setAttribute('role', 'group'); tags.setAttribute('aria-label', 'Reasons (T)');
    TAGS.forEach(function (t) {
      var b = el('button', 'cj-gallery-chip', t); b.type = 'button'; b.dataset.tag = t;
      b.addEventListener('click', function () { toggleTag(t); });
      tags.appendChild(b);
    });
    var lab = el('label', 'cj-gallery-vnote', 'Note (N)'), note = el('textarea'); note.rows = 4; note.maxLength = 1000;
    lab.appendChild(note);
    note.addEventListener('input', function () {
      clearTimeout(noteTimer); noteTimer = setTimeout(saveNote, 700);
    });
    note.addEventListener('blur', saveNote);
    var nav = el('div', 'cj-gallery-vnav');
    var prev = el('button', 'cj-gallery-btn', '← Prev'), next = el('button', 'cj-gallery-btn', 'Next →'),
        close = el('button', 'cj-gallery-btn', 'Close (Esc)');
    prev.type = next.type = close.type = 'button';
    prev.addEventListener('click', function () { step(-1); });
    next.addEventListener('click', function () { step(1); });
    close.addEventListener('click', closeView);
    nav.appendChild(prev); nav.appendChild(next); nav.appendChild(close);
    var open = el('a', 'cj-gallery-btn', 'Open image in a tab'); open.target = '_blank'; open.rel = 'noopener';
    var vmsg = el('p', 'cj-gallery-msg'); vmsg.setAttribute('role', 'status');
    [title, pos, flags, stars, tags, lab, nav, open, vmsg].forEach(function (n) { side.appendChild(n); });
    v.appendChild(stage); v.appendChild(side);
    v.addEventListener('click', function (e) { if (e.target === v || e.target === stage) closeView(); });
    img.addEventListener('load', bar_);
    window.addEventListener('resize', bar_);
    document.body.appendChild(v);
    return {root: v, img: img, bar: bar, title: title, pos: pos, flags: flags, stars: stars, tags: tags, note: note,
            open: open, msg: vmsg};
  }
  function findItem(cid, k) {                           // by comparison, not a selector: k is data from a file
    var all = root.querySelectorAll('[data-item]');
    for (var i = 0; i < all.length; i++) if (all[i].dataset.cid === cid && all[i].dataset.k === k) return all[i];
    return null;
  }
  function bar_() {
    if (!view || view.root.hidden || !cur) return;
    var nm = parseFloat(cur.dataset.nm), w = view.img.clientWidth, nw = view.img.naturalWidth, b = view.bar;
    if (!nm || !w || !nw || cur.hasAttribute('data-sb')) { b.hidden = true; return; }   // data-sb: the montage JPG has its own
    var per = nm * nw / w, best = 0;                  // nm per shown pixel
    NICE.forEach(function (n) { if (n / per <= w * 0.25) best = n; });
    if (!best) { b.hidden = true; return; }
    b.hidden = false;
    b.firstChild.style.width = (best / per) + 'px';
    b.lastChild.textContent = best >= 1000 ? (best / 1000) + ' µm' : best + ' nm';
  }
  function items() {
    return Array.prototype.filter.call(root.querySelectorAll('[data-item]'), function (n) {
      return !n.hidden && !n.closest('[hidden]');
    });
  }
  function refresh() {
    if (!view || view.root.hidden || !cur) return;
    var e = entry(cur.dataset.cid, cur.dataset.k);
    Array.prototype.forEach.call(view.flags.children, function (b) { b.setAttribute('aria-pressed', String(e.flag === b.dataset.flag)); });
    Array.prototype.forEach.call(view.stars.children, function (b) { b.setAttribute('aria-pressed', String(e.stars === parseInt(b.dataset.stars, 10))); });
    Array.prototype.forEach.call(view.tags.children, function (b) { b.setAttribute('aria-pressed', String(e.tags.indexOf(b.dataset.tag) >= 0)); });
    if (document.activeElement !== view.note) view.note.value = e.note;
    view.msg.textContent = msg ? msg.textContent : '';
  }
  function show(n) {
    if (!view) view = buildView();
    saveNote();
    cur = n;
    var list = items(), i = list.indexOf(n);
    view.title.textContent = n.dataset.kind === 'm' ? 'Montage ' + n.dataset.k : (n.title || n.dataset.k);
    view.pos.textContent = (i >= 0 ? (i + 1) + ' of ' + list.length + ' shown' : '');
    view.img.src = n.getAttribute('href');
    view.open.href = n.getAttribute('href');
    view.root.hidden = false;
    document.body.classList.add('cj-gallery-lock');
    view.bar.hidden = true;
    refresh();
    if (view.img.complete) bar_();
  }
  function closeView() {
    if (!view || view.root.hidden) return;
    saveNote();
    view.root.hidden = true; view.img.removeAttribute('src');
    document.body.classList.remove('cj-gallery-lock');
    var back = cur; cur = null;
    if (back) { var t = findItem(back.dataset.cid, back.dataset.k); if (t) t.focus(); }
  }
  function step(d) {
    var list = items(), i = list.indexOf(cur);
    if (i < 0 || !list.length) return;
    show(list[(i + d + list.length) % list.length]);
  }
  function toggleFlag(f) {
    if (!cur) return;
    var e = entry(cur.dataset.cid, cur.dataset.k);
    set(cur.dataset.cid, cur.dataset.k, 'flag', e.flag === f ? '' : f);
  }
  function setStars(n) { if (cur) set(cur.dataset.cid, cur.dataset.k, 'stars', n); }
  function toggleTag(t) {
    if (!cur) return;
    var e = entry(cur.dataset.cid, cur.dataset.k), tags = e.tags.slice(), i = tags.indexOf(t);
    if (i >= 0) tags.splice(i, 1); else tags.push(t);
    set(cur.dataset.cid, cur.dataset.k, 'tags', tags);
  }
  function saveNote() {
    clearTimeout(noteTimer);
    if (!view || !cur) return;
    var e = entry(cur.dataset.cid, cur.dataset.k);
    if (view.note.value !== e.note) set(cur.dataset.cid, cur.dataset.k, 'note', view.note.value);
  }

  root.addEventListener('click', function (e) {
    if (e.button || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
    var a = e.target.closest ? e.target.closest('a[data-k]') : null;
    if (!a || !root.contains(a)) return;
    var n = a;
    if (!a.hasAttribute('data-item')) {                 // a box / the montage image: open the matching list entry
      n = findItem(a.dataset.cid, a.dataset.k);
      if (!n || n.hidden || n.closest('[hidden]')) n = a;     // filtered out: still opens, just not in the next / prev list
    }
    e.preventDefault();
    show(n);
  });
  document.addEventListener('keydown', function (e) {
    if (!view || view.root.hidden || e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key, typing = e.target && (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT');
    if (k === 'Escape') { e.preventDefault(); if (typing) { e.target.blur(); } else closeView(); return; }
    if (typing) return;
    if (k === 'ArrowRight') { e.preventDefault(); step(1); }
    else if (k === 'ArrowLeft') { e.preventDefault(); step(-1); }
    else if (k === 'k' || k === 'K') { e.preventDefault(); toggleFlag('keep'); }
    else if (k === 'm' || k === 'M') { e.preventDefault(); toggleFlag('maybe'); }
    else if (k === 'r' || k === 'R') { e.preventDefault(); toggleFlag('reject'); }
    else if (/^[0-5]$/.test(k)) { e.preventDefault(); setStars(parseInt(k, 10)); }
    else if (k === 't' || k === 'T') { e.preventDefault(); var c = view.tags.querySelector('button'); if (c) c.focus(); }
    else if (k === 'n' || k === 'N') { e.preventDefault(); view.note.focus(); }
  });

  // ---- start: the marks of every conversion on the page
  Promise.all(cids.map(function (cid) {
    return fetch('/gallery/' + cid + '/api/state', {cache: 'no-store', credentials: 'same-origin'})
      .then(function (r) { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then(function (j) { S[cid] = j.state || {}; });
  })).then(paintAll).catch(function () { say('The marks could not be loaded. Reload the page before marking anything.'); });
})();
