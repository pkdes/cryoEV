// CryoEV shell top bar: wordmark (grid-atlas mark + brand.json's wordmark), nav from /cj/modules.json filtered by the
// user's role (/cj/me.json), user chip. On every page that includes this file: the shell's own pages (base.html has
// an empty #cj-bar placeholder of the bar's final height) and history.html as the portal serves it (the bar is
// inserted at the top of <body>; cj.css holds its height free until then, and hides History's own #srv bar).
// Nav: modules with nav:true whose roles include the user's role; status "on" = link, "coming" = greyed, not a link;
// "external" (sandbox) = its own port, marked with an arrow. Dots from the live snapshot, only for roles that may read
// it (the History module's roles). Same-origin requests only; DOM built with textContent / createElementNS.
(function () {
  'use strict';
  var PUB = ['', '443', '8443', '8445', '8446'];   // tailscale serve's https ports
  var SVG = 'http://www.w3.org/2000/svg';
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function svg(tag, attrs) {
    var e = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }
  function getJSON(url) {
    return fetch(url, {cache: 'no-store', credentials: 'same-origin'}).then(function (r) {
      if (!r.ok) throw new Error(url + ': ' + r.status);
      return r.json();
    });
  }
  function hideSrv() {
    var s = document.getElementById('srv');
    if (s) { s.hidden = true; s.style.display = 'none'; }
  }
  function extHref(x) {
    if (!x) return null;
    var pub = location.protocol === 'https:' || PUB.indexOf(location.port) >= 0;
    return (pub ? 'https:' : 'http:') + '//' + location.hostname + ':' + (pub ? x.public_port : x.local_port) + (x.path || '/');
  }
  function activeId(mods) {
    var a = document.body && document.body.dataset.active;
    if (a) return a;
    var p = location.pathname, best = null, len = -1;
    (mods || []).forEach(function (m) {
      var r = m.route;
      if (!r || m.external) return;
      var hit = r === '/' ? p === '/' : p.indexOf(r) === 0;
      if (hit && r.length > len) { best = m.id; len = r.length; }
    });
    return best;
  }
  // the grid-atlas mark: 3 x 3 squares, one lit
  function mark(brand) {
    var a = el('a', 'cj-mark');
    a.href = '/';
    var name = brand ? (brand.wordmark || brand.name || 'CryoEV') : '';   // no brand yet: the mark alone (no flash)
    a.setAttribute('aria-label', (name ? name + ' ' : '') + 'home');
    if (brand && brand.tagline) a.title = brand.tagline;
    var s = svg('svg', {viewBox: '0 0 21 21', 'aria-hidden': 'true', focusable: 'false'});
    for (var k = 0; k < 9; k++) {
      var r = svg('rect', {x: 1 + (k % 3) * 7, y: 1 + Math.floor(k / 3) * 7, width: 5, height: 5, rx: 0.6});
      if (k === 5) r.setAttribute('class', 'on');
      s.appendChild(r);
    }
    a.appendChild(s);
    if (name) a.appendChild(el('b', null, name));
    return a;
  }
  function extIcon() {
    var s = svg('svg', {viewBox: '0 0 24 24', 'class': 'cj-ext', 'aria-hidden': 'true', focusable: 'false'});
    s.appendChild(svg('path', {d: 'M8 16L16 8M10 8h6v6'}));
    return s;
  }
  // the bar element: the template's placeholder, or a new one at the top of <body> (History)
  function barEl() {
    var bar = document.getElementById('cj-bar');
    if (!bar) {
      bar = el('div', 'cj-bar');
      bar.id = 'cj-bar';
      document.body.insertBefore(bar, document.body.firstChild);
    }
    bar.setAttribute('role', 'banner');
    document.documentElement.classList.add('cj-barred');   // cj.css stops holding the space: the bar holds it now
    return bar;
  }
  function fill(bar, brand, mods, me) {
    if (bar.dataset.built) return null;
    bar.dataset.built = '1';
    bar.textContent = '';
    bar.appendChild(mark(brand));
    var nav = el('nav', 'cj-nav');
    nav.setAttribute('aria-label', 'Modules');
    var role = me && me.role, active = activeId(mods);
    (mods || []).forEach(function (m) {
      if (!m.nav || (m.roles || []).indexOf(role) < 0) return;
      if (m.status !== 'on') {
        var off = el('span', 'cj-off', m.name);
        off.dataset.id = m.id;
        off.setAttribute('aria-disabled', 'true');
        off.title = m.name + ': coming';
        nav.appendChild(off);
        return;
      }
      var href = m.external ? extHref(m.external) : m.route;
      if (!href) return;
      var a = el('a', null);
      a.href = href;
      a.dataset.id = m.id;
      if (m.health) { var d = el('span', 'cj-dot'); d.dataset.health = m.health; d.title = 'unknown'; a.appendChild(d); }
      a.appendChild(document.createTextNode(m.name));
      if (m.external) { a.appendChild(extIcon()); a.title = m.name + ' (its own address)'; }
      else if (m.id === active) a.setAttribute('aria-current', 'page');
      nav.appendChild(a);
    });
    bar.appendChild(nav);
    if (me && me.id) {   // Feedback: on every page; the page it was clicked on is prefilled (path only, never the query)
      var fb = el('a', 'cj-feedback-link', 'Feedback');
      fb.href = '/feedback/?page=' + encodeURIComponent(location.pathname);
      fb.title = 'Report a bug or suggest something';
      if (location.pathname.indexOf('/feedback') === 0) fb.setAttribute('aria-current', 'page');
      bar.appendChild(fb);
    }
    if (me && me.id) {
      var u = el('span', 'cj-user');
      u.title = 'Signed in as ' + me.id + ' (' + me.role + ')';
      u.appendChild(el('span', 'cj-av', String(me.id).slice(0, 2)));
      var nm = el('span', 'cj-nm');
      nm.appendChild(el('span', null, me.id));
      nm.appendChild(el('small', null, me.role));
      u.appendChild(nm);
      bar.appendChild(u);
    }
    if (me && me.signout === '/cdn-cgi/access/logout') {   // Cloudflare access mode only (me.json says so)
      var so = el('a', 'cj-signout', 'Sign out');
      so.href = me.signout;
      bar.appendChild(so);
    }
    return bar;
  }
  // the running code's version ("Updated <date time> . <commit>"), small, at the foot of every page
  function stamp(ver, bar) {
    if (!ver || !ver.version || document.getElementById('cj-version')) return;
    var v = el('div', 'cj-version', ver.version);
    v.id = 'cj-version';
    if (document.getElementById('cj-review-frame')) { v.className = 'cj-version cj-version-bar'; bar.appendChild(v); }   // the full-height frame page: no room below
    else document.body.appendChild(v);
  }
  function dots(bar, live) {
    if (!bar || !live) return;
    var age = (Date.now() - Date.parse(live.generated_utc)) / 1000;
    var fresh = isFinite(age) && age <= 3 * (live.interval_s || 30);
    var by = {};
    (live.servers || []).forEach(function (s) { if (s && s.name) by[s.name] = s; });
    Array.prototype.forEach.call(bar.querySelectorAll('.cj-dot[data-health]'), function (d) {
      var s = by[d.dataset.health];
      var st = !fresh || !s ? 'unk' : s.up ? 'up' : 'down';
      d.className = 'cj-dot ' + st;
      d.title = st === 'unk' ? 'unknown' : st;
    });
  }
  function start() {
    hideSrv();
    var bar = barEl();
    if (!bar.firstChild) bar.appendChild(mark(null));    // the wordmark at once; the rest when the JSON arrives
    Promise.all([getJSON('/cj/brand.json'), getJSON('/cj/modules.json'), getJSON('/cj/me.json')])
      .then(function (r) {
        var mods = (r[1] || {}).modules || [], me = r[2] || {};
        fill(bar, r[0], mods, me);
        getJSON('/cj/version.json').then(function (ver) { stamp(ver, bar); }, function () {});
        hideSrv();
        var hist = mods.filter(function (m) { return m.id === 'history'; })[0];
        if (!hist || (hist.roles || []).indexOf(me.role) < 0) return null;   // the snapshot is History's data
        return getJSON('/history/live/live.json').then(function (live) { dots(bar, live); }, function () {});
      })
      .catch(function () { /* not signed in or offline: the page works with the wordmark only */ });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
