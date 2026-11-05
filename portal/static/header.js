// CJ shell top bar: wordmark + nav from /cj/modules.json + user chip, on any page that includes this file
// (home, and history.html as the portal serves it). Hides the history page's own Servers bar (#srv): the nav shows
// the same links, with dots from the live snapshot. Same-origin requests only; DOM built with textContent.
(function () {
  'use strict';
  var PUB = ['', '443', '8443', '8445', '8446'];   // tailscale serve's https ports
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
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
  function current(route) {
    var p = location.pathname;
    if (route === '/') return p === '/';
    return p.indexOf(route) === 0;
  }
  function build(brand, mods, me) {
    if (document.getElementById('cj-bar')) return null;
    var bar = el('header', 'cj-bar');
    bar.id = 'cj-bar';
    var mark = el('a', 'cj-mark', (brand && (brand.wordmark || brand.name)) || 'CJ');
    mark.href = '/';
    mark.title = (brand && brand.tagline) || '';
    bar.appendChild(mark);
    var nav = el('nav', 'cj-nav');
    nav.setAttribute('aria-label', 'Modules');
    var role = me && me.role;
    (mods || []).forEach(function (m) {
      if (!m.nav || m.status !== 'on' || (m.roles || []).indexOf(role) < 0) return;
      var href = m.external ? extHref(m.external) : m.route;
      if (!href) return;
      var a = el('a', null);
      a.href = href;
      a.dataset.id = m.id;
      if (m.health) { var d = el('span', 'cj-dot'); d.dataset.health = m.health; d.title = 'unknown'; a.appendChild(d); }
      a.appendChild(document.createTextNode(m.name));
      if (!m.external && current(m.route)) a.setAttribute('aria-current', 'page');
      nav.appendChild(a);
    });
    bar.appendChild(nav);
    if (me && me.id) { var u = el('span', 'cj-user', me.id); u.title = 'Signed in as ' + me.id + ' (' + me.role + ')'; bar.appendChild(u); }
    document.body.insertBefore(bar, document.body.firstChild);
    return bar;
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
    Promise.all([getJSON('/cj/brand.json'), getJSON('/cj/modules.json'), getJSON('/cj/me.json')])
      .then(function (r) {
        var bar = build(r[0], (r[1] || {}).modules, r[2]);
        hideSrv();
        return getJSON('/history/live/live.json').then(function (live) { dots(bar, live); }, function () {});
      })
      .catch(function () { /* not signed in or offline: the page works without the bar */ });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
