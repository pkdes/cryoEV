// The review shell (/review/): the review tool runs in a same-origin iframe at <base>app/ under the CJ bar.
// - The server put the shell's query (?img=, ?mode=, ?who=; never k) into the iframe's src.
// - Every URL change inside the frame (load, pushState, replaceState, popstate, hashchange) is mirrored to this
//   window's address with history.replaceState: <base>app/?x#y -> <base>?x#y, so reload, bookmarks and Back work.
// - A same-origin link out of the tool (e.g. an error page's Home link) opens in this window instead; and if the
//   frame still ends up on another shell page, this window goes there (never a bar inside the bar).
(function () {
  'use strict';
  var f = document.getElementById('cj-review-frame');
  if (!f) return;
  var base = f.getAttribute('data-base') || '/review/', app = f.getAttribute('data-app') || base + 'app/';
  var title = document.title;
  // navigate this window, only ever to an absolute URL of this origin (never a path: "//host/x" would be another
  // host, and a same-origin pathname can start with "//", e.g. from "/.//evil.example/x")
  function go(href, replace) {
    var u;
    try { u = new URL(href); } catch (e) { return; }
    if (u.origin !== location.origin) return;
    if (replace) location.replace(u.href); else location.assign(u.href);
  }
  function sync() {
    var w = f.contentWindow, l;
    try { l = w.location; if (!l || l.href === 'about:blank') return; } catch (e) { return; }   // not same-origin: leave
    if (l.origin !== location.origin) return;
    var p = l.pathname;
    if (p.indexOf(app) !== 0) { go(l.href, true); return; }
    var sub = p.slice(app.length);
    if (sub === '' || sub === 'index.html') {
      var want = base + l.search + l.hash;
      if (want !== location.pathname + location.search + location.hash) {
        try { history.replaceState(history.state, '', want); } catch (e) { /* ignore */ }
      }
    }
    try { var t = w.document.title; document.title = t ? t + ' · ' + title : title; } catch (e) { /* ignore */ }
  }
  function hook() {
    var w;
    try { w = f.contentWindow; if (!w || w.location.href === 'about:blank') return; } catch (e) { return; }
    ['pushState', 'replaceState'].forEach(function (n) {
      var orig = w.history[n];
      if (typeof orig !== 'function' || orig.cjWrapped) return;
      var wrapped = function () { var r = orig.apply(w.history, arguments); sync(); return r; };
      wrapped.cjWrapped = true;
      w.history[n] = wrapped;
    });
    w.addEventListener('popstate', sync);
    w.addEventListener('hashchange', sync);
    // a same-origin link out of the tool (e.g. the 502 page's Home) opens in this window, not in the frame: the
    // shell's other pages refuse to be framed (X-Frame-Options DENY), so in the frame they would not load at all
    w.document.addEventListener('click', function (ev) {
      if (ev.button || ev.ctrlKey || ev.metaKey || ev.shiftKey || ev.altKey) return;
      var a = ev.target && ev.target.closest ? ev.target.closest('a[href]') : null;
      if (!a || (a.target && a.target !== '_self')) return;
      var u;
      try { u = new URL(a.getAttribute('href'), w.location.href); } catch (e) { return; }
      if (u.origin !== location.origin || u.pathname.indexOf(app) === 0) return;
      ev.preventDefault();
      go(u.href, false);
    }, true);
    sync();
    try { w.focus(); } catch (e) { /* ignore */ }      // keyboard shortcuts go to the tool
  }
  f.addEventListener('load', hook);
  try { if (f.contentDocument && f.contentDocument.readyState === 'complete') hook(); } catch (e) { /* ignore */ }
})();
