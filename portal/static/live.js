// CJLive: the History page's Live-tab rendering (registry/history_page.py, lineChart / runBox / queueBox / dur),
// copied as a small library for the portal's modules. History keeps its own copy; this one uses cj-lv- classes
// (styles in cj.css) and the shell's tokens (--s1..--s4 for series, --line, --mut, --acc, ...).
//
// API (window.CJLive):
//   CJLive.render(el, liveJson, {compact})  draw a live.json snapshot (jobs/monitor.py, schema live/1) into el,
//                                           replacing its content. compact:false = every queue (watcher, running job
//                                           with charts, queue, finished), as History's Live tab; compact:true = one
//                                           small block per running job (epoch bar, ETA, last val mAP), no charts.
//                                           liveJson null = "no snapshot". Returns el.
//   CJLive.lineChart(title, series, defs, fmt) -> <figure> with .draw(); series = {epoch:[..], <key>:[..]},
//                                           defs = [[key, label], ...] (up to 4), fmt = value formatter.
//   CJLive.runBox(running), CJLive.queueBox(queue, stale), CJLive.snapshot(running)   the building blocks
//   CJLive.dur(seconds), CJLive.f3(v), CJLive.isStale(live), CJLive.age(live)      helpers
//   CJLive.redraw()                         redraw every chart still on the page (done on resize already)
// Markup hook, no inline script needed:
//   <div data-cj-live="compact|full" data-cj-live-json="<id of a <script type=application/json>>"
//        data-cj-live-src="/history/live/live.json" data-cj-live-every="30"></div>
//   renders from the embedded JSON at once, then refetches data-cj-live-src (same origin) every N seconds.
// DOM is built with textContent only; nothing from live.json is ever parsed as HTML.
(function (root) {
  'use strict';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var charts = [], opened = {};
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function svgEl(tag, cls) {
    var e = document.createElementNS(SVGNS, tag);
    if (cls) e.setAttribute('class', cls);
    return e;
  }
  function dur(s) {
    if (s == null || isNaN(s)) return '?';
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + ' s';
    var m = Math.round(s / 60);
    if (m < 60) return m + ' min';
    var h = Math.floor(m / 60); m %= 60;
    if (h < 48) return h + ' h' + (m ? ' ' + m + ' min' : '');
    return Math.floor(h / 24) + ' d ' + (h % 24) + ' h';
  }
  function f3(v) { return v == null ? '?' : Number(v).toFixed(3); }
  function sig(v) { return v == null ? '?' : Number(v).toPrecision(3); }
  function age(live) { return live ? (Date.now() - Date.parse(live.generated_utc)) / 1000 : null; }
  function isStale(live) {
    var a = age(live);
    return a == null || !isFinite(a) || a > 3 * ((live && live.interval_s) || 30);
  }
  var CHIP = {done: 'is-done', alive: 'is-done', blocked: 'is-blocked', paused: 'is-blocked', waiting: 'is-blocked',
    'waiting for GPU': 'is-blocked', offline: 'is-blocked', 'access denied': 'is-blocked', stale: 'is-blocked',
    stopped: 'is-blocked', 'no watcher': 'is-blocked', running: 'is-run', cancelled: 'is-cancelled'};
  function chip(t) { return el('span', 'cj-lv-chip ' + (CHIP[t] || ''), t); }
  function stat(box, k, v, s) {
    var d = el('div', 'cj-lv-stat');
    d.appendChild(el('div', 'k', k)); d.appendChild(el('div', 'v', v));
    if (s) d.appendChild(el('div', 's', s));
    box.appendChild(d);
  }

  // a small line chart: one y axis from 0, hairline grid, legend + end labels, crosshair tooltip per epoch, table view
  function lineChart(title, S, defs, fmt) {
    fmt = fmt || sig;
    var fig = el('figure', 'cj-lv-chart'); fig.appendChild(el('figcaption', null, title));
    var leg = el('div', 'cj-lv-leg');
    defs.forEach(function (d, i) {
      var s = el('span'), k = el('span', 'cj-lv-key');
      k.style.borderTopColor = 'var(--s' + (i + 1) + ')'; s.appendChild(k); s.appendChild(document.createTextNode(d[1]));
      leg.appendChild(s);
    });
    fig.appendChild(leg);
    var plot = el('div', 'cj-lv-plot'); fig.appendChild(plot);
    var dt = el('details'); dt.open = !!opened[title];
    dt.addEventListener('toggle', function () { opened[title] = dt.open; });
    dt.appendChild(el('summary', null, 'Table'));
    var tw = el('div'), t = el('table', 'cj-lv-tab'), hr = el('tr');
    hr.appendChild(el('th', null, 'epoch')); defs.forEach(function (d) { hr.appendChild(el('th', null, d[1])); });
    t.appendChild(hr);
    S.epoch.forEach(function (e, j) {
      var tr = el('tr'); tr.appendChild(el('td', null, e));
      defs.forEach(function (d) { tr.appendChild(el('td', null, fmt((S[d[0]] || [])[j]))); });
      t.appendChild(tr);
    });
    tw.appendChild(t); dt.appendChild(tw); fig.appendChild(dt);
    fig.draw = function () {
      var W = plot.clientWidth; if (!W) return;
      var H = 180, ML = 44, MR = 46, MT = 8, MB = 24, pw = W - ML - MR, ph = H - MT - MB;
      plot.textContent = '';
      var svg = svgEl('svg'); svg.setAttribute('width', W); svg.setAttribute('height', H);
      svg.setAttribute('tabindex', '0'); svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', title + ', ' + S.epoch.length + ' epochs; table below');
      var xs = S.epoch, n = xs.length, ymax = 0;
      defs.forEach(function (d) { (S[d[0]] || []).forEach(function (v) { if (v != null && v > ymax) ymax = v; }); });
      var raw = (ymax || 1) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
      var step = (r <= 1 ? 1 : r <= 2 ? 2 : r <= 2.5 ? 2.5 : r <= 5 ? 5 : 10) * mag, top = step * Math.ceil((ymax || 1) / step - 1e-9);
      var x0 = xs[0], x1 = n > 1 ? xs[n - 1] : xs[0] + 1;
      function X(e) { return ML + (n > 1 ? (e - x0) / (x1 - x0) * pw : pw / 2); }
      function Y(v) { return MT + ph - v / top * ph; }
      for (var g = 0; g <= top + step / 2; g += step) {
        var ln = svgEl('line', 'cj-lv-grid'); ln.setAttribute('x1', ML); ln.setAttribute('x2', ML + pw);
        ln.setAttribute('y1', Y(g)); ln.setAttribute('y2', Y(g)); svg.appendChild(ln);
        var tx = svgEl('text', 'cj-lv-axis'); tx.setAttribute('x', ML - 6); tx.setAttribute('y', Y(g) + 4);
        tx.setAttribute('text-anchor', 'end');
        tx.textContent = step < 0.01 ? g.toExponential(0) : String(+g.toFixed(4)); svg.appendChild(tx);
      }
      var xr = (x1 - x0) / 4, xm = Math.pow(10, Math.floor(Math.log10(Math.max(xr, 1)))), xq = xr / xm;
      var xstep = Math.max(1, (xq <= 1 ? 1 : xq <= 2 ? 2 : xq <= 5 ? 5 : 10) * xm);
      for (var e = Math.ceil(x0 / xstep) * xstep; e <= x1; e += xstep) {
        if (n < 2 && e !== x0) continue;
        var xt = svgEl('text', 'cj-lv-axis'); xt.setAttribute('x', X(e)); xt.setAttribute('y', H - 6);
        xt.setAttribute('text-anchor', 'middle'); xt.textContent = e; svg.appendChild(xt);
      }
      var ends = [];
      defs.forEach(function (d, i) {
        var v = S[d[0]] || [], p = '', pen = false, last = null;
        for (var j = 0; j < n; j++) {
          if (v[j] == null) { pen = false; continue; }
          p += (pen ? 'L' : 'M') + X(xs[j]).toFixed(1) + ',' + Y(v[j]).toFixed(1); pen = true; last = j;
        }
        if (last == null) return;
        if (n === 1) {
          var c = svgEl('circle', 'cj-lv-f' + (i + 1)); c.setAttribute('cx', X(xs[0])); c.setAttribute('cy', Y(v[0]));
          c.setAttribute('r', 4); svg.appendChild(c);
        } else {
          var pa = svgEl('path', 'cj-lv-line cj-lv-s' + (i + 1)); pa.setAttribute('d', p); svg.appendChild(pa);
        }
        ends.push({y: Y(v[last]), x: X(xs[last]), t: d[1]});
      });
      ends.sort(function (a, b) { return a.y - b.y; });
      var prev = -1e9;   // end labels; one that would overlap the last is left to the legend
      ends.forEach(function (L) {
        if (L.y - prev < 12) return; prev = L.y;
        var t2 = svgEl('text', 'cj-lv-lab'); t2.setAttribute('x', L.x + 6); t2.setAttribute('y', L.y + 4);
        t2.textContent = L.t; svg.appendChild(t2);
      });
      var xh = svgEl('line', 'cj-lv-xh'); xh.setAttribute('y1', MT); xh.setAttribute('y2', MT + ph);
      xh.style.display = 'none'; svg.appendChild(xh);
      var dots = defs.map(function (d, i) {
        var c = svgEl('circle', 'cj-lv-dot cj-lv-f' + (i + 1)); c.setAttribute('r', 4); c.style.display = 'none';
        svg.appendChild(c); return c;
      });
      var hit = svgEl('rect'); hit.setAttribute('x', ML); hit.setAttribute('y', 0); hit.setAttribute('width', pw);
      hit.setAttribute('height', H); hit.setAttribute('fill', 'transparent'); svg.appendChild(hit);
      plot.appendChild(svg);
      var tip = el('div', 'cj-lv-tip'); tip.style.display = 'none'; plot.appendChild(tip);
      var cur = n - 1;
      function showAt(j) {
        cur = Math.max(0, Math.min(n - 1, j)); var x = X(xs[cur]);
        xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.style.display = '';
        tip.textContent = ''; tip.appendChild(el('div', 'n', 'Epoch ' + xs[cur]));
        defs.forEach(function (d, i) {
          var v = (S[d[0]] || [])[cur], rr = el('div', 'r'), k = el('span', 'cj-lv-key');
          k.style.borderTopColor = 'var(--s' + (i + 1) + ')';
          rr.appendChild(k); rr.appendChild(el('b', null, fmt(v))); rr.appendChild(el('span', 'n', d[1])); tip.appendChild(rr);
          if (v == null) dots[i].style.display = 'none';
          else { dots[i].setAttribute('cx', x); dots[i].setAttribute('cy', Y(v)); dots[i].style.display = ''; }
        });
        tip.style.display = ''; var tw2 = tip.offsetWidth;
        tip.style.left = (x + 12 + tw2 > W ? Math.max(0, x - 12 - tw2) : x + 12) + 'px';
      }
      function hide() { xh.style.display = 'none'; tip.style.display = 'none'; dots.forEach(function (c) { c.style.display = 'none'; }); }
      function near(ev) {
        var b = svg.getBoundingClientRect(), px = ev.clientX - b.left, best = 0;
        xs.forEach(function (e2, j) { if (Math.abs(X(e2) - px) < Math.abs(X(xs[best]) - px)) best = j; });
        return best;
      }
      svg.addEventListener('pointermove', function (ev) { showAt(near(ev)); });
      svg.addEventListener('pointerdown', function (ev) { showAt(near(ev)); });
      svg.addEventListener('pointerleave', hide);
      svg.addEventListener('focus', function () { showAt(cur); });
      svg.addEventListener('blur', hide);
      svg.addEventListener('keydown', function (ev) {
        if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') { ev.preventDefault(); showAt(cur + (ev.key === 'ArrowLeft' ? -1 : 1)); }
      });
    };
    charts.push(fig);
    return fig;
  }

  function epochBar(r) {
    var bar = el('div', 'cj-lv-pbar'), fill = el('i');
    fill.style.width = Math.min(100, Math.round(100 * r.epoch / r.epochs)) + '%';
    bar.appendChild(fill);
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuemin', '0'); bar.setAttribute('aria-valuemax', String(r.epochs));
    bar.setAttribute('aria-valuenow', String(r.epoch));
    return bar;
  }
  function etaText(r) {   // -> [value, note]
    if (r.phase === 'finishing') return ['final validation', null];
    if (r.eta_patience_bound_s != null)
      return ['≤ ' + dur(r.eta_patience_bound_s), 'early stop unless val improves (best epoch ' + r.best_epoch +
        ', patience ' + r.patience + '); all epochs ' + dur(r.eta_s)];
    return [r.eta_s == null ? '?' : dur(r.eta_s), r.eta_capped ? 'time cap: max_hours ' + r.max_hours :
      (r.epoch_median_s != null ? 'median epoch ' + dur(r.epoch_median_s) : 'after epoch 2')];
  }
  function lastOf(a) {
    for (var i = (a || []).length - 1; i >= 0; i--) if (a[i] != null) return a[i];
    return null;
  }
  function warnings(b, r) {
    if (r.cancel_requested) b.appendChild(el('div', 'cj-lv-why', 'Cancel requested: the runner stops the job within ~10 s (the partial run is kept).'));
    if (r.hung) b.appendChild(el('div', 'cj-lv-why', 'Hung? No new epoch for ' + dur(r.last_epoch_age_s) + ' (expected every ' + dur(r.epoch_median_s) + '). Check train.log on the desktop.'));
  }

  function runBox(r) {
    var b = el('div', 'cj-lv-run'), top = el('div', 'cj-lv-top');
    top.appendChild(el('span', 'cj-lv-rid', r.job)); top.appendChild(chip('running'));
    if (r.phase) top.appendChild(el('span', 'cj-lv-chip', r.phase));
    b.appendChild(top);
    if (r.run) b.appendChild(el('div', 'cj-lv-note', 'run ' + r.run));
    warnings(b, r);
    if (r.batch_halved) b.appendChild(el('div', 'cj-lv-why', 'batch ' + r.batch + ' of ' + r.batch_asked + ' asked: Ultralytics halved it after running out of GPU memory in epoch 1.'));
    if (r.epoch != null && r.epochs) { b.appendChild(el('div', 'cj-lv-ep', 'Epoch ' + r.epoch + ' / ' + r.epochs)); b.appendChild(epochBar(r)); }
    else b.appendChild(el('div', 'cj-lv-ep', r.phase === 'setting up' ? 'Setting up: no epoch finished yet' : 'Epoch ?'));
    var st = el('div', 'cj-lv-stats'), eta = etaText(r);
    stat(st, 'Elapsed', dur(r.elapsed_s), r.last_epoch_age_s != null ? 'last epoch ' + dur(r.last_epoch_age_s) + ' ago' : null);
    stat(st, 'ETA', eta[0], eta[1]);
    stat(st, 'Batch', r.batch != null ? (r.batch_halved ? r.batch + ' of ' + r.batch_asked + ' asked' : String(r.batch)) : (r.batch_note || 'unknown'),
      r.batch == null && r.batch_asked != null ? 'asked ' + r.batch_asked : null);
    stat(st, 'GPU memory', r.gpu_reserved_gb != null ? r.gpu_reserved_gb + ' GB' : 'unknown', 'reserved by the trainer');
    stat(st, 'Val mask mAP50-95', f3(r.map50_95), 'mAP50 ' + f3(r.map50));
    b.appendChild(st);
    var S = r.series;
    if (S && S.epoch && S.epoch.length) {
      var cs = el('div', 'cj-lv-charts');
      cs.appendChild(lineChart('Train losses', S, [['box', 'box'], ['seg', 'seg'], ['cls', 'cls'], ['dfl', 'dfl']], sig));
      cs.appendChild(lineChart('Val mask mAP', S, [['map50', 'mAP50'], ['map50_95', 'mAP50-95']], f3));
      b.appendChild(cs);
    }
    return b;
  }

  // compact: one running job as a small block (home's training snapshot)
  function snapshot(r) {
    var b = el('div', 'cj-lv-snap'), top = el('div', 'cj-lv-top');
    top.appendChild(el('span', 'cj-lv-rid', r.job || '?'));
    if (r.phase) top.appendChild(el('span', 'cj-lv-chip is-run', r.phase));
    b.appendChild(top);
    var ep = el('div', 'cj-lv-ep');
    if (r.epoch != null && r.epochs) {
      ep.appendChild(document.createTextNode('Epoch ' + r.epoch + ' '));
      ep.appendChild(el('small', null, '/ ' + r.epochs + ' · ' + Math.min(100, Math.round(100 * r.epoch / r.epochs)) + '%'));
      b.appendChild(ep); b.appendChild(epochBar(r));
    } else {
      ep.textContent = r.phase === 'setting up' ? 'Setting up' : 'Epoch ?';
      b.appendChild(ep);
    }
    var eta = etaText(r), s1 = el('div', 's');
    s1.appendChild(document.createTextNode('ETA ')); s1.appendChild(el('b', null, eta[0]));
    if (r.elapsed_s != null) s1.appendChild(document.createTextNode(' · running ' + dur(r.elapsed_s)));
    b.appendChild(s1);
    var S = r.series || {}, m95 = r.map50_95 != null ? r.map50_95 : lastOf(S.map50_95), m50 = r.map50 != null ? r.map50 : lastOf(S.map50);
    var s2 = el('div', 's');
    s2.appendChild(document.createTextNode('Last val mask mAP50-95 ')); s2.appendChild(el('b', 'cj-lv-map', f3(m95)));
    s2.appendChild(document.createTextNode(' · mAP50 ' + f3(m50)));
    b.appendChild(s2);
    warnings(b, r);
    return b;
  }

  function queueBox(q, stale) {
    var s = el('section', 'cj-lv-q'); s.appendChild(el('h3', null, q.share));
    if (q.reach !== 'ok') {
      var w = el('div', 'cj-lv-w'); w.appendChild(chip(q.reach || 'offline')); w.appendChild(el('span', 'cj-lv-note', q.detail || ''));
      s.appendChild(w); return s;
    }
    var W = q.watcher || {}, w2 = el('div', 'cj-lv-w');
    w2.appendChild(el('span', 'cj-lv-note', 'Watcher')); w2.appendChild(chip(W.display || 'unknown'));
    w2.appendChild(el('span', 'cj-lv-note', (W.age_s != null ? 'heartbeat ' + dur(W.age_s) + ' ago' : 'no heartbeat') +
      (W.machine ? ' · ' + W.machine : '') + (W.detail ? ' · ' + W.detail : '')));
    s.appendChild(w2);
    if (q.running && !stale) s.appendChild(runBox(q.running));
    else if (!stale) s.appendChild(el('div', 'cj-lv-note', 'No job running.'));
    var jobs = q.jobs || [], fin = q.finished || [];
    s.appendChild(el('h4', null, 'Queue (' + jobs.length + ')'));
    if (jobs.length) {
      var t = el('table', 'cj-lv-tab');
      jobs.forEach(function (j) {
        var tr = el('tr'); tr.appendChild(el('td', null, j.job));
        var td = el('td'); td.appendChild(chip(stale && j.state === 'running' ? 'unknown (stale)' : j.state));
        var note = [j.cancel_requested && !j.reason ? 'cancel requested' : null, j.reason,
          j.state === 'done' || j.state === 'blocked' || j.state === 'cancelled' ? (j.pulled ? 'pulled' : 'not pulled') : null]
          .filter(function (x) { return x; }).join(' · ');
        if (note) td.appendChild(el('div', 'cj-lv-note', note));
        tr.appendChild(td); t.appendChild(tr);
      });
      s.appendChild(t);
    } else s.appendChild(el('div', 'cj-lv-note', 'No jobs in q/.'));
    s.appendChild(el('h4', null, 'Recent finished (' + fin.length + ')'));
    fin.forEach(function (f) {
      var d = el('div', 'cj-lv-fin'), tp = el('div', 'cj-lv-top');
      tp.appendChild(el('span', 'cj-lv-rid', f.job)); tp.appendChild(chip(f.status)); d.appendChild(tp);
      var m = f.metrics || {};
      d.appendChild(el('div', 'cj-lv-note', [m.mask_map50_95 != null ? 'mask mAP50-95 ' + f3(m.mask_map50_95) : null,
        m.mask_map50 != null ? 'mAP50 ' + f3(m.mask_map50) : null, f.best_epoch != null ? 'best epoch ' + f.best_epoch : null,
        f.ended ? 'ended ' + f.ended : null, f.pulled ? 'pulled' : 'not pulled'].filter(function (x) { return x; }).join(' · ')));
      if (f.reason) d.appendChild(el('div', 'cj-lv-why', f.reason));
      s.appendChild(d);
    });
    if (!fin.length) s.appendChild(el('div', 'cj-lv-note', 'None yet.'));
    return s;
  }

  function render(target, live, opts) {
    opts = opts || {};
    charts = charts.filter(function (f) { return f.isConnected && !target.contains(f); });
    target.textContent = '';
    if (!live) { target.appendChild(el('div', 'cj-lv-note', 'No live snapshot (is jobs/monitor.py running?)')); return target; }
    var stale = isStale(live), queues = live.queues || [];
    if (opts.compact) {
      if (stale) {
        target.appendChild(el('div', 'cj-lv-note', 'Snapshot stale (' + dur(age(live)) + ' old): no job shown as running.'));
        return target;
      }
      var runs = queues.filter(function (q) { return q && q.reach === 'ok' && q.running; });
      if (!runs.length) target.appendChild(el('div', 'cj-lv-note', 'No job running.'));
      runs.forEach(function (q) { target.appendChild(snapshot(q.running)); });
      return target;
    }
    var meta = el('div', 'cj-lv-meta', 'Updated ' + dur(age(live)) + ' ago · every ' + dur(live.interval_s || 30) +
      ' · monitor on ' + ((live.monitor && live.monitor.machine) || '?'));
    target.appendChild(meta);
    if (stale) target.appendChild(el('div', 'cj-lv-stale', 'Stale: this snapshot is ' + dur(age(live)) +
      ' old (the monitor writes every ' + dur(live.interval_s || 30) + '). Nothing below is current and no job is shown as running.'));
    queues.forEach(function (q) { target.appendChild(queueBox(q, stale)); });
    redraw();
    return target;
  }
  function redraw() {
    charts = charts.filter(function (f) { return f.isConnected; });
    charts.forEach(function (f) { f.draw(); });
  }

  function getJSON(url) {
    return fetch(url, {cache: 'no-store', credentials: 'same-origin'}).then(function (r) {
      if (!r.ok) throw new Error(url + ': ' + r.status);
      return r.json();
    });
  }
  function autoload() {
    Array.prototype.forEach.call(document.querySelectorAll('[data-cj-live]'), function (box) {
      if (box.dataset.cjLiveOn) return;
      box.dataset.cjLiveOn = '1';
      var compact = box.dataset.cjLive === 'compact', data = null;
      var src = box.dataset.cjLiveJson && document.getElementById(box.dataset.cjLiveJson);
      if (src) { try { data = JSON.parse(src.textContent); } catch (e) { data = null; } }
      if (src) render(box, data, {compact: compact});
      var url = box.dataset.cjLiveSrc, every = Math.max(5, Number(box.dataset.cjLiveEvery) || 30);
      if (!url || url.charAt(0) !== '/' || url.charAt(1) === '/') return;     // same origin only
      function load() { getJSON(url).then(function (j) { render(box, j, {compact: compact}); }, function () {}); }
      if (!src) load();
      setInterval(load, every * 1000);
    });
  }
  var rt = null;
  window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(redraw, 150); });
  root.CJLive = {render: render, lineChart: lineChart, runBox: runBox, queueBox: queueBox, snapshot: snapshot,
    dur: dur, f3: f3, age: age, isStale: isStale, redraw: redraw, autoload: autoload};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', autoload);
  else autoload();
})(window);
