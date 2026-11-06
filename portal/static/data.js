/* Data module's Lab graph (served as /data/data.js by portal/modules/data.py; no inline script on the page).
   Reads /data/api/lineage/graph (with this page's ?focus) and draws History's lineage graph: the same layout as
   registry/history_page.py drawGraph (columns, boxes over an SVG edge layer, bezier edges, same-column lanes on the
   left, back-edges routed over the top). Record text only ever goes in through textContent. */
(function () {
  'use strict';
  var canvas = document.getElementById('cj-data-graph-canvas');
  if (!canvas) return;
  var svg = document.getElementById('cj-data-graph-svg'), NS = svg.namespaceURI;
  var histBox = document.getElementById('cj-data-graph-hist'), anBox = document.getElementById('cj-data-graph-an');
  var status = document.getElementById('cj-data-graph-status');
  var BW = 184, BH = 74, GAP = 60, PAD = 30, HEAD = 30, VG = 12, LANE = 7;
  var G = null, drawn = [];

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = String(text);
    return e;
  }
  function svgEl(tag, cls) {
    var e = document.createElementNS(NS, tag);
    if (cls) e.setAttribute('class', cls);
    return e;
  }
  function arrow(x, y, dir) {
    return 'M' + x + ',' + y + 'L' + (x - 8 * dir) + ',' + (y - 4.5) + 'L' + (x - 8 * dir) + ',' + (y + 4.5) + 'Z';
  }
  // an analysis: a prediction set, or its prediction review package (id pred_...); hidden until "Show analyses"
  function isAnalysis(n) {
    return n.kind === 'predictions' || (n.kind === 'package' && n.title.indexOf('pred_') === 0);
  }
  function isProd(n) { return n.kind === 'model' && n.status === 'production'; }

  function box(i, p) {
    var n = G.nodes[i];
    var cls = 'cj-data-graph-node' + (n.part >= 0 ? ' cj-data-graph-part' : '') + (isProd(n) ? ' cj-data-graph-prod' : '')
      + (n.hist ? ' cj-data-graph-dim' : '') + (i === G.focus ? ' cj-data-graph-focus' : '');
    var b;
    if (n.url) { b = el('a', cls); b.href = n.url; } else { b = el('div', cls + ' cj-data-graph-nolink'); }
    b.title = n.kind + ': ' + n.title + (n.url ? '' : ' (no Data page)');
    b.style.left = p.x + 'px'; b.style.top = p.y + 'px'; b.style.width = p.w + 'px'; b.style.height = BH + 'px';
    var l1 = el('span', 'cj-data-graph-l1');
    l1.appendChild(el('span', 'cj-data-graph-tag', n.kind));
    if (isProd(n)) l1.appendChild(el('span', 'cj-data-graph-st cj-data-graph-stprod', '★ production'));
    else if (n.status != null) l1.appendChild(el('span', 'cj-data-graph-st', String(n.status).replace('_', ' ')));
    b.appendChild(l1);
    b.appendChild(el('span', 'cj-data-graph-id', n.title));
    if (n.sub) {
      var sc = n.kind === 'label version' ? (n.sub.indexOf('verified') === 0 ? ' cj-data-graph-ver' : ' cj-data-graph-unrev') : '';
      b.appendChild(el('span', 'cj-data-graph-sub' + sc, n.sub));
    }
    if (i === G.focus) b.setAttribute('aria-current', 'true');
    return b;
  }

  function draw() {
    var hist = histBox.checked, an = anBox.checked, ncol = G.cols.length;
    drawn.forEach(function (b) { b.remove(); }); drawn = [];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var vis = G.nodes.map(function (n) { return (hist || !n.hist) && (an || !isAnalysis(n)); });
    var cols = [], kids = {}, parents = {}, pos = {}, maxr = 0;
    for (var c = 0; c < ncol; c++) cols.push([]);
    G.nodes.forEach(function (n, i) {
      if (!vis[i]) return;
      if (n.part >= 0 && vis[n.part]) { (kids[n.part] = kids[n.part] || []).push(i); return; }
      cols[n.col].push(i);
    });
    G.edges.forEach(function (e) { (parents[e[1]] = parents[e[1]] || []).push(e[0]); });
    var back = G.edges.filter(function (e) { return vis[e[0]] && vis[e[1]] && G.nodes[e[0]].col > G.nodes[e[1]].col; });
    var top = HEAD + back.length * LANE + (back.length ? 10 : 0);
    var ch = {};
    G.edges.forEach(function (e) { (ch[e[0]] = ch[e[0]] || []).push(e[1]); });
    function mean(ks) { return ks.length ? ks.reduce(function (a, k) { return a + pos[k].r; }, 0) / ks.length : 1e9; }
    function byParents(c) {
      return function (i) { return mean((parents[i] || []).filter(function (p) { return pos[p] && pos[p].c < c; })); };
    }
    function place(c, score) {   // order by score (mean row of linked boxes), then build order; parts follow their package
      var o = cols[c].map(function (i, j) { return [score(i), j, i]; });
      o.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
      var r = 0, x = PAD + c * (BW + GAP);
      o.forEach(function (t) {
        pos[t[2]] = {c: c, r: r, x: x, w: BW, y: top + r * (BH + VG)}; r++;
        (kids[t[2]] || []).forEach(function (j) { pos[j] = {c: c, r: r, x: x + 18, w: BW - 18, y: top + r * (BH + VG)}; r++; });
      });
      return r;
    }
    cols.forEach(function (list, c) { place(c, byParents(c)); });
    // 2nd pass: sources near their children
    place(0, function (i) { return mean((ch[i] || []).filter(function (k) { return pos[k] && pos[k].c > 0; })); });
    for (var c2 = 1; c2 < ncol; c2++) place(c2, byParents(c2));
    cols.forEach(function (list) {
      maxr = Math.max(maxr, list.length + list.reduce(function (a, i) { return a + (kids[i] || []).length; }, 0));
    });
    var W = 2 * PAD + ncol * BW + (ncol - 1) * GAP + back.length * LANE + 10, H = top + maxr * (BH + VG) + PAD;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    svg.setAttribute('width', W); svg.setAttribute('height', H);
    G.cols.forEach(function (t, c) {
      var h = el('div', 'cj-data-graph-head', t);
      h.style.left = (PAD + c * (BW + GAP)) + 'px';
      canvas.appendChild(h); drawn.push(h);
    });
    var same = {}, bl = 0;
    G.edges.forEach(function (e) {
      var a = pos[e[0]], b = pos[e[1]];
      if (!a || !b) return;
      var ay = a.y + BH / 2, by = b.y + BH / 2, d, dir, ex, cls = 'cj-data-graph-e';
      if (a.c < b.c) {
        var x1 = a.x + a.w, x2 = b.x, m = (x1 + x2) / 2;
        d = 'M' + x1 + ',' + ay + 'C' + m + ',' + ay + ' ' + m + ',' + by + ' ' + x2 + ',' + by; dir = 1; ex = x2;
      } else if (a.c === b.c) {
        var n = same[a.c] = (same[a.c] || 0) + 1, lx = PAD + a.c * (BW + GAP) - 8 - n * LANE;
        d = 'M' + a.x + ',' + ay + 'H' + lx + 'V' + by + 'H' + b.x; dir = 1; ex = b.x;
      } else {
        var lane = bl++, band = HEAD + 4 + lane * LANE, rx = PAD + ncol * BW + (ncol - 1) * GAP + 6 + lane * LANE,
          tx = b.x + b.w + 10 + lane * LANE;
        d = 'M' + (a.x + a.w) + ',' + ay + 'H' + rx + 'V' + band + 'H' + tx + 'V' + by + 'H' + (b.x + b.w);
        dir = -1; ex = b.x + b.w; cls += ' cj-data-graph-back';
      }
      var g = svgEl('g', (G.nodes[e[0]].hist || G.nodes[e[1]].hist) ? 'cj-data-graph-edim' : null);
      var t = svgEl('title');
      t.textContent = G.nodes[e[0]].title + ' → ' + G.nodes[e[1]].title + ' (' + e[2] + ')';
      g.appendChild(t);
      var p = svgEl('path', cls); p.setAttribute('d', d); g.appendChild(p);
      var h = svgEl('path', cls.replace('cj-data-graph-e', 'cj-data-graph-a')); h.setAttribute('d', arrow(ex, by, dir));
      g.appendChild(h); svg.appendChild(g);
    });
    G.nodes.forEach(function (n, i) {
      if (pos[i]) { var b = box(i, pos[i]); canvas.appendChild(b); drawn.push(b); }
    });
  }

  function labels() {
    var nHist = G.nodes.filter(function (n) { return n.hist; }).length;
    var nAn = G.nodes.filter(isAnalysis).length;
    document.getElementById('cj-data-graph-histl').textContent =
      'Show history (+' + nHist + ' superseded, cancelled or unused)';
    document.getElementById('cj-data-graph-anl').textContent =
      'Show analyses (+' + nAn + ' prediction set' + (nAn === 1 ? '' : 's') + ' and packages)';
  }

  function focusBox() {
    var f = canvas.querySelector('.cj-data-graph-focus'), wrap = document.getElementById('cj-data-graph-wrap');
    if (!f) return;
    wrap.scrollLeft = Math.max(0, f.offsetLeft - (wrap.clientWidth - f.offsetWidth) / 2);
    f.scrollIntoView({block: 'center', inline: 'nearest'});
    f.focus({preventScroll: true});
  }

  function start(data) {
    G = data;
    if (G.error) status.textContent = 'The registry could not be read just now (' + G.error + ').';
    else if (!G.nodes.length) status.textContent = 'Nothing in the registry to draw yet.';
    else status.textContent = G.nodes.length + ' items, ' + G.edges.length + ' links.';
    if (!G.cols.length) return;
    labels();
    if (G.focus != null) {               // show the item asked for even when it starts hidden
      var n = G.nodes[G.focus];
      if (n.hist) histBox.checked = true;
      if (isAnalysis(n)) anBox.checked = true;
    }
    draw();
    focusBox();
    histBox.addEventListener('change', draw);
    anBox.addEventListener('change', draw);
  }

  var focus = new URLSearchParams(location.search).get('focus');
  fetch('/data/api/lineage/graph' + (focus ? '?focus=' + encodeURIComponent(focus) : ''),
        {credentials: 'same-origin', headers: {'Accept': 'application/json'}})
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(start)
    .catch(function (e) { status.textContent = 'Could not load the graph (' + e.message + '). Reload to try again.'; });
})();
