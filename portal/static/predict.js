// Predict module (portal/modules/predict.py serves this file as /predict/predict.js). No inline script anywhere.
// - Set page (#cj-predict-set[data-live="1"]) and conversion page (#cj-predict-conv[data-live="1"]): every few seconds
//   GET the page's JSON (same origin) and swap in the fragments, rendered and escaped by the server; stops when nothing
//   is left to do.
// - Forms: show what is selected, check the upload caps before sending, disable the button while it works.
// - Uploads (form[data-cj-upload]): one request per file, or per piece of at most `chunk` bytes of a big file, into an
//   upload session (POST /predict/api/upload), then one "create" request. A dropped connection is retried, and pressing
//   the button again resumes where it stopped (the session id is kept in sessionStorage). Without JavaScript the form
//   posts as a plain multi-file form (small sets only; the server says so).
// - Progress page (#cj-predict-prep): a set is made in the background; poll its JSON and open the analysis when done.
// - Time line (#cj-predict-est, data-a data-b data-ahead): "about N min once it starts" for the tile count shown.
(function () {
  'use strict';
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;

  function swap(id, frag) {
    var box = document.getElementById(id);
    if (!box || typeof frag !== 'string') return;
    var t = document.createElement('template');
    t.innerHTML = frag;
    box.replaceChildren(t.content);
  }

  function poller(root, url, ids) {
    var every = Math.max(2, parseInt(root.dataset.every, 10) || 5) * 1000, busy = false, timer = null;
    var note = document.getElementById('cj-predict-err');
    var tick = function () {
      if (busy || document.hidden) return;
      busy = true;
      fetch(url, {cache: 'no-store', credentials: 'same-origin'})
        .then(function (r) { if (!r.ok) throw new Error('status ' + r.status); return r.json(); })
        .then(function (j) {
          var h = (j && j.html) || {};
          Object.keys(ids).forEach(function (k) { swap(ids[k], h[k]); });
          if (note) { note.textContent = ''; note.hidden = true; }
          if (!j.live && timer) { clearInterval(timer); timer = null; }
        })
        .catch(function (e) {
          if (note) { note.textContent = 'Could not refresh (' + e.message + '); showing the last state.'; note.hidden = false; }
        })
        .then(function () { busy = false; });
    };
    if (root.dataset.live === '1') timer = setInterval(tick, every);
    document.addEventListener('visibilitychange', function () { if (!document.hidden && timer) tick(); });
  }

  // seconds = a + b * images (the server fits it to finished runs); rewritten here as the tile count changes
  function estimate(n) {
    var el = document.getElementById('cj-predict-est');
    if (!el) return;
    var a = parseFloat(el.dataset.a), b = parseFloat(el.dataset.b), ahead = parseInt(el.dataset.ahead, 10) || 0;
    if (!(n > 0) || isNaN(a) || isNaN(b)) { el.textContent = ''; return; }
    var text = 'about ' + Math.max(1, Math.round((a + b * n) / 60)) + ' min once it starts';
    if (ahead > 0) text += ' · ' + ahead + (ahead === 1 ? ' analysis' : ' analyses') + ' ahead of you';
    el.textContent = text;
  }

  var prep = document.getElementById('cj-predict-prep');
  if (prep && prep.dataset.tid && prep.dataset.live === '1') {
    var pbusy = false, ptimer = null, perr = document.getElementById('cj-predict-err');
    var ptick = function () {
      if (pbusy) return;
      pbusy = true;
      fetch('/predict/api/preparing/' + encodeURIComponent(prep.dataset.tid) + '?html=1', {cache: 'no-store', credentials: 'same-origin'})
        .then(function (r) {
          if (r.status === 404) throw {gone: true};            // the portal restarted: the task is gone
          if (!r.ok) throw new Error('status ' + r.status);
          return r.json();
        })
        .then(function (j) {
          swap('cj-predict-prepstatus', ((j && j.html) || {}).status);
          if (perr) { perr.textContent = ''; perr.hidden = true; }
          if (j.href) { clearInterval(ptimer); location.assign(j.href); }
          else if (!j.live) clearInterval(ptimer);
        })
        .catch(function (e) {
          if (e && e.gone) {
            clearInterval(ptimer);
            swap('cj-predict-prepstatus', '<p class="cj-predict-msg">This preparation was interrupted. Check My analyses; if it isn’t there, start again.</p>');
            if (perr) { perr.textContent = ''; perr.hidden = true; }
            return;
          }
          if (perr) { perr.textContent = 'Could not refresh (' + e.message + '); the analysis is still being prepared.'; perr.hidden = false; }
        })
        .then(function () { pbusy = false; });
    };
    ptimer = setInterval(ptick, 2000);
    ptick();
  }

  var set = document.getElementById('cj-predict-set');
  if (set && set.dataset.ps) {
    poller(set, '/predict/api/set/' + encodeURIComponent(set.dataset.ps) + '?html=1',
           {status: 'cj-predict-status', actions: 'cj-predict-actions'});
  }
  var conv = document.getElementById('cj-predict-conv');
  if (conv && conv.dataset.cid) {
    poller(conv, '/predict/api/convert/' + encodeURIComponent(conv.dataset.cid) + '?html=1',
           {status: 'cj-predict-cstatus', result: 'cj-predict-cresult'});
  }

  document.addEventListener('submit', function (e) {
    var f = e.target;
    if (!f || f.tagName !== 'FORM') return;
    if (f.dataset.sent === '1') { e.preventDefault(); return; }
    f.dataset.sent = '1';
    var b = f.querySelector('button[type="submit"]');
    if (b) { b.disabled = true; if (b.dataset.busy) b.textContent = b.dataset.busy; }
  });

  var sess = document.getElementById('cj-predict-session');
  if (sess) {
    var out = document.getElementById('cj-predict-sel'), max = parseInt(sess.dataset.max, 10) || 1000;   // data-max: the server's limit
    var count = sess.querySelector('input[name="count"]');
    var update = function () {
      var n = 0;
      sess.querySelectorAll('input[name="samples"]:checked').forEach(function (c) { n += parseInt(c.dataset.n, 10) || 0; });
      var mode = sess.querySelector('input[name="how"]:checked');
      if (mode && mode.value === 'count') {
        var k = parseInt(count.value, 10);
        if (k > 0) n = Math.min(n, k);
      }
      out.textContent = n + ' tile' + (n === 1 ? '' : 's') + ' will be analysed' + (n > max ? ' (too many: at most ' + max + ')' : '') + '.';
      estimate(Math.min(n, max));
    };
    sess.addEventListener('input', update);
    sess.addEventListener('change', update);
    if (count) count.addEventListener('focus', function () {
      var r = sess.querySelector('input[name="how"][value="count"]');
      if (r) r.checked = true;
      update();
    });
    update();
  }

  // ---------------------------------------------------------------- uploads, one file (piece) per request
  function mb(n) { return (n / 1048576).toFixed(1) + ' MB'; }
  function wait(ms) { return new Promise(function (ok) { setTimeout(ok, ms); }); }
  function answer(r) {
    return r.json().catch(function () { return {}; }).then(function (j) {
      j = (j && typeof j === 'object') ? j : {};
      j._status = r.status;
      return j;
    });
  }
  function post(url, body, type) {
    return fetch(url, {method: 'POST', credentials: 'same-origin', cache: 'no-store',
                       headers: {'Content-Type': type || 'application/json'},
                       body: type ? body : JSON.stringify(body || {})});
  }

  document.querySelectorAll('form[data-cj-upload]').forEach(function (form) {
    var kind = form.dataset.cjUpload, input = form.querySelector('input[type="file"]');
    var info = form.querySelector('[data-cj-upload-info]'), box = form.querySelector('[data-cj-upload-msg]');
    var btn = form.querySelector('button[type="submit"]'), idle = btn ? btn.textContent : '';
    var maxFiles = parseInt(form.dataset.maxFiles, 10), maxBytes = parseInt(form.dataset.maxBytes, 10);
    var store = 'cj-predict-upload-' + kind;
    if (!input) return;

    var check = function () {
      var fs = Array.prototype.slice.call(input.files || []), big = fs.filter(function (f) { return f.size > maxBytes; });
      var total = fs.reduce(function (a, f) { return a + f.size; }, 0);
      var msg = fs.length + ' file' + (fs.length === 1 ? '' : 's') + ', ' + mb(total);
      var bad = '';
      if (fs.length > maxFiles) bad = 'At most ' + maxFiles + ' files.';
      else if (big.length) bad = big[0].name + ' is over ' + Math.round(maxBytes / 1048576) + ' MB.';
      if (info) info.textContent = bad ? msg + '. ' + bad : msg;
      estimate(bad ? 0 : fs.length);
      input.setCustomValidity(bad);
    };
    input.addEventListener('change', check);

    function say(text) { if (box) { box.textContent = text || ''; box.hidden = !text; } }
    function progress(text) { if (info) info.textContent = text; }
    var noteEl = null;
    function note(text) {                                       // a lasting line under the progress (left-out files)
      if (!noteEl && text && info) {
        noteEl = document.createElement('p');
        noteEl.className = 'cj-mut';
        noteEl.setAttribute('data-cj-upload-note', '');
        info.parentNode.insertBefore(noteEl, info.nextSibling);
      }
      if (noteEl) { noteEl.textContent = text || ''; noteEl.hidden = !text; }
    }
    function busy(on) {
      if (!btn) return;
      btn.disabled = on;
      btn.textContent = on && btn.dataset.busy ? btn.dataset.busy : idle;
    }
    function remember(id) { try { if (id) sessionStorage.setItem(store, id); else sessionStorage.removeItem(store); } catch (x) { /* private mode */ } }
    function remembered() { try { return sessionStorage.getItem(store); } catch (x) { return null; } }

    function sendAll(st, files, onState) {
      var id = st.upload, chunk = st.chunk, total = files.reduce(function (a, f) { return a + f.size; }, 0);
      var sentOf = function (s) {
        return files.reduce(function (a, f) {
          return a + ((s.done || []).indexOf(f.name) >= 0 ? f.size : ((s.received || {})[f.name] || 0));
        }, 0);
      };
      var one = function (f, idx, s, tries) {
        if ((s.done || []).indexOf(f.name) >= 0) return Promise.resolve(s);
        var off = (s.received || {})[f.name] || 0, end = Math.min(f.size, off + chunk);
        progress('Sending file ' + (idx + 1) + ' of ' + files.length + ' (' + f.name + '): ' +
                 Math.floor(100 * sentOf(s) / Math.max(1, total)) + ' % of ' + mb(total));
        return post('/predict/api/upload/' + encodeURIComponent(id) + '/file?name=' + encodeURIComponent(f.name) +
                    '&offset=' + off, f.slice(off, end), 'application/octet-stream')
          .then(answer, function () { return null; })
          .then(function (j) {
            if (j === null) {                                   // the connection dropped: ask where it stopped
              if (tries >= 4) throw {error: 'The connection dropped. Press the button again to resume where it stopped.'};
              return wait(1000 * Math.pow(2, tries))
                .then(function () { return fetch('/predict/api/upload/' + encodeURIComponent(id), {cache: 'no-store', credentials: 'same-origin'}); })
                .then(answer, function () { return {_status: 0}; })
                .then(function (s2) { return one(f, idx, s2._status === 200 ? s2 : s, tries + 1); });
            }
            if (j._status === 200 || j._status === 409) return one(f, idx, j, 0);   // 409: resume from j.received
            throw j;
          });
      };
      var i = 0, s = st;
      var next = function () {
        if (i >= files.length) return Promise.resolve(id);
        return one(files[i], i, s, 0).then(function (s2) {
          s = s2;
          i += 1;
          if (onState) onState(s);
          return next();
        });
      };
      return next();
    }

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      e.stopPropagation();                                      // not the plain-form handler below
      if (form.dataset.sending === '1') return;
      var files = Array.prototype.slice.call(input.files || []);
      var leftOut = 0;
      var noteLeft = function (byServer) {                      // atlas files left out here + those the server left out
        var n = leftOut + byServer;
        note(n ? n + ' file' + (n === 1 ? '' : 's') + ' left out (not converted)' +
             (leftOut ? ', ' + leftOut + ' of them atlas file' + (leftOut === 1 ? '' : 's') : '') + '.' : '');
      };
      if (kind === 'raw') {                                     // whole-grid atlases are never converted: not sent
        var out = {};
        files.forEach(function (f) { if (/atlas/i.test(f.name)) out[f.name.toLowerCase()] = 1; });
        files.forEach(function (f) {                            // an .mrc whose .mrc.mdoc is left out goes too
          var lo = f.name.toLowerCase();
          if (/\.mrc$/.test(lo) && out[lo + '.mdoc']) out[lo] = 1;
        });
        leftOut = files.filter(function (f) { return out[f.name.toLowerCase()]; }).length;
        files = files.filter(function (f) { return !out[f.name.toLowerCase()]; });
        if (leftOut && !files.length) {
          note('');
          say('Only atlas files were chosen: nothing to convert.');
          return;
        }
        noteLeft(0);
      }
      var fields = {};
      new FormData(form).forEach(function (v, k) { if (typeof v === 'string') fields[k] = v; });
      form.dataset.sending = '1';
      busy(true);
      say('');
      progress('Starting…');
      post('/predict/api/upload', {kind: kind, fields: fields, resume: remembered(),
                                   files: files.map(function (f) { return {name: f.name, size: f.size}; })})
        .then(answer)
        .then(function (st) {
          if (st._status !== 200) throw st;
          remember(st.upload);
          if (kind === 'raw') noteLeft((st.skipped || []).length);
          return sendAll(st, files, function (s) { if (kind === 'raw') noteLeft((s.skipped || []).length); });
        })
        .then(function (id) {
          progress('All ' + files.length + ' file' + (files.length === 1 ? '' : 's') + ' sent; starting…');
          return post('/predict/api/upload/' + encodeURIComponent(id) + '/create', {}).then(answer);
        })
        .then(function (j) {
          if (j._status !== 200 || !j.href) throw j;
          remember(null);
          progress('Done: opening…');
          location.assign(j.href);
        })
        .catch(function (j) {
          if (j && j.dropped) remember(null);
          say((j && j.error) || 'The upload stopped. Press the button again to resume where it stopped.');
          progress('');
          check();
          busy(false);
          form.dataset.sending = '0';
        });
    });
  });
  // ---- Conversion settings (round 7 phase 2). Everything the server sends here is escaped HTML or JSON.
  // - Raw form: select[data-cj-recipe] + [data-cj-recipe-readout]: the real values of the chosen recipe
  //   (GET /predict/api/recipe/<name>, a child process on the server, cached); the option labels gain a short summary.
  // - Conversion page: #cj-convset loads GET <data-url> (values used, diff, the Adjust panel). The Adjust panel's sliders
  //   feed a one-tile preview (POST .../preview), shown as before | after; "Re-convert" is a plain form post.
  function setHtml(node, frag) {
    var t = document.createElement('template');
    t.innerHTML = frag || '';
    node.replaceChildren(t.content);
  }
  function getJson(url) {
    return fetch(url, {cache: 'no-store', credentials: 'same-origin'}).then(function (r) {
      return r.json().then(function (j) { j._status = r.status; return j; });
    });
  }
  function plainMsg(e) { return String(e && e.message || e).replace(/[<>&]/g, ''); }

  var rsel = document.querySelector('select[data-cj-recipe]');
  var rbox = document.querySelector('[data-cj-recipe-readout]');
  if (rsel && rbox) {
    var rvals = rbox.querySelector('[data-cj-recipe-values]'), rcache = {}, rchain = Promise.resolve();
    Array.prototype.forEach.call(rsel.options, function (o) {
      o.dataset.base = o.textContent.split(' · ')[0];   // the label before any ' · summary' the server put there
    });
    var rload = function (name) {
      if (rcache[name]) return Promise.resolve(rcache[name]);
      return getJson('/predict/api/recipe/' + encodeURIComponent(name)).then(function (j) {
        if (j._status !== 200) throw new Error(j.error || ('status ' + j._status));
        rcache[name] = j;
        Array.prototype.forEach.call(rsel.options, function (o) {
          if (o.value === name && j.summary) o.textContent = o.dataset.base + ' · ' + j.summary;
        });
        return j;
      });
    };
    var rshow = function () {
      var name = rsel.value;
      if (rcache[name]) { setHtml(rvals, rcache[name].html); return; }
      setHtml(rvals, '<p class="cj-mut">Loading the values of this recipe…</p>');
      rload(name).then(function () { if (rsel.value === name) setHtml(rvals, rcache[name].html); })
        .catch(function (e) {
          if (rsel.value === name) setHtml(rvals, '<p class="cj-mut">The values could not be read (' + plainMsg(e) + ').</p>');
        });
    };
    rsel.addEventListener('change', rshow);
    if (rvals.querySelector('[data-cj-convset-wait]')) rshow();
    // fill in the other options' summaries, one recipe at a time (each is a child process on the server)
    Array.prototype.slice.call(rsel.options, 0, 12).forEach(function (o) {
      if (o.textContent.indexOf(' · ') > -1) return;
      rchain = rchain.then(function () { return rload(o.value); }).catch(function () {});
    });
  }

  var cs = document.getElementById('cj-convset');
  if (cs) {
    var csLoad = function () {
      getJson(cs.dataset.url).then(function (j) {
        if (j._status !== 200) throw new Error(j.error || ('status ' + j._status));
        setHtml(cs, j.html);
        cs.dataset.state = j.state || '';
        csWire();
      }).catch(function (e) {
        setHtml(cs, '<p class="cj-mut">The settings could not be loaded (' + plainMsg(e) + ').</p>');
      });
    };
    csLoad();
    if (cs.dataset.state && cs.dataset.state !== 'done' && cs.dataset.state !== 'failed') {
      var csId = (document.getElementById('cj-predict-conv') || {dataset: {}}).dataset.cid;
      var csPoll = setInterval(function () {
        if (document.hidden || !csId) return;
        getJson('/predict/api/convert/' + encodeURIComponent(csId)).then(function (j) {
          if (j.state === 'done' || j.state === 'failed') { clearInterval(csPoll); csLoad(); }
        }).catch(function () {});
      }, 6000);
    }
  }

  function csWire() {
    var panel = document.getElementById('cj-convset-adjust');
    if (!panel || panel.dataset.disabled) return;
    var form = panel.querySelector('[data-cj-convset-form]');
    var sliders = Array.prototype.slice.call(form.querySelectorAll('[data-cj-slider]'));
    var dn = form.querySelector('[data-cj-denoise]'), dtext = form.querySelector('[data-cj-denoise-text]');
    var msg = panel.querySelector('[data-cj-convset-msg]'), pv = panel.querySelector('[data-cj-pv]');
    var btn = form.querySelector('[data-cj-preview]');
    var say = function (t) { msg.textContent = t || ''; msg.hidden = !t; };
    var sync = function () {
      sliders.forEach(function (s) { s.parentNode.querySelector('[data-cj-out]').textContent = s.value; });
      dtext.textContent = dn.checked ? 'On' : 'Off';
      sliders.forEach(function (s) { if (s.name === 'bandpass_nm') s.disabled = !dn.checked; });
    };
    sliders.forEach(function (s) { s.addEventListener('input', sync); });
    dn.addEventListener('change', sync);
    form.querySelector('[data-cj-reset]').addEventListener('click', function () {
      sliders.forEach(function (s) { s.value = s.dataset.recipe; });
      dn.checked = dn.dataset.recipe === '1';
      sync(); say(''); pv.hidden = true;
    });
    form.addEventListener('submit', function () {
      form.querySelector('[data-cj-reconvert]').disabled = true;
      say('Starting a new conversion…');
    });
    var figure = function (label, src, note) {
      var f = document.createElement('figure'), c = document.createElement('figcaption'), i = document.createElement('img');
      c.textContent = label; i.src = src; i.alt = label; f.appendChild(i); f.appendChild(c);
      if (note) { var n = document.createElement('small'); n.textContent = note; f.appendChild(n); }
      return f;
    };
    var describe = function (p) {
      if (!p) return '';
      return 'clip ' + p.clip_lo_pct + '–' + p.clip_hi_pct + ' %, ' + (p.denoise === false ? 'no denoise' :
        (p.bandpass_nm != null ? 'bandpass ' + p.bandpass_nm + ' nm' : 'denoise on'));
    };
    btn.addEventListener('click', function () {
      var params = {denoise: dn.checked ? 'true' : 'false'};
      sliders.forEach(function (s) { if (!s.disabled) params[s.name] = s.value; });
      btn.disabled = true; say('Making the preview… (a few seconds)');
      fetch(panel.dataset.preview, {method: 'POST', credentials: 'same-origin',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tile: form.querySelector('[data-cj-tile]').value, params: params})})
        .then(function (r) { return r.json().then(function (j) { j._status = r.status; return j; }); })
        .then(function (j) {
          if (!j.ok) { say(j.error || ('The preview failed (' + j._status + ').')); pv.hidden = true; return; }
          say('');
          var wrap = document.createElement('div'), row = document.createElement('div'), cap = document.createElement('p');
          row.className = 'cj-convset-pair';
          row.appendChild(figure('Before: this conversion', j.before, describe(j.conversion_params || j.recipe_params)));
          row.appendChild(figure('After: these settings', j.after, describe(j.effective)));
          cap.className = 'cj-mut';
          cap.textContent = 'Tile ' + j.tile + (j.compute ? ' · rendered on the ' + j.compute : '') +
            (j.warnings && j.warnings.length ? ' · ' + j.warnings.join(' ') : '');
          wrap.appendChild(row); wrap.appendChild(cap);
          pv.replaceChildren(wrap); pv.hidden = false;
        })
        .catch(function (e) { say('The preview failed (' + plainMsg(e) + ').'); })
        .then(function () { btn.disabled = false; });
    });
    sync();
  }
})();
