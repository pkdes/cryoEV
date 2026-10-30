"""
History-tree page for the tracking registry (contract v1, docs/TRACKING_CONTRACT.md). Standard library only.

    python registry/history_page.py [--registry FILE_OR_DIR ...] [--out history.html]

Reads append-only JSONL (one file, or a folder of <entity>.jsonl like registry/samples/) and writes ONE
self-contained static HTML file: inline CSS/JS, no external URLs. The page has a status board (lanes from the
contract's Statuses table), a collapsible history tree (pull -> label versions -> datasets -> job bundles ->
runs -> models, with review packages and inference checks hung off what they relate to), per-record detail
with the status history from repeated lines, filters by entity/status, and id search.

Semantics: the LAST LINE PER id IN FILE ORDER is current (not by `logged`); every line is kept as history.
Links are child -> parent only; children are derived here (no `superseded_by`, `parts`, `scores` needed).
If a later line is a partial status line, fields from earlier lines are kept (merged view) so the record
stays linked; the status always comes from the last line.

Record text reaches the page only as JSON inside <script type="application/json"> with <, >, & escaped,
and the page script builds the DOM with textContent (never innerHTML with record values).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from data_utils.path_guard import protected_problem  # noqa: E402

REAL_DIR = Path.home() / "Desktop" / "cryoEV pujrog" / "CryoAI-laptop" / "registry"
DEFAULT_REGISTRY = REAL_DIR  # one <entity>.jsonl per entity (registry.py); every *.jsonl in it is read
DEFAULT_OUT = REAL_DIR / "history.html"

# Page lanes, from the contract's "Statuses (queue and history view)" table.
LANES = [
    ("queued", {"dataset": {"frozen"}, "job_bundle": {"bundled", "sent"}, "run": {"queued"}}),
    ("staged", {"inference_check": {"staged"}, "review_package": {"built"}}),
    ("in review", {"review_package": {"in_review"}}),
    ("reviewed", {"review_package": {"reviewed"}, "inference_check": {"reviewed"}}),
    ("applied", {"label_version": {"created", "qc_passed"}, "review_package": {"applied"}}),
    ("pushed", {"label_version": {"pushed"}}),
    ("trained", {"run": {"running", "finished", "failed"}, "dataset": {"trained"}, "job_bundle": {"returned"}}),
    ("scored", {"inference_check": {"scored"}, "model": {"candidate", "production", "retired"}}),
    ("closed", {"review_package": {"superseded", "abandoned"}, "label_version": {"superseded"},
                "session": {"excluded"}}),
]
TREE_ENTITIES = ("pull", "roboflow_export", "label_version", "review_package", "inference_check",
                 "dataset", "job_bundle", "run", "model")
WHO_FIELDS = ("who", "reviewer", "reviewers", "recorded_by", "tool")
WHEN_FIELDS = ("logged", "created", "built", "uploaded")


# ---------------------------------------------------------------- reading
# Own small reader with the same semantics as registry/registry.py (piece 4): last line per id wins, plus
# the history list. It can switch to registry.current() / registry.history(id) once that module lands.

def entity_of(rec: dict) -> str | None:
    s = rec.get("schema")
    if isinstance(s, str) and "/" in s:
        return s.split("/", 1)[0]
    if "batch_id" in rec:  # legacy batches.jsonl line (no schema): inference_check
        return "inference_check"
    return None


def input_files(paths) -> list[Path]:
    out = []
    for p in paths:
        p = Path(p)
        out.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    return out


def read_registry(paths):
    """-> (current {key: rec}, history {key: [rec, ...]}, stats). key = '<entity>|<id>'.
    current[key] is the merged view (earlier fields overlaid by later lines); its status is the last line's."""
    history: dict[str, list[dict]] = {}
    order: list[str] = []
    stats = {"lines": 0, "bad_json": 0, "no_id": 0, "label_change_rows": 0}
    for f in input_files(paths):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                if not isinstance(rec, dict):
                    stats["bad_json"] += 1
                    continue
                ent = entity_of(rec)
                if ent == "label_change":
                    stats["label_change_rows"] += 1
                    continue
                rid = rec.get("id", rec.get("batch_id"))
                if rid is None or ent is None:
                    stats["no_id"] += 1
                    continue
                key = f"{ent}|{rid}"
                if key not in history:
                    history[key] = []
                    order.append(key)
                history[key].append(rec)
    current = {}
    for key in order:
        merged: dict = {}
        for rec in history[key]:
            merged.update(rec)
        last = history[key][-1]
        merged["status"] = last.get("status")
        merged["id"] = str(last.get("id", last.get("batch_id")))
        current[key] = merged
    return current, history, stats


# ---------------------------------------------------------------- model

def lane_of(ent: str, status) -> str | None:
    for lane, m in LANES:
        if status in m.get(ent, ()):
            return lane
    return None


def _ids_in(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, str):
                yield x
    elif isinstance(v, dict) and isinstance(v.get("id"), str):  # dataset.source {kind, id}
        yield v["id"]


def build_model(current: dict, history: dict, stats: dict, sources) -> dict:
    by_id: dict[str, list[str]] = {}
    for key, rec in current.items():
        by_id.setdefault(rec["id"], []).append(key)

    def find(ent, rid):
        k = f"{ent}|{rid}"
        return k if isinstance(rid, str) and k in current else None

    # derived reverse links
    superseded_by = {}
    for key, rec in current.items():
        if key.startswith("label_version|") and isinstance(rec.get("supersedes"), str):
            superseded_by.setdefault(rec["supersedes"], []).append(rec["id"])

    lv_for_pkg = {}
    for key, rec in current.items():
        if key.startswith("label_version|") and find("review_package", rec.get("pkg")):
            lv_for_pkg.setdefault(rec["pkg"], key)

    def parent_of(key, rec):
        ent = key.split("|", 1)[0]
        if ent == "roboflow_export":
            return find("roboflow_export", rec.get("derived_from"))
        if ent == "label_version":
            return find("pull", rec.get("base"))
        if ent == "review_package":
            return (find("review_package", rec.get("parent")) or lv_for_pkg.get(rec["id"])
                    or find("model", rec.get("model")) or find(rec.get("base_kind") or "pull", rec.get("base"))
                    or find("pull", rec.get("base")) or find("roboflow_export", rec.get("base")))
        if ent == "inference_check":
            return find("model", rec.get("model"))
        if ent == "dataset":
            src = rec.get("source") if isinstance(rec.get("source"), dict) else {}
            return (find(src.get("kind") or "", src.get("id")) or find("label_version", rec.get("label_version"))
                    or find("roboflow_export", rec.get("export")))
        if ent == "job_bundle":
            return find("dataset", rec.get("dataset"))
        if ent == "run":
            return find("job_bundle", rec.get("job_bundle")) or find("dataset", rec.get("dataset"))
        if ent == "model":
            return find("run", rec.get("run")) or find("dataset", rec.get("dataset"))
        return None

    parent = {}
    for key, rec in current.items():
        if key.split("|", 1)[0] in TREE_ENTITIES:
            parent[key] = parent_of(key, rec)
    for key in list(parent):  # break cycles: a node that reaches itself becomes a root
        seen, p = {key}, parent[key]
        while p:
            if p in seen:
                parent[key] = None
                break
            seen.add(p)
            p = parent.get(p)
    children: dict[str, list[str]] = {}
    for key, p in parent.items():
        if p:
            children.setdefault(p, []).append(key)

    def sort_key(k):
        r = current[k]
        return (str(r.get("created") or r.get("built") or r.get("logged") or ""), r["id"])

    def node(k):
        return {"k": k, "c": [node(c) for c in sorted(children.get(k, []), key=sort_key)]}

    roots = [k for k, p in parent.items() if not p]
    sections = []
    for title, ents in (("Roboflow pulls", ("pull",)), ("Roboflow exports", ("roboflow_export",)),
                        ("Unlinked", tuple(e for e in TREE_ENTITIES if e not in ("pull", "roboflow_export")))):
        rs = sorted((k for k in roots if k.split("|", 1)[0] in ents), key=sort_key)
        if rs:
            sections.append({"title": title, "nodes": [node(k) for k in rs]})
    other: dict[str, list[str]] = {}
    for key in current:
        ent = key.split("|", 1)[0]
        if ent not in TREE_ENTITIES:
            other.setdefault(ent, []).append(key)
    for ent in sorted(other):
        sections.append({"title": f"Other: {ent}", "nodes": [{"k": k, "c": []} for k in sorted(other[ent])],
                         "collapsed": True})

    recs = {}
    for key, rec in current.items():
        ent = key.split("|", 1)[0]
        links = []
        for field, v in rec.items():
            if field in ("id", "schema"):
                continue
            for rid in _ids_in(v):
                for tk in by_id.get(rid, []):
                    if tk != key:
                        links.append([field, tk])
        if ent == "label_version":
            for s in superseded_by.get(rec["id"], []):
                tk = find("label_version", s)
                if tk:
                    links.append(["superseded by (derived)", tk])
        for c in children.get(key, []):
            links.append([f"child {c.split('|', 1)[0]}", c])
        hist = []
        for line in history[key]:
            hist.append({"logged": line.get("logged"), "status": line.get("status"),
                         "by": line.get("recorded_by") or line.get("tool"), "fields": len(line)})
        recs[key] = {
            "e": ent, "id": rec["id"], "s": rec.get("status"), "lane": lane_of(ent, rec.get("status")),
            "who": {f: rec[f] for f in WHO_FIELDS if rec.get(f) not in (None, [], "")},
            "when": {f: rec[f] for f in WHEN_FIELDS if rec.get(f)},
            "gaps": {f: rec[f] for f in rec if f.endswith("_gap") and rec[f]},
            "copy_of": rec.get("copy_of"),
            "sup": rec.get("supersedes") if ent == "label_version" else None,
            "supby": superseded_by.get(rec["id"], []) if ent == "label_version" else [],
            "dim": ent == "label_version" and (rec.get("status") == "superseded" or bool(superseded_by.get(rec["id"]))),
            "prod": ent == "model" and rec.get("status") == "production",
            "links": links, "hist": hist, "raw": rec,
        }
    counts: dict[str, int] = {}
    for r in recs.values():
        counts[r["e"]] = counts.get(r["e"], 0) + 1
    return {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [str(s) for s in sources],
        "stats": stats, "counts": counts,
        "lanes": [lane for lane, _ in LANES],
        "recs": recs, "sections": sections,
        "production": [k for k, r in recs.items() if r["prod"]],
    }


# ---------------------------------------------------------------- HTML

def safe_json(obj) -> str:
    """JSON safe inside a <script> element: no '<', '>', '&' (so no '</script>' or '<!--')."""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e") \
        .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


CSS = r"""
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#6b6560;--card:#fff;--line:#e3e0dc;--acc:#2456c7;--accbg:#e8eefc;
--prod:#0f7a3d;--prodbg:#dcf3e5;--warn:#9a5b00;--warnbg:#fdf0d8;--dim:.5}
@media (prefers-color-scheme:dark){:root{--bg:#141312;--fg:#ece9e5;--mut:#a39e98;--card:#1e1c1b;--line:#35322f;
--acc:#8fb0ff;--accbg:#1f2a44;--prod:#6fdc9c;--prodbg:#163323;--warn:#f2c26b;--warnbg:#3a2c12}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header,main{max-width:1280px;margin:0 auto;padding:12px 16px}
h1{font-size:1.25rem;margin:4px 0}h2{font-size:1.05rem;margin:18px 0 8px}
.meta{color:var(--mut);font-size:.85rem;overflow-wrap:anywhere}
.prodline{margin:8px 0;padding:8px 10px;border-radius:8px;background:var(--prodbg);color:var(--prod);font-weight:600;overflow-wrap:anywhere}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.filters select,.filters input{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);min-height:40px}
.filters input{flex:1 1 180px;min-width:0}
.board{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.lane{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.lane summary{cursor:pointer;list-style:none;display:flex;justify-content:space-between;gap:6px;align-items:baseline}
.lane summary::-webkit-details-marker{display:none}
.lane .n{font-size:1.4rem;font-weight:700}.lane.empty{opacity:.55}
.lane ul{list-style:none;margin:6px 0 0;padding:0}
.layout{display:grid;grid-template-columns:1fr;gap:16px}
@media (min-width:900px){.layout{grid-template-columns:minmax(0,1fr) 400px}#detail{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}}
@media (max-width:899px){#detail{position:fixed;left:0;right:0;bottom:0;max-height:70vh;overflow:auto;z-index:5;border-radius:14px 14px 0 0;box-shadow:0 -6px 24px rgba(0,0,0,.25)}#detail.hidden{display:none}}
.tree ul{list-style:none;margin:0;padding-left:14px;border-left:1px solid var(--line)}
.tree>ul{padding-left:0;border-left:0}
.tree li{margin:2px 0}
.tree summary{cursor:pointer;padding:2px 0}
.leaf{padding-left:16px}
.rec{font:inherit;color:inherit;background:none;border:0;padding:4px 2px;text-align:left;cursor:pointer;display:inline-flex;flex-wrap:wrap;gap:4px 6px;align-items:baseline;max-width:100%;min-height:32px}
.rec:hover .rid,.rec:focus .rid{text-decoration:underline}
.rid{font-family:ui-monospace,Consolas,monospace;font-size:.85rem;overflow-wrap:anywhere;word-break:break-all}
.tag{font-size:.72rem;padding:1px 6px;border-radius:999px;background:var(--accbg);color:var(--acc);white-space:nowrap}
.st{font-size:.72rem;padding:1px 6px;border-radius:999px;border:1px solid var(--line);color:var(--mut);white-space:nowrap}
.prod{background:var(--prodbg);color:var(--prod);font-weight:700}
.gap{background:var(--warnbg);color:var(--warn)}
.note{font-size:.78rem;color:var(--mut)}
.dim{opacity:var(--dim)}.ctx{opacity:.6}
.sel{outline:2px solid var(--acc);border-radius:6px}
.hide{display:none!important}
section.sec{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px;margin-bottom:10px}
section.sec>details>summary{font-weight:600;cursor:pointer}
#detail{background:var(--card);border:1px solid var(--line);padding:12px}
#detail .close{float:right;font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);cursor:pointer}
#detail dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:4px 10px;margin:8px 0}
#detail dt{color:var(--mut);font-size:.85rem}#detail dd{margin:0;overflow-wrap:anywhere}
#detail table{border-collapse:collapse;width:100%;font-size:.85rem}#detail td,#detail th{border-top:1px solid var(--line);padding:3px 4px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
#detail pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.78rem;background:var(--bg);padding:8px;border-radius:8px}
.lnk{font:inherit;color:var(--acc);background:none;border:0;padding:2px 0;cursor:pointer;text-align:left;overflow-wrap:anywhere;word-break:break-all}
"""

JS = r"""
(function(){
var M=JSON.parse(document.getElementById('data').textContent);
var R=M.recs;
function el(t,cls,txt){var e=document.createElement(t);if(cls)e.className=cls;if(txt!=null)e.textContent=String(txt);return e;}
function recButton(k){
  var r=R[k],b=el('button','rec'+(r.dim?' dim':''));b.type='button';b.dataset.k=k;
  b.appendChild(el('span','tag',r.e));b.appendChild(el('span','rid',r.id));
  if(r.s!=null)b.appendChild(el('span','st'+(r.prod?' prod':''),r.prod?'PRODUCTION':r.s));
  if(r.prod&&r.s!=='production')b.appendChild(el('span','st',r.s));
  if(r.sup)b.appendChild(el('span','note','supersedes '+r.sup));
  if(r.supby.length)b.appendChild(el('span','note','superseded by '+r.supby.join(', ')));
  if(Object.keys(r.gaps).length)b.appendChild(el('span','st gap','gap'));
  if(r.copy_of)b.appendChild(el('span','note','copy of '+r.copy_of));
  b.addEventListener('click',function(ev){ev.preventDefault();show(k);});
  return b;
}
function renderNode(n){
  var li=el('li');var r=R[n.k];li.dataset.k=n.k;
  if(n.c.length){
    var d=el('details');d.open=true;var s=el('summary');s.appendChild(recButton(n.k));
    s.appendChild(el('span','note',' ('+n.c.length+')'));d.appendChild(s);
    var ul=el('ul');n.c.forEach(function(c){ul.appendChild(renderNode(c));});d.appendChild(ul);li.appendChild(d);
  }else{var w=el('div','leaf');w.appendChild(recButton(n.k));li.appendChild(w);}
  return li;
}
// header
document.getElementById('gen').textContent='generated '+M.generated+' from '+M.sources.join(', ');
var cnt=Object.keys(M.counts).sort().map(function(e){return e+' '+M.counts[e];}).join(' \u00b7 ');
var st=M.stats;
document.getElementById('counts').textContent=Object.keys(R).length+' records ('+cnt+') from '+st.lines+' lines'+
  (st.label_change_rows?'; '+st.label_change_rows+' label_change rows not shown':'')+
  (st.bad_json?'; '+st.bad_json+' unreadable lines':'')+(st.no_id?'; '+st.no_id+' lines without id/schema':'');
var pl=document.getElementById('prodline');
if(M.production.length){pl.textContent='Production model: ';M.production.forEach(function(k){pl.appendChild(recButton(k));});}
else{pl.textContent='No model is marked production.';pl.classList.add('note');}
// filters
var fe=document.getElementById('fe'),fs=document.getElementById('fs'),fq=document.getElementById('fq');
var ents={},sts={};Object.keys(R).forEach(function(k){ents[R[k].e]=1;if(R[k].s!=null)sts[R[k].s]=1;});
Object.keys(ents).sort().forEach(function(e){var o=el('option',null,e);o.value=e;fe.appendChild(o);});
Object.keys(sts).sort().forEach(function(s){var o=el('option',null,s);o.value=s;fs.appendChild(o);});
function match(k){var r=R[k],q=fq.value.trim().toLowerCase();
  return (!fe.value||r.e===fe.value)&&(!fs.value||String(r.s)===fs.value)&&(!q||r.id.toLowerCase().indexOf(q)>=0);}
// board
var board=document.getElementById('board');var laneItems={};
M.lanes.concat(['other']).forEach(function(l){laneItems[l]=[];});
Object.keys(R).forEach(function(k){var r=R[k];if(r.e==='image'||r.e==='roboflow_image'||r.e==='grid')return;
  if(r.lane)laneItems[r.lane].push(k);else if(r.s!=null&&r.e!=='session')laneItems.other.push(k);});
var laneEls=[];
Object.keys(laneItems).forEach(function(l){if(l==='other'&&!laneItems.other.length)return;
  var d=el('details','lane');var s=el('summary');s.appendChild(el('span',null,l));var n=el('span','n','0');s.appendChild(n);d.appendChild(s);
  var ul=el('ul');laneItems[l].forEach(function(k){var li=el('li');li.dataset.k=k;li.appendChild(recButton(k));ul.appendChild(li);});
  d.appendChild(ul);board.appendChild(d);laneEls.push({d:d,n:n,ul:ul});});
// tree
var tree=document.getElementById('tree');
M.sections.forEach(function(sec){var s=el('section','sec');var d=el('details');d.open=!sec.collapsed;
  var sm=el('summary',null,sec.title+' ('+sec.nodes.length+')');d.appendChild(sm);
  var w=el('div','tree');var ul=el('ul');sec.nodes.forEach(function(n){ul.appendChild(renderNode(n));});
  w.appendChild(ul);d.appendChild(w);s.appendChild(d);tree.appendChild(s);});
function apply(){
  var active=!!(fe.value||fs.value||fq.value.trim());
  laneEls.forEach(function(L){var c=0;Array.prototype.forEach.call(L.ul.children,function(li){var m=match(li.dataset.k);li.classList.toggle('hide',!m);if(m)c++;});
    L.n.textContent=c;L.d.classList.toggle('empty',!c);});
  function walk(li){var m=match(li.dataset.k),any=false;var sub=li.querySelector(':scope>details>ul');
    if(sub)Array.prototype.forEach.call(sub.children,function(c){if(walk(c))any=true;});
    li.classList.toggle('hide',active&&!m&&!any);li.classList.toggle('ctx',active&&!m&&any);
    if(active&&any){var d=li.querySelector(':scope>details');if(d)d.open=true;}
    return m||any;}
  Array.prototype.forEach.call(tree.querySelectorAll('section.sec'),function(sec){var any=false;
    Array.prototype.forEach.call(sec.querySelector('.tree>ul').children,function(li){if(walk(li))any=true;});
    sec.classList.toggle('hide',active&&!any);if(active&&any)sec.querySelector('details').open=true;});
}
[fe,fs].forEach(function(x){x.addEventListener('change',apply);});fq.addEventListener('input',apply);
// detail
var D=document.getElementById('detail');
function kv(dl,k,v){dl.appendChild(el('dt',null,k));var dd=el('dd');
  if(v instanceof Node)dd.appendChild(v);else dd.textContent=(typeof v==='object')?JSON.stringify(v):String(v);dl.appendChild(dd);}
function show(k){var r=R[k];D.textContent='';D.classList.remove('hidden');
  var c=el('button','close','Close');c.type='button';c.addEventListener('click',function(){D.classList.add('hidden');});D.appendChild(c);
  D.appendChild(el('h2',null,r.e+(r.prod?' (PRODUCTION)':'')));D.appendChild(el('div','rid',r.id));
  var dl=el('dl');kv(dl,'status',r.s==null?'(none)':r.s);if(r.lane)kv(dl,'lane',r.lane);
  Object.keys(r.who).forEach(function(f){kv(dl,f,r.who[f]);});Object.keys(r.when).forEach(function(f){kv(dl,f,r.when[f]);});
  Object.keys(r.gaps).forEach(function(f){kv(dl,f,r.gaps[f]);});if(r.copy_of)kv(dl,'copy_of',r.copy_of);
  if(r.sup)kv(dl,'supersedes',r.sup);if(r.supby.length)kv(dl,'superseded by',r.supby.join(', '));
  D.appendChild(dl);
  D.appendChild(el('h3',null,'Links ('+r.links.length+')'));
  if(r.links.length){var t=el('table');r.links.forEach(function(L){var tr=el('tr');tr.appendChild(el('td',null,L[0]));
    var td=el('td');var b=el('button','lnk',R[L[1]].e+': '+R[L[1]].id);b.type='button';b.addEventListener('click',function(){show(L[1]);});
    td.appendChild(b);tr.appendChild(td);t.appendChild(tr);});D.appendChild(t);}
  else D.appendChild(el('div','note','No links to other records in this registry.'));
  D.appendChild(el('h3',null,'Status history ('+r.hist.length+' line'+(r.hist.length===1?'':'s')+')'));
  var h=el('table');var hr=el('tr');['logged','status','by','fields'].forEach(function(x){hr.appendChild(el('th',null,x));});h.appendChild(hr);
  r.hist.forEach(function(x){var tr=el('tr');[x.logged,x.status,x.by,x.fields].forEach(function(v){tr.appendChild(el('td',null,v==null?'':v));});h.appendChild(tr);});
  D.appendChild(h);
  var dd=el('details');dd.appendChild(el('summary',null,'All fields (merged current record)'));dd.appendChild(el('pre',null,JSON.stringify(r.raw,null,1)));D.appendChild(dd);
  Array.prototype.forEach.call(document.querySelectorAll('.rec.sel'),function(x){x.classList.remove('sel');});
  Array.prototype.forEach.call(document.querySelectorAll('.rec'),function(x){if(x.dataset.k===k)x.classList.add('sel');});
  if(window.matchMedia('(min-width:900px)').matches)D.scrollTop=0;
}
apply();
})();
"""


def render_html(model: dict) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Registry history</title>
<style>{CSS}</style></head>
<body>
<header><h1>Registry history</h1>
<div class="meta" id="gen"></div><div class="meta" id="counts"></div>
<div class="prodline" id="prodline"></div>
<div class="filters">
<select id="fe" aria-label="Entity"><option value="">All entities</option></select>
<select id="fs" aria-label="Status"><option value="">All statuses</option></select>
<input id="fq" type="search" placeholder="Search id" aria-label="Search id">
</div></header>
<main>
<h2>Queue / status board</h2>
<div class="board" id="board"></div>
<div class="layout">
<div><h2>History tree</h2><div id="tree"></div></div>
<aside id="detail" class="hidden"><div class="note">Tap a record for its detail.</div></aside>
</div>
</main>
<noscript>This page needs JavaScript (inline, no network).</noscript>
<script type="application/json" id="data">{safe_json(model)}</script>
<script>{JS}</script>
</body></html>
"""


def build(paths, out) -> dict:
    """Read registry paths, write the page to `out`. -> model."""
    out = Path(out)
    problem = protected_problem(out)
    if problem:
        raise SystemExit(f"refusing to write: {problem}")
    files = input_files(paths)
    if any(out.resolve() == f.resolve() for f in files):
        raise SystemExit(f"refusing to write: {out} is an input file")
    current, history, stats = read_registry(files)
    model = build_model(current, history, stats, [str(p) for p in paths])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(render_html(model), encoding="utf-8")
    tmp.replace(out)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", action="append", help=f"JSONL file or folder of *.jsonl (repeatable). "
                    f"Default: {DEFAULT_REGISTRY}")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    paths = a.registry or [str(DEFAULT_REGISTRY)]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"not found: {', '.join(missing)} (try --registry registry/samples)")
    m = build(paths, a.out)
    print(f"wrote {a.out}: {len(m['recs'])} records, {m['stats']['lines']} lines")


if __name__ == "__main__":
    main()
