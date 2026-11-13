"""registry.append(new_only=True): two processes racing for one id -> exactly one wins (checked under the lock).

Both workers first do the old check OUTSIDE the lock (`id in current()`, as `job new` and dataset_from_lv do), then
wait at a file barrier until both have passed it, then append. So both always get past the outside check, and only
the check under the lock can stop the second one.
- control: the same race without the flag appends both lines (the race the flag closes is real in this harness).
- new_only: exactly one appends, the other is refused; one line for the id.
- `job new` racing for one name with different image lists: one job line, and the list file is the winner's.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REG.parent))                   # the repo root: `from registry import ...`
from registry import store as registry  # noqa: E402

WORKER = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from registry import store as registry
root, mode, barrier, me, rec_json = Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5], sys.argv[6]

def wait_barrier():
    (barrier / f"ready_{me}").write_text("x")
    end = time.time() + 60
    while len(list(barrier.glob("ready_*"))) < 2:
        if time.time() > end:
            print("BARRIER_TIMEOUT"); sys.exit(4)
        time.sleep(0.02)

try:
    if mode == "job":
        real_check, hit = registry._check, []
        def check_then_wait(*a, **k):            # new_job: its outside id check has passed, now meet the other one
            out = real_check(*a, **k)
            if not hit:
                hit.append(1)
                wait_barrier()
            return out
        registry._check = check_then_wait
        lst = Path(sys.argv[7])
        line = registry.new_job("check", "laptop", "Race", images=lst, root=root, date="20990101")
    else:
        rec = json.loads(rec_json)
        if rec["id"] in registry.current(root=root):   # the old check, outside the lock
            print("PRECHECK_REFUSED"); sys.exit(3)
        wait_barrier()
        line = registry.append(rec, root=root, new_only=(mode == "flag"))
    print("WON " + json.dumps(line.get("image_list")))
except registry.RegistryError as e:
    print("REFUSED " + str(e))
"""


def sample(entity):
    return json.loads((REG / "samples" / f"{entity}.jsonl").read_text(encoding="utf-8").splitlines()[0])


class TestRace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "registry"

    def tearDown(self):
        self.tmp.cleanup()

    def race(self, mode, rec=None, lists=(None, None)):
        barrier = self.base / f"barrier_{mode}"
        barrier.mkdir()
        procs = []
        for i in range(2):
            args = [sys.executable, "-B", "-c", WORKER, str(REG.parent), str(self.root), mode, str(barrier), str(i),
                    json.dumps(rec or {})] + ([str(lists[i])] if lists[i] else [])
            procs.append(subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}))
        outs = []
        for p in procs:
            out, err = p.communicate(timeout=120)
            self.assertNotIn("Traceback", err, err)
            outs.append(out.strip())
        return outs

    def lines_for(self, entity, id):
        p = self.root / f"{entity}.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
                and json.loads(x).get("id") == id] if p.exists() else []

    def test_control_without_flag_both_append(self):
        rec = dict(sample("dataset"), id="race_control")
        outs = self.race("noflag", rec)
        self.assertEqual(sorted(o.split(" ")[0] for o in outs), ["WON", "WON"], outs)
        self.assertEqual(len(self.lines_for("dataset", "race_control")), 2)      # the duplicate the flag prevents

    def test_new_only_exactly_one_wins(self):
        rec = dict(sample("dataset"), id="race_flag")
        outs = self.race("flag", rec)
        self.assertEqual(sorted(o.split(" ")[0] for o in outs), ["REFUSED", "WON"], outs)
        self.assertIn("already used by dataset/1", next(o for o in outs if o.startswith("REFUSED")))
        self.assertEqual(len(self.lines_for("dataset", "race_flag")), 1)

    def test_new_job_race_one_job_and_the_winners_list(self):
        lists = []
        for i, stems in enumerate((["a_1", "b_2"], ["c_3", "d_4", "e_5"])):
            f = self.base / f"imgs_{i}.txt"
            f.write_text("".join(s + "\n" for s in stems), encoding="utf-8")
            lists.append(f)
        outs = self.race("job", lists=lists)
        self.assertEqual(sorted(o.split(" ")[0] for o in outs), ["REFUSED", "WON"], outs)
        jobs = self.lines_for("job", "job_20990101_race")
        self.assertEqual(len(jobs), 1)
        lf = self.root / jobs[0]["image_list"]["file"]
        self.assertEqual(registry.hashlib.sha256(lf.read_bytes()).hexdigest(), jobs[0]["image_list"]["sha256"])
        self.assertEqual(len(lf.read_text(encoding="utf-8").splitlines()), jobs[0]["n_images"])


class TestNewOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "registry"

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_still_appends_new_versions(self):        # importers rely on this (last line wins)
        run = sample("run")
        registry.append(run, root=self.root)
        registry.append(dict(run, status="failed"), root=self.root)
        self.assertEqual(registry.current(root=self.root)[run["id"]]["status"], "failed")
        with self.assertRaisesRegex(registry.RegistryError, "already used by run/1"):
            registry.append(dict(run, status="finished"), root=self.root, new_only=True)
        with self.assertRaisesRegex(registry.RegistryError, "already used by run/1"):   # any entity owns the id
            registry.append(dict(sample("dataset"), id=run["id"]), root=self.root, new_only=True)
        self.assertEqual(registry.current(root=self.root)[run["id"]]["status"], "failed")

    def test_files_written_only_for_the_winner(self):
        ds = dict(sample("dataset"), id="with_list")
        registry.append(ds, root=self.root, new_only=True, files={"lists/with_list.tsv": b"a_1\ttrain\n"})
        self.assertEqual((self.root / "lists" / "with_list.tsv").read_bytes(), b"a_1\ttrain\n")
        with self.assertRaises(registry.RegistryError):
            registry.append(ds, root=self.root, new_only=True, files={"lists/with_list.tsv": b"other\n"})
        self.assertEqual((self.root / "lists" / "with_list.tsv").read_bytes(), b"a_1\ttrain\n")   # not overwritten
        with self.assertRaises(registry.RegistryError):                    # other content refused even without new_only
            registry.append(dict(ds, id="x2"), root=self.root, files={"lists/with_list.tsv": b"other\n"})
        self.assertNotIn("x2", registry.current(root=self.root))
        with self.assertRaises(registry.RegistryError):                    # outside the registry folder
            registry.append(dict(ds, id="x3"), root=self.root, files={"../escape.tsv": b"x\n"})
        self.assertFalse((self.root.parent / "escape.tsv").exists())

    def test_files_never_leave_the_registry(self):
        ds = dict(sample("dataset"), id="paths")
        bad = ["/escape.tsv", "lists/../../escape.tsv", ""] + \
            (["\\escape.tsv", "C:escape.tsv", "C:/x/escape.tsv"] if os.name == "nt" else [])   # root / drive forms
        for rel in bad:
            with self.subTest(rel=rel), self.assertRaisesRegex(registry.RegistryError, "registry folder"):
                registry.append(ds, root=self.root, files={rel: b"x\n"})
        self.assertNotIn("paths", registry.current(root=self.root))
        registry.append(ds, root=self.root, files={"lists/ok.tsv": b"x\n"})          # control: a plain one is fine
        self.assertEqual((self.root / "lists" / "ok.tsv").read_bytes(), b"x\n")

    @unittest.skipUnless(os.name == "nt", "directory junctions are Windows only")
    def test_files_refused_through_a_junction(self):
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (self.root / "lists").mkdir(parents=True)
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(self.root / "lists" / "j"), str(outside)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not (self.root / "lists" / "j").exists():
            self.skipTest(f"mklink /J unavailable: {r.stdout} {r.stderr}")
        try:
            with self.assertRaisesRegex(registry.RegistryError, "leads outside the registry folder"):
                registry.append(dict(sample("dataset"), id="via_j"), root=self.root, files={"lists/j/x.tsv": b"x\n"})
            self.assertFalse((outside / "x.tsv").exists())
            self.assertNotIn("via_j", registry.current(root=self.root))
        finally:
            os.rmdir(self.root / "lists" / "j")                   # removes the junction only, not its target


if __name__ == "__main__":
    unittest.main()
