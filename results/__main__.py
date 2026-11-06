"""EV counts and sizes for one prediction_set: results.json, per_image.csv, per_sample.csv.

    python -m results --ps <CryoAI-laptop>/predict/<ps_id> [--reviewed <pkg>/reviewed.json] [--out <dir>]

Default --out is <ps>/results. Exit 0 = computed; 2 = written but refused (the reason is printed and is in
results.json `refused`): a sample mixing tile sizes or nm/px, a missing nm/px, or a set still running.
"""

import argparse
import sys
from pathlib import Path

from inference.model_classes import describe
from results.compute import CONF, MULTILAYER_RULE, compute, write_results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m results", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", required=True, type=Path, help="prediction_set folder (has request.json)")
    ap.add_argument("--reviewed", type=Path, default=None, help="reviewed.json (contract section 3)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default <ps>/results)")
    args = ap.parse_args(argv)

    res = compute(args.ps, args.reviewed)
    print(describe(res["model"]))
    print(f"rule: conf >= {CONF} (images neither edited nor reviewed), EV classes {res['rule']['ev_classes']} before layer roles, "
          f"inner layers excluded, multilayer = {MULTILAYER_RULE}")
    n_rev = sum(i["reviewed"] for i in res["images"].values())
    print(f"input: {'reviewed polygons ' + str(args.reviewed) if args.reviewed else 'model predictions'}; "
          f"{len(res['images'])} image(s), {n_rev} reviewed")
    for label, s in res["samples"].items():
        print(f"  {label:<30} images {s['n_images']:>4} (reviewed {s['n_reviewed']})  EVs {s['ev_total']:>6}  "
              f"per image {s['ev_per_image_mean']}  median {s['diam_nm_median']} nm  "
              f"multilayer {s['pct_multilayer_geometry']}% (reviewer {s['pct_multilayer_reviewer']})  "
              f"non-EV {s['nonev_total']}")
    out = write_results(res, args.out or args.ps / "results")
    print(f"wrote {out / 'results.json'}, per_image.csv, per_sample.csv")
    if res["refused"]:
        print(f"REFUSED: {res['refused']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
