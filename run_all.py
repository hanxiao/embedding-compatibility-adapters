"""Run the whole study: main matrix, chaining, hub, low rank, calibration size.

usage: python run_all.py --models v3,v4,v5nano,v5small,qwen3_06b,bgem3,e5large --out results.json
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from align import analyses, data
from align.adapters import fit_affine, fit_procrustes
from align.protocol import CANDIDATES, CONTROLS, Study, align_hits


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    return o


def save(res, path):
    Path(path).write_text(json.dumps(to_py(res), indent=1))


def cal_subset(X, Y, n, block=2500):
    """Balanced n/2 query-role + n/2 passage-role rows from the calibration matrix."""
    h = n // 2
    idx = np.concatenate([np.arange(h), block + np.arange(h)])
    return X[idx], Y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(data.MODELS))
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--stages", default="main,chain,gpa,lowrank,calsize")
    ap.add_argument("--pairs", default="", help="comma separated a>b to restrict the main stage")
    args = ap.parse_args()
    models = args.models.split(",")
    stages = args.stages.split(",")

    t0 = time.time()
    study = Study(args.cache, models=models)
    if Path(args.out).exists():
        res = json.loads(Path(args.out).read_text())
        print(f"resuming from {args.out}: {len(res.get('pairs', {}))} pairs, "
              f"stages {sorted(set(res) - {'meta', 'native'})}", flush=True)
    else:
        res = dict(meta=dict(models=models, tasks=study.tasks, seed=0,
                             n_val={t: int(study.val[t].size) for t in study.tasks},
                             n_test={t: int(study.test[t].size) for t in study.tasks}))
    if "native" not in res:
        res["native"] = study.run_native("test")
        save(res, args.out)

    pairs = [(s, t) for s in models for t in models if s != t]
    if args.pairs:
        want = {tuple(p.split(">")) for p in args.pairs.split(",")}
        pairs = [p for p in pairs if p in want]

    if "main" in stages:
        res.setdefault("pairs", {})
        nat_test = {m: study.eval_native(m, "test") for m in models}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            if key in res["pairs"]:
                continue
            print(f"\n[{key}]  ({time.time() - t0:.0f}s)", flush=True)
            r = study.run_pair(src, tgt)
            r["features"] = analyses.predict_features(study.cal[src], study.cal[tgt])
            for m in ("procrustes", "affine"):
                r[m]["failures"] = align_hits(nat_test[src], nat_test[tgt],
                                              r[m]["hits"], r[m]["qids"], study.tasks)
            for m in r:
                r[m].pop("hits", None)
                r[m].pop("qids", None)
            res["pairs"][key] = r
            save(res, args.out)

    if "sweep" in stages:
        print("\n[sweep] every candidate, val and test", flush=True)
        out = {}
        for src, tgt in pairs:
            X, Y = study.cal[src], study.cal[tgt]
            row = {}
            for family, fn in (("affine", lambda: [fit_affine(X, Y, c, s)
                                                   for c in (True, False) for s in (True, False)]),
                               ("ridgecca", lambda: CANDIDATES["ridgecca"](X, Y))):
                for ad in fn():
                    key = ad.meta().get("kind") if family == "affine" else f"k={ad.k}"
                    row[key] = dict(val=study.mean(study.eval_method(src, tgt, ad, "val")),
                                    test=study.mean(study.eval_method(src, tgt, ad, "test")))
            out[f"{src}->{tgt}"] = row
            print(f"  {src}->{tgt}: " + " ".join(f"{k}:{v['test']:.3f}" for k, v in row.items()),
                  flush=True)
        res["sweep"] = out
        save(res, args.out)

    if "chain" in stages:
        out = []
        for a, b, c in itertools.permutations(models, 3):
            W1 = fit_procrustes(study.cal[a], study.cal[b]).A
            W2 = fit_procrustes(study.cal[b], study.cal[c]).A
            chained = analyses.eval_map(study, a, c, lambda q: q @ W1 @ W2)
            direct = res["pairs"][f"{a}->{c}"]["procrustes"]["test"]
            out.append(dict(src=a, mid=b, tgt=c,
                            chained=float(np.mean(list(chained.values()))),
                            direct=float(np.mean(list(direct.values())))))
            print(f"  chain {a}->{b}->{c}: {out[-1]['chained']:.4f} vs direct "
                  f"{out[-1]['direct']:.4f}", flush=True)
        res["chain"] = out
        save(res, args.out)

    if "gpa" in stages:
        print("\n[gpa]", flush=True)
        R = analyses.fit_gpa({m: study.cal[m] for m in models}, models)
        hub = {}
        for src, tgt in pairs:
            v = analyses.eval_map(study, src, tgt, lambda q, s=src, t=tgt: (q @ R[s]) @ R[t].T)
            hub[f"{src}->{tgt}"] = dict(test=v, test_mean=float(np.mean(list(v.values()))))
            print(f"  hub {src}->{tgt}: {hub[f'{src}->{tgt}']['test_mean']:.4f}", flush=True)
        res["hub"] = hub
        save(res, args.out)

    if "lowrank" in stages:
        print("\n[lowrank]", flush=True)
        ks = [16, 32, 64, 128, 256, 384, 512, 768, 1024]
        out = {}
        for src, tgt in pairs:
            ds, dt = study.cal[src].shape[1], study.cal[tgt].shape[1]
            row = {}
            for k in ks:
                if k > min(ds, dt):
                    continue
                Ps = analyses.pca_basis(study.cal[src], k)
                Pt = analyses.pca_basis(study.cal[tgt], k)
                X = study.cal[src] @ Ps
                Y = study.cal[tgt] @ Pt
                W = fit_procrustes(X, Y).A

                def tf(q, Ps=Ps, W=W):
                    return (q @ Ps) @ W

                v = analyses.eval_map(study, src, tgt, tf)
                row[k] = float(np.mean(list(v.values())))
            out[f"{src}->{tgt}"] = row
            print(f"  {src}->{tgt}: " + " ".join(f"{k}:{v:.3f}" for k, v in row.items()), flush=True)
        res["lowrank"] = out
        save(res, args.out)

    if "calsize" in stages:
        print("\n[calsize]", flush=True)
        ns = [100, 250, 500, 1000, 2500, 5000]
        out = {}
        for src, tgt in pairs:
            row = {}
            for n in ns:
                X, Y = cal_subset(study.cal[src], study.cal[tgt], n)
                W = fit_procrustes(X, Y).A
                v = analyses.eval_map(study, src, tgt, lambda q, W=W: q @ W)
                row[n] = float(np.mean(list(v.values())))
            out[f"{src}->{tgt}"] = row
            print(f"  {src}->{tgt}: " + " ".join(f"{k}:{v:.3f}" for k, v in row.items()), flush=True)
        res["calsize"] = out
        save(res, args.out)

    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
