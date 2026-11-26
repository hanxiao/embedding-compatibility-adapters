"""Run the complete study across all models and stages.

Stages:
  main       56 directed pairs x all adapter families + controls + failure recovery
  sweep      per-candidate ablation (affine variants, ridge lambda/scale, CCA rank)
  multiseed  5 random val/test splits for statistical stability
  deployment query-side vs corpus-side conversion + progressive hybrid index migration
  role_cal   query-only vs passage-only vs mixed calibration ablation
  chain      all A->B->C triples for both orthogonal Procrustes and Affine (Plato's Key)
  gpa        full-5k 40-iter Generalized Procrustes hub (both uncentred and Affine GPA)
  lowrank    one-sided low-rank Affine (no corpus re-indexing) + two-sided PCA-Procrustes
  calsize    calibration budget curve (50..5000) for Procrustes, Affine, Ridge
  latency    wall-clock fit and inference throughput benchmark

usage:
    python run_all.py --models v3,v4,v5nano,v5small,qwen3_06b,bgem3,e5large,arctic_m --out paper/results.json
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from align import analyses, data
from align.adapters import (
    fit_affine,
    fit_lowrank_affine,
    fit_procrustes,
    fit_ridge,
)
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_py(res), indent=1))


def cal_subset(X, Y, n, block=2500):
    """Balanced n/2 query-role + n/2 passage-role rows from the 5,000 calibration matrix."""
    h = n // 2
    idx = np.concatenate([np.arange(h), block + np.arange(h)])
    return X[idx], Y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(data.MODELS))
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="paper/results.json")
    ap.add_argument(
        "--stages",
        default="main,sweep,multiseed,deployment,role_cal,chain,gpa,lowrank,calsize,latency",
    )
    ap.add_argument("--pairs", default="", help="comma-separated a>b to restrict pairs")
    ap.add_argument("--force", action="store_true", help="recompute stages even if present")
    args = ap.parse_args()
    models = args.models.split(",")
    stages = args.stages.split(",")

    t0 = time.time()
    study = Study(args.cache, models=models)
    if Path(args.out).exists() and not args.force:
        res = json.loads(Path(args.out).read_text())
        print(
            f"resuming from {args.out}: {len(res.get('pairs', {}))} pairs, "
            f"stages {sorted(set(res) - {'meta', 'native'})}",
            flush=True,
        )
    else:
        res = dict(
            meta=dict(
                models=models,
                model_info={m: data.MODELS[m] for m in models},
                tasks=study.tasks,
                seed=0,
                n_val={t: int(study.val[t].size) for t in study.tasks},
                n_test={t: int(study.test[t].size) for t in study.tasks},
            )
        )

    if "native" not in res:
        res["native"] = study.run_native("test")
        res["anisotropy"] = {m: analyses.anisotropy(study.cal[m]) for m in models}
        res["anisotropy_query"] = {m: analyses.anisotropy(study.cal_q[m]) for m in models}
        res["anisotropy_passage"] = {m: analyses.anisotropy(study.cal_p[m]) for m in models}
        save(res, args.out)

    pairs = [(s, t) for s in models for t in models if s != t]
    if args.pairs:
        want = {tuple(p.split(">")) for p in args.pairs.split(",")}
        pairs = [p for p in pairs if p in want]

    # ------------------------------------------------------------------ 1. main
    if "main" in stages:
        res.setdefault("pairs", {})
        nat_test = {m: study.eval_native(m, "test") for m in models}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            if key in res["pairs"] and not args.force:
                continue
            print(f"\n[main {key}] ({time.time() - t0:.0f}s)", flush=True)
            r = study.run_pair(src, tgt)
            r["features"] = analyses.predict_features(study.cal[src], study.cal[tgt])
            for m in ("procrustes", "affine", "affine-cs", "ridge"):
                r[m]["failures"] = align_hits(
                    nat_test[src], nat_test[tgt], r[m]["hits"], r[m]["qids"], study.tasks
                )
            for m in list(r.keys()):
                if isinstance(r[m], dict):
                    r[m].pop("hits", None)
                    r[m].pop("qids", None)
            res["pairs"][key] = r
            save(res, args.out)

    # ------------------------------------------------------------------ 2. sweep
    if "sweep" in stages and ("sweep" not in res or args.force):
        print(f"\n[sweep] ({time.time() - t0:.0f}s)", flush=True)
        out = {}
        for src, tgt in pairs:
            X, Y = study.cal[src], study.cal[tgt]
            row = {}
            cands = (
                [fit_affine(X, Y, c, s) for c in (True, False) for s in (True, False)]
                + [fit_affine(X, Y, True, True, ls_scale=True)]
                + [fit_ridge(X, Y, lam=l, rescale=r) for l in (1e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1) for r in (False, True)]
                + CANDIDATES["ridgecca"](X, Y)
            )
            for ad in cands:
                mt = ad.meta()
                if mt["kind"] == "ridgecca":
                    ck = f"cca_k={mt['k']}"
                elif mt["kind"].startswith("ridge"):
                    ck = f"{mt['kind']}_lam={mt['lam']:g}"
                else:
                    ck = mt["kind"]
                row[ck] = dict(
                    val=study.mean(study.eval_method(src, tgt, ad, "val")),
                    test=study.mean(study.eval_method(src, tgt, ad, "test")),
                )
            out[f"{src}->{tgt}"] = row
        res["sweep"] = out
        save(res, args.out)
        print(f"  sweep done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ 3. multiseed
    if "multiseed" in stages and ("multiseed" not in res or args.force):
        print(f"\n[multiseed] 5 seeds ({time.time() - t0:.0f}s)", flush=True)
        fast_fams = {
            k: CANDIDATES[k]
            for k in ("procrustes", "affine-cs", "affine", "ridge", "ridgecca")
        }
        fast_fams["identity"] = CONTROLS["identity"]
        # Pre-fit all pair candidates once (calibration is fixed; only val/test split varies)
        fitted = {}
        for src, tgt in pairs:
            X, Y = study.cal[src], study.cal[tgt]
            fitted[(src, tgt)] = {k: fn(X, Y) for k, fn in fast_fams.items()}

        ms = {}
        for seed in range(5):
            study.set_seed(seed)
            nat_s = {m: study.mean(study.eval_native(m, "test")) for m in models}
            fam_scores = {k: [] for k in fast_fams}
            fam_ratios = {k: [] for k in fast_fams}
            for src, tgt in pairs:
                for k, cands in fitted[(src, tgt)].items():
                    if len(cands) == 1:
                        ad = cands[0]
                    else:
                        vals = [study.mean(study.eval_method(src, tgt, a, "val")) for a in cands]
                        ad = cands[int(np.argmax(vals))]
                    sc = study.mean(study.eval_method(src, tgt, ad, "test"))
                    fam_scores[k].append(sc)
                    fam_ratios[k].append(sc / nat_s[src])
            ms[str(seed)] = {
                k: dict(ndcg=float(np.mean(fam_scores[k])), ratio_src=float(np.mean(fam_ratios[k])))
                for k in fast_fams
            }
            print(
                f"  seed {seed}: "
                + " ".join(f"{k}={ms[str(seed)][k]['ndcg']:.4f}" for k in fast_fams),
                flush=True,
            )
        study.set_seed(0)
        res["multiseed"] = ms
        save(res, args.out)

    # ------------------------------------------------------------------ 4. deployment
    if "deployment" in stages and ("deployment" not in res or args.force):
        print(f"\n[deployment: query-side vs corpus-side + hybrid migration] ({time.time() - t0:.0f}s)", flush=True)
        dep = {}
        for idx_m, qry_m in pairs:
            # User has documents frozen in idx_m, wants to search with qry_m queries.
            # Option A (query-side): fit qry_m -> idx_m, map queries into frozen idx_m space.
            Xq, Yq = study.cal[qry_m], study.cal[idx_m]
            # Option B (corpus-side): fit idx_m -> qry_m, matrix-multiply idx_m corpus into qry_m space.
            Xc, Yc = study.cal[idx_m], study.cal[qry_m]

            row = {}
            for name, fit_fn in (
                ("procrustes", lambda A, B: fit_procrustes(A, B)),
                ("affine-cs", lambda A, B: fit_affine(A, B, True, True)),
                ("ridge", lambda A, B: fit_ridge(A, B, lam=1e-4, rescale=True)),
            ):
                ad_q = fit_fn(Xq, Yq)
                ad_c = fit_fn(Xc, Yc)
                q_ndcg = study.mean(study.eval_method(qry_m, idx_m, ad_q, "test"))
                c_res = analyses.eval_corpus_map(study, idx_m, qry_m, ad_c.transform_source, "test")
                c_ndcg = float(np.mean(list(c_res.values())))
                row[name] = dict(query_side=q_ndcg, corpus_side=c_ndcg)
                if name == "affine-cs":
                    row["hybrid_affine"] = analyses.eval_hybrid_migration(
                        study, idx_m, qry_m, ad_c.transform_source,
                        fracs=(0.0, 0.25, 0.5, 0.75, 1.0), split="test", seed=0,
                    )
                if name == "procrustes":
                    row["hybrid_procrustes"] = analyses.eval_hybrid_migration(
                        study, idx_m, qry_m, ad_c.transform_source,
                        fracs=(0.0, 0.25, 0.5, 0.75, 1.0), split="test", seed=0,
                    )
            dep[f"idx={idx_m}|qry={qry_m}"] = row
        res["deployment"] = dep
        save(res, args.out)
        # Print canonical v3 <-> v5nano summary
        for key in ("idx=v3|qry=v5nano", "idx=v5nano|qry=v3"):
            if key in dep:
                print(f"  {key}: {dep[key]}", flush=True)

    # ------------------------------------------------------------------ 5. role_cal
    if "role_cal" in stages and ("role_cal" not in res or args.force):
        print(f"\n[role_cal: query vs passage vs mixed calibration] ({time.time() - t0:.0f}s)", flush=True)
        rc = {}
        for src, tgt in pairs:
            row = {}
            for role, X, Y in (
                ("mixed_5000", study.cal[src], study.cal[tgt]),
                ("mixed_2500", *cal_subset(study.cal[src], study.cal[tgt], 2500)),
                ("query_2500", study.cal_q[src], study.cal_q[tgt]),
                ("passage_2500", study.cal_p[src], study.cal_p[tgt]),
            ):
                ad_p = fit_procrustes(X, Y)
                ad_a = fit_affine(X, Y, True, True)
                # Evaluate both query-side (src query -> tgt corpus) and corpus-side (src corpus -> tgt query)
                q_p = study.mean(study.eval_method(src, tgt, ad_p, "test"))
                q_a = study.mean(study.eval_method(src, tgt, ad_a, "test"))
                c_p = float(np.mean(list(analyses.eval_corpus_map(study, src, tgt, ad_p.transform_source, "test").values())))
                c_a = float(np.mean(list(analyses.eval_corpus_map(study, src, tgt, ad_a.transform_source, "test").values())))
                row[role] = dict(q_proc=q_p, q_aff=q_a, c_proc=c_p, c_aff=c_a)
            rc[f"{src}->{tgt}"] = row
        res["role_cal"] = rc
        save(res, args.out)
        for role in ("mixed_5000", "mixed_2500", "query_2500", "passage_2500"):
            qp = np.mean([r[role]["q_proc"] for r in rc.values()])
            qa = np.mean([r[role]["q_aff"] for r in rc.values()])
            cp = np.mean([r[role]["c_proc"] for r in rc.values()])
            ca = np.mean([r[role]["c_aff"] for r in rc.values()])
            print(f"  {role:12s} query-side proc={qp:.4f} aff={qa:.4f} | corpus-side proc={cp:.4f} aff={ca:.4f}", flush=True)

    # ------------------------------------------------------------------ 6. chain
    if "chain" in stages and ("chain" not in res or args.force):
        print(f"\n[chain: all triples, Procrustes + Affine] ({time.time() - t0:.0f}s)", flush=True)
        W_proc = {(a, b): fit_procrustes(study.cal[a], study.cal[b]) for a, b in pairs}
        W_aff = {(a, b): fit_affine(study.cal[a], study.cal[b], True, True) for a, b in pairs}
        out_p, out_a = [], []
        for a, b, c in itertools.permutations(models, 3):
            # Procrustes chain
            p1, p2 = W_proc[(a, b)], W_proc[(b, c)]
            ch_p = analyses.eval_map(study, a, c, lambda q, p1=p1, p2=p2: p2.transform_source(p1.transform_source(q)))
            dir_p = res["pairs"][f"{a}->{c}"]["procrustes"]["test_mean"]
            out_p.append(dict(src=a, mid=b, tgt=c, chained=float(np.mean(list(ch_p.values()))), direct=dir_p))
            # Affine chain
            a1, a2 = W_aff[(a, b)], W_aff[(b, c)]
            ch_a = analyses.eval_map(study, a, c, lambda q, a1=a1, a2=a2: a2.transform_source(a1.transform_source(q)))
            dir_a = res["pairs"][f"{a}->{c}"]["affine-cs"]["test_mean"]
            out_a.append(dict(src=a, mid=b, tgt=c, chained=float(np.mean(list(ch_a.values()))), direct=dir_a))
        res["chain"] = out_p
        res["chain_affine"] = out_a
        save(res, args.out)
        rp = [c["chained"] / c["direct"] for c in out_p if c["direct"] > 0]
        ra = [c["chained"] / c["direct"] for c in out_a if c["direct"] > 0]
        print(f"  procrustes chain ({len(out_p)} triples): median={np.median(rp):.4f} p10={np.percentile(rp, 10):.4f}", flush=True)
        print(f"  affine     chain ({len(out_a)} triples): median={np.median(ra):.4f} p10={np.percentile(ra, 10):.4f}", flush=True)

    # ------------------------------------------------------------------ 7. gpa
    if "gpa" in stages and ("gpa_affine" not in res or args.force):
        print(f"\n[gpa: full 5k, 40 iters, Procrustes + Affine GPA] ({time.time() - t0:.0f}s)", flush=True)
        R = analyses.fit_gpa({m: study.cal[m] for m in models}, models, iters=40)
        hub = {}
        for src, tgt in pairs:
            v = analyses.eval_map(study, src, tgt, lambda q, s=src, t=tgt: (q @ R[s]) @ R[t].T)
            hub[f"{src}->{tgt}"] = dict(test=v, test_mean=float(np.mean(list(v.values()))))
        res["hub"] = hub

        ag = analyses.fit_affine_gpa({m: study.cal[m] for m in models}, models, iters=40)
        hub_a = {}
        for src, tgt in pairs:
            scale = ag["rms"][tgt] / max(ag["rms"][src], 1e-12)
            tf = lambda q, s=src, t=tgt, sc=scale: (
                ((q - ag["mu"][s]) @ ag["R"][s]) @ ag["R"][t].T * sc + ag["mu"][t]
            )
            v = analyses.eval_map(study, src, tgt, tf)
            hub_a[f"{src}->{tgt}"] = dict(test=v, test_mean=float(np.mean(list(v.values()))))
        res["hub_affine"] = hub_a
        save(res, args.out)
        dp = np.mean([hub[k]["test_mean"] - res["pairs"][k]["procrustes"]["test_mean"] for k in hub])
        da = np.mean([hub_a[k]["test_mean"] - res["pairs"][k]["affine-cs"]["test_mean"] for k in hub_a])
        print(f"  hub procrustes mean={np.mean([v['test_mean'] for v in hub.values()]):.4f} (delta {dp:+.4f})", flush=True)
        print(f"  hub affine     mean={np.mean([v['test_mean'] for v in hub_a.values()]):.4f} (delta {da:+.4f})", flush=True)

    # ------------------------------------------------------------------ 8. lowrank
    if "lowrank" in stages and ("lowrank_affine" not in res or args.force):
        print(f"\n[lowrank: full 5k, one-sided Affine + two-sided PCA-Procrustes] ({time.time() - t0:.0f}s)", flush=True)
        ks = [16, 32, 64, 128, 256, 384, 512, 768, 1024]
        # Precompute centred calibration matrices & full PCA bases ONCE per model (8 SVDs total!)
        mu_m, Xc_m, Vt_m = {}, {}, {}
        for m in models:
            X = np.asarray(study.cal[m], dtype=np.float64)
            mu_m[m] = X.mean(0)
            Xc_m[m] = X - mu_m[m]
            _, _, Vt_m[m] = np.linalg.svd(Xc_m[m], full_matrices=False)
        lr_aff, lr_proc = {}, {}
        for src, tgt in pairs:
            ds, dt = study.cal[src].shape[1], study.cal[tgt].shape[1]
            row_a, row_p = {}, {}
            for k in ks:
                if k > min(ds, dt):
                    continue
                Ps = Vt_m[src][:k].T
                Pt = Vt_m[tgt][:k].T
                # One-sided low-rank affine (Plato's Key protocol, zero corpus touch!)
                Xk, Yk = Xc_m[src] @ Ps, Xc_m[tgt] @ Pt
                U, _, Vt = np.linalg.svd(Xk.T @ Yk, full_matrices=False)
                Wk = U @ Vt
                s = float(np.linalg.norm(Yk) / max(np.linalg.norm(Xk), 1e-12))
                A_eff = (Ps @ Wk @ Pt.T).astype(np.float32)
                from align.adapters import Affine
                ad = Affine(A_eff, s, mu_m[src], mu_m[tgt], name=f"lowrank-affine-{k}")
                row_a[k] = study.mean(study.eval_method(src, tgt, ad, "test"))
                # Two-sided PCA-Procrustes (for comparison)
                Ps32, Pt32 = Ps.astype(np.float32), Pt.astype(np.float32)
                W = fit_procrustes(study.cal[src] @ Ps32, study.cal[tgt] @ Pt32).A
                vp = analyses.eval_map(
                    study, src, tgt,
                    lambda q, Ps=Ps32, W=W: (q @ Ps) @ W,
                    corpus_transform=lambda C, Pt=Pt32: C @ Pt,
                )
                row_p[k] = float(np.mean(list(vp.values())))
            lr_aff[f"{src}->{tgt}"] = row_a
            lr_proc[f"{src}->{tgt}"] = row_p
        res["lowrank_affine"] = lr_aff
        res["lowrank"] = lr_proc
        save(res, args.out)
        for k in (16, 32, 64, 128, 256, 384, 512, 768):
            va = [r[k] for r in lr_aff.values() if k in r]
            print(f"  lowrank_affine k={k:4d}: mean={np.mean(va):.4f} (n={len(va)})", flush=True)

    # ------------------------------------------------------------------ 9. calsize
    if "calsize" in stages and ("calsize_affine" not in res or args.force):
        print(f"\n[calsize: 50..5000, Procrustes + Affine + Ridge] ({time.time() - t0:.0f}s)", flush=True)
        ns = [50, 100, 250, 500, 1000, 2500, 5000]
        cs_p, cs_a, cs_r = {}, {}, {}
        for src, tgt in pairs:
            rp, ra, rr = {}, {}, {}
            for n in ns:
                X, Y = cal_subset(study.cal[src], study.cal[tgt], n)
                rp[n] = study.mean(study.eval_method(src, tgt, fit_procrustes(X, Y), "test"))
                ra[n] = study.mean(study.eval_method(src, tgt, fit_affine(X, Y, True, True), "test"))
                rr[n] = study.mean(study.eval_method(src, tgt, fit_ridge(X, Y, lam=1e-4, rescale=True), "test"))
            cs_p[f"{src}->{tgt}"] = rp
            cs_a[f"{src}->{tgt}"] = ra
            cs_r[f"{src}->{tgt}"] = rr
        res["calsize"] = cs_p
        res["calsize_affine"] = cs_a
        res["calsize_ridge"] = cs_r
        save(res, args.out)
        for n in ns:
            print(
                f"  n={n:4d}: proc={np.mean([r[n] for r in cs_p.values()]):.4f} "
                f"aff={np.mean([r[n] for r in cs_a.values()]):.4f} "
                f"ridge={np.mean([r[n] for r in cs_r.values()]):.4f}",
                flush=True,
            )

    # ------------------------------------------------------------------ 10. latency
    if "latency" in stages and ("latency" not in res or args.force):
        print(f"\n[latency benchmark] ({time.time() - t0:.0f}s)", flush=True)
        X, Y = study.cal["v5nano"], study.cal["v3"]  # 768 -> 1024
        # Fit times (median of 5 runs)
        t_proc, t_aff, t_ridge = [], [], []
        for _ in range(5):
            ts = time.perf_counter(); fit_procrustes(X, Y); t_proc.append((time.perf_counter() - ts) * 1000)
            ts = time.perf_counter(); ad = fit_affine(X, Y, True, True); t_aff.append((time.perf_counter() - ts) * 1000)
            ts = time.perf_counter(); fit_ridge(X, Y, 1e-4, True); t_ridge.append((time.perf_counter() - ts) * 1000)
        # Single-query latency (1000 queries one by one)
        q1 = X[:1].astype(np.float32)
        ts = time.perf_counter()
        for _ in range(1000):
            _ = ad.transform_source(q1)
        q_us = (time.perf_counter() - ts) * 1e6 / 1000
        # Batch 100k vectors corpus conversion throughput
        batch = np.tile(X, (20, 1)).astype(np.float32)  # 100,000 x 768
        ts = time.perf_counter()
        _ = ad.transform_source(batch)
        dt_100k = time.perf_counter() - ts
        res["latency"] = dict(
            pair="v5nano->v3 (768->1024, n=5000)",
            fit_procrustes_ms=float(np.median(t_proc)),
            fit_affine_ms=float(np.median(t_aff)),
            fit_ridge_ms=float(np.median(t_ridge)),
            single_query_us=float(q_us),
            corpus_100k_s=float(dt_100k),
            corpus_1m_est_s=float(dt_100k * 10),
        )
        save(res, args.out)
        print(f"  latency: {res['latency']}", flush=True)

    print(f"\nALL DONE in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
