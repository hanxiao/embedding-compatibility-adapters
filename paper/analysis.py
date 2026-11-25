"""Turn results.json into the numbers and figures of the report."""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL_ORDER = ["v5small", "v5nano", "qwen3_06b", "v4", "v3", "e5large", "bgem3"]
ORDER = MODEL_ORDER
SHORT = {"v5small": "v5s", "v5nano": "v5n", "qwen3_06b": "q3-0.6", "v4": "v4",
         "v3": "v3", "e5large": "e5l", "bgem3": "bm3"}
FAMILIES = ["procrustes", "affine", "ridgecca", "sgd-linear-infonce",
            "sgd-mlp-infonce", "krr-rbf"]
CONTROLS = ["random", "shuffled", "identity"]
AFFINE_LABELS = {"affine-c-s": "centre + scale", "affine-c": "centre",
                 "affine-s": "scale", "affine": "neither"}


def m(v):
    return float(np.mean(list(v.values())))


def load(path):
    return json.loads(Path(path).read_text())


def native_means(D):
    return {k: v["test_mean"] for k, v in D["native"].items()}


def rows(D):
    nat = native_means(D)
    out = []
    for key, p in D["pairs"].items():
        s, t = key.split("->")
        r = dict(src=s, tgt=t, source=nat[s], target=nat[t],
                 expanding=D["native"][s]["test_mean"] <= D["native"][t]["test_mean"])
        for f in FAMILIES + CONTROLS:
            if f in p:
                r[f] = p[f]["test_mean"]
        r["features"] = p.get("features", {})
        for f in ("procrustes", "affine"):
            if f in p and "failures" in p[f]:
                r[f + "_fail"] = p[f]["failures"]
        out.append(r)
    return out


def summarise(D, out):
    nat = native_means(D)
    R = rows(D)
    L = []
    add = L.append
    add("# results summary\n")
    add("## native nDCG@10, 650 queries / 13 tasks (test split: 429 queries)\n")
    add("| model | test | all |")
    add("|---|---|---|")
    for k in sorted(nat, key=lambda x: -nat[x]):
        add(f"| {k} | {nat[k]:.4f} | {D['native'][k]['all_mean']:.4f} |")
    add("")

    add("## per-pair, procrustes (test split)\n")
    add("| pair | source | adapted | target | /source | /target | d_s>d_t |")
    add("|---|---|---|---|---|---|---|")
    for r in R:
        a = r.get("procrustes", float("nan"))
        add(f"| {r['src']}->{r['tgt']} | {r['source']:.4f} | {a:.4f} | {r['target']:.4f} | "
            f"{a/r['source']:.3f} | {a/r['target']:.3f} | {'Y' if not r['expanding'] else ''} |")
    add("")

    add("## families, mean over pairs (ratio to source, ratio to target)\n")
    add("| method | nDCG | /source | /target | wins vs procrustes | losses |")
    add("|---|---|---|---|---|---|")
    for f in FAMILIES + CONTROLS:
        v = [r[f] for r in R if f in r]
        if not v:
            continue
        rs = [r[f] / r["source"] for r in R if f in r]
        rt = [r[f] / r["target"] for r in R if f in r]
        w = sum(1 for r in R if f in r and r[f] > r["procrustes"] + 1e-9)
        l = sum(1 for r in R if f in r and r[f] < r["procrustes"] - 1e-9)
        add(f"| {f} | {np.mean(v):.4f} | {np.mean(rs):.3f} | {np.mean(rt):.3f} | {w} | {l} |")
    add("")

    exp = [r for r in R if r["expanding"]]
    comp = [r for r in R if not r["expanding"]]
    add(f"## by direction: expanding (d_s<=d_t) n={len(exp)}, compressing n={len(comp)}\n")
    add("| method | expanding | compressing |")
    add("|---|---|---|")
    for f in FAMILIES:
        if not exp or not comp:
            break
        add(f"| {f} | {np.mean([r[f] for r in exp]):.4f} | {np.mean([r[f] for r in comp]):.4f} |")
    add("")

    if "sweep" in D:
        add("## affine variants and cca rank, mean over pairs (test split)\n")
        keys = {}
        for k, row in D["sweep"].items():
            for kk, vv in row.items():
                keys.setdefault(kk, []).append(vv["test"])
        add("| candidate | test | n |")
        add("|---|---|---|")
        for k, v in sorted(keys.items()):
            add(f"| {k} | {np.mean(v):.4f} | {len(v)} |")
        add("")
        add("### selection gap: val-optimal vs test-optimal, per candidate set\n")
        add("| pair | affine val pick | affine val | affine best test | cca val pick | cca val | cca best test |")
        add("|---|---|---|---|---|---|---|")
        for k, row in list(D["sweep"].items()):
            for fam in ("affine", "ridgecca"):
                pass
        add("")

    if "chain" in D:
        ch = D["chain"]
        rel = [c["chained"] / c["direct"] for c in ch if c["direct"] > 0]
        add(f"## chaining: {len(ch)} triples, chained/direct median {np.median(rel):.3f}, "
            f"p10 {np.percentile(rel,10):.3f}, p90 {np.percentile(rel,90):.3f}, "
            f"min {np.min(rel):.3f}\n")
        add("| triple | chained | direct | ratio |")
        add("|---|---|---|---|")
        for c in sorted(ch, key=lambda c: c["chained"] / max(c["direct"], 1e-9))[:8]:
            add(f"| {c['src']}->{c['mid']}->{c['tgt']} | {c['chained']:.4f} | "
                f"{c['direct']:.4f} | {c['chained']/c['direct']:.3f} |")
        add("")

    if "hub" in D:
        h = D["hub"]
        pw = {k: r["procrustes"]["test_mean"] for k, r in D["pairs"].items()}
        diffs = [h[k]["test_mean"] - pw[k] for k in h if k in pw]
        add(f"## hub (GPA, 7 maps) vs pairwise procrustes: mean delta "
            f"{np.mean(diffs):+.4f}, hub wins {sum(d>0 for d in diffs)}/{len(diffs)}\n")
        add("| pair | hub | pairwise | delta |")
        add("|---|---|---|---|")
        for k in sorted(h, key=lambda k: h[k]["test_mean"] - pw.get(k, 0))[:6]:
            add(f"| {k} | {h[k]['test_mean']:.4f} | {pw[k]:.4f} | "
                f"{h[k]['test_mean']-pw[k]:+.4f} |")
        add("")

    if "lowrank" in D:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from align.data import MODELS as _M
        lr = {k: v for k, v in D["lowrank"].items()
              if min(_M[k.split("->")[0]]["dim"], _M[k.split("->")[1]]["dim"]) >= 1024}
        add(f"## low rank: test nDCG vs k principal components per model "
            f"({len(lr)} pairs whose widths are all >= 1024)\n")
        add("| k | mean | /full |")
        add("|---|---|---|")
        for k in sorted({int(x) for r in lr.values() for x in r}):
            v = [r[str(k)] for r in lr.values() if str(k) in r]
            full = [r[str(max(int(x) for x in r))] for r in lr.values() if r]
            add(f"| {k} | {np.mean(v):.4f} | {np.mean([a/b for a,b in zip(v,full)]):.3f} |")
        add("")

    if "calsize" in D:
        cs = D["calsize"]
        add("## calibration size\n")
        add("| n | mean |")
        add("|---|---|")
        for n in sorted({int(x) for r in cs.values() for x in r}):
            v = [r[str(n)] for r in cs.values() if str(n) in r]
            add(f"| {n} | {np.mean(v):.4f} |")
        add("")

    add("## predictors\n")
    feats = ["rho", "cka", "nagree", "top_energy", "eff_rank", "dim_ratio", "resid_gap"]
    rec = {(r["src"], r["tgt"]): r for r in R}
    und = {}
    for (a, b), r in rec.items():
        if (b, a) in rec:
            r1 = r["procrustes"] / r["source"]
            r2 = rec[(b, a)]["procrustes"] / rec[(b, a)]["source"]
            und[(a, b)] = dict(sym=(r1 + r2) / 2, anti=(r1 - r2) / 2, feats=r["features"])
    y = np.array([r["procrustes"] / r["source"] for r in R])
    sym = np.array([u["sym"] for u in und.values()])
    anti = np.array([u["anti"] for u in und.values()])
    add("| feature | r(adapted/source) | r(symmetric) | r(antisymmetric) |")
    add("|---|---|---|---|")
    for f in feats:
        x = np.array([r["features"].get(f, np.nan) for r in R])
        xs = np.array([u["feats"].get(f, np.nan) for u in und.values()])
        ok, ok2 = ~np.isnan(x), ~np.isnan(xs)
        c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
        add(f"| {f} | {c(x[ok], y[ok]):+.3f} | {c(xs[ok2], sym[ok2]):+.3f} | {c(xs[ok2], anti[ok2]):+.3f} |")
    add("")
    if len(anti):
        add(f"antisymmetric share of the variance in adapted/source: {np.var(anti)/np.var(y):.3f}")
        add(f"range of the antisymmetric part: {anti.min():+.3f} to {anti.max():+.3f}\n")
    else:
        add("(no reverse pairs yet)\n")

    add("## recoverable failures (procrustes)\n")
    add("| pair | recoverable | n | loss | n | target-only kept |")
    add("|---|---|---|---|---|---|")
    for r in R:
        f = r.get("procrustes_fail")
        if f:
            add(f"| {r['src']}->{r['tgt']} | {f['recoverable']:.3f} | {f['n_recoverable']} | "
                f"{f['loss']:.3f} | {f['n_target_hits']} | {f['keep_target_only']:.3f} |")
    add("")
    agg = {}
    for r in R:
        f = r.get("procrustes_fail")
        if f:
            for k in ("n_recoverable", "n_target_hits", "n_target_only"):
                agg[k] = agg.get(k, 0) + f[k]
    add(f"pooled: recoverable {np.mean([r['procrustes_fail']['recoverable'] for r in R if 'procrustes_fail' in r]):.3f}, "
        f"loss {np.mean([r['procrustes_fail']['loss'] for r in R if 'procrustes_fail' in r]):.3f}\n")

    Path(out).write_text("\n".join(L))
    print("\n".join(L[:80]))


# ------------------------------------------------------------------ latex tables
def _f(x, n=4):
    return f"{x:.{n}f}"


def _m(name):
    """model key to latex-safe label"""
    return name.replace("_", r"\_")


def _pair(a, b):
    return f"{_m(a)}$\\to${_m(b)}"


def emit_tables(D, d):
    from scipy.stats import wilcoxon
    d = Path(d)
    d.mkdir(exist_ok=True)
    nat = native_means(D)
    R = rows(D)
    T = lambda name, body: (d / f"{name}.tex").write_text(body)

    # native
    b = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"model & family & dim & all 649 & held-out 429 \\", r"\midrule"]
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from align.data import MODELS
    for k in sorted(nat, key=lambda x: -nat[x]):
        b.append(f"{MODELS[k]['label']} & {MODELS[k]['family']} & {MODELS[k]['dim']} & "
                 f"{_f(D['native'][k]['all_mean'])} & {_f(nat[k])} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_native", "\n".join(b))

    # adapted matrix, diagonal = native
    models = [m for m in ORDER if m in nat]
    b = [r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule",
         "source $\\backslash$ target & " + " & ".join(SHORT[m] for m in models) + r" \\",
         r"\midrule"]
    for s in models:
        cells = []
        for t in models:
            if s == t:
                cells.append(f"\\textit{{{_f(nat[s], 3)}}}")
            else:
                p = D["pairs"].get(f"{s}->{t}")
                cells.append(_f(p["procrustes"]["test_mean"], 3) if p else "--")
        b.append(f"{SHORT[s]} & " + " & ".join(cells) + r" \\")
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_matrix", "\n".join(b))

    # families
    b = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"method & nDCG@10 & /source & /target & wins & losses \\", r"\midrule"]
    for f in FAMILIES + CONTROLS:
        v = [r[f] for r in R if f in r]
        if not v:
            continue
        w = l = 0
        for r in R:
            if f not in r:
                continue
            pt = D["pairs"][f"{r['src']}->{r['tgt']}"]
            a = np.array([pt[f]["test"][t] for t in pt[f]["test"]])
            c = np.array([pt["procrustes"]["test"][t] for t in pt[f]["test"]])
            if np.allclose(a, c):
                continue
            try:
                p = wilcoxon(a, c).pvalue
            except ValueError:
                continue
            if p < 0.05:
                if a.mean() > c.mean():
                    w += 1
                else:
                    l += 1
        b.append(f"{f.replace('-infonce','').replace('sgd-','SGD-')} & {_f(np.mean(v))} & "
                 f"{_f(np.mean([r[f]/r['source'] for r in R if f in r]), 3)} & "
                 f"{_f(np.mean([r[f]/r['target'] for r in R if f in r]), 3)} & {w} & {l} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_families", "\n".join(b))

    # by direction
    exp = [r for r in R if r["expanding"]]
    comp = [r for r in R if not r["expanding"]]
    b = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"method & expanding & compressing & difference \\", r"\midrule"]
    for f in FAMILIES:
        if not exp or not comp:
            continue
        a = np.mean([r[f] for r in exp])
        c = np.mean([r[f] for r in comp])
        b.append(f"{f.replace('-infonce','')} & {_f(a)} & {_f(c)} & {_f(a-c)} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_direction", "\n".join(b))

    if "sweep" in D:
        agg = {}
        for row in D["sweep"].values():
            for k, v in row.items():
                agg.setdefault(k, []).append(v["test"])
        b = [r"\begin{tabular}{lrr}", r"\toprule",
             r"operating point & nDCG@10 & pairs \\", r"\midrule"]
        for k in ["affine-c-s", "affine-c", "affine-s", "affine"]:
            if k in agg:
                b.append(f"affine, {AFFINE_LABELS[k]} & {_f(np.mean(agg[k]))} & {len(agg[k])} \\\\")
        for k in sorted([k for k in agg if k.startswith("k=")], key=lambda x: int(x[2:])):
            b.append(f"ridge CCA {k} & {_f(np.mean(agg[k]))} & {len(agg[k])} \\\\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_sweep", "\n".join(b))

    # predictors
    rec = {(r["src"], r["tgt"]): r for r in R}
    und = {}
    for (a, b), r in rec.items():
        if (b, a) in rec:
            r1 = r["procrustes"] / r["source"]
            r2 = rec[(b, a)]["procrustes"] / rec[(b, a)]["source"]
            und[(a, b)] = dict(sym=(r1 + r2) / 2, anti=(r1 - r2) / 2, feats=r["features"])
    y = np.array([r["procrustes"] / r["source"] for r in R])
    sym = np.array([u["sym"] for u in und.values()])
    anti = np.array([u["anti"] for u in und.values()])
    cor = lambda a, b: (float(np.corrcoef(a, b)[0, 1])
                        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan"))
    idn = np.array([r["identity"] for r in R])
    gain = np.array([r["affine"] - r["identity"] for r in R])
    b = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"feature & $r$(identity) & $r$(gain over identity) & $r$(antisymmetric) \\",
         r"\midrule"]
    for f in ["rho", "cka", "nagree", "top_energy", "eff_rank", "dim_ratio", "resid_gap"]:
        x = np.array([r["features"].get(f, np.nan) for r in R])
        xs = np.array([u["feats"].get(f, np.nan) for u in und.values()])
        ok, ok2 = ~np.isnan(x), ~np.isnan(xs)
        b.append(f"{f.replace('_','-')} & {cor(x[ok], idn[ok]):+.3f} & "
                 f"{cor(x[ok], gain[ok]):+.3f} & {cor(xs[ok2], anti[ok2]):+.3f} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_predictors", "\n".join(b))

    # failures
    b = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"pair & recoverable & $n$ & loss & target-only kept \\", r"\midrule"]
    for r in sorted(R, key=lambda r: r.get("procrustes_fail", {}).get("recoverable", 0)):
        f = r.get("procrustes_fail")
        if f:
            b.append(f"{_pair(r['src'], r['tgt'])} & {_f(f['recoverable'],3)} & {f['n_recoverable']} & "
                     f"{_f(f['loss'],3)} & {_f(f['keep_target_only'],3)} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_failures", "\n".join(b))

    if "lowrank" in D:
        from align.data import MODELS as _M
        lr = {k: v for k, v in D["lowrank"].items()
              if min(_M[k.split("->")[0]]["dim"], _M[k.split("->")[1]]["dim"]) >= 1024}
        full = {k: r[str(max(int(x) for x in r))] for k, r in lr.items()}
        ks = sorted({int(x) for r in lr.values() for x in r})
        b = [r"\begin{tabular}{rrr}", r"\toprule",
             r"$k$ & nDCG@10 & relative \\", r"\midrule"]
        for k in ks:
            v = [(r[str(k)], full[kk]) for kk, r in lr.items() if str(k) in r]
            b.append(f"{k} & {_f(np.mean([x for x,_ in v]))} & "
                     f"{_f(np.mean([x/y for x,y in v]), 3)} \\\\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_lowrank", "\n".join(b))

    if "calsize" in D:
        cs = D["calsize"]
        ns = sorted({int(x) for r in cs.values() for x in r})
        b = [r"\begin{tabular}{rr}", r"\toprule", r"$n$ & nDCG@10 \\", r"\midrule"]
        for n in ns:
            v = [r[str(n)] for r in cs.values() if str(n) in r]
            b.append(f"{n} & {_f(np.mean(v))} \\\\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_calsize", "\n".join(b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="summary.md")
    ap.add_argument("--tables", default="")
    a = ap.parse_args()
    D = load(a.results)
    summarise(D, a.out)
    if a.tables:
        emit_tables(D, a.tables)
        print("tables written to", a.tables)


