"""Turn paper/results.json into summary.md and all LaTeX tables in paper/tabs/."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from align.data import MODELS

MODEL_ORDER = ["v5small", "v5nano", "qwen3_06b", "arctic_m", "v4", "v3", "bgem3", "e5large"]
SHORT = {
    "v5small": "v5s",
    "v5nano": "v5n",
    "qwen3_06b": "q3",
    "arctic_m": "arc",
    "v4": "v4",
    "v3": "v3",
    "bgem3": "bge",
    "e5large": "e5",
}
FAMILIES = [
    "procrustes",
    "affine-cs",
    "src-centred",
    "affine-ls",
    "affine",
    "ridge",
    "whitened-proc",
    "ridgecca",
    "sgd-linear-infonce",
    "sgd-resmlp-infonce",
    "krr-rbf",
]
CONTROLS = ["identity", "shuffled", "random"]

METHOD_LABELS = {
    "procrustes": "Orthogonal Procrustes (ours)",
    "affine-cs": "Affine, centre + scale (Plato's Key, fixed)",
    "src-centred": "Source-centred Procrustes (ours, fixed)",
    "affine-ls": "Affine, centre + LS scale (fixed)",
    "affine": "Affine, val-selected (Plato's Key)",
    "ridge": "Ridge regression, val-selected",
    "whitened-proc": "Whitened Procrustes, val-selected",
    "ridgecca": "Ridge CCA (two-sided), val-selected",
    "sgd-linear-infonce": "SGD linear, InfoNCE",
    "sgd-resmlp-infonce": "SGD 3-layer residual MLP, InfoNCE",
    "krr-rbf": "Kernel ridge, RBF",
    "identity": "Identity, pad or truncate",
    "shuffled": "Procrustes on shuffled pairs",
    "random": "Random semi-orthogonal map",
}


def load(path):
    return json.loads(Path(path).read_text())


def native_means(D):
    return {k: v["test_mean"] for k, v in D["native"].items()}


def rows(D):
    nat = native_means(D)
    out = []
    for key, p in D["pairs"].items():
        s, t = key.split("->")
        ds, dt = MODELS[s]["dim"], MODELS[t]["dim"]
        r = dict(
            key=key,
            src=s,
            tgt=t,
            source=nat[s],
            target=nat[t],
            d_s=ds,
            d_t=dt,
            dim_dir="expand" if ds < dt else ("same" if ds == dt else "compress"),
            upgrade=nat[s] < nat[t],
        )
        for f in FAMILIES + CONTROLS:
            if f in p:
                r[f] = p[f]["test_mean"]
        r["features"] = p.get("features", {})
        for f in ("procrustes", "affine", "affine-cs", "src-centred", "ridge"):
            if f in p and "failures" in p[f]:
                r[f + "_fail"] = p[f]["failures"]
        out.append(r)
    return out


def _f(x, n=4):
    return f"{x:.{n}f}"


def _cor(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _wilcoxon_counts(D, R, method, ref="procrustes"):
    w = l = 0
    for r in R:
        if method not in r or ref not in r:
            continue
        pt = D["pairs"][r["key"]]
        tasks = list(pt[method]["test"])
        a = np.array([pt[method]["test"][t] for t in tasks])
        c = np.array([pt[ref]["test"][t] for t in tasks])
        if np.allclose(a, c):
            continue
        try:
            pval = wilcoxon(a, c).pvalue
        except ValueError:
            continue
        if pval < 0.05:
            if a.mean() > c.mean():
                w += 1
            else:
                l += 1
    return w, l


def summarise(D, out_path):
    nat = native_means(D)
    aniso = D.get("anisotropy", {})
    R = rows(D)
    L = []
    add = L.append

    add(f"# Results summary ({len(nat)} models, {len(R)} directed pairs)\n")

    add("## Native nDCG@10 (13 NanoBEIR tasks, 649 queries / 429 test)\n")
    add("| model | family | backbone | dim | kappa | all 649 | test 429 |")
    add("|---|---|---|---:|---:|---:|---:|")
    for k in sorted(nat, key=lambda x: -nat[x]):
        m = MODELS[k]
        add(
            f"| {m['label']} | {m['family']} | {m['backbone']} | {m['dim']} | "
            f"{aniso.get(k, float('nan')):.2f} | {D['native'][k]['all_mean']:.4f} | {nat[k]:.4f} |"
        )
    add("")

    add("## Adapter families (mean over all directed pairs, Wilcoxon vs Procrustes)\n")
    add("| method | nDCG@10 | /source | /target | raw W/L | sig W/L |")
    add("|---|---:|---:|---:|---:|---:|")
    for f in FAMILIES + CONTROLS:
        v = [r[f] for r in R if f in r]
        if not v:
            continue
        rs = [r[f] / r["source"] for r in R if f in r]
        rt = [r[f] / r["target"] for r in R if f in r]
        rw = sum(1 for r in R if f in r and r[f] > r["procrustes"] + 1e-9)
        rl = sum(1 for r in R if f in r and r[f] < r["procrustes"] - 1e-9)
        sw, sl = _wilcoxon_counts(D, R, f, "procrustes")
        add(
            f"| {METHOD_LABELS.get(f, f)} | {np.mean(v):.4f} | {np.mean(rs):.3f} | "
            f"{np.mean(rt):.3f} | {rw}/{rl} | {sw}/{sl} |"
        )
    add("")

    if "multiseed" in D:
        ms = D["multiseed"]
        add("## Multi-seed stability (5 random val/test splits)\n")
        add("| method | mean nDCG@10 | std | mean /source |")
        add("|---|---:|---:|---:|")
        for f in next(iter(ms.values())).keys():
            vals = [ms[s][f]["ndcg"] for s in ms]
            rats = [ms[s][f]["ratio_src"] for s in ms]
            add(f"| {f} | {np.mean(vals):.4f} | {np.std(vals):.4f} | {np.mean(rats):.3f} |")
        add("")

    add("## True dimensional direction (d_s vs d_t)\n")
    buckets = {
        "expanding (d_s < d_t)": [r for r in R if r["dim_dir"] == "expand"],
        "same-dim (d_s = d_t)": [r for r in R if r["dim_dir"] == "same"],
        "compressing (d_s > d_t)": [r for r in R if r["dim_dir"] == "compress"],
    }
    add("| method | " + " | ".join(f"{k} (n={len(v)})" for k, v in buckets.items()) + " |")
    add("|---|---:|---:|---:|")
    for f in ("procrustes", "affine-cs", "affine", "ridge", "whitened-proc", "ridgecca"):
        if any(f in r for r in R):
            vals = [f"{np.mean([r[f] for r in grp]):.4f}" for grp in buckets.values()]
            add(f"| {f} | " + " | ".join(vals) + " |")
    add("")

    add("## Quality direction (upgrade nat[s]<nat[t] vs downgrade nat[s]>nat[t])\n")
    up = [r for r in R if r["upgrade"]]
    dn = [r for r in R if not r["upgrade"]]
    add(f"| method | upgrade (n={len(up)}) | downgrade (n={len(dn)}) | diff |")
    add("|---|---:|---:|---:|")
    for f in ("procrustes", "affine-cs", "affine", "ridge", "ridgecca"):
        u = np.mean([r[f] for r in up])
        d = np.mean([r[f] for r in dn])
        add(f"| {f} | {u:.4f} | {d:.4f} | {u - d:+.4f} |")
    add("")

    if "deployment" in D:
        dep = D["deployment"]
        add("## Deployment: query-side vs corpus-side & hybrid progressive migration\n")
        for m in ("procrustes", "affine-cs", "ridge"):
            qs = np.mean([v[m]["query_side"] for v in dep.values()])
            cs = np.mean([v[m]["corpus_side"] for v in dep.values()])
            add(f"- **{m}** mean over {len(dep)} pairs: query-side `{qs:.4f}` vs corpus-side `{cs:.4f}` (delta `{cs - qs:+.4f}`)")
        for mode in ("uncalibrated", "zscore_top100"):
            fracs = ["0.0", "0.25", "0.5", "0.75", "1.0"]
            means = [np.mean([v["hybrid_affine"][mode][fr] for v in dep.values()]) for fr in fracs]
            add(f"- **hybrid affine ({mode})** mean: " + ", ".join(f"{int(float(fr)*100)}%={val:.4f}" for fr, val in zip(fracs, means)))
        add("\n### Canonical examples: jina-v3 <-> jina-v5-nano\n")
        for key in ("idx=v3|qry=v5nano", "idx=v5nano|qry=v3", "idx=v3|qry=v5small", "idx=v5small|qry=v3"):
            if key in dep:
                v = dep[key]
                add(f"- `{key}`: query-side aff={v['affine-cs']['query_side']:.4f} proc={v['procrustes']['query_side']:.4f} | "
                    f"corpus-side aff={v['affine-cs']['corpus_side']:.4f} proc={v['procrustes']['corpus_side']:.4f}")
                add(f"  - hybrid uncalibrated : " + " ".join(f"{int(float(fr)*100)}%={val:.4f}" for fr, val in v["hybrid_affine"]["uncalibrated"].items()))
                add(f"  - hybrid zscore_top100: " + " ".join(f"{int(float(fr)*100)}%={val:.4f}" for fr, val in v["hybrid_affine"]["zscore_top100"].items()))
        add("")

    if "role_cal" in D:
        rc = D["role_cal"]
        add("## Role-matched calibration ablation\n")
        add("| calibration set | query-side proc | query-side aff | corpus-side proc | corpus-side aff |")
        add("|---|---:|---:|---:|---:|")
        for role in ("mixed_5000", "mixed_2500", "query_2500", "passage_2500"):
            qp = np.mean([r[role]["q_proc"] for r in rc.values()])
            qa = np.mean([r[role]["q_aff"] for r in rc.values()])
            cp = np.mean([r[role]["c_proc"] for r in rc.values()])
            ca = np.mean([r[role]["c_aff"] for r in rc.values()])
            add(f"| {role} | {qp:.4f} | {qa:.4f} | {cp:.4f} | {ca:.4f} |")
        add("")

    if "chain" in D:
        add("## Chaining (all A -> B -> C triples)\n")
        for label, key in (("Procrustes", "chain"), ("Affine (Plato's Key)", "chain_affine")):
            if key not in D:
                continue
            ch = D[key]
            rel = [c["chained"] / c["direct"] for c in ch if c["direct"] > 0]
            add(
                f"- **{label}** ({len(ch)} triples): median `{np.median(rel):.4f}`, "
                f"p10 `{np.percentile(rel, 10):.4f}`, min `{np.min(rel):.4f}`, "
                f">=0.95: `{100 * np.mean(np.array(rel) >= 0.95):.1f}%`"
            )
        add("")

    if "hub" in D:
        add("## Shared Platonic hub (GPA, 40 iters, full 5k)\n")
        for label, hkey, refk in (("Uncentred GPA vs Procrustes", "hub", "procrustes"),
                                  ("Affine GPA vs Affine-CS", "hub_affine", "affine-cs"),
                                  ("Source-centred GPA vs Source-centred", "hub_src_centred", "src-centred")):
            if hkey not in D:
                continue
            h = D[hkey]
            pw = {k: r[refk]["test_mean"] for k, r in D["pairs"].items()}
            hv = np.mean([h[k]["test_mean"] for k in h])
            pv = np.mean([pw[k] for k in h])
            wins = sum(1 for k in h if h[k]["test_mean"] > pw[k])
            add(f"- **{label}**: hub `{hv:.4f}` vs pairwise `{pv:.4f}` (delta `{hv - pv:+.4f}`, wins `{wins}/{len(h)}`)")
        add("")

    if "lowrank_affine" in D:
        lr = D["lowrank_affine"]
        add("## One-sided low-rank Affine (zero corpus re-indexing)\n")
        add("| k | mean nDCG@10 | /full affine-cs | pairs |")
        add("|---:|---:|---:|---:|")
        for k in (16, 32, 64, 128, 256, 384, 512, 768):
            vals, rats = [], []
            for pk, row in lr.items():
                if str(k) in row:
                    vals.append(row[str(k)])
                    rats.append(row[str(k)] / D["pairs"][pk]["affine-cs"]["test_mean"])
            if vals:
                add(f"| {k} | {np.mean(vals):.4f} | {np.mean(rats):.3f} | {len(vals)} |")
        add("")

    if "calsize" in D:
        add("## Calibration size\n")
        add("| n | Procrustes | Affine (Plato's Key) | Source-centred | Ridge |")
        add("|---:|---:|---:|---:|---:|")
        for n in (50, 100, 250, 500, 1000, 2500, 5000):
            p = np.mean([r[str(n)] for r in D["calsize"].values() if str(n) in r])
            a = np.mean([r[str(n)] for r in D.get("calsize_affine", {}).values() if str(n) in r])
            sc = np.mean([r[str(n)] for r in D.get("calsize_src_centred", {}).values() if str(n) in r])
            ri = np.mean([r[str(n)] for r in D.get("calsize_ridge", {}).values() if str(n) in r])
            add(f"| {n} | {p:.4f} | {a:.4f} | {sc:.4f} | {ri:.4f} |")
        add("")

    # Predictors
    rec = {(r["src"], r["tgt"]): r for r in R}
    und = {}
    for (a, b), r in rec.items():
        if (b, a) in rec:
            r1 = r["procrustes"] / r["source"]
            r2 = rec[(b, a)]["procrustes"] / rec[(b, a)]["source"]
            und[(a, b)] = dict(sym=(r1 + r2) / 2, anti=(r1 - r2) / 2, feats=r["features"])
    y_proc = np.array([r["procrustes"] / r["source"] for r in R])
    y_id = np.array([r["identity"] for r in R])
    y_gain_id = np.array([r["affine"] - r["identity"] for r in R])
    y_aff_gain = np.array([r["affine"] - r["procrustes"] for r in R])
    anti = np.array([u["anti"] for u in und.values()])
    add("## Predictors\n")
    add("| feature | r(identity) | r(affine - id) | r(affine - proc) | r(proc/src) | r(antisym) |")
    add("|---|---:|---:|---:|---:|---:|")
    for f in ("rho", "kappa_source", "kappa_target", "cka", "nagree", "eff_rank", "dim_ratio", "resid_gap"):
        x = np.array([r["features"].get(f, np.nan) for r in R])
        xs = np.array([u["feats"].get(f, np.nan) for u in und.values()])
        add(
            f"| {f} | {_cor(x, y_id):+.3f} | {_cor(x, y_gain_id):+.3f} | "
            f"{_cor(x, y_aff_gain):+.3f} | {_cor(x, y_proc):+.3f} | {_cor(xs, anti):+.3f} |"
        )
    add(f"\nantisymmetric share of variance in procrustes/source: `{np.var(anti)/np.var(y_proc):.3f}`")
    y_aff = np.array([r["affine"] / r["source"] for r in R])
    anti_a = np.array([
        0.5 * (rec[(a, b)]["affine"] / rec[(a, b)]["source"] - rec[(b, a)]["affine"] / rec[(b, a)]["source"])
        for (a, b) in und
    ])
    add(f"antisymmetric share of variance in affine/source: `{np.var(anti_a)/np.var(y_aff):.3f}`\n")

    # Failure / dark-memory recovery
    add("## Recoverable failures (dark memories)\n")
    for m in ("procrustes", "src-centred", "affine", "affine-cs"):
        recs = [r[m + "_fail"]["recoverable"] for r in R if m + "_fail" in r]
        losses = [r[m + "_fail"]["loss"] for r in R if m + "_fail" in r]
        if recs:
            add(f"- **{m}**: mean recoverable `{np.mean(recs):.3f}` (range `{np.min(recs):.3f}` to `{np.max(recs):.3f}`), mean target-hit loss `{np.mean(losses):.3f}`")
    add("")

    if "latency" in D:
        add("## Wall-clock latency\n")
        for k, v in D["latency"].items():
            add(f"- `{k}`: `{v}`")

    Path(out_path).write_text("\n".join(L))
    print("\n".join(L[:100]))


def emit_tables(D, out_dir):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    nat = native_means(D)
    aniso = D.get("anisotropy", {})
    R = rows(D)
    T = lambda name, body: (d / f"{name}.tex").write_text(body)

    # 1. Native table
    b = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"model & family & backbone & dim & $\kappa$ & all 649 & test 429 \\",
        r"\midrule",
    ]
    for k in sorted(nat, key=lambda x: -nat[x]):
        m = MODELS[k]
        b.append(
            f"{m['label']} & {m['family']} & {m['backbone']} & {m['dim']} & "
            f"{aniso.get(k, 0.0):.2f} & {_f(D['native'][k]['all_mean'])} & {_f(nat[k])} \\\\"
        )
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_native", "\n".join(b))

    # 2. 8x8 Matrix (Affine val-selected, diagonal = native)
    models = [m for m in MODEL_ORDER if m in nat]
    for mkey, fname in (("affine", "tab_matrix_affine"), ("procrustes", "tab_matrix")):
        b = [
            r"\begin{tabular}{l" + "r" * len(models) + "}",
            r"\toprule",
            r"source $\backslash$ target & " + " & ".join(SHORT[m] for m in models) + r" \\",
            r"\midrule",
        ]
        for s in models:
            cells = []
            for t in models:
                if s == t:
                    cells.append(f"\\textit{{{_f(nat[s], 3)}}}")
                else:
                    p = D["pairs"].get(f"{s}->{t}")
                    cells.append(_f(p[mkey]["test_mean"], 3) if p and mkey in p else "--")
            b.append(f"{SHORT[s]} & " + " & ".join(cells) + r" \\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T(fname, "\n".join(b))

    # 3. Families table
    ms = D.get("multiseed", {})
    b = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"method & nDCG@10 & /source & /target & wins & losses \\",
        r"\midrule",
    ]
    for f in FAMILIES + CONTROLS:
        v = [r[f] for r in R if f in r]
        if not v:
            continue
        sw, sl = _wilcoxon_counts(D, R, f, "procrustes")
        rs = np.mean([r[f] / r["source"] for r in R if f in r])
        rt = np.mean([r[f] / r["target"] for r in R if f in r])
        w_str = "--" if f == "procrustes" else str(sw)
        l_str = "--" if f == "procrustes" else str(sl)
        b.append(
            f"{METHOD_LABELS.get(f, f)} & {_f(np.mean(v))} & {_f(rs, 3)} & {_f(rt, 3)} & {w_str} & {l_str} \\\\"
        )
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_families", "\n".join(b))

    # 4. True dimensional direction table
    exp = [r for r in R if r["dim_dir"] == "expand"]
    sam = [r for r in R if r["dim_dir"] == "same"]
    cmp_ = [r for r in R if r["dim_dir"] == "compress"]
    b = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        f"method & $d_s < d_t$ ({len(exp)}) & $d_s = d_t$ ({len(sam)}) & $d_s > d_t$ ({len(cmp_)}) \\\\",
        r"\midrule",
    ]
    dir_labels = {
        "procrustes": "Orthogonal Procrustes",
        "src-centred": "Source-centred Procrustes",
        "affine-cs": "Affine (fixed, Plato's Key)",
        "affine": "Affine (val-selected)",
        "ridge": "Ridge regression",
        "whitened-proc": "Whitened Procrustes",
        "ridgecca": "Ridge CCA (two-sided)",
    }
    for f in ("procrustes", "src-centred", "affine-cs", "affine", "ridge", "whitened-proc", "ridgecca"):
        b.append(
            f"{dir_labels[f]} & "
            f"{_f(np.mean([r[f] for r in exp]))} & "
            f"{_f(np.mean([r[f] for r in sam]))} & "
            f"{_f(np.mean([r[f] for r in cmp_]))} \\\\"
        )
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_direction", "\n".join(b))

    # 5. Predictors table
    rec = {(r["src"], r["tgt"]): r for r in R}
    und = {}
    for (a, b_), r in rec.items():
        if (b_, a) in rec:
            r1 = r["procrustes"] / r["source"]
            r2 = rec[(b_, a)]["procrustes"] / rec[(b_, a)]["source"]
            und[(a, b_)] = dict(anti=(r1 - r2) / 2, feats=r["features"])
    y_id = np.array([r["identity"] for r in R])
    y_gain_id = np.array([r["affine"] - r["identity"] for r in R])
    y_aff_gain = np.array([r["affine"] - r["procrustes"] for r in R])
    anti = np.array([u["anti"] for u in und.values()])
    b = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"feature & $r$(identity) & $r$(map gain) & $r$(affine $-$ rot) & $r$(antisym) \\",
        r"\midrule",
    ]
    feat_names = {
        "rho": r"frame correlation $\rho$",
        "kappa_source": r"source anisotropy $\kappa_s$",
        "kappa_target": r"target anisotropy $\kappa_t$",
        "cka": "linear CKA",
        "nagree": "10-NN agreement",
        "eff_rank": "effective rank",
        "dim_ratio": r"dim ratio $d_s/d_t$",
        "resid_gap": "fit residual gap",
    }
    for f, lbl in feat_names.items():
        x = np.array([r["features"].get(f, np.nan) for r in R])
        xs = np.array([u["feats"].get(f, np.nan) for u in und.values()])
        b.append(
            f"{lbl} & {_cor(x, y_id):+.3f} & {_cor(x, y_gain_id):+.3f} & "
            f"{_cor(x, y_aff_gain):+.3f} & {_cor(xs, anti):+.3f} \\\\"
        )
    b += [r"\bottomrule", r"\end{tabular}"]
    T("tab_predictors", "\n".join(b))

    # 6. Deployment & Hybrid migration table
    if "deployment" in D:
        dep = D["deployment"]
        b = [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"scenario & 0\% & 25\% & 50\% & 75\% & 100\% \\",
            r"\midrule",
        ]
        fracs = ["0.0", "0.25", "0.5", "0.75", "1.0"]
        for lbl, key, mode in (
            (r"v3 index $\to$ v5-nano (raw cosine)", "idx=v3|qry=v5nano", "uncalibrated"),
            (r"v3 index $\to$ v5-nano (top-100 z-score)", "idx=v3|qry=v5nano", "zscore_top100"),
            (r"v5-nano index $\to$ v3 (raw cosine)", "idx=v5nano|qry=v3", "uncalibrated"),
            (r"v5-nano index $\to$ v3 (top-100 z-score)", "idx=v5nano|qry=v3", "zscore_top100"),
        ):
            if key in dep:
                vals = [dep[key]["hybrid_affine"][mode][fr] for fr in fracs]
                b.append(lbl + " & " + " & ".join(_f(v) for v in vals) + r" \\")
        b.append(r"\midrule")
        for lbl, mode in (
            (f"Mean over {len(dep)} pairs (raw cosine)", "uncalibrated"),
            (f"Mean over {len(dep)} pairs (top-100 z-score)", "zscore_top100"),
        ):
            vals = [np.mean([v["hybrid_affine"][mode][fr] for v in dep.values()]) for fr in fracs]
            b.append(lbl + " & " + " & ".join(_f(v) for v in vals) + r" \\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_hybrid", "\n".join(b))

    # 7. Role calibration table
    if "role_cal" in D:
        rc = D["role_cal"]
        b = [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"calibration set & query (rot) & query (aff) & corpus (rot) & corpus (aff) \\",
            r"\midrule",
        ]
        for role, lbl in (
            ("mixed_5000", "Mixed (2,500 query + 2,500 passage)"),
            ("mixed_2500", "Mixed (1,250 query + 1,250 passage)"),
            ("query_2500", "Query-role only (2,500)"),
            ("passage_2500", "Passage-role only (2,500)"),
        ):
            qp = np.mean([r[role]["q_proc"] for r in rc.values()])
            qa = np.mean([r[role]["q_aff"] for r in rc.values()])
            cp = np.mean([r[role]["c_proc"] for r in rc.values()])
            ca = np.mean([r[role]["c_aff"] for r in rc.values()])
            b.append(f"{lbl} & {_f(qp)} & {_f(qa)} & {_f(cp)} & {_f(ca)} \\\\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_role_cal", "\n".join(b))

    # 8. Low-rank one-sided Affine table
    if "lowrank_affine" in D:
        lr = D["lowrank_affine"]
        b = [r"\begin{tabular}{rrrr}", r"\toprule", r"rank $k$ & nDCG@10 & relative to full & pairs \\", r"\midrule"]
        for k in (16, 32, 64, 128, 256, 384, 512, 768):
            vals, rats = [], []
            for pk, row in lr.items():
                if str(k) in row:
                    vals.append(row[str(k)])
                    rats.append(row[str(k)] / D["pairs"][pk]["affine-cs"]["test_mean"])
            if vals:
                b.append(f"{k} & {_f(np.mean(vals))} & {_f(np.mean(rats), 3)} & {len(vals)} \\\\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_lowrank", "\n".join(b))

    # 9. Calibration size table
    if "calsize" in D:
        b = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$n$ & Procrustes & Affine (fixed) & Source-centred & Ridge \\", r"\midrule"]
        for n in (50, 100, 250, 500, 1000, 2500, 5000):
            p = np.mean([r[str(n)] for r in D["calsize"].values() if str(n) in r])
            a = np.mean([r[str(n)] for r in D.get("calsize_affine", {}).values() if str(n) in r])
            sc = np.mean([r[str(n)] for r in D.get("calsize_src_centred", {}).values() if str(n) in r])
            ri = np.mean([r[str(n)] for r in D.get("calsize_ridge", {}).values() if str(n) in r])
            b.append(f"{n:,} & {_f(p)} & {_f(a)} & {_f(sc)} & {_f(ri)} \\\\")
        b += [r"\bottomrule", r"\end{tabular}"]
        T("tab_calsize", "\n".join(b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="paper/results.json")
    ap.add_argument("--out", default="paper/summary.md")
    ap.add_argument("--tables", default="paper/tabs")
    a = ap.parse_args()
    D = load(a.results)
    summarise(D, a.out)
    if a.tables:
        emit_tables(D, a.tables)
        print("tables written to", a.tables)
