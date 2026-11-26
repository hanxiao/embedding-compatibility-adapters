"""Embed NanoBEIR (13 tasks) and the 5,000 AG News calibration texts with a local model.

Mirrors the file naming of the Jina API cache so align/data.py can load either.
Writes into ./cache, or $ECA_CACHE. Uses length-bucketed batching on GPU.

usage:
    python embed_local.py qwen3_06b
    python embed_local.py bgem3 e5large nomic_v15
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT = Path(os.environ.get("ECA_CACHE", "cache"))
OUT.mkdir(exist_ok=True)

TASKS = [
    "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018",
    "NanoHotpotQA", "NanoMSMARCO", "NanoNFCorpus", "NanoNQ",
    "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoArguAna", "NanoSciFact",
    "NanoTouche2020",
]
SYMMETRIC = {"NanoQuoraRetrieval", "NanoSCIDOCS", "NanoArguAna"}

INSTR_WEB = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)

MODELS = {
    "qwen3_06b": dict(
        hf="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        query_instr=INSTR_WEB,
        doc_instr=None,
        batch_size=128,
        max_len=512,
    ),
    "bgem3": dict(
        hf="BAAI/bge-m3",
        dim=1024,
        query_instr=None,
        doc_instr=None,
        batch_size=128,
        max_len=512,
    ),
    "e5large": dict(
        hf="intfloat/multilingual-e5-large-instruct",
        dim=1024,
        query_instr=INSTR_WEB,
        doc_instr=None,
        batch_size=128,
        max_len=512,
    ),
    "arctic_m": dict(
        hf="Snowflake/snowflake-arctic-embed-m-v1.5",
        dim=768,
        query_instr="Represent this sentence for searching relevant passages: ",
        doc_instr=None,
        batch_size=256,
        max_len=512,
    ),
}


def embed(model, texts, instruction, batch_size=128):
    """Length-bucketed batching so padding does not dominate GPU throughput."""
    n = len(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((n, model.get_sentence_embedding_dimension()), dtype=np.float32)
    done = 0
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        batch = [texts[j] for j in idx]
        with torch.no_grad():
            e = model.encode(
                batch,
                prompt=instruction,
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=len(batch),
            )
        out[idx] = e.astype(np.float32)
        done += len(idx)
        if done % 8192 < batch_size:
            print(f"    {done}/{n}", flush=True)
    return out


def embed_model(key):
    t0 = time.time()
    cfg = MODELS[key]
    print(f"\n[{key}] {cfg['hf']}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
    model = SentenceTransformer(cfg["hf"], trust_remote_code=True, device=device,
                                model_kwargs=kwargs)
    model.max_seq_length = cfg["max_len"]
    bs = cfg["batch_size"]
    print(f"  device={device} dim={model.get_sentence_embedding_dimension()} bs={bs}", flush=True)

    for task in TASKS:
        tt = "text-matching" if task in SYMMETRIC else None
        q_tt = tt or "retrieval_query"
        c_tt = tt or "retrieval_passage"
        for side, tt_name, instr in (
            ("queries", q_tt, cfg["query_instr"]),
            ("corpus", c_tt, cfg["doc_instr"]),
        ):
            f = OUT / f"nanobeir2_{task}_{side}_{key}_{tt_name}_local.npy"
            if f.exists():
                print(f"  skip {f.name}", flush=True)
                continue
            ds = load_dataset("sentence-transformers/NanoBEIR-en", side, split=task)
            texts = ds["text"]
            print(f"  {task} {side}: {len(texts)}", flush=True)
            np.save(f, embed(model, texts, instr, batch_size=bs))

    texts = json.load(open(OUT / "texts_5000.json"))
    for side, instr in (
        ("retquery", cfg["query_instr"]),
        ("retpassage", cfg["doc_instr"]),
    ):
        role = "retrieval_query" if side == "retquery" else "retrieval_passage"
        f = OUT / f"cal5k_{side}_{key}_{role}_embs.npy"
        if f.exists():
            print(f"  skip {f.name}", flush=True)
            continue
        part = texts[:2500] if side == "retquery" else texts[2500:]
        print(f"  calibration {side}: {len(part)}", flush=True)
        np.save(f, embed(model, part, instr, batch_size=bs))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{key}] done in {time.time() - t0:.1f}s", flush=True)


def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS)
    for key in keys:
        embed_model(key)


if __name__ == "__main__":
    main()
