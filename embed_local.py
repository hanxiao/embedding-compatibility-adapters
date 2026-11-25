"""Embed NanoBEIR (13 tasks) and the 5000 calibration texts with a local model.

Mirrors the file naming of the jina API cache so align/data.py can glob either.
Writes into ./cache, or $ECA_CACHE. Needs a GPU for the 0.6B and up models.

usage: python embed_local.py qwen3_06b|bgem3|e5large
"""
import json
import os
import sys
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

INSTR = ("Instruct: Given a web search query, retrieve relevant passages that "
         "answer the query\nQuery: ")

MODELS = {
    "qwen3_06b": dict(hf="Qwen/Qwen3-Embedding-0.6B", dim=1024, query_instr=INSTR),
    "bgem3": dict(hf="BAAI/bge-m3", dim=1024, query_instr=None),
    "e5large": dict(hf="intfloat/multilingual-e5-large-instruct", dim=1024, query_instr=INSTR),
}


def embed(model, texts, instruction, batch_size=64):
    """Length-bucketed batching so padding does not dominate."""
    n = len(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((n, model.get_sentence_embedding_dimension()), dtype=np.float32)
    done = 0
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        batch = [texts[j] for j in idx]
        with torch.no_grad():
            e = model.encode(batch, prompt=instruction, normalize_embeddings=False,
                             show_progress_bar=False, convert_to_numpy=True,
                             batch_size=len(batch))
        out[idx] = e.astype(np.float32)
        done += len(idx)
        if done % 8192 < batch_size:
            print(f"    {done}/{n}", flush=True)
    return out


def main():
    key = sys.argv[1]
    cfg = MODELS[key]
    print(f"[{key}] {cfg['hf']}", flush=True)

    model = SentenceTransformer(cfg["hf"], trust_remote_code=True,
                                device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 512
    print(f"  dim={model.get_sentence_embedding_dimension()}", flush=True)

    for task in TASKS:
        tt = "text-matching" if task in SYMMETRIC else None
        q_tt = tt or "retrieval_query"
        c_tt = tt or "retrieval_passage"
        for side, tt_name, instr in (
            ("queries", q_tt, cfg["query_instr"]),
            ("corpus", c_tt, None),
        ):
            f = OUT / f"nanobeir2_{task}_{side}_{key}_{tt_name}_local.npy"
            if f.exists():
                print(f"  skip {f.name}", flush=True)
                continue
            ds = load_dataset("sentence-transformers/NanoBEIR-en", side, split=task)
            texts = ds["text"]
            print(f"  {task} {side}: {len(texts)}", flush=True)
            np.save(f, embed(model, texts, instr))
            print(f"    saved {f.name}", flush=True)

    texts = json.load(open(OUT / "texts_5000.json"))
    for side, instr in (("retquery", cfg["query_instr"]), ("retpassage", None)):
        f = OUT / f"cal5k_{side}_{key}_{'retrieval_query' if side == 'retquery' else 'retrieval_passage'}_embs.npy"
        if f.exists():
            print(f"  skip {f.name}", flush=True)
            continue
        part = texts[:2500] if side == "retquery" else texts[2500:]
        print(f"  calibration {side}: {len(part)}", flush=True)
        np.save(f, embed(model, part, instr))
        print(f"    saved {f.name}", flush=True)

    print(f"[{key}] done", flush=True)


if __name__ == "__main__":
    main()
