"""Run the RadGraph entity / relation extractor on a list of report texts.

Wraps :class:`radgraph.RadGraph` so the CLI and the Python API don't have
to know about CUDA placement, multi-GPU sharding, or warmup details.

Single-GPU usage::

    from radlabels.radgraph_runner import run_radgraph
    annotations = run_radgraph(["FINDINGS: Small left pleural effusion. ..."])

Multi-GPU usage (data-parallel, one process per GPU)::

    annotations = run_radgraph(texts, gpus=[0, 1, 2])

The model is loaded lazily once per process and cached, so repeated calls
in the same process do not pay the warmup cost twice.
"""
from __future__ import annotations

import functools
import os
import time
from typing import Sequence


@functools.lru_cache(maxsize=2)
def _get_model(model_type: str, cuda: int):
    """Load (and cache) a RadGraph model on the given CUDA device.

    ``cuda=-1`` means CPU. Results are cached per (model_type, cuda) pair;
    at most two models are held in memory simultaneously.
    """
    # The upstream radgraph package compiles with torch.dynamo on first call,
    # which requires triton to be installed. We don't need the dynamo speedup,
    # so suppress errors and fall back to eager execution.
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
    except Exception:
        pass
    from radgraph import RadGraph

    model = RadGraph(model_type=model_type, cuda=cuda, batch_size=1)
    try:
        _ = model(["warmup text"])
    except Exception:
        pass
    return model


def _resolve_cuda(gpu: int | None) -> int:
    """Resolve a single GPU index, falling back to CPU when CUDA is unavailable."""
    if gpu is not None and gpu >= 0:
        return gpu
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def run_radgraph(
    texts: Sequence[str],
    *,
    gpu: int | None = None,
    gpus: Sequence[int] | None = None,
    model_type: str = "modern-radgraph-xl",
    progress: bool = True,
) -> list[dict]:
    """Run RadGraph on a list of report texts.

    Parameters
    ----------
    texts
        Free-text radiology reports. Empty strings are replaced with
        ``"None"`` because the upstream model rejects empty input.
    gpu
        Single GPU index to use. If ``None``, autodetects (CUDA if
        available, else CPU).
    gpus
        Multiple GPU indices for data-parallel inference. Mutually
        exclusive with ``gpu``. When provided, spawns one worker process
        per GPU using ``torch.multiprocessing``.
    model_type
        RadGraph model type. Defaults to ``"modern-radgraph-xl"``.
    progress
        Whether to print a progress line periodically.

    Returns
    -------
    list[dict]
        One annotation per input text, in the same order. Each annotation
        is a ``{"0": {...}}`` dict ready to feed into
        :func:`radlabels.matcher.label_study`.
    """
    if gpus is not None and gpu is not None:
        raise ValueError("Pass either `gpu` or `gpus`, not both.")
    if gpus is not None and len(gpus) > 1:
        return _run_multi_gpu(list(texts), list(gpus), model_type, progress)

    cuda = _resolve_cuda(gpu if gpu is not None else (gpus[0] if gpus else None))
    model = _get_model(model_type, cuda)

    out: list[dict] = []
    t0 = time.perf_counter()
    n = len(texts)
    for i, text in enumerate(texts, 1):
        try:
            res = model([text or "None"])
            if isinstance(res, dict):
                out.append(res)
            elif isinstance(res, list) and res:
                out.append(res[0])
            else:
                out.append({})
        except Exception as e:
            out.append({"_error": str(e)})

        if progress and (i % 50 == 0 or i == n):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            print(f"  radgraph {i}/{n}  {rate:.1f} rep/s", flush=True)

    return out


# ------------------------------------------------------------------ #
#                          MULTI-GPU PATH                            #
# ------------------------------------------------------------------ #
def _worker(gpu_id: int, indexed_shard: list[tuple[int, str]],
            out_path: str, model_type: str) -> None:
    """One worker process: run RadGraph on its shard and write a JSON file."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import orjson
    import torch

    torch.set_num_threads(2)
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
    except Exception:
        pass
    from radgraph import RadGraph

    model = RadGraph(model_type=model_type, cuda=0, batch_size=1)
    try:
        _ = model(["warmup text"])
    except Exception:
        pass

    results: dict[int, dict] = {}
    for idx, text in indexed_shard:
        try:
            res = model([text or "None"])
            if isinstance(res, dict):
                results[idx] = res
            elif isinstance(res, list) and res:
                results[idx] = res[0]
            else:
                results[idx] = {}
        except Exception as e:
            results[idx] = {"_error": str(e)}

    with open(out_path, "wb") as f:
        f.write(orjson.dumps({str(k): v for k, v in results.items()}))


def _run_multi_gpu(texts: list[str], gpus: list[int], model_type: str,
                   progress: bool) -> list[dict]:
    """Data-parallel RadGraph: one process per GPU."""
    import tempfile
    import orjson
    import torch.multiprocessing as mp

    if progress:
        print(f"Running RadGraph on {len(gpus)} GPUs ({gpus}) over {len(texts)} reports",
              flush=True)

    indexed = list(enumerate(texts))
    shards: list[list[tuple[int, str]]] = [[] for _ in gpus]
    for i, item in enumerate(indexed):
        shards[i % len(gpus)].append(item)

    with tempfile.TemporaryDirectory(prefix="radlabels_radgraph_") as td:
        shard_paths = [f"{td}/shard_gpu{g}.json" for g in gpus]
        ctx = mp.get_context("spawn")
        procs = []
        t0 = time.perf_counter()
        for gpu, shard, outp in zip(gpus, shards, shard_paths):
            p = ctx.Process(target=_worker, args=(gpu, shard, outp, model_type))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        merged: dict[int, dict] = {}
        for p_ in shard_paths:
            with open(p_, "rb") as f:
                shard = orjson.loads(f.read())
            for k, v in shard.items():
                merged[int(k)] = v

    if progress:
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s ({len(texts)/max(elapsed,1e-9):.1f} rep/s)",
              flush=True)
    return [merged[i] for i in range(len(texts))]
