#!/usr/bin/env python
"""Agent-friendly predict+eval utility.

This module exposes a programmatic API `run_predict_and_eval` suitable for
integration in agent frameworks (e.g. LangGraph). It also provides a thin CLI
wrapper for ad-hoc use.
"""

# Move one level up to import src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from typing import Any, Dict, Iterable, List, Optional

import torch
from hydra.utils import instantiate

from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM
from src.metrics import get_all_metrics
from src.utils import safe_get_source


def load_lm_and_tokenizer(ckpt: str, cfg: Optional[dict] = None, device: str = "cpu"):
    """Load Lightning checkpoint and return underlying HF LM, tokenizer, and torch.device.

    This mirrors the loading logic used by run_predict_and_eval so callers (agents)
    can reuse a single implementation.
    """
    repo_root = Path(__file__).resolve().parents[1]
    # cfg may be an OmegaConf mapping or a plain dict. Normalize to a plain dict
    if cfg is None:
        # try to load config if available; fall back to empty dict
        try:
            cfg = OmegaConf.load(repo_root / "config" / "default.yaml")
        except Exception:
            cfg = {}

    if OmegaConf.is_config(cfg):
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg_dict = {}
    else:
        cfg_dict = cfg if isinstance(cfg, dict) else {}

    model_id = cfg_dict.get("model_id")

    tokenizer = create_tokenizer(model_id)

    # If no encoder config provided, it's likely the caller only needs the HF LM/tokenizer
    # for free-text generation (agent use-case). In that case, load the HF model directly
    # from `model_id` instead of instantiating the full Lightning module which requires
    # an `encoder` config with `token_dim`.
    encoder_cfg = cfg_dict.get("encoder")
    if encoder_cfg is None:
        # try to infer model_id from checkpoint if missing
        if model_id is None:
            try:
                ck = torch.load(ckpt, map_location="cpu")
                for key in ("hyper_parameters", "hparams", "hyper_params"):
                    if key in ck:
                        meta = ck[key]
                        model_id = meta.get("model_id") if isinstance(meta, dict) else None
                        if model_id:
                            break
            except Exception:
                model_id = None

        if model_id is None:
            raise RuntimeError("Unable to determine model_id for HF LM load; provide encoder config or model_id in cfg")

        try:
            from transformers import AutoModelForCausalLM

            lm = AutoModelForCausalLM.from_pretrained(model_id)
            use_cuda = device in ("cuda", "gpu") and torch.cuda.is_available()
            torch_device = torch.device("cuda" if use_cuda else "cpu")
            lm = lm.to(torch_device)
            return lm, tokenizer, torch_device
        except Exception:
            # fall through to trying the Lightning load path
            pass

    model_kwargs = {
        "encoder": encoder_cfg,
        "model_id": model_id,
        "tokenizer": tokenizer,
        "peft_config": None,
        "learning_rate": cfg_dict.get("lr"),
        "metrics_select_sources": None,
    }

    ld = MindLLM.load_from_checkpoint(ckpt, map_location="cpu", strict=False, **model_kwargs)
    ld.eval()

    use_cuda = device in ("cuda", "gpu") and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    ld = ld.to(torch_device)

    return ld.lm, tokenizer, torch_device

def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = _move_batch_to_device(v, device)
        else:
            batch[k] = v
    return batch


def _serialize_metric_value(val: Any) -> Any:
    if isinstance(val, torch.Tensor):
        try:
            return val.item()
        except Exception:
            return val.tolist()
    if isinstance(val, dict):
        return {k: _serialize_metric_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_serialize_metric_value(v) for v in val]
    return val


def evaluate_predictions(
    predictions: List[Dict[str, Any]] | str,
    select_sources: Optional[Iterable[str]] = None,
    skip_heavy: Optional[bool] = None,
    default_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a list of prediction dicts or a path to saved predictions.

    Each prediction dict should have keys: `id`, `pred`, `target`, and optional `source`.
    Returns a mapping source -> {metric_name: value}.
    """
    # load if path provided
    if isinstance(predictions, (str, Path)):
        p = Path(predictions)
        if not p.exists():
            raise FileNotFoundError(str(p))
        if p.suffix in (".pt", ".pth"):
            all_predictions = torch.load(str(p))
        else:
            all_predictions = json.loads(open(str(p), "r", encoding="utf-8").read())
    else:
        all_predictions = predictions

    # determine skip_heavy behavior: if not provided, default to False here.
    skip = False if skip_heavy is None else bool(skip_heavy)

    # If a default_source is provided, ensure the metrics factory includes it
    allowed = None
    if select_sources is not None:
        allowed = set(select_sources)
        if default_source is not None:
            allowed.add(default_source)
    else:
        if default_source is not None:
            allowed = {default_source}

    metrics = get_all_metrics(allowed, skip_heavy=skip)

    # Update metrics per-source
    for item in all_predictions:
        src = item.get("source") or default_source
        pred = item.get("pred")
        target = item.get("target")
        if src is not None and src in metrics:
            for name, metric in metrics[src].items():
                try:
                    # update expects batch lists; provide single-item lists
                    metric.update([pred], [target])
                except Exception:
                    pass

    # compute metrics
    results: Dict[str, Any] = {}
    for source_key, metric_dict in metrics.items():
        results[source_key] = {}
        for name, metric in metric_dict.items():
            try:
                val = metric.compute()
            except Exception as e:
                val = str(e)
            results[source_key][name] = _serialize_metric_value(val)

    return results


def compute_metrics_from_file(
    predictions_path: str,
    select_sources: Optional[Iterable[str]] = None,
    skip_heavy: Optional[bool] = None,
    default_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Load predictions from a path and compute metrics. Convenience wrapper."""
    return evaluate_predictions(
        predictions_path, select_sources=select_sources, skip_heavy=skip_heavy, default_source=default_source
    )


def run_predict_and_eval(
    ckpt: str,
    config_path: Optional[str] = None,
    select_sources: Optional[Iterable[str]] = None,
    device: str = "cpu",
    out: str = "outputs/predictions.pt",
    return_predictions: bool = False,
    default_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Run prediction and evaluate metrics; returns a summary dict.

    This function is safe to call from agent code. It returns structured
    results rather than printing them, making it easy to integrate into
    pipelines.
    """

    repo_root = Path(__file__).resolve().parents[1]
    config_path = config_path or (repo_root / "config" / "default.yaml")
    assert Path(config_path).exists(), f"Config not found at {config_path}"

    # Use Hydra compose so `defaults:` entries (like encoder) are resolved
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = str(repo_root / "config")
    # If caller supplied a full path to a single YAML file, still prefer hydra compose
    # `initialize_config_dir` accepts absolute paths for the config directory.
    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name="default")

    # select_sources precedence: explicit argument -> config
    if select_sources is None:
        select_sources = cfg.get("data", {}).get("select_sources", None)

    model_id = cfg.get("model_id")

    # allow omegaconf objects in pickle if available
    try:
        from omegaconf import DictConfig

        torch.serialization.add_safe_globals([DictConfig])
    except Exception:
        pass

    # Allow CLI `device` to override trainer config. If user requests GPU, set trainer to use GPU.
    if device in ("gpu", "cuda"):
        # ensure trainer config will attempt to use GPU
        if not hasattr(cfg, "trainer"):
            cfg.trainer = {}
        # override trainer accelerator/devices when requested
        try:
            cfg.trainer.accelerator = "gpu"
            # if devices not present, default to 1
            if not getattr(cfg.trainer, "devices", None):
                cfg.trainer.devices = 1
        except Exception:
            # cfg may be a plain dict
            if isinstance(cfg, dict):
                cfg.setdefault("trainer", {})
                cfg["trainer"]["accelerator"] = "gpu"
                cfg["trainer"].setdefault("devices", 1)

    # Tokenizer
    tokenizer = create_tokenizer(model_id)

    # DataModule
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else cfg.data
    if select_sources is not None:
        # make a shallow copy so we don't mutate the composed config
        if isinstance(data_cfg, dict):
            data_cfg = dict(data_cfg)
            data_cfg["select_sources"] = list(select_sources)
        else:
            data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
            data_cfg["select_sources"] = list(select_sources)

    datamodule = instantiate(data_cfg, tokenizer=tokenizer)
    datamodule.setup(stage="predict")

    # instantiate Trainer from config so Lightning handles moving model to GPU if configured
    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg, dict) else cfg.trainer
    trainer = instantiate(trainer_cfg)

    # model kwargs
    peft_config = None
    model_kwargs = {
        "encoder": cfg.get("encoder") if isinstance(cfg, dict) else cfg.encoder,
        "model_id": model_id,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
        "learning_rate": cfg.get("lr") if isinstance(cfg, dict) else cfg.lr,
        "metrics_select_sources": list(select_sources) if select_sources is not None else None,
    }

    # Load checkpoint to CPU then hand to Trainer to place on device
    model = MindLLM.load_from_checkpoint(ckpt, map_location="cpu", strict=False, **model_kwargs)
    model.eval()

    # Use Lightning Trainer to run predict - this will move the model to GPU if trainer is configured
    predictions = trainer.predict(model=model, datamodule=datamodule)

    # `predictions` is a list (per dataloader / per batch). Flatten and process
    all_predictions: List[Dict[str, Any]] = []

    # Ensure metrics registry includes default_source if provided so we can update it
    allowed_sources = None
    if select_sources is not None:
        allowed_sources = set(select_sources)
        if default_source is not None:
            allowed_sources.add(default_source)
    else:
        if default_source is not None:
            allowed_sources = {default_source}

    metrics = get_all_metrics(allowed_sources)

    # diagnostic counters
    missing_source_count = 0
    reattributed_count = 0

    # predictions may be nested: list of list or list of tuples returned from predict_step
    for entry in predictions:
        # handle nested lists
        if isinstance(entry, list):
            iter_outs = entry
        else:
            iter_outs = [entry]

        for batch_out in iter_outs:
            if batch_out is None:
                continue
            # support both (preds, targets, sample_ids) and (preds, targets, sample_ids, source)
            if isinstance(batch_out, (list, tuple)) and len(batch_out) in (3, 4):
                if len(batch_out) == 4:
                    preds, targets, sample_ids, source = batch_out
                else:
                    preds, targets, sample_ids = batch_out
                    source = None
            elif isinstance(batch_out, dict):
                preds = batch_out.get("preds")
                targets = batch_out.get("targets")
                sample_ids = batch_out.get("sample_ids")
                source = batch_out.get("source")
            else:
                # unrecognized format
                continue

            # Normalize iterables
            try:
                n_items = len(preds)
            except Exception:
                n_items = 1

            # normalize sample_ids to a list
            if isinstance(sample_ids, (list, tuple)):
                sample_id_list = list(sample_ids)
            else:
                sample_id_list = [sample_ids] * n_items

            # normalize per-sample sources: source may be a list, None, or single string
            if isinstance(source, (list, tuple)):
                sources_list = list(source)
                if len(sources_list) < n_items:
                    sources_list = sources_list + [sources_list[-1]] * (n_items - len(sources_list))
                elif len(sources_list) > n_items:
                    sources_list = sources_list[:n_items]
            elif source is None:
                sources_list = [None] * n_items
            else:
                sources_list = [source] * n_items

            # Update metrics per-sample (use default_source when needed)
            for i in range(n_items):
                sid = sample_id_list[i]
                p = preds[i]
                t = targets[i] if isinstance(targets, (list, tuple)) else targets
                orig_s = sources_list[i]
                assigned_s = orig_s if orig_s is not None else default_source

                if orig_s is None:
                    missing_source_count += 1
                    if default_source is not None:
                        reattributed_count += 1

                if assigned_s is not None and assigned_s in metrics:
                    for name, metric in metrics[assigned_s].items():
                        try:
                            metric.update([p], [t])
                        except Exception:
                            pass

                # save assigned source so the saved predictions are attributed for future runs
                all_predictions.append({"id": sid, "pred": p, "target": t, "source": assigned_s})

    # Diagnostic log
    total = len(all_predictions)
    unique_sources = sorted({x.get("source") for x in all_predictions if x.get("source") is not None})
    print(f"Predictions: {total}, missing source: {missing_source_count}, reattributed: {reattributed_count}")
    print(f"Sources seen: {unique_sources}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_predictions, str(out_path))

    # Evaluate predictions using the reusable function. Respect config toggle `metrics.skip_heavy`.
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else cfg.metrics
    skip_heavy = metrics_cfg.get("skip_heavy", True) if isinstance(metrics_cfg, dict) else metrics_cfg.skip_heavy
    results = evaluate_predictions(
        all_predictions, select_sources=select_sources, skip_heavy=skip_heavy, default_source=default_source
    )

    # save metrics JSON next to predictions unless config specifies an explicit path
    metrics_out_cfg = metrics_cfg.get("output_metrics", "") if isinstance(metrics_cfg, dict) else metrics_cfg.output_metrics
    if metrics_out_cfg:
        metrics_out = Path(metrics_out_cfg)
    else:
        metrics_out = out_path.with_suffix(".metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    open(metrics_out, "w", encoding="utf-8").write(json.dumps(results, indent=2, ensure_ascii=False))

    ret: Dict[str, Any] = {
        "predictions_path": str(out_path),
        "num_predictions": len(all_predictions),
        "metrics": results,
    }
    if return_predictions:
        ret["predictions"] = all_predictions

    return ret


def _cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="mindllm-base.ckpt", help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument(
        "--select-sources",
        default=None,
        help='JSON list of dataset sources to select, e.g. "[\"coco-caption\",\"vqa-v2\"]". If omitted, uses config default.'
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"], help="Device to run on")
    parser.add_argument("--out", default="outputs/predictions.pt", help="Path to save predictions (torch.save)")
    parser.add_argument("--config", default=None, help="Path to override config/default.yaml")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate using an existing predictions file (do not run model)")
    parser.add_argument("--predictions", default=None, help="Path to saved predictions (.pt/.pth) or JSON file when using --evaluate-only")
    parser.add_argument("--default-source", default=None, help="Default source key to attribute predictions that lack a `source` field")
    args = parser.parse_args()

    select_sources = json.loads(args.select_sources) if args.select_sources is not None else None

    if args.evaluate_only:
        assert args.predictions is not None, "--predictions PATH is required with --evaluate-only"
        p = Path(args.predictions)
        if p.suffix in (".pt", ".pth"):
            all_predictions = torch.load(str(p))
        else:
            all_predictions = json.loads(open(str(p), "r", encoding="utf-8").read())

        # If a config was provided, read metrics.skip_heavy and metrics.output_metrics
        repo_root = Path(__file__).resolve().parents[1]
        config_path = args.config or (repo_root / "config" / "default.yaml")
        cfg = OmegaConf.load(str(config_path)) if Path(config_path).exists() else {}
        metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else cfg.metrics
        skip_heavy = metrics_cfg.get("skip_heavy", True) if isinstance(metrics_cfg, dict) else metrics_cfg.skip_heavy

        results = evaluate_predictions(
            all_predictions, select_sources=select_sources, skip_heavy=skip_heavy, default_source=args.default_source
        )

        # write metrics JSON next to predictions or to configured location
        metrics_out_cfg = metrics_cfg.get("output_metrics", "") if isinstance(metrics_cfg, dict) else metrics_cfg.output_metrics
        if metrics_out_cfg:
            metrics_out = Path(metrics_out_cfg)
        else:
            metrics_out = p.with_suffix(".metrics.json")
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        open(metrics_out, "w", encoding="utf-8").write(json.dumps(results, indent=2, ensure_ascii=False))
        print("Metric results:")
        for source, metric_res in results.items():
            print(f"Source: {source}")
            for name, val in metric_res.items():
                print(f"  {name}: {val}")
        sys.exit(0)

    res = run_predict_and_eval(
        ckpt=args.ckpt,
        config_path=args.config,
        select_sources=select_sources,
        device=args.device,
        out=args.out,
        return_predictions=False,
        default_source=args.default_source,
    )

    # print formatted results
    print("Metric results:")
    for source, metric_res in res["metrics"].items():
        print(f"Source: {source}")
        for name, val in metric_res.items():
            print(f"  {name}: {val}")
    print(f"Saved {res['num_predictions']} predictions to {res['predictions_path']}")


if __name__ == "__main__":
    _cli_main()
