#!/usr/bin/env python
"""Simple agent runner for MindLLM's LM component.

Protocol:
- The agent prompt should ask the model to output either an action JSON or a final_answer JSON.
  Example action JSON: {"action": "get_dataset", "args": {"subfolder": "roi"}}
  Example final answer JSON: {"final_answer": "The result is ..."}

- agent_runner will parse JSON from the model output. If it finds an action, it will call a local tool
  (function) with the provided args and append the tool result as an observation. The model is then
  invoked again with the updated conversation. Loop until a final_answer is produced or max steps reached.

This file is intentionally small and synchronous to be easy to call from agent frameworks.
"""

from __future__ import annotations

# Move one level up to import src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from typing import Any, Callable, Dict, List, Optional

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM


def _load_model_and_tokenizer(ckpt: str, cfg: Optional[dict] = None, device: str = "cpu"):
    # reuse shared loader from scripts.predict_and_eval to avoid duplicate code
    try:
        from scripts.predict_and_eval import load_lm_and_tokenizer

        return load_lm_and_tokenizer(ckpt=ckpt, cfg=cfg, device=device)
    except Exception:
        # fallback to local logic if import fails
        repo_root = Path(__file__).resolve().parents[1]
        cfg = cfg or OmegaConf.load(repo_root / "config" / "default.yaml")
        # Convert OmegaConf structures to plain dicts when possible so key access is safe
        try:
            cfg_dict = cfg if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg_dict = cfg

        model_id = cfg_dict.get("model_id") if isinstance(cfg_dict, dict) else getattr(cfg, "model_id", None)
        tokenizer = create_tokenizer(model_id)
        model_kwargs = {
            "encoder": cfg_dict.get("encoder") if isinstance(cfg_dict, dict) else getattr(cfg, "encoder", None),
            "model_id": model_id,
            "tokenizer": tokenizer,
            "peft_config": None,
            "learning_rate": cfg_dict.get("lr") if isinstance(cfg_dict, dict) else getattr(cfg, "lr", None),
            "metrics_select_sources": None,
        }
        ld = MindLLM.load_from_checkpoint(ckpt, map_location="cpu", strict=False, **model_kwargs)
        ld.eval()
        use_cuda = device in ("cuda", "gpu") and torch.cuda.is_available()
        torch_device = torch.device("cuda" if use_cuda else "cpu")
        ld = ld.to(torch_device)
        return ld.lm, tokenizer, torch_device


def _move_value_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(v, device) for v in value]
    return value


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: _move_value_to_device(v, device) for k, v in batch.items()}


def _tensor_to_python(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _select_entry(container, idx: int):
    if container is None:
        return None
    if isinstance(container, (list, tuple)):
        if len(container) == 0:
            return None
        value = container[idx] if idx < len(container) else container[-1]
    else:
        value = container
    if torch.is_tensor(value):
        return _tensor_to_python(value)
    return value


def _stream_predictions_from_model(
    ckpt: str,
    cfg_path: Path,
    select_sources: Optional[List[str]],
    device_choice: Optional[str],
    default_source: Optional[str],
):
    """Yield per-sample prediction dicts by running the MindLLM predict pipeline inline."""

    config_dir = str(cfg_path.parent)
    config_name = cfg_path.stem

    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name=config_name)

    if select_sources is None:
        data_cfg_obj = cfg.get("data", {}) if isinstance(cfg, dict) else cfg.data
        select_sources = data_cfg_obj.get("select_sources") if isinstance(data_cfg_obj, dict) else data_cfg_obj.select_sources

    model_id = cfg.get("model_id") if isinstance(cfg, dict) else cfg.model_id
    tokenizer = create_tokenizer(model_id)

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else cfg.data
    if isinstance(data_cfg, dict):
        data_cfg = dict(data_cfg)
    else:
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    if select_sources is not None:
        data_cfg["select_sources"] = list(select_sources)

    datamodule = instantiate(data_cfg, tokenizer=tokenizer)
    datamodule.setup(stage="predict")
    predict_loader = datamodule.predict_dataloader()
    if predict_loader is None:
        return

    peft_config = None
    model_kwargs = {
        "encoder": cfg.get("encoder") if isinstance(cfg, dict) else cfg.encoder,
        "model_id": model_id,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
        "learning_rate": cfg.get("lr") if isinstance(cfg, dict) else cfg.lr,
        "metrics_select_sources": list(select_sources) if select_sources is not None else None,
    }

    model = MindLLM.load_from_checkpoint(ckpt, map_location="cpu", strict=False, **model_kwargs)
    model.eval()

    use_cuda = (device_choice in ("cuda", "gpu")) and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(torch_device)

    for batch_idx, batch in enumerate(predict_loader):
        batch = _move_batch_to_device(batch, torch_device)
        with torch.no_grad():
            preds, targets, sample_ids, per_sample_sources = model.predict_step(batch, batch_idx)

        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        n_items = len(preds)

        if isinstance(sample_ids, (list, tuple)):
            sample_id_list = list(sample_ids)
        else:
            sample_id_list = [sample_ids] * n_items

        if isinstance(per_sample_sources, (list, tuple)):
            sources_list = list(per_sample_sources)
            if len(sources_list) < n_items and len(sources_list) > 0:
                sources_list = sources_list + [sources_list[-1]] * (n_items - len(sources_list))
            elif len(sources_list) > n_items:
                sources_list = sources_list[:n_items]
        else:
            sources_list = [per_sample_sources] * n_items

        for i in range(n_items):
            sample_id = _tensor_to_python(sample_id_list[i])
            pred_val = _tensor_to_python(preds[i])
            target_val = _select_entry(targets, i)
            assigned_source = sources_list[i] if sources_list and i < len(sources_list) else None
            if assigned_source is None:
                assigned_source = default_source

            if isinstance(batch, dict):
                metadata = {
                    "subject": _select_entry(batch.get("subject"), i),
                    "question_id": _select_entry(batch.get("question_id"), i),
                    "sample_or_coco_id": sample_id,
                }
            else:
                metadata = {"sample_or_coco_id": sample_id}

            sample_entry = {
                "id": sample_id,
                "pred": pred_val,
                "target": target_val,
                "source": assigned_source,
            }
            sample_entry["metadata"] = {k: v for k, v in metadata.items() if v is not None}
            yield sample_entry


def default_tools() -> Dict[str, Callable[..., Any]]:
    """Return a small toolset. Extend this dict for your agent.
    - echo: returns the args as a string (useful for debugging)
    """

    def t_echo(**kwargs):
        return {"echo": kwargs}

    return {"echo": t_echo}


def build_tools_from_cfg(agent_cfg: dict, cfg_root: Optional[dict] = None) -> Dict[str, Callable[..., Any]]:
    """Construct tools based on the `agent.tools` config section.

    agent_cfg is expected to be a mapping-like object (OmegaConf or dict) containing `tools`.
    """
    tools: Dict[str, Callable[..., Any]] = {}

    tools_cfg = agent_cfg.get("tools") if isinstance(agent_cfg, dict) else agent_cfg.tools

    # echo
    echo_cfg = tools_cfg.get("echo") if isinstance(tools_cfg, dict) else tools_cfg.get("echo")
    if echo_cfg and echo_cfg.get("enabled", False):
        def t_echo(**kwargs):
            return {"echo": kwargs}

        tools["echo"] = t_echo

    # predict_and_eval: wire into existing script helper if enabled
    pa_cfg = tools_cfg.get("predict_and_eval") if isinstance(tools_cfg, dict) else tools_cfg.get("predict_and_eval")
    if pa_cfg and pa_cfg.get("enabled", False):
        try:
            from scripts.predict_and_eval import run_predict_and_eval

            def t_predict_and_eval(**kwargs):
                # allow overriding defaults from config with kwargs
                local_ckpt = kwargs.get("ckpt") or (cfg_root.get("checkpoint") if isinstance(cfg_root, dict) else cfg_root.checkpoint)
                out = kwargs.get("out") or pa_cfg.get("out")
                return_predictions = kwargs.get("return_predictions") if "return_predictions" in kwargs else pa_cfg.get("return_predictions", False)
                return run_predict_and_eval(ckpt=local_ckpt, config_path=None, select_sources=None, device=kwargs.get("device", "cpu"), out=out, return_predictions=return_predictions)

            tools["predict_and_eval"] = t_predict_and_eval
        except Exception:
            # If import fails, skip registering the tool
            pass

    return tools


def run_agent_loop(
    lm,
    tokenizer,
    tools: Dict[str, Callable[..., Any]],
    system_prompt: str,
    user_prompt: str,
    device: torch.device,
    max_steps: int = 5,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Run a simple agent loop. Returns final JSON (parsed) or last model text.

    The model is asked to emit JSON. If it emits an action, run the tool and append the tool's
    result to the conversation. Repeat until the model returns a final_answer JSON.
    """

    conversation: List[Dict[str, str]] = []
    conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_prompt})

    print(f"[agent] start loop: prompt=" + (user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt), flush=True)

    for step in range(max_steps):
        prompt_text = "\n".join([f"[{m['role'].upper()}] {m['content']}" for m in conversation])

        # tokenize and generate with the LM
        try:
            print(f"[agent] step {step}: generating...", flush=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = lm.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), max_new_tokens=max_new_tokens)
            out_text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"[agent] step {step}: model output (truncated) = " + (out_text[:300] + "..." if len(out_text) > 300 else out_text), flush=True)
        except Exception as e:
            print(f"[agent] step {step}: generation failed: {e}", flush=True)
            return {"error": f"generation_error: {e}", "step": step}

        # try to parse JSON from the model output
        parsed = None
        try:
            parsed = json.loads(out_text)
        except Exception:
            # try to extract a JSON substring
            start = out_text.find("{")
            end = out_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(out_text[start : end + 1])
                except Exception:
                    parsed = None

        if parsed is None:
            # treat as final text answer
            print(f"[agent] step {step}: no JSON parsed; returning final text", flush=True)
            return {"final_text": out_text, "step": step}

        # if parsed contains final_answer, return it
        if "final_answer" in parsed:
            print(f"[agent] step {step}: final_answer found", flush=True)
            return {"final_answer": parsed["final_answer"], "step": step}

        # expect an action
        if "action" in parsed and "args" in parsed:
            action = parsed["action"]
            args = parsed["args"]
            tool = tools.get(action)
            if tool is None:
                observation = {"error": f"unknown tool {action}"}
            else:
                try:
                    print(f"[agent] step {step}: invoking tool '{action}' with args={args}", flush=True)
                    observation = tool(**args) if isinstance(args, dict) else tool(args)
                    print(f"[agent] step {step}: tool '{action}' returned (truncated) = " + (json.dumps(observation)[:300] + "..." if isinstance(observation, (dict, list)) and len(json.dumps(observation)) > 300 else str(observation)), flush=True)
                except Exception as e:
                    observation = {"error": str(e)}

            # append observation to conversation and continue
            conversation.append({"role": "assistant", "content": out_text})
            conversation.append({"role": "tool", "content": json.dumps(observation)})
            continue

        # fallback: return parsed
        return {"parsed": parsed, "step": step}

    return {"error": "max_steps_exceeded", "conversation": conversation}


def _cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="mindllm-base.ckpt", help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--config", default=None, help="Path to override config/default.yaml")
    parser.add_argument("--device", default=None, choices=["cpu", "gpu", "cuda"], help="Optional device override")
    parser.add_argument("--system", default=None, help="Optional system prompt override")
    parser.add_argument("--predictions", default=None, help="Optional path to existing predictions (.pt/.pth or JSON). If omitted, the script runs the MindLLM predict pipeline inline (same as scripts/predict_and_eval.py) and uses those fresh results")
    parser.add_argument("--predict-out", default="outputs/agent_runner_predictions.pt", help="Where to store the intermediate predictions generated inline before the agent runs")
    parser.add_argument("--select-sources", default=None, help='JSON list of dataset sources to select, e.g. "[\"coco-caption\",\"vqa-v2\"]". Defaults to config value when omitted.')
    parser.add_argument("--default-source", default=None, help="Default dataset source key to assign when predictions lack a source field")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional override for max agent steps")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config is not None else (repo_root / "config" / "default.yaml")
    cfg = OmegaConf.load(cfg_path)

    # agent config
    agent_cfg = cfg.get("agent") if isinstance(cfg, dict) else cfg.agent
    device_choice = args.device or (agent_cfg.get("device") if isinstance(agent_cfg, dict) else agent_cfg.device)
    system_prompt = args.system or (agent_cfg.get("system_prompt") if isinstance(agent_cfg, dict) else agent_cfg.system_prompt)
    max_steps = args.max_steps or (agent_cfg.get("max_steps") if isinstance(agent_cfg, dict) else agent_cfg.max_steps)
    max_new_tokens = agent_cfg.get("max_new_tokens") if isinstance(agent_cfg, dict) else agent_cfg.max_new_tokens

    # build tools from config
    tools = build_tools_from_cfg(agent_cfg if isinstance(agent_cfg, dict) else OmegaConf.to_container(agent_cfg), cfg)

    # parse optional select_sources override
    select_sources = json.loads(args.select_sources) if args.select_sources is not None else None

    # load LM and tokenizer
    lm, tokenizer, torch_device = _load_model_and_tokenizer(args.ckpt, cfg=cfg, device=device_choice or "cpu")

    # Always run over dataset inputs (non-interactive). If a predictions path was provided, load it;
    # otherwise run the MindLLM predict pipeline inline (same logic as scripts/predict_and_eval.py) so
    # the agent always starts from freshly generated captions derived from the fMRI inputs.
    loaded_inputs = None
    if args.predictions:
        p = Path(args.predictions)
        if p.suffix in (".pt", ".pth"):
            loaded_inputs = torch.load(str(p))
        else:
            loaded_inputs = json.loads(open(str(p), "r", encoding="utf-8").read())

    inline_buffer: Optional[List[Dict[str, Any]]] = [] if not args.predictions else None

    def iter_samples():
        if loaded_inputs is not None:
            for entry in loaded_inputs:
                yield entry
            return

        print("[agent] streaming inline predictions (MindLLM -> captions)...", flush=True)
        for sample_entry in _stream_predictions_from_model(
            ckpt=args.ckpt,
            cfg_path=cfg_path,
            select_sources=select_sources,
            device_choice=device_choice,
            default_source=args.default_source,
        ):
            if inline_buffer is not None:
                inline_buffer.append(sample_entry)
            yield sample_entry

    sample_iter = iter_samples()

    results = []
    any_sample = False
    for idx, item in enumerate(sample_iter):
        any_sample = True
        # choose a field to use as the prompt: prefer 'prompt', then 'pred', then join 'target'
        if isinstance(item, dict) and item.get("prompt"):
            user_prompt = item.get("prompt")
        elif isinstance(item, dict) and item.get("pred"):
            user_prompt = item.get("pred")
        elif isinstance(item, dict) and item.get("target"):
            targ = item.get("target")
            if isinstance(targ, (list, tuple)):
                user_prompt = " \n ".join([str(x) for x in targ])
            else:
                user_prompt = str(targ)
        else:
            user_prompt = str(item)

        res = run_agent_loop(lm, tokenizer, tools, system_prompt, user_prompt, torch_device, max_steps=max_steps, max_new_tokens=max_new_tokens)
        record = {"index": idx, "input_id": item.get("id") if isinstance(item, dict) else None, "agent_result": res}
        results.append(record)

        # Create a LangGraph-compatible node for this sample so downstream pipelines can ingest it.
        # Node includes basic identifiers, original input/prediction, and agent output.
        node = {
            "node_id": f"sample-{idx}" if not (isinstance(item, dict) and item.get("id")) else str(item.get("id")),
            "original_input": {},
            "original_prediction": None,
            "agent": record["agent_result"],
        }
        if isinstance(item, dict):
            # copy over common fields if present (image path, embedding, prompt, pred, target)
            for k in ("image", "image_path", "embedding", "prompt", "pred", "target", "source"):
                if k in item:
                    node["original_input"][k] = item.get(k)
            # preserve a top-level prediction if available
            if "pred" in item:
                node["original_prediction"] = item.get("pred")

        # append node to jsonl file for LangGraph ingestion
        nodes_path = repo_root / "outputs" / "langgraph_nodes.jsonl"
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(nodes_path, "a", encoding="utf-8") as nf:
                nf.write(json.dumps(node, ensure_ascii=False) + "\n")
            print(f"[agent] wrote LangGraph node for sample {idx} to {nodes_path}", flush=True)
        except Exception as e:
            print(f"[agent] failed writing LangGraph node for sample {idx}: {e}", flush=True)

    if not any_sample:
        print("No inputs found (provide --predictions or ensure dataset yields samples).")
        sys.exit(1)

    if not args.predictions and inline_buffer:
        predict_out_path = Path(args.predict_out)
        if not predict_out_path.is_absolute():
            predict_out_path = repo_root / predict_out_path
        predict_out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inline_buffer, str(predict_out_path))
        print(f"[agent] saved {len(inline_buffer)} streamed predictions to {predict_out_path}", flush=True)

    out_file = repo_root / "outputs" / "agent_auto_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    open(out_file, "w", encoding="utf-8").write(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {len(results)} agent results to {out_file}")
    sys.exit(0)


if __name__ == "__main__":
    _cli_main()
