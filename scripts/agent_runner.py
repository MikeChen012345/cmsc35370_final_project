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
from typing import Any, Callable, Dict, Iterator, List, Optional, TypedDict

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool, tool

from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM
from src.utils import adapt_voxels


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
    # Hugging Face tokenizers return BatchEncoding objects that expose `.to()`
    # but do not register as dicts; handle those explicitly so their tensors
    # follow the model to GPU during manual predict loops.
    if hasattr(value, "to"):
        try:
            return value.to(device)
        except Exception:
            pass
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


class PipelineState(TypedDict, total=False):
    batch: Dict[str, Any]
    batch_idx: int
    fmri_embeddings: torch.Tensor | None
    encoder_summary: List[Optional[Dict[str, Any]]]
    samples: List[Dict[str, Any]]
    default_source: Optional[str]
    fewshot_prefix: Optional[Dict[str, torch.Tensor]]


class AgentPipelineState(TypedDict, total=False):
    prediction_iter: Iterator[Dict[str, Any]]
    sample: Optional[Dict[str, Any]]
    current_index: int
    next_index: int
    done: bool
    agent_record: Optional[Dict[str, Any]]


def _summarize_embeddings(fmri_token_embeds: torch.Tensor | None) -> List[Optional[Dict[str, Any]]]:
    if fmri_token_embeds is None:
        return []
    summaries: List[Optional[Dict[str, Any]]] = []
    try:
        enc_cpu = fmri_token_embeds.detach().cpu()
    except Exception:
        return summaries
    for tensor in enc_cpu:
        mean_val = tensor.mean().item()
        std_val = tensor.std(unbiased=False).item()
        summaries.append({"shape": list(tensor.shape), "mean": mean_val, "std": std_val})
    return summaries


def _load_example_embedding(entry: Dict[str, Any]) -> Optional[torch.Tensor]:
    tensor = None
    if "embedding_path" in entry:
        path = Path(entry["embedding_path"])
        if path.exists():
            tensor = torch.load(str(path))
    elif "embedding" in entry:
        tensor = torch.tensor(entry["embedding"], dtype=torch.float32)

    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _build_fewshot_prompt(fewshot_examples: Optional[List[Dict[str, Any]]]) -> str:
    if not fewshot_examples:
        return ""

    lines: List[str] = []
    for idx, example in enumerate(fewshot_examples):
        caption = (example.get("caption") or "").strip()
        summary = example.get("summary")
        if summary is None:
            emb_tensor = _load_example_embedding(example)
            if emb_tensor is not None:
                summary_list = _summarize_embeddings(emb_tensor)
                summary = summary_list[0] if summary_list else None
        parts: List[str] = [f"Example {idx + 1}:"]
        if summary:
            parts.append(
                "embedding stats="
                + ", ".join(
                    f"{k}={summary[k]:.4f}" for k in ("mean", "std") if k in summary and isinstance(summary[k], (int, float))
                )
            )
        if caption:
            parts.append(f"caption=\"{caption}\"")
        lines.append(" ".join(parts))

    return "\n".join(lines)


def _prepare_fewshot_prefix(model: MindLLM, tokenizer, fewshot_examples: Optional[List[Dict[str, Any]]], device: torch.device):
    prompt = _build_fewshot_prompt(fewshot_examples)
    if not prompt.strip():
        return None

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    embeds = model.lm.get_input_embeddings()(encoded["input_ids"])
    return {"embeds": embeds, "mask": encoded.get("attention_mask"), "text": prompt}


def build_fewshot_prefix_from_pair_dict(
    model: MindLLM,
    tokenizer,
    pair_dict: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
):
    """Create a prompt/embedding prefix from a {caption: embedding} dictionary.

    Retrieval systems can call this helper after they gather a handful of example
    embedding-caption pairs. The resulting prefix is prepended to the prompt tokens
    so the LM "sees" both the summarized embedding stats and captions before
    decoding the next embedding.
    """

    if not pair_dict:
        return None
    fewshot_examples: List[Dict[str, Any]] = []
    for caption, embedding in pair_dict.items():
        fewshot_examples.append({"caption": caption, "embedding": embedding})
    return _prepare_fewshot_prefix(model, tokenizer, fewshot_examples, device)


def _ensure_list_length(value: Any, n_items: int) -> List[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) == 0:
            return [None] * n_items
        if len(values) < n_items:
            values = values + [values[-1]] * (n_items - len(values))
        elif len(values) > n_items:
            values = values[:n_items]
        return values
    return [value] * n_items


def _apply_prefix_to_inputs(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix: Optional[Dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prefix:
        return inputs_embeds, attention_mask
    prefix_embeds = prefix.get("embeds")
    prefix_mask = prefix.get("mask")
    if prefix_embeds is None or prefix_mask is None:
        return inputs_embeds, attention_mask
    prefix_embeds = prefix_embeds.to(inputs_embeds.device)
    prefix_mask = prefix_mask.to(attention_mask.device)
    batch_size = inputs_embeds.size(0)
    if prefix_embeds.size(0) != batch_size:
        prefix_embeds = prefix_embeds.repeat(batch_size, 1, 1)
        prefix_mask = prefix_mask.repeat(batch_size, 1)
    new_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
    new_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    return new_embeds, new_mask


def _generate_from_embeddings(
    model: MindLLM,
    batch: Dict[str, Any],
    fmri_token_embeds: torch.Tensor,
    max_new_tokens: int,
    fewshot_prefix: Optional[Dict[str, torch.Tensor]] = None,
) -> List[str]:
    chat = batch["prompt"]
    inputs_embeds, attention_mask, _, position_ids, _ = model.prepare_multimodal_inputs(
        chat["input_ids"], chat["attention_mask"], None, fmri_token_embeds
    )

    generated_tokens = []
    with torch.no_grad():
        for i in range(batch["voxels"].size(0)):
            sample_embeds = inputs_embeds[i : i + 1]
            sample_mask = attention_mask[i : i + 1]
            sample_embeds, sample_mask = _apply_prefix_to_inputs(sample_embeds, sample_mask, fewshot_prefix)
            out = model.lm.generate(
                input_ids=None,
                inputs_embeds=sample_embeds,
                attention_mask=sample_mask,
                max_new_tokens=max_new_tokens,
            )
            generated_tokens.append(out.squeeze(0).tolist())
    return model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def _build_inference_graph(
    model: MindLLM,
    max_new_tokens: int,
    *,
    default_source: Optional[str] = None,
    fewshot_prefix: Optional[Dict[str, torch.Tensor]] = None,
) -> Any:
    graph = StateGraph(PipelineState)

    def encoder_node(state: PipelineState) -> Dict[str, Any]:
        batch = state["batch"]
        encoder_inputs = dict(batch)
        encoder_inputs["voxels"] = adapt_voxels(batch["voxels"])
        with torch.no_grad():
            fmri_token_embeds = model.forward(**encoder_inputs, return_fmri_tokens=True)
        return {
            "fmri_embeddings": fmri_token_embeds,
            "encoder_summary": _summarize_embeddings(fmri_token_embeds),
        }

    def llm_node(state: PipelineState) -> Dict[str, Any]:
        batch = state["batch"]
        fmri_token_embeds = state.get("fmri_embeddings")
        if fmri_token_embeds is None:
            return {"samples": []}
        preds = _generate_from_embeddings(
            model,
            batch,
            fmri_token_embeds,
            max_new_tokens,
            fewshot_prefix=fewshot_prefix,
        )
        targets = batch.get("answers")
        sample_ids = _ensure_list_length(batch.get("sample_or_coco_id"), len(preds))
        sources = _ensure_list_length(batch.get("source"), len(preds))
        encoder_summary = state.get("encoder_summary") or []

        subjects = batch.get("subject")
        question_ids = batch.get("question_id")

        entries: List[Dict[str, Any]] = []
        for i in range(len(preds)):
            sid = _tensor_to_python(sample_ids[i])
            metadata = {
                "subject": _select_entry(subjects, i),
                "question_id": _select_entry(question_ids, i),
                "sample_or_coco_id": sid,
            }
            target_val = _select_entry(targets, i)
            source_val = _select_entry(sources, i)
            if source_val is None and default_source is not None:
                source_val = default_source
            summary_val = encoder_summary[i] if i < len(encoder_summary) else None
            entries.append(
                {
                    "id": sid,
                    "pred": preds[i],
                    "target": target_val,
                    "source": source_val,
                    "encoder_summary": summary_val,
                    "metadata": {k: v for k, v in metadata.items() if v is not None},
                }
            )
        return {"samples": entries, "fmri_embeddings": None}

    graph.add_node("encoder", encoder_node)
    graph.add_node("llm", llm_node)
    graph.add_edge("encoder", "llm")
    graph.add_edge("llm", END)
    graph.set_entry_point("encoder")

    return graph.compile()


def _stream_predictions_from_model(
    ckpt: str,
    cfg_path: Path,
    select_sources: Optional[List[str]],
    device_choice: Optional[str],
    default_source: Optional[str],
    max_new_tokens: int,
    fewshot_examples: Optional[List[Dict[str, Any]]],
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

    fewshot_prefix = _prepare_fewshot_prefix(model, tokenizer, fewshot_examples, torch_device)
    if fewshot_prefix and fewshot_prefix.get("text"):
        print("[agent] using few-shot prefix:\n" + fewshot_prefix["text"], flush=True)

    inference_graph = _build_inference_graph(
        model,
        max_new_tokens=max_new_tokens,
        default_source=default_source,
        fewshot_prefix=fewshot_prefix,
    )

    for batch_idx, batch in enumerate(predict_loader):
        batch = _move_batch_to_device(batch, torch_device)
        state = inference_graph.invoke({"batch": batch, "batch_idx": batch_idx})
        for sample_entry in state.get("samples", []):
            if sample_entry.get("source") is None:
                sample_entry["source"] = default_source
            yield sample_entry


def _resolve_checkpoint(cfg_root: Optional[dict]) -> Optional[str]:
    if cfg_root is None:
        return None
    if isinstance(cfg_root, dict):
        return cfg_root.get("checkpoint")
    return getattr(cfg_root, "checkpoint", None)


def _invoke_tool(tool_obj: Any, args: Any) -> Any:
    payload = args if isinstance(args, dict) else {"input": args}
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(payload)
    if callable(tool_obj):
        return tool_obj(**payload)
    raise TypeError(f"Tool object {tool_obj} is not callable")


def build_tools_from_cfg(agent_cfg: dict, cfg_root: Optional[dict] = None) -> Dict[str, BaseTool]:
    """Temporarily return no tools regardless of config so user can supply real ones later."""
    return {}


def _build_agent_workflow_graph(on_sample: Callable[[Dict[str, Any], int], Dict[str, Any]]):
    graph = StateGraph(AgentPipelineState)

    def prediction_node(state: AgentPipelineState) -> Dict[str, Any]:
        iterator = state.get("prediction_iter")
        if iterator is None:
            return {"done": True}
        try:
            sample = next(iterator)
        except StopIteration:
            return {"done": True}

        idx = state.get("next_index", 0)
        return {
            "sample": sample,
            "current_index": idx,
            "next_index": idx + 1,
            "done": False,
        }

    def prediction_router(state: AgentPipelineState) -> str:
        return "stop" if state.get("done") else "agent"

    graph.add_node("prediction", prediction_node)
    graph.add_conditional_edges("prediction", prediction_router, {"stop": END, "agent": "agent"})

    def agent_node(state: AgentPipelineState) -> Dict[str, Any]:
        sample = state.get("sample")
        if sample is None:
            return {}
        idx = state.get("current_index", 0)
        record = on_sample(sample, idx)
        return {"agent_record": record, "sample": None}

    graph.add_node("agent", agent_node)
    graph.add_edge("agent", "prediction")
    graph.set_entry_point("prediction")

    return graph.compile()


def run_agent_loop(
    lm,
    tokenizer,
    tools: Dict[str, BaseTool],
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

    print(f"[agent] start loop: prompt=" + (user_prompt[:1000] + "..." if len(user_prompt) > 1000 else user_prompt), flush=True)

    for step in range(max_steps):
        prompt_text = "\n".join([f"[{m['role'].upper()}] {m['content']}" for m in conversation])

        # tokenize and generate with the LM
        try:
            print(f"[agent] step {step}: generating...", flush=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = lm.generate(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), max_new_tokens=max_new_tokens)
            out_text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"[agent] step {step}: model output (truncated) = " + (out_text[:1000] + "..." if len(out_text) > 1000 else out_text), flush=True)
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
                    observation = _invoke_tool(tool, args)
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
    agent_cfg_dict = agent_cfg if isinstance(agent_cfg, dict) else OmegaConf.to_container(agent_cfg, resolve=True)
    if not isinstance(agent_cfg_dict, dict):
        agent_cfg_dict = dict(agent_cfg_dict)
    device_choice = args.device or agent_cfg_dict.get("device")
    system_prompt = args.system or agent_cfg_dict.get("system_prompt")
    max_steps = args.max_steps or agent_cfg_dict.get("max_steps")
    if system_prompt is None:
        system_prompt = ""
    if max_steps is None:
        max_steps = 5
    max_new_tokens = agent_cfg_dict.get("max_new_tokens", 128)
    fewshot_examples: Optional[List[Dict[str, Any]]] = None

    # build tools from config
    tools = build_tools_from_cfg(agent_cfg_dict, cfg)

    # parse optional select_sources override
    select_sources = json.loads(args.select_sources) if args.select_sources is not None else None

    # load LM and tokenizer
    lm, tokenizer, torch_device = _load_model_and_tokenizer(args.ckpt, cfg=cfg, device=device_choice or "cpu")

    # Always run over dataset inputs (non-interactive). If a predictions path was provided, load it;
    # otherwise run the MindLLM predict pipeline inline (same logic as scripts/predict_and_eval.py) so
    # the agent always starts from freshly generated captions derived from the fMRI inputs.
    loaded_inputs: Optional[List[Dict[str, Any]]] = None
    inline_buffer: Optional[List[Dict[str, Any]]] = None
    prediction_iter: Iterator[Dict[str, Any]]
    if args.predictions:
        p = Path(args.predictions)
        if p.suffix in (".pt", ".pth"):
            loaded_inputs = torch.load(str(p))
        else:
            loaded_inputs = json.loads(open(str(p), "r", encoding="utf-8").read())
        prediction_iter = iter(loaded_inputs)
    else:
        inline_buffer = []

        def streaming_iter():
            print("[agent] streaming inline predictions (MindLLM -> captions)...", flush=True)
            for sample_entry in _stream_predictions_from_model(
                ckpt=args.ckpt,
                cfg_path=cfg_path,
                select_sources=select_sources,
                device_choice=device_choice,
                default_source=args.default_source,
                max_new_tokens=max_new_tokens,
                fewshot_examples=fewshot_examples,
            ):
                if inline_buffer is not None:
                    inline_buffer.append(sample_entry)
                yield sample_entry

        prediction_iter = streaming_iter()

    results: List[Dict[str, Any]] = []
    nodes_path = repo_root / "outputs" / "langgraph_nodes.jsonl"
    nodes_path.parent.mkdir(parents=True, exist_ok=True)

    def emit_langgraph_nodes(item: Dict[str, Any], idx: int, agent_result: Dict[str, Any]):
        base_node_id = str(item.get("id")) if item.get("id") is not None else f"sample-{idx}"
        metadata = dict(item.get("metadata") or {})
        encoder_node = {
            "node_id": f"{base_node_id}-encoder",
            "type": "encoder",
            "metadata": metadata,
            "source": item.get("source"),
            "output_summary": item.get("encoder_summary"),
        }
        lm_node = {
            "node_id": f"{base_node_id}-lm",
            "type": "llm",
            "inputs": {"encoder": encoder_node["node_id"]},
            "prediction": item.get("pred"),
            "target": item.get("target"),
            "source": item.get("source"),
        }
        agent_node = {
            "node_id": f"{base_node_id}-agent",
            "type": "agent",
            "inputs": {"llm": lm_node["node_id"]},
            "agent_result": agent_result,
            "original_prediction": item.get("pred"),
            "target": item.get("target"),
        }
        agent_node["original_input"] = {
            k: item.get(k)
            for k in ("image", "image_path", "embedding", "prompt", "pred", "target", "source")
            if k in item
        }
        try:
            with open(nodes_path, "a", encoding="utf-8") as nf:
                for node in (encoder_node, lm_node, agent_node):
                    nf.write(json.dumps(node, ensure_ascii=False) + "\n")
            print(f"[agent] wrote LangGraph encoder/llm/agent nodes for sample {idx} to {nodes_path}", flush=True)
        except Exception as e:
            print(f"[agent] failed writing LangGraph nodes for sample {idx}: {e}", flush=True)

    def handle_sample(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        if not isinstance(sample, dict):
            raise ValueError("Prediction iterator must yield dict entries")
        pred = sample.get("pred")
        if pred is None:
            raise ValueError("Prediction samples must include a 'pred' field")
        user_prompt = f"""
        {pred}
        """
        agent_res = run_agent_loop(
            lm,
            tokenizer,
            tools,
            system_prompt,
            user_prompt,
            torch_device,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
        )
        record = {"index": idx, "input_id": sample.get("id"), "agent_result": agent_res}
        results.append(record)
        emit_langgraph_nodes(sample, idx, agent_res)
        return record

    agent_graph = _build_agent_workflow_graph(handle_sample)
    agent_graph.invoke({"prediction_iter": prediction_iter, "next_index": 0})

    if not results:
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
