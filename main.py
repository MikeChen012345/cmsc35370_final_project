from lightning import seed_everything
import os
import hydra
from hydra.utils import instantiate
from peft import LoraConfig, TaskType
import torch
from pathlib import Path

from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM

import logging

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    seed_everything(cfg.seed)

    if cfg.early_stop:
        cfg.trainer.callbacks.append(
            {
                "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                "monitor": "val/token_loss",
                "patience": 10,
            }
        )
    trainer = instantiate(cfg.trainer)

    tokenizer = create_tokenizer(cfg.model_id)
    data_module = instantiate(cfg.data, tokenizer=tokenizer)

    if cfg.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
    else:
        peft_config = None

    model_kwargs = {
        "encoder": cfg.encoder,
        "model_id": cfg.model_id,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
        "learning_rate": cfg.lr,
        # pass metric selection from config into the model so metrics are limited
        # to the same dataset sources selected for the DataModule.
        "metrics_data.task": cfg.data.task,
    }

    if cfg.checkpoint is None:
        model = MindLLM(**model_kwargs)
    else:
        model = MindLLM.load_from_checkpoint(
            cfg.checkpoint, strict=False, **model_kwargs
        )

    model.strict_loading = False
    if cfg.stage == "fit":
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=(
                os.path.join(
                    cfg.output_dir, "mindllm", cfg.resume_id, "checkpoints", "last.ckpt"
                )
                if cfg.resume_id is not None
                else None
            ),
        )
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")
    elif cfg.stage == "validate":
        trainer.validate(model=model, datamodule=data_module)
    elif cfg.stage == "test":
        trainer.test(model=model, datamodule=data_module)
    elif cfg.stage == "predict":
        predictions = trainer.predict(
            model=model,
            datamodule=data_module,
        )
        all_predictions = []

        # diagnostic counters
        missing_source_count = 0

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
                    assigned_s = orig_s

                    if orig_s is None:
                        missing_source_count += 1

                    # save assigned source so the saved predictions are attributed for future runs
                    all_predictions.append({"id": sid, "pred": p, "target": t, "source": assigned_s})

        # Diagnostic log
        total = len(all_predictions)
        unique_sources = sorted({x.get("source") for x in all_predictions if x.get("source") is not None})
        print(f"Predictions: {total}, missing source: {missing_source_count}")
        print(f"Sources seen: {unique_sources}")

        out_path = Path("outputs/predictions.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_predictions, str(out_path))


if __name__ == "__main__":
    main()
