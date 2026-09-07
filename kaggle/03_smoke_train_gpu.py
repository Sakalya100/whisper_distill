"""Step 3 -- cut the student to 4 decoder layers and prove it trains on 5 hours.

    Accelerator: GPU T4 x2      Internet: required      Quota: ~1 h

This is the cheap failure detector. If it does not work on 5 hours it will not work on 200,
and you will have spent one quota-hour finding out instead of forty.

THREE THINGS THIS MUST PROVE
  1. Loss goes down.
  2. The output is Hinglish-shaped **including Latin-script English words**. "Hindi-shaped"
     is not the bar -- the whole point is code-switching.
  3. Checkpoint-and-resume through a private HF Hub repo works, before a real run needs it.

Deliberately NOT tested here: vocabulary pruning. One variable at a time -- prove the
decoder cut trains first. Pruning needs token counts from the full pseudo-label corpus
anyway, which does not exist yet at step 3.
"""

import sys
from pathlib import Path

CACHE = Path("/kaggle/input/vaani-hi-mel-cache")   # rename to your dataset slug
HUB_REPO = "Sakalya100/whisper-distill-hinglish-smoke"
MAX_STEPS = 400
BATCH = 16

sys.path.insert(0, "/kaggle/working/whisper_distill/src")


class MelCacheDataset:
    """Memory-mapped reads straight off /kaggle/input. No audio decoding at train time."""

    def __init__(self, cache_dir: Path, *, max_teacher_wer: float = 0.20):
        from whisper_distill.data.pack import open_shards

        self.index, self.shards = open_shards(cache_dir)
        self.records = self.index.filtered(max_teacher_wer=max_teacher_wer)
        print(f"{len(self.records)} clips after filtering "
              f"(of {len(self.index.records)} cached)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        import numpy as np
        import torch

        r = self.records[i]
        mels, tokens = self.shards[r.shard]
        return {
            "input_features": torch.from_numpy(np.asarray(mels[r.row], dtype=np.float32)),
            "labels": torch.from_numpy(np.asarray(tokens[r.row], dtype=np.int64)),
        }


def collate(batch: list[dict]) -> dict:
    import torch

    return {
        "input_features": torch.stack([b["input_features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def main() -> None:
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
    from kaggle_secrets import UserSecretsClient

    from whisper_distill.config import DEFAULT
    from whisper_distill.modeling.student import build_student

    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

    ds = MelCacheDataset(CACHE)
    if len(ds) < 100:
        raise SystemExit(f"only {len(ds)} clips -- run step 1 and 2 for more audio first")

    # token_counts=None skips vocabulary pruning on purpose. One variable at a time.
    model, _ = build_student(DEFAULT.student, token_counts=None)
    proc = WhisperProcessor.from_pretrained(DEFAULT.student.init_model_id)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nstudent: {total / 1e6:.1f}M params, {trainable / 1e6:.1f}M trainable "
          f"(encoder frozen: {DEFAULT.student.freeze_encoder})")
    print(f"decoder layers: {len(model.model.decoder.layers)}")
    print(f"encoder positions: {model.model.encoder.embed_positions.weight.shape[0]}")

    args = Seq2SeqTrainingArguments(
        output_dir="/kaggle/working/smoke",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=40,
        fp16=True,
        logging_steps=10,          # heartbeat: silent cells get killed
        save_steps=100,
        save_total_limit=1,        # /kaggle/working is capped ~20 GB
        report_to=[],
        # Sessions die. /kaggle/working does not survive a fresh session, so the Hub is
        # the checkpoint store. `all_checkpoints` pushes optimiser state, not just weights,
        # which is what makes a resume actually resume.
        push_to_hub=True,
        hub_model_id=HUB_REPO,
        hub_strategy="all_checkpoints",
        hub_private_repo=True,
        # Trainer's default multi-GPU path on T4x2 is naive model parallelism and leaves
        # one GPU idle. For a real run, launch this under torchrun/accelerate for DDP.
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=ds, data_collator=collate
    )

    # Resume from the Hub if a previous session got partway. This is the line that turns a
    # disconnect from a lost day into a lost few minutes.
    resume = None
    try:
        from huggingface_hub import snapshot_download
        resume = snapshot_download(HUB_REPO, repo_type="model",
                                   allow_patterns=["checkpoint-*/*"])
        print(f"resuming from {resume}")
    except Exception as e:  # noqa: BLE001 - first run has nothing to resume from
        print(f"no prior checkpoint ({type(e).__name__}); starting fresh")

    result = trainer.train(resume_from_checkpoint=resume)
    print(f"\nfinal train loss: {result.training_loss:.4f}")

    # --- proof 2: does it emit Hinglish, including Latin-script English?
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT -- looking for Latin-script English, not just Devanagari")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(ds))):
            feats = ds[i]["input_features"].unsqueeze(0).to(model.device, torch.float16)
            ids = model.generate(feats, max_new_tokens=100)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
            ref = ds.records[i].text
            print(f"\n  teacher: {ref[:100]}")
            print(f"  student: {hyp[:100]}")

    trainer.push_to_hub("smoke test complete")
    print("\nSmoke test done. Check: loss decreased, and English survived above.")


if __name__ == "__main__":
    main()
