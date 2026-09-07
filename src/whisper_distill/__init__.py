"""Distil a small Hinglish dictation model from a Hindi Whisper teacher.

Layout mirrors the pipeline stages in docs/01-kaggle-execution-plan.md:

    data/        acquisition, VAD segmentation, shard packing   (CPU session)
    labeling/    teacher wrappers, fused label+mel pass, filters (GPU session)
    modeling/    student surgery: decoder cut, vocab prune, window slice
    training/    collation, distillation loss, train loop, Hub checkpoint IO
    evaluation/  WER/CER, entity accuracy, false-trigger rate    (CPU session)
    itn/         rule-based inverse text normalisation            (CPU, no model)
"""

__version__ = "0.1.0"
