"""MyBuddy lightweight research codebase.

This package implements the core modules described in BuddyVQA/MyBuddy:
- streaming video frame encoding
- active, semi-active and low-active memory
- visual information retrieval with caption/filter/embodied cues
- historical QA buffer and question router
- hierarchical rephrasing and answer generation
- train / infer / evaluate entry points

The implementation is deliberately dependency-light. It uses deterministic text and
numeric features to make the full pipeline runnable without proprietary MLLM APIs.
Production deployments can replace the mock encoders and mock MLLMs through the
same interfaces.
"""

__all__ = [
    "config",
    "data",
    "models",
    "memory",
    "retrieval",
    "router",
    "pipeline",
    "train",
    "infer",
    "evaluate",
]

# ---------------------------------------------------------------------------
# Package design notes
# ---------------------------------------------------------------------------
# This codebase is organized to mirror the paper-level method rather than to be
# a single monolithic script. The goal is to make every module replaceable while
# preserving the same data flow.
#
# 1. data.py
#    Defines BuddyVQA-style videos, frames, timestamped QA records, JSONL I/O,
#    synthetic data generation, and dataset splitting.
#
# 2. models.py
#    Provides deterministic local stand-ins for expensive vision-language
#    models: frame encoder, captioner, summarizer, embodied cue detector,
#    hierarchical rephraser, and answer model.
#
# 3. memory.py
#    Implements Active Memory, Semi-Active Memory with DBSCAN compression, and
#    Low-Active Memory summaries.
#
# 4. retrieval.py
#    Implements Visual Information Retrieval: on-demand active descriptions,
#    semi-active filtering, low-active summary filtering, and gaze/hand cues.
#
# 5. router.py
#    Implements the Historical QA Buffer and a trainable lightweight Question
#    Router for chained QA selection.
#
# 6. pipeline.py
#    Wires all modules into the online streaming QA procedure described by the
#    paper: memory update, retrieval, routing, rephrasing, answering, and buffer
#    update.
#
# 7. train.py
#    Trains the router and saves config/checkpoints/metrics.
#
# 8. infer.py
#    Runs end-to-end streaming inference and emits auditable JSONL records.
#
# 9. evaluate.py
#    Computes deterministic local metrics by overall accuracy, category,
#    chained status, deictic status, and router behavior.
#
# The paper version can be reproduced more closely by replacing the lightweight
# local model classes with the intended MLLM backbones. The interfaces are kept
# deliberately small: encode_frame, describe_frames, summarize, detect, rephrase,
# and answer. This makes it straightforward to plug in GPT, Gemini, Qwen, or a
# fine-tuned local model while leaving the streaming memory and routing logic
# untouched.
#
# The project is dependency-light by design. It runs without PyTorch, OpenCV, or
# external APIs, which is useful for verifying the system architecture and for
# unit testing on CPU-only machines. A production version would add video frame
# extraction, real embeddings, batched captioning, and API/model backends.
#
# The JSONL dataset format stores one complete video per line. Each line includes
# a video_id, sampled frames, and timestamped QA records. The synthetic generator
# intentionally creates high rates of deictic and chained questions to exercise
# the companion-specific logic.
#
# All command-line entry points can be run with python -m mybuddy.<module>. This
# avoids import-path issues and makes the zipped project immediately runnable.
#
# Additional ablations can be implemented by toggling modules inside
# MyBuddyPipeline or by replacing them with identity/no-op variants. For example,
# w/o Historical QA corresponds to forcing the router prediction to unchained;
# w/o Hierarchical Rephrasing corresponds to bypassing HierarchicalRephraser;
# w/o Eye & Hand Detection corresponds to clearing embodied_cues in retrieval;
# and w/o memory tiers corresponds to modifying MultiLevelMemory outputs.
#
# The checkpoint format is JSON for readability. It stores router weights, bias,
# threshold, and vectorizer dimension. This is intentionally transparent for
# debugging and paper supplement inspection.
#
# End of package design notes.
