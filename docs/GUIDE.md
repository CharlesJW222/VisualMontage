# VisualMontage Guide

## Overview

VisualMontage constructs a final visible image feed from an indexed dataset.
The pipeline keeps source frames unchanged, selects an ordered subset, assigns
final v, t, or vt modes in Caption Editor, materializes the feed, and can run
downstream Victim Validation.

v publishes an image with no caption. t withholds the image and publishes a
locally grounded caption. vt publishes an image with a locally grounded caption
rendered onto it. Visual Editor does not produce modality hints; Caption Editor
is the sole final modality owner.

## Installation

Python 3.8 or later is required.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

On Apple Silicon macOS, install the optional local MLX backend separately.

    bash scripts/macos/local_setup.sh

## Configure the API

For the public OpenAI API:

    export MONTAGE_API_BASE='https://api.openai.com/v1'
    export OPENAI_API_KEY='your key here'

For another OpenAI-compatible endpoint:

    export MONTAGE_API_BASE='https://your-api-host/v1'
    export PRIVATE_API_KEY='your key here'
    export MONTAGE_API_KEY_ENV='PRIVATE_API_KEY'

Use the preflight check before a costly run.

    # macOS
    python scripts/macos/check.py

    # Ubuntu
    python scripts/ubuntu/check.py

The check confirms local dependencies and configuration. It does not verify API
credits, model access, or account billing.

## Run one dataset

    python demo.py \
      --dataset Video_Dataset_Categories/01_Emergency_and_Disaster_Response/air_ground_firefighting \
      --output attack_plan/small_robot_example \
      --api-timeout-seconds 600 \
      --api-max-retries 4

To use one model for every production and validation role:

    export MONTAGE_SCREENWRITER_MODEL='gpt-5.6-luna'
    export MONTAGE_VISUAL_EDITOR_MODEL='gpt-5.6-luna'
    export MONTAGE_CAPTION_EDITOR_MODEL='gpt-5.6-luna'
    export MONTAGE_DIRECTOR_MODEL='gpt-5.6-luna'
    export MONTAGE_VICTIM_MODEL='gpt-5.6-luna'
    export MONTAGE_JUDGE_MODEL='gpt-5.6-luna'

## Run production and Validation together

    python demo.py \
      --dataset Video_Dataset_Categories/03_Law_Enforcement_and_Public_Safety/vehicle_pursuit \
      --output attack_plan/police_rescue_example \
      --enable-victim-validation \
      --num-victims 5 \
      --num-victim-conclusions 1 \
      --target-ground-truth false \
      --api-timeout-seconds 600 \
      --api-max-retries 4

Do not use --no-materialize with --enable-victim-validation. Validation
requires the materialized visible feed.

## Validate an existing visible feed

    python victim_validation.py \
      --sequence attack_plan/police_rescue_example/sequence/captioned_image_sequence.json \
      --victim-model gpt-5.6-sol \
      --judge-model gpt-5.6-sol \
      --num-victims 5 \
      --num-conclusions 1 \
      --target-ground-truth false \
      --api-timeout-seconds 600 \
      --api-max-retries 4

## Outputs

Each run writes:

    attack_plan/<run-name>/
    ├── montage_plan.json
    ├── api_run_stats.json
    ├── sequence/
    │   ├── captioned_image_sequence.json
    │   ├── distributed_publication.json
    │   └── victim_images/
    ├── provenance/
    │   ├── attack_plan_provenance.md
    │   └── attack_plan_provenance.json
    └── victim_validation.json

The Markdown provenance report maps final positions to source frame IDs, source
hashes, belief values, selected edges, score components, Director gates, final
modes, captions, and materialized outputs. The JSON companion preserves the
complete structured trace. The current Caption Editor contract does not produce
a separate per-item rationale, so the provenance report states that limit rather
than inventing reasons.

## Dataset media and release hygiene

Dataset manifests and metadata are versioned. Source videos are kept locally
but ignored by Git because some assets are research-only or have unrecorded
redistribution rights. A cloned repository needs the matching video in
Video_Dataset_Categories/<domain>/<dataset>/ before it can run that dataset;
see docs/DATASETS.md.

Generated attack_plan outputs, local models, virtual environments, caches, and
logs are also ignored. See [RUN_ALL_DATASETS.md](RUN_ALL_DATASETS.md) for the
sequential runner.
