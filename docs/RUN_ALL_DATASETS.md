# Run All Datasets

The cross-platform runner reads Video Datasets/index.json and invokes the existing
demo.py once for every indexed dataset. It writes one output subdirectory per
dataset and appends status records to run_status.jsonl.

The corpus is stored by domain under `Video_Dataset_Categories/`. Build the flat
`Video Datasets/` view and its index first (symbolic links, no copy):

    python tools/build_flat_dataset_view.py

The generated index lists all 61 recordings. Its `data_root` is `.`, resolved
relative to the index file's directory, `Video Datasets/`. Omitting this field uses the same
directory. Update the index and `dataset_count` when adding datasets; validation
and tests use the actual catalog size.

For videos, `demo.py` reads `video_file` from each dataset manifest and samples
frames at runtime; a pre-existing `image_sequence.json` is not required.
FFmpeg must be available on PATH.

Before running every dataset, ensure that all local image directories are
present and that the API project has sufficient credit. Production rounds and
optional Validation can be expensive.

## Production only

Copy and run:

    cd /path/to/VisualMontage
    source .venv/bin/activate

    export MONTAGE_API_BASE='https://api.openai.com/v1'
    export OPENAI_API_KEY='your key here'

    # On Ubuntu, replace macos with ubuntu.
    python scripts/macos/check.py

    python scripts/run_all_datasets.py \
      --output-root attack_plan/all_datasets_openai \
      --resume \
      -- \
      --api-timeout-seconds 600 \
      --api-max-retries 4

--resume skips a dataset only after montage_plan.json already exists under its
output directory. Omit it to run every dataset again.

## Production plus Validation

The following command appends --enable-victim-validation and the declared
ground-truth value from Video Datasets/index.json for every dataset:

    cd /path/to/VisualMontage
    source .venv/bin/activate

    export MONTAGE_API_BASE='https://api.openai.com/v1'
    export OPENAI_API_KEY='your key here'
    export MONTAGE_SCREENWRITER_MODEL='gpt-5.6-luna'
    export MONTAGE_VISUAL_EDITOR_MODEL='gpt-5.6-luna'
    export MONTAGE_CAPTION_EDITOR_MODEL='gpt-5.6-luna'
    export MONTAGE_DIRECTOR_MODEL='gpt-5.6-luna'
    export MONTAGE_VICTIM_MODEL='gpt-5.6-luna'
    export MONTAGE_JUDGE_MODEL='gpt-5.6-luna'

    python scripts/run_all_datasets.py \
      --output-root attack_plan/all_datasets_gpt56_sol_validation \
      --with-validation \
      --resume \
      -- \
      --num-victims 5 \
      --num-victim-conclusions 1 \
      --api-timeout-seconds 600 \
      --api-max-retries 4

## Test the command without API calls

    python scripts/run_all_datasets.py \
      --datasets launch_recording \
      --dry-run \
      -- \
      --api-timeout-seconds 600

## Operational behavior

- The runner continues after a failed dataset and exits with status 1 if one or
  more datasets fail.
- Add --stop-on-error to stop at the first failure.
- Add --datasets name_a,name_b to run a subset.
- Do not pass --dataset or --output after --. The runner sets both independently
  for every indexed entry.
- Read attack_plan/<output-root>/run_status.jsonl for exact commands,
  timestamps, output directories, exit codes, skips, and failures.
