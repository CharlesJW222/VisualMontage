# VisualMontage

Code release for the paper **"VisualMontage: Manipulating the Beliefs of
Vision-Language Agents with Authentic Image Sequences"**.

VisualMontage studies *visual cognitive collusion*: coordinated publishers
select, order, and faithfully describe **authentic, unmodified** source images
so that a vision-language agent reading the resulting feed accepts a specified
false conclusion. No pixel is edited and every caption is locally truthful; the
false belief comes from the relations the victim infers *between* records.

```text
 source video ──► candidate frames
                       │
        ┌──────────────┴───────────────────────────────────────────┐
        │  Cinematic Montage Studio (four prompt-driven VLM roles) │
        │  Screenwriter ─► Visual Editor ─► Caption Editor         │
        │        ▲              ▲                ▲                 │
        │        └────────── Director (audit / revise) ────────────┘
        └──────────────┬───────────────────────────────────────────┘
                       ▼
        Distributed publication (local simulator, separate posts)
                       ▼
        Victim validation: N victim VLMs ─► majority ─► AI Judge
                       ▼
        ASR · HC-ASR · Conf.+ · DDR · provenance record
```

| Role | Responsibility | Code |
| --- | --- | --- |
| Screenwriter | Designs the abstract relational screenplay for the target | `src/production/screenwriter.py` |
| Visual Editor | Selects and orders authentic frames (path through the relational emergence graph) | `src/production/visual_editor.py` |
| Caption Editor | Sole owner of the final `v` / `t` / `vt` mode; writes locally grounded captions | `src/production/caption_editor.py` |
| Director | Audits local compliance, target belief and singleton leakage; drives bounded revision rounds | `src/production/director.py` |
| Publisher | Deterministic local materialization and distributed-publication manifest | `src/environment/` |
| Victim / Judge | Independent downstream assessment, majority vote, AI Judge, metrics | `victim_validation.py` |

All role prompts live in one file, [`configs/agents.json`](configs/agents.json).

## Responsible use

This is a research artifact for measuring and defending against a cognitive
failure mode of vision-language agents.

- The publisher is a **local simulator**. Nothing is posted to any external
  platform; every run records `external_publication_performed: false` and
  `research_only: true`.
- Target claims are **false by construction** (`ground_truth: false`). They are
  experimental targets, never factual annotations of the source footage.
- Source media is **not redistributed** here. See [docs/DATASETS.md](docs/DATASETS.md).
- This release contains the method and its victim-side evaluation
  (construction, simulated publication, victim panel, majority vote and AI
  Judge). The scripts behind the individual tables and figures of the paper are
  not part of it.

Please use the code to evaluate and harden models, not to mislead people.

## Repository layout

```text
demo.py                      End-to-end entry point: construct → publish → validate → provenance
generate_montage_plan.py     Construction phase only (four-role production team)
victim_validation.py         Victim panel, majority vote, AI Judge, ASR / HC-ASR / DDR
distributed_publish.py       Add the local publication layer to an existing sequence
tools.py                     Config loading and production-team assembly
configs/                     agents.json (all prompts) and settings.json (models, defaults)
src/production/              Screenwriter, Visual Editor, Caption Editor, Director
src/environment/             Image publisher and distributed publisher (local simulator)
src/utils/                   Data loader, video sampler, LLM client, provenance, runtime
scripts/                     Batch runners, re-validation, macOS/Ubuntu helpers
tools/                       Project validator and flat dataset-view builder
Video_Dataset_Categories/    Corpus manifests (separate dataset package): 8 domains, 61 recordings, 620 declared targets
tests/                       Unit and integration tests
docs/                        Pipeline guide, batch-run guide, dataset notes
```

## Installation

Python 3.8 or later. Video input needs an `ffmpeg` executable on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A conda environment file (`environment.yml`) is provided as an alternative.
The optional Apple-Silicon local backend is installed with
`bash scripts/macos/local_setup.sh` (`requirements-macos-local.txt`).

## Configure model access

Every role talks to an OpenAI-compatible Chat Completions endpoint. Credentials
are read from environment variables only and are never written to outputs.

```bash
# Public OpenAI API
export MONTAGE_API_BASE='https://api.openai.com/v1'
export OPENAI_API_KEY='your key here'

# Any other OpenAI-compatible endpoint (hosted gateway, local vLLM server, …)
export MONTAGE_API_BASE='https://your-api-host/v1'
export PRIVATE_API_KEY='your key here'
export MONTAGE_API_KEY_ENV='PRIVATE_API_KEY'
```

[`.env.example`](.env.example) lists every supported variable; the code does
not read `.env` by itself, so load it with `set -a; source .env; set +a`.
Default models and sampling temperatures are in `configs/settings.json` and can
be overridden per role with `MONTAGE_*_MODEL` variables or `demo.py` flags.
Run the preflight check before a costly run:

```bash
python scripts/macos/check.py     # or scripts/ubuntu/check.py
```

## Data

The corpus is distributed as a separate dataset package. Unpack it into the
repository root so that `Video_Dataset_Categories/` sits next to `demo.py`.
`Video_Dataset_Categories/<domain>/<dataset_id>/` holds, for each recording,
`dataset.json` (the declared false targets and run settings), `metadata.json`
(source, rights, SHA-256, duration) and construction notes. The video files
themselves are not redistributed. [docs/DATASETS.md](docs/DATASETS.md)
lists every recording with its source page, sampling interval and target
counts, and explains how to place the media.

## Quick start

Construct one montage and evaluate it with five victims and an AI Judge:

```bash
python demo.py \
  --dataset Video_Dataset_Categories/04_Aviation_and_Aerospace_Operations/launch_recording \
  --target-id M00 \
  --frame-interval 2 \
  --screenwriter-model gpt-5.6-luna --visual-editor-model gpt-5.6-luna \
  --caption-editor-model gpt-5.6-luna --director-model gpt-5.6-luna \
  --enable-victim-validation --num-victims 5 --num-victim-conclusions 1 \
  --victim-model gpt-5.6-luna --judge-model gpt-5.4 \
  --target-ground-truth false \
  --api-timeout-seconds 900 --api-max-retries 6 \
  --output attack_plan/launch_recording/M00
```

Common variations:

```bash
# Every declared target of one recording (one sub-directory per target)
python scripts/run_multi_target.py \
  --dataset Video_Dataset_Categories/04_Aviation_and_Aerospace_Operations/launch_recording \
  --model gpt-5.6-luna

# Construction only, no victim calls
python demo.py --dataset <dataset_dir> --target-id M00 --output attack_plan/example

# Re-evaluate an existing feed with a different victim (no reconstruction)
python victim_validation.py \
  --sequence attack_plan/launch_recording/M00/sequence/captioned_image_sequence.json \
  --output attack_plan/launch_recording/M00/victim_validation_other.json \
  --target-ground-truth false --num-victims 5 --num-conclusions 1 \
  --victim-model <victim> --judge-model <judge>

# Preview the commands for the whole corpus without calling a model
python tools/build_flat_dataset_view.py
python scripts/run_all_datasets.py --dry-run
```

Useful `demo.py` behaviour:

- `--target-id` selects one target; without it every declared target runs in
  its own sub-directory. Legacy single-`target` manifests are still accepted.
- `--video` / `--frame-interval` sample the candidate pool deterministically
  (`frame_000001`, …). The video is decoded once per invocation and the full
  pool is sent to the Screenwriter and Visual Editor with no silent truncation,
  so choose an interval that fits the provider's image limit.
- `--min-output-frames` / `--max-output-frames` bound the sequence length *K*
  chosen by the Visual Editor (default 5–30).
- The stage order is Screenwriter → Director → Visual Editor → Director →
  Caption Editor → Director, each stage revising until acceptance or its round
  limit.

## Outputs

Each target directory (`attack_plan/<dataset_id>/<target_id>/`) contains:

| File | Content |
| --- | --- |
| `montage_plan.json` | Accepted screenplay, ordered frames, captions and modes, Director gate metrics, agent history |
| `sequence/captioned_image_sequence.json` | The victim-visible feed (order, publisher, timestamp, image, caption) |
| `sequence/distributed_publication.json`, `sequence/index.html` | Simulated publication manifest and a browsable preview |
| `sequence/victim_images/` | Materialized images exactly as shown to the victim |
| `candidate_frames/`, `sampling_metadata.json` | Sampled pool with timestamps and hashes (video mode) |
| `victim_validation.json` | Victim responses, majority, Judge adjudication, ASR / HC-ASR / DDR |
| `api_run_stats.json`, `run_progress.json` | Call, token and retry accounting; live progress snapshot |
| `provenance/` | Human- and machine-readable provenance of the run |

`attack_plan/` and all other generated directories are ignored by Git.

## Reproducing the main results

| Paper | What | How |
| --- | --- | --- |
| Table 2, VisualMontage columns | ASR, HC-ASR and Conf.+ per victim | Build the cohort with `demo.py` (or `scripts/run_all_datasets.py`), then evaluate each feed with `victim_validation.py` (five victims, one conclusion, threshold 0.8) |
| Table 9 | Corpus statistics | `Video_Dataset_Categories/` manifests and [docs/DATASETS.md](docs/DATASETS.md) |
| Fig. 10–14 | Prompts of the four roles | `configs/agents.json` |

Model outputs are stochastic and hosted model snapshots change, so a rerun
reproduces the protocol and the qualitative findings rather than bit-identical
numbers. The reported results were computed on a frozen cohort of 544 generated
feeds.

## Tests

```bash
python -m unittest discover -s tests -v
```

Of the 89 tests, 85 pass on a fresh copy. Two tests in
`test_run_all_datasets.py` and one in `test_validate_project.py` need the
dataset package, the flat view (`python tools/build_flat_dataset_view.py`) and
the source videos. The fourth is the first item under
[Known issues](#known-issues).

## Documentation

| Topic | Document |
| --- | --- |
| Pipeline guide | [docs/GUIDE.md](docs/GUIDE.md) |
| Batch runs over indexed datasets | [docs/RUN_ALL_DATASETS.md](docs/RUN_ALL_DATASETS.md) |
| Datasets and media | [docs/DATASETS.md](docs/DATASETS.md) |

## Known issues

- `tests/test_production_team.py::test_visual_accept_normalizes_conflict_and_stops_without_beam`
  fails with `Caption Editor must preserve the Visual Editor frame order
  exactly`: the test fixture and the current order check disagree.
- Seven per-recording `README.md` files still show an earlier wording of the
  target list. `dataset.json` is always the authoritative target source.

## License and citation

A license file will be added later. Until then the code is provided for peer
review only. Citation details will be added after the review period.
