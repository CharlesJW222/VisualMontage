from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from src.production.director import MontagePlan


SCHEMA_VERSION = "visual_montage.attack_plan_provenance.v1"


def _number(value: Any) -> Optional[float]:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _path_record(path: Any, dataset_root: str) -> Dict[str, Optional[str]]:
    if path is None:
        return {"absolute_path": None, "dataset_relative_path": None}
    value = Path(str(path)).expanduser().resolve()
    try:
        relative = str(value.relative_to(Path(dataset_root).expanduser().resolve()))
    except ValueError:
        relative = None
    return {"absolute_path": str(value), "dataset_relative_path": relative}


def _decision_record(decision: Any) -> Dict[str, Any]:
    return {
        "action": decision.action,
        "local_compliance": decision.local_compliance,
        "target_belief": _number(decision.target_belief),
        "auditor_beliefs": [_number(value) for value in decision.auditor_beliefs],
        "singleton_leakage": _number(decision.singleton_leakage),
        "modality_loads": {
            name: _number(value)
            for name, value in sorted((decision.modality_loads or {}).items())
        },
        "lc_components": dict(decision.lc_components or {}),
        "revise_owner": decision.revise_owner or None,
        "spurious_edges": list(decision.spurious_edges or []),
        "violations": list(decision.violations or []),
        "critique": decision.critique,
        "agent_id": decision.agent_id,
        "model": decision.model,
    }


def _mode_meaning(modality: str) -> str:
    return {
        "v": "Image only: the original image is published without a caption.",
        "t": "Text only: the original image is withheld and only the caption for this frame is published.",
        "vt": "Image and text: the image is published with the caption for this frame rendered onto it.",
    }.get(modality, "Unknown modality.")


def _compact_history(history: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in history:
        phase = str(row.get("phase") or "")
        if phase not in ("screenwriter", "visual_editor", "caption_editor"):
            continue
        draft = row.get("draft") or {}
        director = row.get("director") or {}
        compact.append(
            {
                "phase": phase,
                "round": row.get("round"),
                "candidate": row.get("candidate"),
                "order": list(draft.get("ordered_unit_ids") or []),
                "action": director.get("action"),
                "target_belief": _number(director.get("target_belief")),
                "singleton_leakage": _number(director.get("singleton_leakage")),
                "search_score": _number(row.get("search_score")),
                "pruned": row.get("pruned"),
                "invalid_output": row.get("invalid_output"),
                "duplicate_order": row.get("duplicate_order"),
                "critique": director.get("critique"),
            }
        )
    return compact


def build_attack_plan_provenance(
    plan: "MontagePlan",
    output_dir: str,
    *,
    materialized: Optional[Mapping[str, Any]] = None,
    distributed_publication: Optional[Mapping[str, Any]] = None,
    victim_validation: Optional[Any] = None,
) -> Dict[str, Any]:
    dataset = plan.dataset
    root = Path(output_dir).expanduser().resolve()
    units = {unit.unit_id: unit for unit in dataset.image_units}
    captions = {item["unit_id"]: dict(item) for item in plan.caption_sequence.units}
    publication = {item["unit_id"]: dict(item) for item in plan.publication_plan}
    materialization_record = dict(materialized or {"skipped": True})
    materialized_items = materialization_record.get("items") or []
    sequence_path = materialization_record.get("captioned_image_sequence")
    if not materialized_items and sequence_path:
        try:
            payload = json.loads(
                Path(str(sequence_path)).read_text(encoding="utf-8")
            )
            materialized_items = list(payload.get("items") or [])
        except (OSError, ValueError, TypeError):
            materialized_items = []
    output_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in materialized_items:
        unit_id = str(item.get("unit_id") or item.get("source_frame_id") or "")
        if unit_id:
            output_items.setdefault(unit_id, []).append(dict(item))

    selected_ids = list(plan.visual_sequence.ordered_unit_ids)
    sequence = []
    for position, unit_id in enumerate(selected_ids, start=1):
        unit = units[unit_id]
        source_images = [
            {**_path_record(path, dataset.root), "sha256": image_hash}
            for path, image_hash in zip(unit.image_paths, unit.image_sha256)
        ]
        caption = captions[unit_id]
        mode = caption["modality"]
        rendered = []
        for item in output_items.get(unit_id, []):
            rendered.append(
                {
                    "feed_item_id": item.get("feed_item_id"),
                    "sequence_position": item.get("sequence_position"),
                    "image_path": item.get("image_path"),
                    "rendered_image_sha256": item.get("rendered_image_sha256"),
                    "image_withheld": item.get("image_withheld"),
                    "subtitle_burned": item.get("subtitle_burned"),
                    "subtitle_font_size": item.get("subtitle_font_size"),
                }
            )
        sequence.append(
            {
                "sequence_position": position,
                "unit_id": unit_id,
                "timestamp_seconds": getattr(unit, "timestamp_seconds", None),
                "source_video": getattr(unit, "source_video", None),
                "source_images": source_images,
                "publication_schedule": publication[unit_id],
                "relational_beliefs": {
                    "status": "not_used",
                    "singleton_belief": None,
                    "edge_from_previous": None,
                    "edge_to_next": None,
                },
                "visual_selection": {
                    "selected_order": position,
                    "sequencing_strategy": plan.visual_sequence.sequencing_strategy,
                },
                "final_modality_and_caption": {
                    "modality": mode,
                    "caption": caption.get("caption") or "",
                    "mode_meaning": _mode_meaning(mode),
                    "caption_strategy": plan.caption_sequence.caption_strategy,
                    "per_item_model_rationale": None,
                    "rationale_note": (
                        "The current agent output contract neither requires nor stores a per-frame modality rationale. This record "
                        "keeps only the final decision and the global caption strategy and does not invent explanations that were never generated."
                    ),
                },
                "materialized_output": rendered,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Read-only provenance record built from the existing plan output; it calls no model and does not change "
            "frame selection, scoring, gating or publication results."
        ),
        "attack_plan": {
            "output_directory": str(root),
            "montage_plan_path": str(root / "montage_plan.json"),
            "dataset_id": dataset.dataset_id,
            "dataset_root": dataset.root,
            "target": dataset.target.to_dict(),
            "settings": dataset.settings.to_dict(),
            "agent_models": plan.agent_models,
        },
        "selection_score": {
            "status": "not_used",
            "formula": None,
            "selected_adjacent_omega_sum": None,
            "alpha": None,
            "alpha_times_continuity_penalty": None,
            "beta": None,
            "selected_frame_count": len(selected_ids),
            "beta_times_selected_frame_count": None,
            "computed_search_score": None,
            "selected_edges": [],
        },
        "final_sequence": sequence,
        "stage_gates": {
            "screenwriter": _decision_record(plan.screenplay_decision),
            "visual_editor": _decision_record(plan.visual_sequence_decision),
            "caption_editor": _decision_record(plan.caption_sequence_decision),
        },
        "caption_editor": {
            "caption_strategy": plan.caption_sequence.caption_strategy,
            "final_gate": _decision_record(plan.caption_sequence_decision),
            "trace_limit": (
                "The current agent output contract does not produce per-frame modality explanations. The exporter records the final "
                "modality and the global caption strategy verbatim."
            ),
        },
        "candidate_trace": _compact_history(plan.debate_history),
        "full_debate_history": list(plan.debate_history),
        "materialization": materialization_record,
        "distributed_publication": dict(
            distributed_publication or {"skipped": True}
        ),
        "victim_validation": (
            victim_validation
            if victim_validation is not None
            else {"status": "not_run_or_not_available"}
        ),
    }


def _escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _markdown(provenance: Mapping[str, Any]) -> str:
    plan = provenance["attack_plan"]
    score = provenance["selection_score"]
    lines = [
        "# Attack Plan Provenance Record",
        "",
        "This report only records existing pipeline output; it calls no model and does not change the attack plan.",
        "",
        "## Run summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Dataset ID | %s |" % _escape(plan["dataset_id"]),
        "| Target relation | %s |" % _escape(plan["target"]["label"]),
        "| Output directory | `%s` |" % plan["output_directory"],
        "| Target threshold tau | %s |" % plan["settings"]["tau"],
        "| Singleton-leakage threshold epsilon | %s |" % plan["settings"]["epsilon"],
        "| Selected frames | %s |" % score["selected_frame_count"],
        "| Computed selection score | %s |" % (
            score["computed_search_score"]
            if score["computed_search_score"] is not None else "not_used"
        ),
        "",
        "## Per-item provenance of the final sequence",
        "",
        "| Position | Frame ID | Modality | Caption | Singleton belief | Previous-hop omega | Source SHA 256 |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in provenance["final_sequence"]:
        source = row["source_images"][0] if row["source_images"] else {}
        incoming = row["relational_beliefs"]["edge_from_previous"] or {}
        final = row["final_modality_and_caption"]
        lines.append(
            "| %d | %s | %s | %s | %s | %s | `%s` |"
            % (
                row["sequence_position"],
                _escape(row["unit_id"]),
                final["modality"],
                _escape(final["caption"]),
                row["relational_beliefs"]["singleton_belief"],
                incoming.get("omega", "—"),
                source.get("sha256", ""),
            )
        )
    lines.extend(["", "### Detailed record per item", ""])
    for row in provenance["final_sequence"]:
        source = row["source_images"][0] if row["source_images"] else {}
        relation = row["relational_beliefs"]
        final = row["final_modality_and_caption"]
        lines.extend(
            [
                "#### %02d %s" % (row["sequence_position"], row["unit_id"]),
                "",
                "- **Source image:** `%s`"
                % (source.get("dataset_relative_path") or source.get("absolute_path") or "not available"),
                "- **Source SHA 256:** `%s`" % source.get("sha256", "not available"),
                "- **Source video:** %s" % (row["source_video"] or "not available"),
                "- **Sampling time (seconds):** %s" % (
                    row["timestamp_seconds"]
                    if row["timestamp_seconds"] is not None else "not available"
                ),
                "- **Singleton belief:** %s" % relation["singleton_belief"],
                "- **Edge from the previous frame:** %s" % json.dumps(relation["edge_from_previous"], ensure_ascii=False),
                "- **Edge to the next frame:** %s" % json.dumps(relation["edge_to_next"], ensure_ascii=False),
                "- **Final modality:** `%s` — %s"
                % (final["modality"], final["mode_meaning"]),
                "- **Final caption:** %s" % (final["caption"] or "_none_"),
                "- **Modality record note:** %s" % final["rationale_note"],
                "- **Materialized output:** %s" % json.dumps(row["materialized_output"], ensure_ascii=False),
                "",
            ]
        )
    lines.extend([
        "## Selection score",
        "",
        "not_used: this pipeline run computed no singleton, pair or sequence-search score. Frame selection and order were proposed by the Visual Editor and reviewed by the Director.",
        "",
    ])
    lines.extend(
        [
            "## Director gate results",
            "",
            "| Stage | Action | Target belief | Singleton leakage | Revision owner | Critique |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for stage, decision in provenance["stage_gates"].items():
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                stage,
                decision["action"],
                decision["target_belief"],
                decision["singleton_leakage"],
                _escape(decision["revise_owner"] or "none"),
                _escape(decision["critique"]),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate history",
            "",
            "| Stage | Round | Candidate | Order | Action | belief | Leakage | Selection score |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in provenance["candidate_trace"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                item["phase"],
                item["round"] or "",
                item["candidate"] or "",
                _escape(", ".join(item["order"])),
                item["action"] or "",
                item["target_belief"] if item["target_belief"] is not None else "",
                item["singleton_leakage"] if item["singleton_leakage"] is not None else "",
                item["search_score"] if item["search_score"] is not None else "",
            )
        )
    lines.extend(
        [
            "",
            "## Record boundary",
            "",
            "The current Caption Editor contract records the final modality, the final caption and the global caption strategy, but does not require a separate per-frame rationale. The provenance record keeps this boundary explicit and does not invent explanations.",
            "",
            "The complete machine-readable record is `attack_plan_provenance.json` in this directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_attack_plan_provenance(
    plan: "MontagePlan",
    output_dir: str,
    *,
    materialized: Optional[Mapping[str, Any]] = None,
    distributed_publication: Optional[Mapping[str, Any]] = None,
    victim_validation: Optional[Any] = None,
) -> Dict[str, str]:
    provenance_dir = Path(output_dir).expanduser().resolve() / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    record = build_attack_plan_provenance(
        plan,
        output_dir,
        materialized=materialized,
        distributed_publication=distributed_publication,
        victim_validation=victim_validation,
    )
    json_path = provenance_dir / "attack_plan_provenance.json"
    markdown_path = provenance_dir / "attack_plan_provenance.md"
    _atomic_write(json_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(markdown_path, _markdown(record) + "\n")
    return {
        "directory": str(provenance_dir),
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
