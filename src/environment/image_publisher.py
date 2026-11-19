from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

from PIL import Image, ImageDraw, ImageFont

from ..production.director import MontagePlan
from ..utils.data_loader import MontageError, file_sha256


class ImagePublisher:
    def materialize(self, plan: MontagePlan, output_dir: str) -> Dict[str, Any]:
        root = Path(output_dir).expanduser().resolve()
        sequence_root = root / "sequence"
        if sequence_root.exists():
            shutil.rmtree(str(sequence_root))
        victim_root = sequence_root / "victim_images"
        victim_root.mkdir(parents=True, exist_ok=True)

        lookup = {unit.unit_id: unit for unit in plan.dataset.image_units}
        publication = {
            item["unit_id"]: item for item in plan.publication_plan
        }
        private_items = []
        flat_position = 0
        for item in plan.caption_sequence.units:
            unit = lookup[item["unit_id"]]
            for frame_index, image_value in enumerate(unit.image_paths, start=1):
                flat_position += 1
                source = Path(image_value)
                expected_hash = unit.image_sha256[frame_index - 1]
                modality = item["modality"]
                caption = item["caption"] if modality in ("t", "vt") else ""
                subtitle_font_size = None
                victim_image = None
                rendered_hash = None
                if modality == "t":
                    media_type = "text"
                elif modality == "vt":
                    media_type = "image"
                    victim_image = victim_root / (
                        "frame_%04d.png" % flat_position
                    )
                    subtitle_font_size = self._render_captioned_image(
                        source, victim_image, caption
                    )
                else:
                    media_type = "image"
                    suffix = source.suffix.lower() or ".bin"
                    victim_image = victim_root / (
                        "frame_%04d%s" % (flat_position, suffix)
                    )
                    shutil.copy2(str(source), str(victim_image))
                    if file_sha256(victim_image) != expected_hash:
                        raise MontageError(
                            "Materialized visual-only image hash mismatch"
                        )
                if victim_image is not None:
                    rendered_hash = file_sha256(victim_image)
                private_items.append(
                    {
                        "feed_item_id": "montage_%04d" % flat_position,
                        "sequence_position": flat_position,
                        "publisher_id": publication[unit.unit_id]["account_id"],
                        "display_timestamp": publication[unit.unit_id]["timestamp"],
                        "media_type": media_type,
                        "image_path": (
                            str(victim_image) if victim_image is not None else None
                        ),
                        "unit_id": unit.unit_id,
                        "source_frame_id": unit.unit_id,
                        "source_image_path": str(source),
                        "source_image_sha256": expected_hash,
                        "rendered_image_sha256": rendered_hash,
                        "modality": modality,
                        "caption": caption or None,
                        "image_withheld": modality == "t",
                        "subtitle_burned": modality == "vt",
                        "subtitle_font_size": subtitle_font_size,
                    }
                )

        private_feed = {
            "schema_version": "visual_montage.captioned_image_sequence.v2",
            "target": plan.dataset.target.to_dict(),
            "agent_models": plan.agent_models,
            "total_items": len(private_items),
            "items": private_items,
            "research_only": True,
            "publication_allowed": False,
        }

        private_path = sequence_root / "captioned_image_sequence.json"
        private_path.write_text(
            json.dumps(private_feed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_feed_html(
            sequence_root / "index.html",
            private_feed,
            "Captioned montage audit",
        )
        return {
            "root": str(sequence_root),
            "captioned_image_sequence": str(private_path),
            "victim_images": str(victim_root),
            "audit_overview": str(sequence_root / "index.html"),
            "item_count": len(private_items),
        }

    @classmethod
    def _render_captioned_image(
        cls, source: Path, destination: Path, caption: str
    ) -> int:
        try:
            with Image.open(str(source)) as opened:
                image = opened.convert("RGBA")
        except OSError as exc:
            raise MontageError("Cannot open source image %s: %s" % (source, exc))

        width, height = image.size
        font_size = max(12, min(24, int(round(width * 0.022))))
        font = cls._load_font(font_size)
        draw = ImageDraw.Draw(image)
        horizontal_margin = max(6, int(round(width * 0.04)))
        vertical_margin = max(5, int(round(height * 0.025)))
        padding_x = max(7, int(round(width * 0.012)))
        padding_y = max(5, int(round(height * 0.012)))
        maximum_text_width = max(20, width - 2 * (horizontal_margin + padding_x))
        lines = cls._wrap_caption(draw, caption, font, maximum_text_width, 3)
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_heights = [box[3] - box[1] for box in line_boxes]
        line_gap = max(2, font_size // 5)
        text_height = sum(line_heights) + line_gap * (len(lines) - 1)
        text_width = max(box[2] - box[0] for box in line_boxes)
        box_width = text_width + 2 * padding_x
        box_height = text_height + 2 * padding_y
        left = max(horizontal_margin, (width - box_width) // 2)
        right = min(width - horizontal_margin, left + box_width)
        top = max(vertical_margin, height - vertical_margin - box_height)
        bottom = min(height - vertical_margin, top + box_height)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=max(3, font_size // 3),
            fill=(0, 0, 0, 178),
        )
        text_y = top + padding_y
        for line, line_height, box in zip(lines, line_heights, line_boxes):
            line_width = box[2] - box[0]
            text_x = (width - line_width) // 2
            overlay_draw.text(
                (text_x, text_y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 220),
            )
            text_y += line_height + line_gap

        rendered = Image.alpha_composite(image, overlay).convert("RGB")
        try:
            rendered.save(str(destination), format="PNG", optimize=True)
        except OSError as exc:
            raise MontageError(
                "Cannot save captioned image %s: %s" % (destination, exc)
            )
        return font_size

    @staticmethod
    def _load_font(font_size: int):
        candidates = [
            os.getenv("MONTAGE_SUBTITLE_FONT"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                try:
                    return ImageFont.truetype(candidate, font_size)
                except OSError:
                    continue
        return ImageFont.load_default()

    @classmethod
    def _wrap_caption(
        cls,
        draw: ImageDraw.ImageDraw,
        caption: str,
        font,
        maximum_width: int,
        maximum_lines: int,
    ) -> List[str]:
        words = caption.split()
        lines: List[str] = []
        current = ""
        while words:
            word = words.pop(0)
            candidate = (current + " " + word).strip()
            if cls._text_width(draw, candidate, font) <= maximum_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = word
            else:
                current = cls._fit_text(draw, word, font, maximum_width)
            if len(lines) == maximum_lines - 1:
                remainder = " ".join([current] + words).strip()
                lines.append(
                    cls._fit_text(draw, remainder, font, maximum_width, True)
                )
                return lines
        if current:
            lines.append(current)
        return lines or [""]

    @staticmethod
    def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0]

    @classmethod
    def _fit_text(
        cls,
        draw: ImageDraw.ImageDraw,
        text: str,
        font,
        maximum_width: int,
        add_ellipsis: bool = False,
    ) -> str:
        suffix = "..." if add_ellipsis else ""
        result = text
        while result and cls._text_width(
            draw, result.rstrip() + suffix, font
        ) > maximum_width:
            result = result[:-1]
        return result.rstrip() + suffix

    @staticmethod
    def _write_feed_html(
        path: Path, feed: Mapping[str, Any], title: str
    ) -> None:
        rows = []
        for item in feed.get("items", []):
            caption = item.get("caption") or ""
            image_html = ""
            if item.get("image_path"):
                image = Path(str(item["image_path"]))
                try:
                    relative = image.relative_to(path.parent)
                except ValueError:
                    relative = image
                image_html = '<img src="%s" alt="feed item">' % html.escape(
                    str(relative)
                )
            rows.append(
                '<article>%s<p>%s</p></article>'
                % (image_html, html.escape(str(caption)))
            )
        document = (
            '<!doctype html><html><head><meta charset="utf-8"><title>%s</title>'
            "<style>body{max-width:960px;margin:24px auto;font-family:sans-serif}"
            "article{margin:0 0 24px}img{max-width:100%%;height:auto}</style></head>"
            "<body><h1>%s</h1>%s</body></html>"
            % (html.escape(title), html.escape(title), "".join(rows))
        )
        path.write_text(document, encoding="utf-8")
