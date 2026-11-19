from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .data_loader import ImageUnit, MontageError, file_sha256

JPEG_QUALITY = 6


def sample_video(
    video_path: str, output_dir: str, frame_interval: float = 1.0
) -> List[ImageUnit]:
    try:
        interval = float(frame_interval)
    except (TypeError, ValueError) as exc:
        raise MontageError("frame_interval must be a finite number > 0") from exc
    if not math.isfinite(interval) or interval <= 0:
        raise MontageError("frame_interval must be a finite number > 0")
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise MontageError("Video file not found: %s" % source)
    executable = shutil.which("ffmpeg")
    if not executable:
        raise MontageError(
            "Video sampling requires FFmpeg; install ffmpeg and make it available on PATH"
        )

    root = Path(output_dir).expanduser().resolve()
    pool = root / "candidate_frames"
    if pool.is_symlink() or (pool.exists() and not pool.is_dir()):
        raise MontageError("Candidate frame output must be a directory: %s" % pool)
    frame_filter = (
        "setpts=PTS-STARTPTS,fps=fps=1/%.17g:start_time=0:round=up:eof_action=pass"
        % interval
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sampling-", dir=str(root)) as temporary:
            staged = Path(temporary) / "candidate_frames"
            staged.mkdir()
            completed = subprocess.run(
                [
                    executable, "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
                    "-vf", frame_filter, "-fps_mode", "passthrough",
                    "-c:v", "mjpeg", "-q:v", str(JPEG_QUALITY),
                    "-threads", "1", str(staged / "frame_%06d.jpg"),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                check=False,
            )
            if completed.returncode:
                raise MontageError(
                    "Cannot read/sample video %s: %s"
                    % (source, completed.stderr.strip()[-2000:])
                )
            images = sorted(staged.glob("frame_*.jpg"))
            if not images:
                raise MontageError("Video sampling extracted zero frames: %s" % source)
            units = []
            frames = []
            for index, image in enumerate(images):
                timestamp = index * interval
                digest = file_sha256(image)
                destination = pool / image.name
                unit = ImageUnit(
                    unit_id=image.stem,
                    image_paths=[str(destination)],
                    image_sha256=[digest],
                    timestamp_seconds=timestamp,
                    source_video=str(source),
                )
                units.append(unit)
                frames.append({
                    "frame_id": unit.unit_id,
                    "timestamp_seconds": timestamp,
                    "image_path": str(destination.relative_to(root)),
                    "image_sha256": digest,
                    "source_video": str(source),
                })
            if pool.exists():
                shutil.rmtree(str(pool))
            staged.rename(pool)
            metadata = {
                "source_video": str(source),
                "frame_interval_seconds": interval,
                "frame_count": len(units),
                "sampler": "ffmpeg_fps",
                "image_format": "jpeg",
                "jpeg_quality": JPEG_QUALITY,
                "sampling_filter": frame_filter,
                "timestamp_reference": "uniform sampling grid from video start",
                "frames": frames,
            }
            metadata_path = root / "sampling_metadata.json"
            staged_metadata = Path(temporary) / "sampling_metadata.json"
            staged_metadata.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            staged_metadata.replace(metadata_path)
            return units
    except OSError as exc:
        raise MontageError("Cannot sample video %s: %s" % (source, exc)) from exc
