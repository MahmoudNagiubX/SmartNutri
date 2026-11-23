from __future__ import annotations

import io
import ipaddress
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from smartnutri_backend.schemas import ImageQuality


class CameraUnavailable(RuntimeError):
    pass


class ImageQualityError(RuntimeError):
    def __init__(self, message: str, quality: ImageQuality | None = None):
        super().__init__(message)
        self.quality = quality


@dataclass(frozen=True)
class CapturedImage:
    image_bytes: bytes
    source: str
    simulated: bool
    captured_at: str
    quality: ImageQuality
    quality_score: float


@dataclass(frozen=True)
class InspectedImage:
    image_bytes: bytes
    quality: ImageQuality
    score: float


class CameraService(ABC):
    source = "unknown"
    simulated = False

    def __init__(
        self,
        *,
        max_image_bytes: int = 8 * 1024 * 1024,
        min_width: int = 320,
        min_height: int = 240,
        max_dimension: int = 4096,
        min_brightness: float = 18.0,
        max_brightness: float = 238.0,
        min_edge_variance: float = 4.0,
        max_clipped_fraction: float = 0.65,
        capture_retries: int = 2,
        burst_count: int = 1,
        timeout_s: float = 4.0,
    ) -> None:
        if max_image_bytes < 1 or min_width < 1 or min_height < 1 or max_dimension < 1:
            raise ValueError("image dimensions and byte limit must be positive")
        if capture_retries < 0 or burst_count < 1 or timeout_s <= 0:
            raise ValueError("camera retries, burst count, or timeout are invalid")
        self.max_image_bytes = max_image_bytes
        self.min_width = min_width
        self.min_height = min_height
        self.max_dimension = max_dimension
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_edge_variance = min_edge_variance
        self.max_clipped_fraction = max_clipped_fraction
        self.capture_retries = capture_retries
        self.burst_count = burst_count
        self.timeout_s = timeout_s

    @abstractmethod
    def capture_raw(self) -> bytes:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {"configured": True, "source": self.source, "simulated": self.simulated}

    def capture_best(self) -> CapturedImage:
        best: InspectedImage | None = None
        last_quality: ImageQuality | None = None
        capture_error: Exception | None = None
        for _attempt in range(self.capture_retries + 1):
            batch: list[InspectedImage] = []
            for _frame in range(self.burst_count):
                try:
                    raw = self.capture_raw()
                    inspected = self._inspect(raw)
                    last_quality = inspected.quality
                    if inspected.quality.acceptable:
                        batch.append(inspected)
                except ImageQualityError as exc:
                    last_quality = exc.quality
                    capture_error = exc
                except (CameraUnavailable, OSError, TimeoutError, urllib.error.URLError) as exc:
                    capture_error = exc
            if batch:
                candidate = max(batch, key=lambda item: item.score)
                if best is None or candidate.score > best.score:
                    best = candidate
                # A full burst was inspected; do not keep capturing after usable frames exist.
                break

        if best is None:
            if last_quality is not None:
                details = ", ".join(last_quality.reasons) or "image quality was unacceptable"
                raise ImageQualityError(details, last_quality)
            if isinstance(capture_error, ImageQualityError):
                raise capture_error
            raise CameraUnavailable(
                f"camera capture failed ({type(capture_error).__name__ if capture_error else 'no frames'})"
            )
        return CapturedImage(
            image_bytes=best.image_bytes,
            source=self.source,
            simulated=self.simulated,
            captured_at=datetime.now(timezone.utc).isoformat(),
            quality=best.quality,
            quality_score=best.score,
        )

    def _inspect(self, raw: bytes) -> InspectedImage:
        if not isinstance(raw, bytes) or not raw:
            raise ImageQualityError("camera returned an empty image")
        if len(raw) > self.max_image_bytes:
            raise ImageQualityError("image exceeds the configured byte limit")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                image_format = (opened.format or "").upper()
                if image_format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageQualityError("image format must be JPEG, PNG, or WEBP")
                width, height = opened.size
                if width < self.min_width or height < self.min_height:
                    raise ImageQualityError("image resolution is below the configured minimum")
                if max(width, height) > self.max_dimension:
                    raise ImageQualityError("image dimension exceeds the configured maximum")
                if width < 5 or height < 5:
                    raise ImageQualityError("image dimensions are too small for quality analysis")
                reasons: list[str] = []
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except ImageQualityError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ImageQualityError("camera returned an invalid image") from exc

        width, height = image.size
        gray = image.convert("L")
        stats = ImageStat.Stat(gray)
        brightness = float(stats.mean[0])
        histogram = gray.histogram()
        clipped_fraction = sum(histogram[:6] + histogram[250:]) / float(width * height)
        edges = gray.filter(ImageFilter.FIND_EDGES).crop((2, 2, width - 2, height - 2))
        edge_variance = float(ImageStat.Stat(edges).var[0])
        if brightness < self.min_brightness:
            reasons.append("image_underexposed")
        if brightness > self.max_brightness:
            reasons.append("image_overexposed")
        if clipped_fraction > self.max_clipped_fraction:
            reasons.append("too_many_clipped_pixels")
        if edge_variance < self.min_edge_variance:
            reasons.append("image_blurry_or_textureless")

        quality = ImageQuality(
            acceptable=not reasons,
            reasons=reasons,
            width=width,
            height=height,
            format=image_format,
            byte_count=len(raw),
            brightness_mean=brightness,
            clipped_pixel_fraction=clipped_fraction,
            edge_variance=edge_variance,
        )
        score = edge_variance / (1.0 + clipped_fraction)
        score -= abs(brightness - 128.0) / 255.0
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92, optimize=True)
        normalized = output.getvalue()
        if len(normalized) > self.max_image_bytes:
            raise ImageQualityError("normalized image exceeds the configured byte limit", quality)
        return InspectedImage(normalized, quality, score)


class MockCameraService(CameraService):
    source = "mock"
    simulated = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def capture_raw(self) -> bytes:
        image = Image.new("RGB", (640, 480), (222, 220, 211))
        draw = ImageDraw.Draw(image)
        draw.ellipse((90, 40, 550, 440), fill=(242, 239, 229), outline=(85, 78, 62), width=8)
        draw.rounded_rectangle((160, 120, 310, 320), radius=26, fill=(220, 184, 95))
        draw.ellipse((315, 120, 475, 290), fill=(143, 83, 50))
        draw.polygon([(210, 330), (325, 315), (370, 385), (230, 405)], fill=(211, 164, 71))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=94)
        return output.getvalue()


class FileCameraService(CameraService):
    source = "file"
    simulated = False

    def __init__(self, image_path: str | Path, **kwargs: Any) -> None:
        self.image_path = Path(image_path).expanduser().resolve()
        super().__init__(**kwargs)

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "file_available": self.image_path.is_file(),
        }

    def capture_raw(self) -> bytes:
        if not self.image_path.is_file():
            raise CameraUnavailable("configured camera image file is unavailable")
        if self.image_path.stat().st_size > self.max_image_bytes:
            raise ImageQualityError("configured image file exceeds the byte limit")
        return self.image_path.read_bytes()


class IPSnapshotCameraService(CameraService):
    source = "ip_snapshot"
    simulated = False

    def __init__(self, snapshot_url: str, **kwargs: Any) -> None:
        self.snapshot_url = snapshot_url
        self._validate_url(snapshot_url)
        super().__init__(**kwargs)

    @staticmethod
    def _validate_url(snapshot_url: str) -> None:
        parsed = urllib.parse.urlsplit(snapshot_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("camera URL must be an HTTP(S) snapshot URL")
        if parsed.username or parsed.password:
            raise ValueError("camera URL credentials must not be embedded in the URL")
        host = parsed.hostname.casefold()
        is_lan_host = host in {"localhost", "localhost.localdomain"} or host.endswith(".local")
        try:
            is_lan_host = is_lan_host or ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if not is_lan_host:
            raise ValueError("camera snapshot URL must point to a private LAN or .local host")

    def status(self) -> dict[str, Any]:
        return {**super().status(), "transport": "http_snapshot", "reachable": None}

    def capture_raw(self) -> bytes:
        request = urllib.request.Request(
            self.snapshot_url,
            headers={"Accept": "image/jpeg,image/png,image/webp", "User-Agent": "SmartNutri/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(declared_size) > self.max_image_bytes:
                    raise ImageQualityError("camera response exceeds the byte limit")
                raw = response.read(self.max_image_bytes + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CameraUnavailable("phone camera snapshot request failed") from exc
        if len(raw) > self.max_image_bytes:
            raise ImageQualityError("camera response exceeds the byte limit")
        return raw


def save_captured_image(image: CapturedImage, image_dir: str | Path, scan_id: str) -> Path:
    """Atomically save a single selected frame under the local runtime data directory."""
    directory = Path(image_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(character for character in scan_id if character.isalnum() or character == "-")
    if not safe_id:
        raise ValueError("scan_id does not contain a safe filename component")
    target = directory / f"{safe_id}-capture.jpg"
    temporary = target.with_suffix(".jpg.tmp")
    temporary.write_bytes(image.image_bytes)
    temporary.replace(target)
    return target
