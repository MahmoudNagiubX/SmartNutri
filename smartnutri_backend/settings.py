from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def _default_data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home()
    return root / "SmartNutri"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().casefold() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().casefold() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=_default_data_dir)
    scale_mode: str = "real"
    scale_base_url: str | None = None
    camera_mode: str = "mock"
    camera_snapshot_url: str | None = None
    camera_allowed_host: str | None = None
    camera_file_path: Path | None = None
    segmentation_mode: str = "mock"
    checkpoint_dir: Path | None = None
    foodseg_root: Path | None = None
    preprocessing_config_path: Path = Path(__file__).resolve().parents[1] / "configs" / "image_processing" / "baseline.json"
    control_token: str | None = field(default=None, repr=False)
    scale_device_token: str | None = field(default=None, repr=False)
    cors_origins: tuple[str, ...] = ()
    auto_scan_enabled: bool = False
    scale_poll_interval_s: float = 0.25
    placement_threshold_g: float = 20.0
    removal_threshold_g: float = 10.0
    stable_confirmations: int = 2
    removal_confirmations: int = 3
    post_stability_delay_s: float = 1.0
    weight_change_tolerance_g: float = 5.0
    stale_reading_timeout_s: float = 2.0
    correction_weight_tolerance_g: float = 0.05
    camera_timeout_s: float = 4.0
    capture_retries: int = 2
    capture_burst_count: int = 1
    max_image_bytes: int = 8 * 1024 * 1024
    min_image_width: int = 320
    min_image_height: int = 240
    max_image_dimension: int = 4096
    min_brightness: float = 18.0
    max_brightness: float = 238.0
    min_edge_variance: float = 4.0
    max_clipped_pixel_fraction: float = 0.65
    min_classification_confidence: float = 0.55
    portion_review_uncertainty_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.scale_mode not in {"real", "mock", "disabled"}:
            raise ValueError("scale_mode must be real, mock, or disabled")
        if self.camera_mode not in {"mock", "file", "ip_snapshot"}:
            raise ValueError("camera_mode must be mock, file, or ip_snapshot")
        if self.segmentation_mode not in {"mock", "segformer"}:
            raise ValueError("segmentation_mode must be mock or segformer")
        positive = (
            "scale_poll_interval_s", "placement_threshold_g", "removal_threshold_g",
            "camera_timeout_s", "post_stability_delay_s", "min_brightness",
            "stale_reading_timeout_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.removal_threshold_g >= self.placement_threshold_g:
            raise ValueError("removal threshold must be lower than placement threshold")
        for name in ("weight_change_tolerance_g", "correction_weight_tolerance_g"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.stable_confirmations < 1 or self.removal_confirmations < 1:
            raise ValueError("confirmation counts must be positive")
        if self.capture_retries < 0 or self.capture_burst_count < 1:
            raise ValueError("capture retries/count are invalid")
        if not 0 < self.max_image_bytes <= 32 * 1024 * 1024:
            raise ValueError("max_image_bytes must be in (0, 32 MiB]")
        if (
            self.min_image_width < 5
            or self.min_image_height < 5
            or self.max_image_dimension < max(self.min_image_width, self.min_image_height)
        ):
            raise ValueError("image dimensions are inconsistent")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("brightness limits must satisfy 0 <= min < max <= 255")
        if not math.isfinite(self.min_edge_variance) or self.min_edge_variance < 0:
            raise ValueError("min_edge_variance must be finite and non-negative")
        if not 0 <= self.max_clipped_pixel_fraction <= 1:
            raise ValueError("max_clipped_pixel_fraction must be in [0, 1]")
        if not 0 <= self.min_classification_confidence <= 1:
            raise ValueError("min_classification_confidence must be in [0, 1]")
        if not 0 <= self.portion_review_uncertainty_fraction <= 1:
            raise ValueError("portion_review_uncertainty_fraction must be in [0, 1]")
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            if "*" in origin or parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("CORS origins must be explicit HTTP(S) origins without wildcards")

    @property
    def database_path(self) -> Path:
        return self.data_dir / "smartnutri.sqlite3"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @classmethod
    def from_env(cls) -> "Settings":
        scale_url = os.environ.get("SMARTNUTRI_SCALE_URL") or None
        data_dir = Path(os.environ.get("SMARTNUTRI_DATA_DIR", str(_default_data_dir()))).expanduser()
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "SMARTNUTRI_CORS_ORIGINS",
                "",
            ).split(",")
            if origin.strip()
        )
        optional_path = lambda key: Path(os.environ[key]).expanduser() if os.environ.get(key) else None
        return cls(
            data_dir=data_dir,
            scale_mode=os.environ.get("SMARTNUTRI_SCALE_MODE", "real"),
            scale_base_url=scale_url,
            camera_mode=os.environ.get("SMARTNUTRI_CAMERA_MODE", "mock"),
            camera_snapshot_url=os.environ.get("SMARTNUTRI_CAMERA_URL") or None,
            camera_allowed_host=os.environ.get("SMARTNUTRI_CAMERA_ALLOWED_HOST") or None,
            camera_file_path=optional_path("SMARTNUTRI_CAMERA_FILE"),
            segmentation_mode=os.environ.get("SMARTNUTRI_CV_MODE", "mock"),
            checkpoint_dir=optional_path("SMARTNUTRI_CHECKPOINT_DIR"),
            foodseg_root=optional_path("SMARTNUTRI_FOODSEG_ROOT"),
            control_token=os.environ.get("SMARTNUTRI_CONTROL_TOKEN") or None,
            scale_device_token=os.environ.get("SMARTNUTRI_SCALE_DEVICE_TOKEN") or None,
            cors_origins=origins,
            auto_scan_enabled=_env_bool("SMARTNUTRI_AUTO_SCAN", default=False),
            scale_poll_interval_s=float(os.environ.get("SMARTNUTRI_SCALE_POLL_INTERVAL", "0.25")),
            placement_threshold_g=float(os.environ.get("SMARTNUTRI_PLACEMENT_THRESHOLD_G", "20")),
            removal_threshold_g=float(os.environ.get("SMARTNUTRI_REMOVAL_THRESHOLD_G", "10")),
            stable_confirmations=int(os.environ.get("SMARTNUTRI_STABLE_CONFIRMATIONS", "2")),
            removal_confirmations=int(os.environ.get("SMARTNUTRI_REMOVAL_CONFIRMATIONS", "3")),
            post_stability_delay_s=float(os.environ.get("SMARTNUTRI_SETTLING_DELAY", "1.0")),
            weight_change_tolerance_g=float(os.environ.get("SMARTNUTRI_WEIGHT_CHANGE_TOLERANCE_G", "5.0")),
            stale_reading_timeout_s=float(os.environ.get("SMARTNUTRI_STALE_READING_TIMEOUT_S", "2.0")),
            correction_weight_tolerance_g=float(os.environ.get("SMARTNUTRI_CORRECTION_WEIGHT_TOLERANCE_G", "0.05")),
            camera_timeout_s=float(os.environ.get("SMARTNUTRI_CAMERA_TIMEOUT", "4.0")),
            capture_retries=int(os.environ.get("SMARTNUTRI_CAPTURE_RETRIES", "2")),
            capture_burst_count=int(os.environ.get("SMARTNUTRI_CAPTURE_BURST_COUNT", "1")),
            min_classification_confidence=float(
                os.environ.get("SMARTNUTRI_MIN_CLASSIFICATION_CONFIDENCE", "0.55")
            ),
            portion_review_uncertainty_fraction=float(
                os.environ.get("SMARTNUTRI_PORTION_REVIEW_UNCERTAINTY_FRACTION", "0.20")
            ),
        )
