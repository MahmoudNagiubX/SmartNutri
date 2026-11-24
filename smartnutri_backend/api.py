from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from smartnutri_backend import __version__
from smartnutri_backend.agent import NutritionDecisionAgent
from smartnutri_backend.camera import (
    FileCameraService,
    IPSnapshotCameraService,
    MockCameraService,
    UnconfiguredCameraService,
)
from smartnutri_backend.nutrition import NutritionService
from smartnutri_backend.nutrition_catalog import CURATED_NUTRITION_RECORDS
from smartnutri_backend.portion import ExperimentalVisualProportionEstimator
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.scale_service import MockScaleClient, ScaleMonitoringService
from smartnutri_backend.schemas import (
    AgentDecisionResponse,
    CameraStatusResponse,
    FoodPredictionsResponse,
    HealthResponse,
    MealCorrectionRequest,
    MealResponse,
    MockPlacementResponse,
    MockPlacementRequest,
    NutritionRecord,
    NutritionRecordResponse,
    PlateDefaultRequest,
    PlateProfile,
    PlateProfileCreate,
    PortionEstimate,
    PortionEstimationResult,
    PortionResponse,
    ScaleCalibrationResponse,
    ScanResponse,
    ScanNutritionResponse,
    ScaleCalibrationRequest,
    ScaleStatusResponse,
    TareConfigurationResponse,
    TareRequest,
)
from smartnutri_backend.settings import Settings
from smartnutri_backend.vision import MockSegmentationService, SegFormerInferenceService
from smartnutri_backend.workflow import ScanWorkflow


@dataclass
class Runtime:
    settings: Settings
    repository: SQLiteRepository
    scale_client: Any = None
    monitor: Any = None
    camera: Any = None
    inference: Any = None
    workflow: Any = None
    nutrition: Any = None


def _camera(settings: Settings) -> Any:
    options = {
        "max_image_bytes": settings.max_image_bytes,
        "min_width": settings.min_image_width,
        "min_height": settings.min_image_height,
        "max_dimension": settings.max_image_dimension,
        "min_brightness": settings.min_brightness,
        "max_brightness": settings.max_brightness,
        "min_edge_variance": settings.min_edge_variance,
        "max_clipped_fraction": settings.max_clipped_pixel_fraction,
        "capture_retries": settings.capture_retries,
        "burst_count": settings.capture_burst_count,
        "timeout_s": settings.camera_timeout_s,
    }
    if settings.camera_mode == "mock":
        return MockCameraService(**options)
    if settings.camera_mode == "file" and settings.camera_file_path is not None:
        return FileCameraService(settings.camera_file_path, **options)
    if settings.camera_mode == "ip_snapshot" and settings.camera_snapshot_url:
        if not settings.camera_allowed_host:
            raise ValueError("SMARTNUTRI_CAMERA_ALLOWED_HOST must explicitly allow the phone IP and port")
        return IPSnapshotCameraService(
            settings.camera_snapshot_url,
            allowed_host=settings.camera_allowed_host,
            **options,
        )
    return UnconfiguredCameraService(**options)


def _scale_client(settings: Settings) -> Any:
    if settings.scale_mode == "disabled":
        return None
    if settings.scale_mode == "mock":
        return MockScaleClient()
    if not settings.scale_base_url:
        return None
    from smartnutri_weight.client import WeightClient

    return WeightClient(
        settings.scale_base_url,
        timeout_s=2.0,
        device_token=settings.scale_device_token,
    )


def _safe_scan(scan: dict[str, Any], *, meal: dict[str, Any] | None = None) -> dict[str, Any]:
    scan_id = scan["id"]
    return {
        key: scan.get(key)
        for key in (
            "id", "placement_key", "state", "decision", "source", "simulated",
            "device_session_id", "measurement_id", "measurement_timestamp_ms",
            "stable_weight_g", "measurement_received_at", "verified_measurement_id",
            "verified_weight_g", "verified_timestamp_ms", "verified_received_at",
            "capture_placement_key",
            "tare_mode", "plate_profile_id", "measured_gross_weight_g",
            "plate_tare_weight_g", "measured_net_food_weight_g", "capture_source",
            "capture_simulated", "image_quality", "predictions", "portions",
            "nutrition_totals", "result_message", "error_code", "created_at", "updated_at",
        )
    } | {
        "capture_url": f"/api/v1/scans/{scan_id}/capture.jpg" if scan.get("capture_path") else None,
        "annotated_url": f"/api/v1/scans/{scan_id}/annotated.jpg" if scan.get("annotated_path") else None,
        "mask_url": f"/api/v1/scans/{scan_id}/mask.png" if scan.get("capture_path") else None,
        "meal": meal,
    }


def _safe_nutrition_record(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository = SQLiteRepository(resolved_settings.database_path)
        camera = _camera(resolved_settings)
        if resolved_settings.segmentation_mode == "mock":
            inference = MockSegmentationService()
        else:
            inference = SegFormerInferenceService(
                checkpoint_dir=resolved_settings.checkpoint_dir,
                foodseg_root=resolved_settings.foodseg_root,
                # Real inference loads the exact preprocessing config embedded in checkpoint metadata.
                preprocessing_config_path=None,
                device="cpu",
            )
        nutrition = NutritionService(repository)
        nutrition.seed_catalog(CURATED_NUTRITION_RECORDS)
        workflow = ScanWorkflow(
            repository=repository,
            camera=camera,
            inference=inference,
            portion_estimator=ExperimentalVisualProportionEstimator(),
            agent=NutritionDecisionAgent(
                min_classification_confidence=resolved_settings.min_classification_confidence,
                max_portion_uncertainty_fraction=resolved_settings.portion_review_uncertainty_fraction,
            ),
            nutrition=nutrition,
            image_dir=resolved_settings.image_dir,
        )
        client = _scale_client(resolved_settings)
        scale_service = ScaleMonitoringService(
            client=client,
            repository=repository,
            workflow=workflow,
            settings=resolved_settings,
            allow_simulated=resolved_settings.scale_mode == "mock",
        )
        runtime = Runtime(
            settings=resolved_settings,
            repository=repository,
            scale_client=client,
            monitor=scale_service,
            camera=camera,
            inference=inference,
            workflow=workflow,
            nutrition=nutrition,
        )
        app.state.runtime = runtime
        await scale_service.start()
        try:
            yield
        finally:
            await scale_service.stop()
            repository.close()

    app = FastAPI(
        title="SmartNutri Backend API",
        description="Local scale, camera, segmentation, portion and meal API.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-SmartNutri-Control-Token"],
        max_age=600,
    )

    def runtime(request: Request) -> Runtime:
        return request.app.state.runtime

    def require_control(request: Request, supplied_token: str | None) -> None:
        active = runtime(request)
        if active.settings.scale_mode == "mock":
            return
        expected = active.settings.control_token
        if not expected:
            raise HTTPException(status_code=503, detail="SMARTNUTRI_CONTROL_TOKEN is required for control operations")
        if supplied_token is None or not hmac.compare_digest(expected, supplied_token):
            raise HTTPException(status_code=403, detail="control token is missing or invalid")

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> dict[str, Any]:
        active = runtime(request)
        try:
            active.repository.get_setting("health_probe", True)
            database_status = "ok"
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {
            "status": "ok",
            "service": "smartnutri-backend",
            "version": __version__,
            "database": database_status,
            "auto_scan_enabled": active.settings.auto_scan_enabled,
            "scale_configured": active.scale_client is not None,
            "camera_mode": active.settings.camera_mode,
            "segmentation_mode": active.settings.segmentation_mode,
            "scan_state": active.monitor.state.value,
        }

    @app.get("/api/v1/scale/status", response_model=ScaleStatusResponse, tags=["scale"])
    async def scale_status(request: Request) -> ScaleStatusResponse:
        active = runtime(request)
        if active.scale_client is None:
            return ScaleStatusResponse(
                configured=False,
                status="not_configured",
                state=active.monitor.state,
                last_error=active.monitor.last_error,
            )
        try:
            raw = await asyncio.to_thread(active.scale_client.get_weight)
            reading = active.monitor.to_observation(raw)
        except Exception as exc:
            return ScaleStatusResponse(
                configured=True,
                status="unavailable",
                source="mock" if active.settings.scale_mode == "mock" else "hx711",
                simulated=active.settings.scale_mode == "mock",
                state=active.monitor.state,
                last_error=type(exc).__name__,
            )
        return ScaleStatusResponse(
            configured=True,
            status=reading.status,
            source=reading.source,
            simulated=reading.simulated,
            sensor_connected=reading.sensor_connected,
            stable=reading.stable,
            tared=reading.tared,
            calibrated=reading.calibrated,
            weight_g=reading.weight_g,
            state=active.monitor.state,
            last_error=active.monitor.last_error,
        )

    @app.get("/api/v1/scale/tare-config", response_model=TareConfigurationResponse, tags=["scale"])
    async def tare_configuration(request: Request) -> dict[str, Any]:
        return runtime(request).repository.get_tare_configuration()

    @app.post("/api/v1/scale/tare", response_model=TareConfigurationResponse, tags=["scale"])
    async def tare_scale(
        payload: TareRequest,
        request: Request,
        control_token: str | None = Header(default=None, alias="X-SmartNutri-Control-Token"),
    ) -> dict[str, Any]:
        require_control(request, control_token)
        active = runtime(request)
        try:
            return await asyncio.to_thread(
                active.monitor.configure_tare,
                payload.mode,
                confirmation=payload.confirm,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=503, detail="scale device authentication is not configured") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/scale/calibrate", response_model=ScaleCalibrationResponse, tags=["scale"])
    async def calibrate_scale(
        payload: ScaleCalibrationRequest,
        request: Request,
        control_token: str | None = Header(default=None, alias="X-SmartNutri-Control-Token"),
    ) -> dict[str, Any]:
        require_control(request, control_token)
        active = runtime(request)
        if active.scale_client is None:
            raise HTTPException(status_code=503, detail="scale is not configured")
        if active.settings.scale_mode != "real":
            raise HTTPException(status_code=409, detail="calibration requires real HX711 hardware")
        try:
            reading = await asyncio.to_thread(
                active.scale_client.calibrate,
                payload.known_weight_g,
                confirmation=payload.confirm,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=503, detail="scale device authentication is not configured") from exc
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"calibration failed ({type(exc).__name__})") from exc
        return {"reading": reading.__dict__, "simulated": False}

    @app.get("/api/v1/plates", response_model=list[PlateProfile], tags=["plates"])
    async def list_plates(request: Request) -> list[dict[str, Any]]:
        return runtime(request).repository.list_plate_profiles()

    @app.post("/api/v1/plates", response_model=PlateProfile, status_code=status.HTTP_201_CREATED, tags=["plates"])
    async def create_plate(payload: PlateProfileCreate, request: Request) -> dict[str, Any]:
        active = runtime(request)
        if payload.make_default and active.monitor.monitor.latched:
            raise HTTPException(status_code=409, detail="remove the current plate before changing the default plate profile")
        reference_image_path = None
        if payload.reference_image_path:
            candidate = Path(payload.reference_image_path).expanduser().resolve()
            root = active.settings.image_dir.resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise HTTPException(status_code=422, detail="reference image must be an existing backend runtime image")
            try:
                if candidate.stat().st_size > active.settings.max_image_bytes:
                    raise HTTPException(status_code=413, detail="reference image exceeds the configured size limit")
                with Image.open(candidate) as image:
                    if (image.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
                        raise HTTPException(status_code=422, detail="reference image format must be JPEG, PNG, or WEBP")
                    image.verify()
            except HTTPException:
                raise
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=422, detail="reference image is invalid") from exc
            reference_image_path = str(candidate)
        try:
            profile = active.repository.create_plate_profile(
                payload.name,
                payload.empty_plate_weight_g,
                reference_image_path=reference_image_path,
                make_default=payload.make_default,
            )
            if payload.make_default:
                active.monitor.refresh_saved_plate_profile()
            return profile
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/plates/default", response_model=PlateProfile, tags=["plates"])
    async def get_default_plate(request: Request) -> dict[str, Any]:
        plate = runtime(request).repository.get_default_plate()
        if plate is None:
            raise HTTPException(status_code=404, detail="no default plate profile is registered")
        return plate

    @app.put("/api/v1/plates/default", response_model=PlateProfile, tags=["plates"])
    async def set_default_plate(payload: PlateDefaultRequest, request: Request) -> dict[str, Any]:
        active = runtime(request)
        if active.monitor.monitor.latched:
            raise HTTPException(status_code=409, detail="remove the current plate before changing the default plate profile")
        try:
            profile = active.repository.set_default_plate(payload.plate_profile_id)
            active.monitor.refresh_saved_plate_profile()
            return profile
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="plate profile not found") from exc

    @app.get("/api/v1/camera/status", response_model=CameraStatusResponse, tags=["camera"])
    async def camera_status(request: Request) -> dict[str, Any]:
        return runtime(request).camera.status()

    @app.get("/api/v1/scans", response_model=list[ScanResponse], tags=["scanning"])
    async def list_scans(request: Request, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
        active = runtime(request)
        return [
            _safe_scan(scan, meal=active.repository.get_meal_by_scan(scan["id"]))
            for scan in active.repository.list_scans(limit)
        ]

    @app.get("/api/v1/scans/active", response_model=ScanResponse | None, tags=["scanning"])
    async def active_scan(request: Request) -> dict[str, Any] | None:
        active = runtime(request)
        placement_key = active.monitor.monitor.active_placement_key
        if placement_key is None:
            return None
        scan = active.repository.get_scan_by_placement(placement_key)
        return _safe_scan(scan, meal=active.repository.get_meal_by_scan(scan["id"])) if scan else None

    @app.get("/api/v1/scans/{scan_id}", response_model=ScanResponse, tags=["scanning"])
    async def get_scan(scan_id: str, request: Request) -> dict[str, Any]:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return _safe_scan(scan, meal=active.repository.get_meal_by_scan(scan_id))

    @app.get("/api/v1/scans/{scan_id}/foods", response_model=FoodPredictionsResponse, tags=["vision"])
    async def get_food_predictions(scan_id: str, request: Request) -> dict[str, Any]:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        predictions = scan.get("predictions") or []
        source = predictions[0]["source"] if predictions else active.settings.segmentation_mode
        simulated = bool(
            scan.get("simulated")
            or scan.get("capture_simulated")
            or source == "mock"
            or any(item.get("simulated") for item in predictions)
        )
        return {
            "scan_id": scan_id,
            "predictions": predictions,
            "source": source,
            "simulated": simulated,
        }

    @app.get("/api/v1/scans/{scan_id}/portions", response_model=PortionResponse, tags=["portions"])
    async def get_portion_estimates(scan_id: str, request: Request) -> dict[str, Any]:
        scan = runtime(request).repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return {
            "scan_id": scan_id,
            "portions": scan.get("portions"),
            "simulated": bool(
                scan.get("simulated")
                or scan.get("capture_simulated")
                or (scan.get("portions") or {}).get("simulated")
                or any(item.get("simulated") for item in (scan.get("predictions") or []))
            ),
        }

    @app.get("/api/v1/scans/{scan_id}/nutrition", response_model=ScanNutritionResponse, tags=["nutrition"])
    async def get_scan_nutrition(scan_id: str, request: Request) -> dict[str, Any]:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return {
            "scan_id": scan_id,
            "measured_net_food_weight_g": scan.get("measured_net_food_weight_g"),
            "totals": scan.get("nutrition_totals"),
            "meal": active.repository.get_meal_by_scan(scan_id),
            "estimated": True,
            "simulated": bool(
                scan.get("simulated")
                or scan.get("capture_simulated")
                or (scan.get("portions") or {}).get("simulated")
                or any(item.get("simulated") for item in (scan.get("predictions") or []))
            ),
        }

    @app.get(
        "/api/v1/scans/{scan_id}/mask.png",
        responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
        tags=["vision"],
    )
    async def semantic_mask(scan_id: str, request: Request) -> Response:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        path = (active.settings.image_dir / "scans" / f"{scan_id}-mask.png").resolve()
        if not path.is_relative_to(active.settings.image_dir.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="segmentation mask is unavailable")
        return FileResponse(path, media_type="image/png")

    @app.get(
        "/api/v1/scans/{scan_id}/mask/{class_id}",
        responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
        tags=["vision"],
    )
    async def class_mask(scan_id: str, class_id: int, request: Request) -> Response:
        if not 1 <= class_id <= 103:
            raise HTTPException(status_code=422, detail="class_id must be in 1..103")
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        path = (active.settings.image_dir / "scans" / f"{scan_id}-mask.png").resolve()
        if not path.is_relative_to(active.settings.image_dir.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="segmentation mask is unavailable")
        try:
            with Image.open(path) as image:
                if image.mode != "L":
                    raise ValueError("mask must be grayscale class IDs")
                binary = image.point(lambda value: 255 if value == class_id else 0)
                output = io.BytesIO()
                binary.save(output, format="PNG")
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="stored segmentation mask is invalid") from exc
        return Response(content=output.getvalue(), media_type="image/png")

    @app.get(
        "/api/v1/scans/{scan_id}/capture.jpg",
        responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}},
        tags=["camera"],
    )
    async def captured_image(scan_id: str, request: Request) -> Response:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None or not scan.get("capture_path"):
            raise HTTPException(status_code=404, detail="captured image is unavailable")
        path = Path(scan["capture_path"]).resolve()
        if not path.is_relative_to(active.settings.image_dir.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="captured image is unavailable")
        return FileResponse(path, media_type="image/jpeg")

    @app.get(
        "/api/v1/scans/{scan_id}/annotated.jpg",
        responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}},
        tags=["vision"],
    )
    async def annotated_image(scan_id: str, request: Request) -> Response:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None or not scan.get("annotated_path"):
            raise HTTPException(status_code=404, detail="annotated image is unavailable")
        path = Path(scan["annotated_path"]).resolve()
        if not path.is_relative_to(active.settings.image_dir.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="annotated image is unavailable")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/v1/scans/{scan_id}/recapture", response_model=ScanResponse, tags=["camera"])
    @app.post("/api/v1/camera/capture/{scan_id}", response_model=ScanResponse, tags=["camera"])
    async def recapture(scan_id: str, request: Request,
        control_token: str | None = Header(default=None, alias="X-SmartNutri-Control-Token")) -> dict[str, Any]:
        require_control(request, control_token)
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        if scan.get("state") != "NEEDS_REVIEW" or scan.get("error_code") not in {
            "camera_unavailable", "image_quality_rejected"
        }:
            raise HTTPException(status_code=409, detail="this scan is not waiting for a camera retry")
        if active.monitor.monitor.active_placement_key != scan["placement_key"]:
            raise HTTPException(status_code=409, detail="recapture requires the same plate to remain on the scale")
        try:
            raw = await asyncio.to_thread(active.scale_client.get_weight)
            observation = active.monitor.to_observation(raw)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="stable scale reading is unavailable") from exc
        if not observation.usable_stable:
            raise HTTPException(status_code=409, detail="recapture requires a stable scale reading")
        try:
            await active.workflow.process(scan_id=scan_id, observation=observation)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail="scan is already processing") from exc
        finally:
            active.workflow.finish(scan_id)
        updated = active.repository.get_scan(scan_id)
        assert updated is not None
        return _safe_scan(updated, meal=active.repository.get_meal_by_scan(scan_id))

    @app.post("/api/v1/scans/mock-placement", response_model=MockPlacementResponse, tags=["scanning"])
    async def mock_placement(payload: MockPlacementRequest, request: Request) -> dict[str, Any]:
        active = runtime(request)
        if active.settings.scale_mode != "mock":
            raise HTTPException(status_code=404, detail="mock scan endpoint is disabled")
        try:
            result = await active.monitor.inject_mock_placement(payload.gross_weight_g)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        scan = result["scan"]
        return {
            "scan": _safe_scan(scan, meal=result["meal"]),
            "simulated": True,
            "simulation_notice": "All scale values in this response are synthetic test inputs.",
        }

    @app.get("/api/v1/nutrition/records", response_model=list[NutritionRecordResponse], tags=["nutrition"])
    async def list_nutrition_records(request: Request, class_id: int | None = None) -> list[dict[str, Any]]:
        if class_id is not None and not 1 <= class_id <= 103:
            raise HTTPException(status_code=422, detail="class_id must be in 1..103")
        return runtime(request).nutrition.list_records(class_id=class_id)

    @app.post(
        "/api/v1/nutrition/records",
        response_model=NutritionRecordResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["nutrition"],
    )
    async def create_nutrition_record(
        payload: NutritionRecord,
        request: Request,
        control_token: str | None = Header(default=None, alias="X-SmartNutri-Control-Token"),
    ) -> dict[str, Any]:
        require_control(request, control_token)
        try:
            result = runtime(request).nutrition.register(payload)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _safe_nutrition_record(result)

    @app.get("/api/v1/meals", response_model=list[MealResponse], tags=["meals"])
    async def meal_history(request: Request, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
        return runtime(request).repository.list_meals(limit)

    @app.get("/api/v1/meals/{meal_id}", response_model=MealResponse, tags=["meals"])
    async def get_meal(meal_id: str, request: Request) -> dict[str, Any]:
        meal = runtime(request).repository.get_meal(meal_id)
        if meal is None:
            raise HTTPException(status_code=404, detail="meal not found")
        return meal

    @app.post("/api/v1/meals/{meal_id}/corrections", response_model=MealResponse, tags=["meals"])
    async def correct_meal(
        meal_id: str,
        payload: MealCorrectionRequest,
        request: Request,
        control_token: str | None = Header(default=None, alias="X-SmartNutri-Control-Token"),
    ) -> dict[str, Any]:
        require_control(request, control_token)
        active = runtime(request)
        meal = active.repository.get_meal(meal_id)
        if meal is None:
            raise HTTPException(status_code=404, detail="meal not found")
        if len({item.class_id for item in payload.items}) != len(payload.items):
            raise HTTPException(status_code=422, detail="each semantic class may appear once")
        total = sum(item.weight_g for item in payload.items)
        if abs(total - meal["measured_net_food_weight_g"]) > active.settings.correction_weight_tolerance_g:
            raise HTTPException(status_code=422, detail="corrected portions must sum to measured net food weight")
        selected_records: dict[int, Any] = {}
        for item in payload.items:
            if item.preparation_state is not None and item.nutrition_record_id is None:
                raise HTTPException(status_code=422, detail="preparation state requires a selected nutrition record")
            if item.nutrition_record_id is None:
                continue
            selected = active.nutrition.lookup_for_class(
                item.class_id,
                nutrition_record_id=item.nutrition_record_id,
                simulated=meal["simulated"],
            )
            if selected is None:
                raise HTTPException(status_code=422, detail="nutrition record is missing, unverified, or does not match this food class")
            _, record = selected
            if item.preparation_state is not None and item.preparation_state != record.preparation_state:
                raise HTTPException(status_code=422, detail="preparation state does not match the selected nutrition record")
            if record.requires_preparation_confirmation and item.preparation_state != record.preparation_state:
                raise HTTPException(status_code=422, detail="selected nutrition record requires explicit preparation confirmation")
            selected_records[item.class_id] = record
        portions = [
            PortionEstimate(
                class_id=item.class_id,
                label=item.label,
                estimated_weight_g=item.weight_g,
                method="user_correction",
                confidence=1,
                uncertainty_g=0,
                review_required=False,
                simulated=meal["simulated"],
            )
            for item in payload.items
        ]
        calculation = active.nutrition.calculate(
            portions,
            simulated=meal["simulated"],
            nutrition_record_ids={
                item.class_id: item.nutrition_record_id
                for item in payload.items
                if item.nutrition_record_id is not None
            },
        )
        nutrition_items = {item.class_id: item for item in calculation.items}
        corrected_items = []
        for portion in portions:
            nutrition = nutrition_items.get(portion.class_id)
            corrected_items.append(
                {
                    "class_id": portion.class_id,
                    "food_label": selected_records[portion.class_id].food_identity
                    if portion.class_id in selected_records else portion.label,
                    "estimated_weight_g": portion.estimated_weight_g,
                    "nutrition_record_id": nutrition.nutrition_record_id if nutrition else None,
                    "preparation_state": selected_records[portion.class_id].preparation_state
                    if portion.class_id in selected_records else None,
                    "nutrients": nutrition.nutrients.model_dump(mode="json")
                    if nutrition and nutrition.nutrients else None,
                    "needs_review": nutrition is None or nutrition.nutrients is None,
                }
            )
        totals = calculation.totals.model_dump(mode="json") if calculation.totals else None
        correction_result = PortionEstimationResult(
            measured_net_weight_g=meal["measured_net_food_weight_g"],
            method="user_correction",
            estimates=portions,
            allocation_complete=True,
            review_required=not calculation.complete,
            simulated=meal["simulated"],
        )
        canonical_items = sorted(
            (item.model_dump(mode="json") for item in payload.items),
            key=lambda item: item["class_id"],
        )
        fingerprint = hashlib.sha256(
            json.dumps(canonical_items, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        try:
            return active.repository.replace_corrected_meal(
                meal_id=meal_id,
                items=corrected_items,
                totals=totals,
                state="saved" if calculation.complete else "needs_review",
                scan_update={
                    "state": "COMPLETED" if calculation.complete else "NEEDS_REVIEW",
                    "decision": "SAVE_MEAL" if calculation.complete else "REQUEST_FOOD_PREPARATION",
                    "portions": correction_result.model_dump(mode="json"),
                    "nutrition_totals": totals,
                    "result_message": "User portion correction was saved and nutrition recalculated."
                    if calculation.complete else "Correction saved; a verified nutrition mapping is still missing.",
                    "error_code": None,
                },
                correction_fingerprint=fingerprint,
                weight_tolerance_g=active.settings.correction_weight_tolerance_g,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except sqlite3.Error as exc:
            raise HTTPException(
                status_code=500,
                detail="correction transaction failed; no changes were saved",
            ) from exc

    @app.get("/api/v1/agents/decisions/{scan_id}", response_model=AgentDecisionResponse, tags=["agent"])
    async def agent_decision(scan_id: str, request: Request) -> dict[str, Any]:
        active = runtime(request)
        scan = active.repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return {
            "scan_id": scan_id,
            "action": scan.get("decision"),
            "rationale": scan.get("result_message"),
            "state": scan.get("state"),
            "simulated": scan.get("simulated"),
        }

    return app


app = create_app()
