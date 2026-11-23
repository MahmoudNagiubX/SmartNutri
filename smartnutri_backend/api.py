from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from smartnutri_backend import __version__
from smartnutri_backend.repository import SQLiteRepository
from smartnutri_backend.schemas import (
    PlateDefaultRequest,
    PlateProfile,
    PlateProfileCreate,
    ScaleStatusResponse,
    TareConfigurationResponse,
)
from smartnutri_backend.settings import Settings


@dataclass
class Runtime:
    settings: Settings
    repository: SQLiteRepository
    scale_client: Any = None
    monitor: Any = None
    camera: Any = None
    inference: Any = None
    workflow: Any = None


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository = SQLiteRepository(resolved_settings.database_path)
        runtime = Runtime(settings=resolved_settings, repository=repository)
        app.state.runtime = runtime
        try:
            yield
        finally:
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

    @app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime: Runtime = request.app.state.runtime
        try:
            runtime.repository.get_setting("health_probe", True)
            database_status = "ok"
        except Exception:
            database_status = "unavailable"
        if database_status != "ok":
            raise HTTPException(status_code=503, detail="database unavailable")
        return {
            "status": "ok",
            "service": "smartnutri-backend",
            "version": __version__,
            "database": database_status,
            "auto_scan_enabled": runtime.settings.auto_scan_enabled,
            "scale_configured": bool(
                runtime.settings.scale_mode == "mock"
                or runtime.settings.scale_base_url
            ),
            "camera_mode": runtime.settings.camera_mode,
            "segmentation_mode": runtime.settings.segmentation_mode,
        }

    @app.get("/api/v1/scale/status", response_model=ScaleStatusResponse)
    async def scale_status(request: Request) -> ScaleStatusResponse:
        runtime: Runtime = request.app.state.runtime
        client = runtime.scale_client
        if client is None:
            return ScaleStatusResponse(
                configured=False,
                status="not_configured",
                state=(runtime.monitor.state if runtime.monitor is not None else "IDLE"),
            )
        try:
            reading = await asyncio.to_thread(client.get_weight)
        except Exception as exc:
            return ScaleStatusResponse(
                configured=True,
                status="unavailable",
                source="hx711" if runtime.settings.scale_mode == "real" else "mock",
                simulated=runtime.settings.scale_mode == "mock",
                state=(runtime.monitor.state if runtime.monitor is not None else "IDLE"),
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
            state=(runtime.monitor.state if runtime.monitor is not None else "IDLE"),
        )

    @app.get("/api/v1/scale/tare-config", response_model=TareConfigurationResponse)
    async def tare_configuration(request: Request) -> dict[str, Any]:
        runtime: Runtime = request.app.state.runtime
        return runtime.repository.get_tare_configuration()

    @app.get("/api/v1/plates", response_model=list[PlateProfile])
    async def list_plates(request: Request) -> list[dict[str, Any]]:
        return request.app.state.runtime.repository.list_plate_profiles()

    @app.post(
        "/api/v1/plates",
        response_model=PlateProfile,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_plate(payload: PlateProfileCreate, request: Request) -> dict[str, Any]:
        try:
            return request.app.state.runtime.repository.create_plate_profile(
                payload.name,
                payload.empty_plate_weight_g,
                make_default=payload.make_default,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/plates/default", response_model=PlateProfile)
    async def get_default_plate(request: Request) -> dict[str, Any]:
        plate = request.app.state.runtime.repository.get_default_plate()
        if plate is None:
            raise HTTPException(status_code=404, detail="no default plate profile is registered")
        return plate

    @app.put("/api/v1/plates/default", response_model=PlateProfile)
    async def set_default_plate(
        payload: PlateDefaultRequest,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return request.app.state.runtime.repository.set_default_plate(payload.plate_profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="plate profile not found") from exc

    return app


app = create_app()
