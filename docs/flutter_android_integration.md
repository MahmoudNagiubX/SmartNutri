# Flutter Android Integration Contract

BE-02 prepares the backend contract for a native Flutter/Dart Android client. It does not create a Flutter project or app code. The API is versioned REST over HTTP with JSON request and response bodies; image resources use PNG or JPEG. FastAPI publishes the contract at `/openapi.json`, with Pydantic response schemas that can feed an OpenAPI-to-Dart model generator.

## Mobile-first flow

| App step | Backend contract |
| --- | --- |
| Home and scale status | `GET /api/v1/health`, `GET /api/v1/scale/status`, `GET /api/v1/scale/tare-config`, and `GET /api/v1/plates` |
| Select plate and tare | `GET /api/v1/plates/default`; select with `PUT /api/v1/plates/default`; register profiles with `POST /api/v1/plates`. An explicit tare uses `POST /api/v1/scale/tare` and `X-SmartNutri-Control-Token` in real mode. |
| Automatic scan progress | Poll `GET /api/v1/scans/active`; a `null` response means there is not yet a latched scan. The response is a `ScanResponse` when a scan is active. |
| Food results and segmentation | `GET /api/v1/scans/{scan_id}` and `/foods`; display `annotated_url` or `/annotated.jpg` over the captured image. `/mask.png` returns the semantic class-ID mask, and `/mask/{class_id}` returns one binary food mask. |
| Portion estimates and uncertainty | `GET /api/v1/scans/{scan_id}/portions`; each estimate reports grams, confidence, uncertainty in grams, and whether review is required. |
| Calories and macros | `GET /api/v1/scans/{scan_id}/nutrition` or inspect the meal. Calories, protein, carbohydrates, and fat are nullable until verified nutrition mappings and usable portion values exist. Values are explicitly estimates. |
| Quick corrections | `POST /api/v1/meals/{meal_id}/corrections`, with corrected item class IDs/weights and the control-token header in real mode. Corrected portions must sum to measured net weight. |
| Meal history | `GET /api/v1/meals?limit=50` and `GET /api/v1/meals/{meal_id}`. |

FoodSeg103 uses class `0` for background and `1..103` for food. Use returned class IDs and labels as a pair; do not infer IDs from display order. For example, this dataset maps rice to 66, chicken duck to 48, french fries to 3, and potato to 70.

## Bounded status polling

WebSockets are not part of the initial contract. A Dart client should keep at most one status request in flight, poll `GET /api/v1/scans/active` at a 1-second interval and back off to no more than 5 seconds, and stop after a 120-second deadline or after the scan reaches `COMPLETED`, `NEEDS_REVIEW`, or `WAITING_FOR_REMOVAL`. On timeout, show a retry/refresh action instead of polling indefinitely. Once a scan ID is available, refresh `GET /api/v1/scans/{scan_id}` to read its latest state and result. Cancellation or leaving the progress screen should cancel the pending timer/request.

## Camera boundary

Camera capture remains a backend camera-service responsibility. The automatic scan workflow asks the configured camera adapter for an image; the Flutter app is not required to open the Android camera, upload a frame, or run segmentation. `GET /api/v1/camera/status` reports whether the configured adapter is mock, file, phone snapshot, or unconfigured. For a real phone snapshot source, configure the backend's existing private `SMARTNUTRI_CAMERA_URL` adapter and keep its capture service available on the trusted LAN.

## JSON and OpenAPI contract

- API prefix: `/api/v1`; JSON uses standard UTF-8 HTTP request/response bodies.
- OpenAPI document: `GET /openapi.json` on the selected base URL. JSON routes reference named response schemas such as `ScanResponse`, `MealResponse`, `NutritionRecordResponse`, `PortionResponse`, and `ScanNutritionResponse`.
- Image routes return `image/jpeg` or `image/png` bytes and are described as binary in OpenAPI. Treat their URLs as relative to the configured API base URL.
- Optional query bounds: scan and meal history `limit` is `1..200`; nutrition record `class_id` is `1..103`.
- Control operations in real mode use `X-SmartNutri-Control-Token`. Keep the token in protected app configuration and out of source control. Mock-mode operations remain marked synthetic.
- Native Flutter HTTP clients do not use browser CORS. CORS is disabled by default; only enable explicit origins if a separate browser client is intentionally deployed.

Nutrition record payloads require a food identity, preparation state, non-negative kcal/protein/carbohydrate/fat per 100 g, source, and version. Verified records must map to a FoodSeg103 class. `POST /api/v1/nutrition/records` returns the stored record ID and creation time. The backend does not seed records from an unverified database.

## Trusted Wi-Fi binding

The supported launcher is `python -m smartnutri_backend.server`. It binds to `127.0.0.1:8000` by default. To make it reachable by an Android device, set `SMARTNUTRI_API_HOST` to the laptop's current private Wi-Fi IPv4 address (RFC1918 only) and optionally set `SMARTNUTRI_API_PORT`. The launcher rejects `0.0.0.0`, public IP addresses, hostnames, and non-IPv4 values. Bind to the Wi-Fi interface address, not every interface.

Example PowerShell setup after checking the laptop Wi-Fi IPv4 address:

```powershell
$env:SMARTNUTRI_API_HOST = '192.168.1.23'
$env:SMARTNUTRI_API_PORT = '8000'
$env:SMARTNUTRI_SCALE_MODE = 'mock'
$env:SMARTNUTRI_AUTO_SCAN = 'false'
& (Join-Path $backendEnv 'Scripts\python.exe') -m smartnutri_backend.server
```

Replace the example address with the laptop's actual private Wi-Fi IPv4. For real hardware, use the real scale/camera settings instead of mock mode. On Windows, allow inbound TCP only on the **Private** firewall profile, local address equal to the laptop Wi-Fi IP, port 8000, and remote address limited to the Android device or trusted home subnet. Do not add a router port-forward, bind a public interface, or expose this API to the internet. The host's LAN firewall scope is important because read endpoints are not user-authenticated; real hardware and meal/nutrition write operations require the control token.

If the phone needs the laptop's address by name, configure a stable mDNS/DNS name such as `smartnutri-host.local` that resolves to the same Wi-Fi IPv4, and keep the server bound to that IPv4. Do not use the laptop's `127.0.0.1` address from the phone; that address refers to the phone itself.

## Android networking configuration

Declare `android.permission.INTERNET` in the Android manifest. It is a normal install-time permission and does not require a runtime prompt. Android 9 (API 28) and newer disable cleartext HTTP by default. Prefer HTTPS when a trusted local certificate can be provisioned. If the local prototype uses `http://smartnutri-host.local:8000`, allow cleartext only for that exact host in a **debug-only** Network Security Configuration; leave the app-wide base policy and release build cleartext-disabled:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <base-config cleartextTrafficPermitted="false" />
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="false">smartnutri-host.local</domain>
    </domain-config>
</network-security-config>
```

Reference that file through `android:networkSecurityConfig` in the debug app manifest. If the app instead uses a literal IP URL, verify Android's policy on the target device; use a dedicated local DNS name for a narrow domain rule, or keep any broader cleartext exception confined to the debug manifest. Do not put a global `usesCleartextTraffic="true"` override in the release manifest.

Starting with Android 17 (API 37), apps targeting SDK 37 or higher need the `ACCESS_LOCAL_NETWORK` runtime permission before direct TCP connections or `.local` resolution. Declare and request it when that target is adopted, and handle denial/revocation. The temporary Android 16 target-36 path is deprecated; do not add the local-network permission for targets below 37. Recheck Android's current guidance when the Flutter app is implemented because target SDK behavior can change.

Android platform references: [network security configuration and cleartext policy](https://developer.android.com/privacy-and-security/security-config), [manifest permissions](https://developer.android.com/guide/topics/manifest/uses-permission-element), and [local-network permission](https://developer.android.com/privacy-and-security/local-network-permission).
