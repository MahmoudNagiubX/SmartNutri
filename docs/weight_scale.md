# SmartNutri ESP32 weighing scale

This module is independent of the food-image model and its training process. It reads an HX711 load-cell amplifier, reports grams over USB Serial and (when configured) a small HTTP API. It does not access datasets, checkpoints, or training code. The PlatformIO platform is pinned to `espressif32@7.0.1`, the version used for the verified build.

## Hardware and wiring

| HX711 / load cell | ESP32 DevKit V1 |
| --- | --- |
| HX711 VCC | 3V3 |
| HX711 GND | GND |
| HX711 DOUT / DT | GPIO32 |
| HX711 SCK / CLK | GPIO33 |
| Load cell bridge wires | HX711 E+, E-, A+, A- terminals |

Use the load-cell terminal labels and its own datasheet to map its wires; wire colors are not universal. Power the HX711 module from 3.3 V so its DOUT logic is safe for the ESP32. Keep SCK low except while clocking a conversion. The firmware reads signed 24-bit samples, selects channel A/gain 128 with the 25th clock, and treats repeated absence of a ready sample as `sensor_disconnected`.

The ESP32 is USB-powered. The smartphone camera is powered independently, and the LED ring light must remain on its independent supply; do not route ring-light current through the ESP32. The laptop and ESP32 communicate over Wi-Fi after credentials are configured. Camera streaming is outside this scale module.

## Build and flash

The PlatformIO project is in `firmware/weight_scale/` and targets the ESP32 DevKit V1 board ID `esp32dev` with the Arduino framework. PlatformIO Core 6.2.0 built this project against Espressif32 7.0.1 and Arduino-ESP32 2.0.17. PlatformIO and its toolchain are not Python project dependencies; use an isolated PlatformIO installation so the CUDA-enabled Python environment is unchanged.

From PowerShell at the repository root:

```powershell
if (-not (Test-Path firmware\weight_scale\include\secrets.h)) { Copy-Item firmware\weight_scale\include\secrets.example.h firmware\weight_scale\include\secrets.h }
# Edit secrets.h only if you want Wi-Fi. Leave its values empty for USB-only mode.
$pioRoot = Join-Path $env:LOCALAPPDATA 'SmartNutriPlatformIO'
if (-not (Test-Path $pioRoot)) { py -3.12 -m venv $pioRoot }
& (Join-Path $pioRoot 'Scripts\python.exe') -m pip install platformio==6.2.0
$pio = Join-Path $pioRoot 'Scripts\pio.exe'
& $pio run -d firmware\weight_scale
& $pio check -d firmware\weight_scale
& $pio device list
& $pio run -d firmware\weight_scale -t upload --upload-port COM5
& $pio device monitor --port COM5 --baud 115200
```

Replace `COM5` with the port shown by `pio device list`. Close Serial Monitor before uploading. The `secrets.h` file is ignored by Git. Do not commit Wi-Fi credentials. No board has been flashed as part of this implementation.

On boot, Serial prints help and status JSON once per second. USB operation does not require Wi-Fi. For network access, set `SMARTNUTRI_WIFI_SSID`, `SMARTNUTRI_WIFI_PASSWORD`, and a unique `SMARTNUTRI_SCALE_CONTROL_TOKEN` of at least 32 characters in the ignored local `secrets.h`; set the same token in the backend environment. The firmware retries the connection every 10 seconds and starts the HTTP listener on connection. Wi-Fi status and reconnect messages are printed to Serial. Without a valid local control token, HTTP tare/calibration are disabled.

## Tare and calibration

Calibration changes persistent ESP32 Preferences values. Use a verified physical mass and send the explicit confirmation token only when ready to write the calibration:

1. Mount the load cell and platform. Leave the platform empty and mechanically unloaded.
2. Open Serial Monitor at 115200 baud and send `tare CONFIRM`, or send an authenticated HTTP `POST /tare` with `confirm=TARE`. The operation starts asynchronously, collects eight HX711 conversions in the regular sample loop, then saves the raw offset in Preferences. The main loop remains responsive while samples are collected.
3. Place a known physical mass on the platform, wait for it to settle, then send `calibrate 500 CONFIRM` for a 500 g mass. Replace `500` with the verified mass in grams. The exact `CONFIRM` token is required. Calibration collects eight samples in the normal loop and saves the signed raw-counts-per-gram factor in Preferences.
4. Remove the mass and verify the reported reading. If the direction is reversed, the signed factor handles it; if the magnitude is wrong, repeat tare and calibration with a known mass.

Calibration is limited to a positive known mass no greater than 5000 g. The firmware rejects calibration until the scale is tared and the explicit confirmation and control token are present. HTTP controls require the `X-SmartNutri-Control-Token` header. For example, after configuring the same token in firmware and the current PowerShell session:

```powershell
$env:SMARTNUTRI_SCALE_CONTROL_TOKEN = 'replace-with-the-local-32-character-token'
curl.exe -X POST http://ESP32_IP/calibrate -H "X-SmartNutri-Control-Token: $env:SMARTNUTRI_SCALE_CONTROL_TOKEN" -H "Content-Type: application/x-www-form-urlencoded" --data "known_weight_g=500&confirm=CALIBRATE"
```

Replace `ESP32_IP` with the address printed after Wi-Fi connects. Keep the known mass on the platform during this request. `POST /tare` should only be used after removing all load, and requires both the header and `confirm=TARE` form field.

To inspect the API manually, use `curl.exe http://ESP32_IP/weight` for a reading. A tare request must include the configured token header and explicit confirmation, for example:

```powershell
curl.exe -X POST http://ESP32_IP/tare -H "X-SmartNutri-Control-Token: $env:SMARTNUTRI_SCALE_CONTROL_TOKEN" -H "Content-Type: application/x-www-form-urlencoded" --data "confirm=TARE"
```

Both tare and calibration complete asynchronously; the Python client polls `GET /weight` and returns after success or raises with the terminal operation state.

## API contract

`GET /weight` returns JSON with these fields:

| Field | Meaning |
| --- | --- |
| `ok` | Sensor is connected, tare and calibration are available, and there is no sensor error, overload, or pending tare/calibration operation. |
| `source` / `simulated` | Firmware reports `hx711` / `false`; mock client readings report `mock` / `true`. |
| `sensor_connected`, `tared`, `calibrated` | Hardware readiness flags. |
| `stable`, `status` | Stability result and one of `sensor_disconnected`, `sensor_error`, `overload`, `not_tared`, `not_calibrated`, `taring`, `calibrating`, `warming_up`, `unstable`, or `stable`. |
| `unit`, `weight_g`, `weight_delta_g` | Grams, current filtered reading, and signed change from the last accepted reading. Weight and delta are `null` until available. |
| `stability_range_g`, `stability_threshold_g` | Max-minus-min range in the active window and its configured threshold. Range is `null` before the first weight sample. |
| `sample_count`, `window_size`, `timestamp_ms` | Window fill, configured window length, and ESP32 uptime in milliseconds. |
| `operation`, `operation_status`, `operation_ok`, `operation_message` | Latest tare/calibration state. `operation_status` is `idle`, `pending`, `succeeded`, or `failed`; `operation_ok` is null until completion. |

`POST /tare` and valid `POST /calibrate` requests return HTTP `202` with the reading fields and operation fields. While samples are collected, status is `pending` and `operation_ok` is null. The Python client polls `GET /weight` until the operation succeeds or fails. Failed starts return a non-2xx response; asynchronous failures appear in the operation fields. Calibration requests must include form-encoded `known_weight_g=<grams>&confirm=CALIBRATE`.

Defaults in `platformio.ini` are a 10-sample window, a 2.0 g max-minus-min stability threshold, and a 100 ms sample period. Build flags can be changed for a separate firmware experiment. A weight becomes stable only after the whole window is populated and its range is at most the threshold. Filtering is a median over up to five raw samples followed by an EMA with alpha 0.35; it reduces short spikes but cannot correct mechanical vibration, creep, off-center loading, or poor mounting. Values whose absolute magnitude exceeds the 5 kg rated capacity are reported as `overload` with no usable weight. Near-saturated 24-bit ADC values are reported as `sensor_error`. Small negative readings after tare are retained as diagnostic values; the host should not record a negative addition as food mass.

## Independent Python client and mock mode

`smartnutri_weight.WeightClient` uses only the Python standard library and has no import or runtime dependency on training code. It retries idempotent GET requests once by default, never automatically retries tare/calibration POSTs, and provides `get_stable_weight()` to wait for a stable real reading while rejecting mock data and sensor errors:

```python
from smartnutri_weight import WeightClient

# Hardware client: set the base URL printed by the ESP32.
import os
scale = WeightClient(
    "http://192.168.1.42",
    control_token=os.environ["SMARTNUTRI_SCALE_CONTROL_TOKEN"],
)
reading = scale.get_weight()
if reading.simulated:
    raise RuntimeError("Do not use a simulated reading as a real measurement")
print(reading.status, reading.weight_g, reading.unit)

# Local-only mock example: this value is synthetic and always marked simulated.
mock = WeightClient(mock_weight_g=125.0).get_weight()
assert mock.source == "mock" and mock.simulated is True
```

Mock mode makes no network request; mock payloads have `sensor_connected=false`, `source="mock"`, `simulated=true`, and `status="mock"`. The backend stability and delta tests use only such explicitly simulated values. Client-side calibration also requires `confirmation="CALIBRATE"` and is disabled in mock mode.

`GET /weight` is read-only. Direct HTTP tare/calibration require a shared secret of at least 32 characters and the exact action confirmation. This authenticates control requests but does not encrypt HTTP traffic; use only a trusted isolated LAN and never expose the device publicly. Keep the secret out of the Flutter app, source control, logs, screenshots, and shared setup instructions.

## Testing and hardware validation

Run the full Python suite and package syntax check from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall smartnutri_weight
```

The scale tests validate API payload parsing, mock labeling, asynchronous operation polling, explicit calibration confirmation, bounded GET reconnect retries, stable-read rejection states, invalid sensor states, the rolling stability rule, and signed gram deltas. They do not emulate HX711 electrical timing or prove load-cell accuracy; successful stable acquisition still requires hardware validation.

### Hardware verification checklist (not run)

1. With power disconnected, check HX711 VCC/GND and DOUT/SCK pin routing. Verify load-cell E+/E-/A+/A- mapping from that cell's markings/datasheet; do not use color assumptions.
2. Connect the ESP32 over USB, open Serial at 115200, and confirm the startup log and `source="hx711"`, `simulated=false` status. A ready sensor should leave `sensor_disconnected`; no weight is usable until tare and calibration are loaded or performed.
3. With the platform empty, send `tare CONFIRM`; wait for `operation_status="succeeded"`. Power-cycle and verify `tared=true` and saved calibration flags persist without automatic re-tare.
4. Perform confirmed calibration using a verified mass, then check at a second known mass without exceeding the 5 kg rating. Record error, repeatability, sign, and off-center response. Do not deliberately overload the load cell to test the software guard.
5. Apply and remove a small mass. Confirm the output passes through `warming_up`/`unstable` before `stable`, that readings settle within the configured threshold/window, and that `weight_delta_g` has the expected sign. Small negative unloaded drift is diagnostic; never record it as added food mass.
6. For sensor-loss behavior, power down before changing wiring, disconnect/reconnect the HX711, and verify `sensor_disconnected`/recovery behavior. Do not hot-plug load-cell wires.
7. With configured Wi-Fi and the local token available only on the trusted laptop, call `GET /weight`, authenticated `POST /tare` with `confirm=TARE` on an empty platform, and authenticated `POST /calibrate` with `confirm=CALIBRATE` on a known mass. Restart the access point and confirm the device reconnects and serves the API again. Keep the API on a trusted LAN.

No physical calibration, flashing, or hardware test was performed here.

## References

- [Arduino-ESP32 Preferences API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/preferences.html) for persistent calibration values.
- [Arduino-ESP32 Wi-Fi API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/wifi.html) and [network events](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/network.html) for connection behavior.
- [Arduino-ESP32 WebServer example](https://github.com/espressif/arduino-esp32/blob/master/libraries/WebServer/examples/WebServer/WebServer.ino) for route registration, `begin()`, and `handleClient()` usage.
- [PlatformIO ESP32 Dev Module](https://docs.platformio.org/en/stable/boards/espressif32/esp32dev.html) for the `esp32dev` board environment.
- [HX711 datasheet](https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/73/HX711.pdf) for 24-bit conversion timing and channel/gain selection.
