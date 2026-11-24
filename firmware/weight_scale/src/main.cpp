#include <Arduino.h>
#include <Preferences.h>
#include <WebServer.h>
#include <WiFi.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "scale_logic.h"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef SMARTNUTRI_WIFI_SSID
#define SMARTNUTRI_WIFI_SSID ""
#endif
#ifndef SMARTNUTRI_WIFI_PASSWORD
#define SMARTNUTRI_WIFI_PASSWORD ""
#endif
#ifndef SMARTNUTRI_SCALE_CONTROL_TOKEN
#define SMARTNUTRI_SCALE_CONTROL_TOKEN ""
#endif
#ifndef SMARTNUTRI_STABILITY_WINDOW
#define SMARTNUTRI_STABILITY_WINDOW 10
#endif
#ifndef SMARTNUTRI_STABILITY_THRESHOLD_G
#define SMARTNUTRI_STABILITY_THRESHOLD_G 2.0
#endif
#ifndef SMARTNUTRI_SAMPLE_PERIOD_MS
#define SMARTNUTRI_SAMPLE_PERIOD_MS 100
#endif

namespace {

constexpr uint8_t kHx711DoutPin = 32;
constexpr uint8_t kHx711SckPin = 33;
constexpr size_t kFilterCapacity = 5;
constexpr size_t kStabilityWindow = SMARTNUTRI_STABILITY_WINDOW;
constexpr uint8_t kAveragingSamples = 8;
constexpr uint32_t kSensorTimeoutMs = 1500;
constexpr uint32_t kOperationTimeoutMs = 3000;
constexpr uint32_t kWifiRetryMs = 10000;
constexpr uint32_t kCommandLogPeriodMs = 1000;
constexpr float kStabilityThresholdG = SMARTNUTRI_STABILITY_THRESHOLD_G;
constexpr float kRatedCapacityG = 5000.0f;
constexpr int32_t kAdcNearSaturation = 8388600;

static_assert(kStabilityWindow > 0, "stability window must be positive");
static_assert(kStabilityWindow <= 32, "stability window must not exceed 32 samples");
static_assert(SMARTNUTRI_SAMPLE_PERIOD_MS > 0, "sample period must be positive");
static_assert(kStabilityThresholdG >= 0.0f, "stability threshold must be non-negative");

enum class OperationType : uint8_t { None, Tare, Calibration };
enum class OperationState : uint8_t { Idle, Pending, Succeeded, Failed };

Preferences preferences;
WebServer server(80);
smartnutri::MedianEmaFilter<kFilterCapacity> rawFilter;
smartnutri::StabilityWindow<kStabilityWindow> stabilityWindow;

bool preferencesReady = false;
bool hasTare = false;
bool hasCalibration = false;
bool sensorHasSample = false;
bool hasWeightReading = false;
bool hasPreviousWeight = false;
bool sensorError = false;
bool overloadDetected = false;
bool wifiWasConnected = false;
bool wifiDisabledByConfig = false;
bool hasLoggedSensorError = false;
bool httpRoutesConfigured = false;
OperationType activeOperation = OperationType::None;
OperationType lastOperation = OperationType::None;
OperationState lastOperationState = OperationState::Idle;
const char* lastOperationMessage = "idle";

int32_t tareOffsetRaw = 0;
float calibrationFactorRawPerGram = 0.0f;
float filteredRaw = 0.0f;
float currentWeightG = 0.0f;
float previousWeightG = 0.0f;
float currentDeltaG = 0.0f;
uint32_t lastSensorSampleMs = 0;
uint32_t operationStartedMs = 0;
uint32_t nextWifiAttemptMs = 0;
uint32_t nextSerialLogMs = 0;
int64_t operationRawSum = 0;
uint8_t operationSampleCount = 0;
float operationKnownWeightG = 0.0f;

bool tryReadHx711(int32_t& raw) {
  if (digitalRead(kHx711DoutPin) != LOW) {
    return false;
  }

  uint32_t value = 0;
  noInterrupts();
  for (uint8_t bit = 0; bit < 24; ++bit) {
    digitalWrite(kHx711SckPin, HIGH);
    delayMicroseconds(1);
    value = (value << 1) | static_cast<uint32_t>(digitalRead(kHx711DoutPin) == HIGH);
    digitalWrite(kHx711SckPin, LOW);
    delayMicroseconds(1);
  }
  // The 25th pulse selects channel A with gain 128 for the next conversion.
  digitalWrite(kHx711SckPin, HIGH);
  delayMicroseconds(1);
  digitalWrite(kHx711SckPin, LOW);
  interrupts();

  if (value & 0x00800000UL) {
    value |= 0xFF000000UL;
  }
  raw = static_cast<int32_t>(value);
  return true;
}

bool sensorConnected() {
  return sensorHasSample && (millis() - lastSensorSampleMs <= kSensorTimeoutMs);
}

void clearWeightWindows() {
  rawFilter.reset();
  stabilityWindow.reset();
  hasWeightReading = false;
  hasPreviousWeight = false;
  currentDeltaG = 0.0f;
}

const char* operationTypeName(OperationType operation) {
  switch (operation) {
    case OperationType::Tare: return "tare";
    case OperationType::Calibration: return "calibrate";
    default: return "none";
  }
}

const char* operationStateName(OperationState state) {
  switch (state) {
    case OperationState::Pending: return "pending";
    case OperationState::Succeeded: return "succeeded";
    case OperationState::Failed: return "failed";
    default: return "idle";
  }
}

void finishOperation(bool success, const char* message) {
  activeOperation = OperationType::None;
  lastOperationState = success ? OperationState::Succeeded : OperationState::Failed;
  lastOperationMessage = message;
  operationSampleCount = 0;
  operationRawSum = 0;
  clearWeightWindows();
}

bool startOperation(OperationType operation) {
  if (activeOperation != OperationType::None) {
    return false;
  }
  lastOperation = operation;
  lastOperationState = OperationState::Failed;
  if (!preferencesReady) {
    lastOperationMessage = "preferences_unavailable";
    return false;
  }
  if (!sensorConnected()) {
    lastOperationMessage = "sensor_disconnected";
    return false;
  }
  if (sensorError || overloadDetected) {
    lastOperationMessage = sensorError ? "sensor_error" : "overload";
    return false;
  }

  activeOperation = operation;
  lastOperationState = OperationState::Pending;
  lastOperationMessage = "collecting_eight_samples";
  operationStartedMs = millis();
  operationRawSum = 0;
  operationSampleCount = 0;
  clearWeightWindows();
  return true;
}

bool startTare() {
  return startOperation(OperationType::Tare);
}

bool startCalibration(float knownWeightG, const String& confirmation) {
  if (confirmation != "CALIBRATE") {
    if (activeOperation == OperationType::None) {
      lastOperation = OperationType::Calibration;
      lastOperationState = OperationState::Failed;
      lastOperationMessage = "explicit_confirmation_required";
    }
    Serial.println(F("Calibration refused: explicit CALIBRATE confirmation is required."));
    return false;
  }
  if (!isfinite(knownWeightG) || knownWeightG <= 0.0f || knownWeightG > kRatedCapacityG) {
    if (activeOperation == OperationType::None) {
      lastOperation = OperationType::Calibration;
      lastOperationState = OperationState::Failed;
      lastOperationMessage = "known_weight_out_of_range";
    }
    Serial.println(F("Calibration refused: known weight must be in (0, 5000] grams."));
    return false;
  }
  if (!hasTare) {
    if (activeOperation == OperationType::None) {
      lastOperation = OperationType::Calibration;
      lastOperationState = OperationState::Failed;
      lastOperationMessage = "tare_required";
    }
    Serial.println(F("Calibration refused: tare the empty scale first."));
    return false;
  }
  if (!startOperation(OperationType::Calibration)) {
    return false;
  }
  operationKnownWeightG = knownWeightG;
  return true;
}

void completeOperationFromSamples() {
  const int32_t average = static_cast<int32_t>(operationRawSum / kAveragingSamples);
  if (activeOperation == OperationType::Tare) {
    if (preferences.putLong("offset_raw", average) == 0 ||
        preferences.putBool("has_tare", true) == 0) {
      finishOperation(false, "preferences_write_failed");
      return;
    }
    tareOffsetRaw = average;
    hasTare = true;
    finishOperation(true, "tare_saved");
    return;
  }

  const float factor = (static_cast<float>(average) - static_cast<float>(tareOffsetRaw)) /
                       operationKnownWeightG;
  if (!isfinite(factor) || fabsf(factor) < 0.000001f) {
    finishOperation(false, "calibration_signal_invalid");
    return;
  }
  if (preferences.putFloat("cal_factor", factor) == 0 ||
      preferences.putBool("has_cal", true) == 0) {
    finishOperation(false, "preferences_write_failed");
    return;
  }
  calibrationFactorRawPerGram = factor;
  hasCalibration = true;
  finishOperation(true, "calibration_saved");
}

void collectOperationSample(int32_t raw) {
  operationRawSum += raw;
  ++operationSampleCount;
  if (operationSampleCount >= kAveragingSamples) {
    completeOperationFromSamples();
  }
}

void expireStaleSensor() {
  const uint32_t now = millis();
  if (activeOperation != OperationType::None && now - operationStartedMs > kOperationTimeoutMs) {
    finishOperation(false, "sample_timeout");
  }
  if (sensorHasSample && now - lastSensorSampleMs > kSensorTimeoutMs) {
    sensorHasSample = false;
    clearWeightWindows();
    if (activeOperation != OperationType::None) {
      finishOperation(false, "sensor_disconnected");
    }
  }
}

void acceptRawSample(int32_t raw) {
  const uint32_t now = millis();
  sensorHasSample = true;
  lastSensorSampleMs = now;

  if (raw >= kAdcNearSaturation || raw <= -kAdcNearSaturation) {
    sensorError = true;
    overloadDetected = false;
    clearWeightWindows();
    if (activeOperation != OperationType::None) {
      finishOperation(false, "adc_saturated");
    }
    return;
  }
  sensorError = false;

  if (activeOperation != OperationType::None) {
    collectOperationSample(raw);
    return;
  }

  filteredRaw = rawFilter.update(static_cast<float>(raw));

  if (!hasTare || !hasCalibration || fabsf(calibrationFactorRawPerGram) < 0.000001f) {
    return;
  }

  const float weight = (filteredRaw - static_cast<float>(tareOffsetRaw)) /
                       calibrationFactorRawPerGram;
  if (!isfinite(weight)) {
    sensorError = true;
    overloadDetected = false;
    clearWeightWindows();
    return;
  }
  if (fabsf(weight) > kRatedCapacityG) {
    overloadDetected = true;
    clearWeightWindows();
    return;
  }
  overloadDetected = false;
  if (hasPreviousWeight) {
    currentDeltaG = smartnutri::weightDeltaGrams(weight, previousWeightG);
  } else {
    currentDeltaG = 0.0f;
    hasPreviousWeight = true;
  }
  previousWeightG = weight;
  currentWeightG = weight;
  hasWeightReading = true;
  stabilityWindow.add(weight);
}

String nullableFloat(float value, bool valid, uint8_t decimals = 2) {
  return valid && isfinite(value)
             ? String(value, static_cast<unsigned int>(decimals))
             : String("null");
}

const char* currentStatus() {
  if (!sensorConnected()) return "sensor_disconnected";
  if (sensorError) return "sensor_error";
  if (activeOperation == OperationType::Tare) return "taring";
  if (activeOperation == OperationType::Calibration) return "calibrating";
  if (overloadDetected) return "overload";
  if (!hasTare) return "not_tared";
  if (!hasCalibration) return "not_calibrated";
  if (!hasWeightReading || stabilityWindow.sampleCount() < kStabilityWindow) return "warming_up";
  if (!stabilityWindow.isStable(kStabilityThresholdG)) return "unstable";
  return "stable";
}

String weightJson() {
  const bool connected = sensorConnected();
  const bool hasWeight = connected && !sensorError && !overloadDetected &&
                         activeOperation == OperationType::None && hasTare &&
                         hasCalibration && hasWeightReading;
  float rangeG = 0.0f;
  const bool hasRange = hasWeight && stabilityWindow.range(rangeG);
  const char* status = currentStatus();
  const bool ok = connected && !sensorError && !overloadDetected &&
                  activeOperation == OperationType::None && hasTare && hasCalibration;

  String json = "{";
  json += "\"ok\":" + String(ok ? "true" : "false");
  json += ",\"source\":\"hx711\",\"simulated\":false";
  json += ",\"sensor_connected\":" + String(connected ? "true" : "false");
  json += ",\"tared\":" + String(hasTare ? "true" : "false");
  json += ",\"calibrated\":" + String(hasCalibration ? "true" : "false");
  json += ",\"stable\":" + String(strcmp(status, "stable") == 0 ? "true" : "false");
  json += ",\"status\":\"" + String(status) + "\",\"unit\":\"g\"";
  json += ",\"weight_g\":" + nullableFloat(currentWeightG, hasWeight, 2);
  json += ",\"weight_delta_g\":" + nullableFloat(currentDeltaG, hasWeight && hasPreviousWeight, 2);
  json += ",\"stability_range_g\":" + nullableFloat(rangeG, hasRange, 2);
  json += ",\"stability_threshold_g\":" + String(kStabilityThresholdG, 2u);
  json += ",\"sample_count\":" + String(stabilityWindow.sampleCount());
  json += ",\"window_size\":" + String(kStabilityWindow);
  json += ",\"timestamp_ms\":" + String(millis());
  const OperationType reportedOperation = activeOperation != OperationType::None
                                              ? activeOperation
                                              : lastOperation;
  json += ",\"operation\":\"" + String(operationTypeName(reportedOperation)) + "\"";
  json += ",\"operation_status\":\"" + String(operationStateName(lastOperationState)) + "\"";
  json += ",\"operation_ok\":";
  if (lastOperationState == OperationState::Pending || lastOperationState == OperationState::Idle) {
    json += "null";
  } else {
    json += lastOperationState == OperationState::Succeeded ? "true" : "false";
  }
  json += ",\"operation_message\":\"" + String(lastOperationMessage) + "\"";
  json += "}";
  return json;
}

void handleWeight() {
  server.send(200, "application/json", weightJson());
}

bool requireControlToken() {
  const String expected = SMARTNUTRI_SCALE_CONTROL_TOKEN;
  if (expected.length() < 32) {
    server.send(503, "application/json", "{\"ok\":false,\"error\":\"scale_control_token_not_configured\"}");
    return false;
  }
  const char* headerName = "X-SmartNutri-Control-Token";
  if (!server.hasHeader(headerName)) {
    server.send(403, "application/json", "{\"ok\":false,\"error\":\"control_token_required\"}");
    return false;
  }
  const String supplied = server.header(headerName);
  if (supplied.length() != expected.length()) {
    server.send(403, "application/json", "{\"ok\":false,\"error\":\"control_token_invalid\"}");
    return false;
  }
  uint8_t mismatch = 0;
  for (size_t index = 0; index < expected.length(); ++index) {
    mismatch |= static_cast<uint8_t>(expected[index] ^ supplied[index]);
  }
  if (mismatch != 0) {
    server.send(403, "application/json", "{\"ok\":false,\"error\":\"control_token_invalid\"}");
    return false;
  }
  return true;
}

void handleTare() {
  if (!requireControlToken()) return;
  if (!server.hasArg("confirm") || server.arg("confirm") != "TARE") {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"explicit_TARE_confirmation_required\"}");
    return;
  }
  Serial.println(F("Authenticated HTTP tare requested with explicit TARE confirmation."));
  const bool started = startTare();
  const int code = started ? 202 : (activeOperation != OperationType::None ? 409 : 503);
  server.send(code, "application/json", weightJson());
}

void handleCalibration() {
  if (!requireControlToken()) return;
  if (!server.hasArg("known_weight_g") || !server.hasArg("confirm")) {
    server.send(400, "application/json", "{\"ok\":false,\"error\":\"known_weight_g_and_confirm_required\"}");
    return;
  }
  const float knownWeightG = server.arg("known_weight_g").toFloat();
  const String confirmation = server.arg("confirm");
  const bool started = startCalibration(knownWeightG, confirmation);
  const int code = started ? 202 : (activeOperation != OperationType::None ? 409 :
                                   (sensorConnected() ? 400 : 503));
  server.send(code, "application/json", weightJson());
}

void handleNotFound() {
  server.send(404, "application/json", "{\"ok\":false,\"error\":\"not_found\"}");
}

void startHttpServer() {
  if (!httpRoutesConfigured) {
    const char* headerKeys[] = {"X-SmartNutri-Control-Token"};
    server.collectHeaders(headerKeys, 1);
    server.on("/weight", HTTP_GET, handleWeight);
    server.on("/tare", HTTP_POST, handleTare);
    server.on("/calibrate", HTTP_POST, handleCalibration);
    server.onNotFound(handleNotFound);
    httpRoutesConfigured = true;
  }
  // begin() is re-run after Wi-Fi reconnects; route handlers are registered once.
  server.begin();
  Serial.println(F("HTTP API listening on port 80: GET /weight, POST /tare, POST /calibrate"));
}

void serviceWifi() {
  if (wifiDisabledByConfig) return;
  const uint32_t now = millis();
  if (WiFi.status() == WL_CONNECTED) {
    if (!wifiWasConnected) {
      wifiWasConnected = true;
      startHttpServer();
      Serial.print(F("Wi-Fi connected; scale API at http://"));
      Serial.println(WiFi.localIP());
    }
    server.handleClient();
    return;
  }

  if (wifiWasConnected) {
    wifiWasConnected = false;
    Serial.println(F("Wi-Fi disconnected; API paused while USB serial remains available."));
  }
  if (static_cast<int32_t>(now - nextWifiAttemptMs) >= 0) {
    nextWifiAttemptMs = now + kWifiRetryMs;
    Serial.println(F("Attempting Wi-Fi connection..."));
    WiFi.disconnect();
    WiFi.begin(SMARTNUTRI_WIFI_SSID, SMARTNUTRI_WIFI_PASSWORD);
  }
}

void printHelp() {
  Serial.println(F("Commands: status | tare CONFIRM | calibrate <known_weight_g> CONFIRM | help"));
  Serial.println(F("Tare/calibration require the exact confirmation token; HTTP controls also require a secret header."));
}

void handleSerialCommand(String command) {
  command.trim();
  if (command.length() == 0) return;
  if (command == "help") {
    printHelp();
  } else if (command == "status") {
    Serial.println(weightJson());
  } else if (command == "tare CONFIRM") {
    startTare();
    Serial.println(weightJson());
  } else if (command == "tare") {
    Serial.println(F("Tare refused. Enter: tare CONFIRM"));
  } else if (command.startsWith("calibrate ")) {
    const String args = command.substring(10);
    const int separator = args.indexOf(' ');
    if (separator <= 0 || args.substring(separator + 1) != "CONFIRM") {
      Serial.println(F("Calibration refused. Enter: calibrate <known_weight_g> CONFIRM"));
      return;
    }
    const float knownWeightG = args.substring(0, separator).toFloat();
    startCalibration(knownWeightG, args.substring(separator + 1));
    Serial.println(weightJson());
  } else {
    Serial.println(F("Unknown command."));
    printHelp();
  }
}

void serviceSerial() {
  if (Serial.available()) {
    handleSerialCommand(Serial.readStringUntil('\n'));
  }
  const uint32_t now = millis();
  if (static_cast<int32_t>(now - nextSerialLogMs) >= 0) {
    nextSerialLogMs = now + kCommandLogPeriodMs;
    const char* status = currentStatus();
    if (strcmp(status, "sensor_disconnected") == 0 && !hasLoggedSensorError) {
      Serial.println(F("ERROR: HX711 not responding (DOUT has not produced a sample). Check power, wiring and load cell."));
      hasLoggedSensorError = true;
    } else if (strcmp(status, "sensor_error") == 0 && !hasLoggedSensorError) {
      Serial.println(F("ERROR: HX711 returned a saturated or invalid ADC value; weight is suppressed."));
      hasLoggedSensorError = true;
    } else if (strcmp(status, "overload") == 0 && !hasLoggedSensorError) {
      Serial.println(F("ERROR: measured magnitude exceeds the 5 kg software capacity; weight is suppressed."));
      hasLoggedSensorError = true;
    } else if (strcmp(status, "sensor_disconnected") != 0) {
      if (strcmp(status, "sensor_error") != 0 && strcmp(status, "overload") != 0) {
        hasLoggedSensorError = false;
      }
    }
    Serial.println(weightJson());
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(250);
  Serial.println();
  Serial.println(F("SmartNutri ESP32 weighing scale booting (HX711 source; no simulated readings)."));

  pinMode(kHx711DoutPin, INPUT_PULLUP);
  pinMode(kHx711SckPin, OUTPUT);
  digitalWrite(kHx711SckPin, LOW);

  preferencesReady = preferences.begin("smartnutri", false);
  if (preferencesReady) {
    tareOffsetRaw = preferences.getLong("offset_raw", 0);
    calibrationFactorRawPerGram = preferences.getFloat("cal_factor", 0.0f);
    hasTare = preferences.getBool("has_tare", false);
    hasCalibration = preferences.getBool("has_cal", false) &&
                     isfinite(calibrationFactorRawPerGram) &&
                     fabsf(calibrationFactorRawPerGram) >= 0.000001f;
    Serial.println(F("Calibration settings loaded from ESP32 Preferences."));
  } else {
    Serial.println(F("ERROR: Preferences could not open; readings remain uncalibrated and settings cannot be saved."));
  }

  if (SMARTNUTRI_WIFI_SSID[0] == '\0') {
    wifiDisabledByConfig = true;
    Serial.println(F("Wi-Fi credentials are empty; USB Serial control is active."));
  } else {
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    nextWifiAttemptMs = 0;
  }
  printHelp();
}

void loop() {
  const uint32_t now = millis();
  expireStaleSensor();
  static uint32_t nextSampleMs = 0;
  if (static_cast<int32_t>(now - nextSampleMs) >= 0) {
    nextSampleMs = now + SMARTNUTRI_SAMPLE_PERIOD_MS;
    int32_t raw = 0;
    if (tryReadHx711(raw)) {
      acceptRawSample(raw);
    }
  }
  serviceWifi();
  serviceSerial();
  delay(1);
}
