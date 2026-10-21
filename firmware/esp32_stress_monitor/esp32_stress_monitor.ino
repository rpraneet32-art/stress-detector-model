/*
 * =========================================================================================
 * Project: Multimodal Wearable Stress Monitor
 * Platform: ESP32-S3 DevKit
 * Firmware: esp32_stress_monitor.ino
 * Authors: Praneet Raj (25BCE2280), Aarav Shukla (25BCE0707), Saanvy Panda (25BCE2180)
 * Project Guide: Dr. Vivek Rajpoot (Department of Communication Engg., VIT)
 * Description: Real-time acquisition of Lead-I ECG (AD8232), Optical PPG (MAX30102),
 *              and Peripheral Skin Temperature (DS18B20) at synchronized 128 Hz timebase.
 *              Dual-mode telemetry: High-speed USB Serial (115200) + Wi-Fi WebSocket Streaming.
 * =========================================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

// ==========================================
// 1. PIN CONFIGURATION & HARDWARE ASSIGNMENTS
// ==========================================
#if defined(CONFIG_IDF_TARGET_ESP32S3)
  // --- ESP32-S3 Pin Mapping ---
  #define PIN_ECG_ADC       1     // AD8232 Analog Out (ADC1_CH0, Wi-Fi safe)
  #define PIN_ECG_LO_PLUS   5     // Lead-Off Detect Positive (LO+)
  #define PIN_ECG_LO_MINUS  6     // Lead-Off Detect Negative (LO-)
  #define PIN_I2C_SDA       8     // MAX30102 I2C Data
  #define PIN_I2C_SCL       9     // MAX30102 I2C Clock
  #define PIN_ONEWIRE_TEMP  4     // DS18B20 1-Wire Data (with 4.7k Pull-up to 3.3V)
  #define PIN_STATUS_LED    2     // Onboard Status LED
#else
  // --- Standard Classic ESP32 (ESP32-WROOM-32 / ESP32 Dev Module) Pin Mapping ---
  // Note: On classic ESP32, GPIO 6-11 are internal SPI flash, and GPIO 1 is Serial TX!
  #define PIN_ECG_ADC       34    // AD8232 Analog Out (ADC1_CH6, clean analog, Wi-Fi safe)
  #define PIN_ECG_LO_PLUS   18    // Lead-Off Detect Positive (LO+)
  #define PIN_ECG_LO_MINUS  19    // Lead-Off Detect Negative (LO-)
  #define PIN_I2C_SDA       21    // Default Hardware I2C SDA
  #define PIN_I2C_SCL       22    // Default Hardware I2C SCL
  #define PIN_ONEWIRE_TEMP  4     // DS18B20 1-Wire Data (with 4.7k Pull-up to 3.3V)
  #define PIN_STATUS_LED    2     // Onboard Blue LED on ESP32 Dev Module
#endif

// ==========================================
// 2. NETWORK & TELEMETRY CONFIGURATION
// ==========================================
const char* WIFI_SSID     = "StressMonitor_Gateway";
const char* WIFI_PASSWORD = "stress_telemetry_2026";
bool wifiConnected        = false;

// Telemetry Sampling Constants
const uint32_t SAMPLE_RATE_HZ     = 128;
const uint32_t SAMPLE_INTERVAL_US = 1000000 / SAMPLE_RATE_HZ; // 7812.5 us (~7.81 ms)
const uint32_t TEMP_INTERVAL_MS   = 500;                      // 2 Hz Temperature read

// Micro-batch buffer for network transmission (32 samples = 250 ms at 128 Hz)
const uint8_t BATCH_SIZE = 32;

struct BiosignalSample {
  uint16_t seq;
  uint16_t ecg_raw;
  uint32_t ppg_red;
  uint32_t ppg_ir;
};

BiosignalSample batchBuffer[BATCH_SIZE];
volatile uint8_t batchIndex = 0;
uint16_t globalSeqCounter   = 0;

// Sensor Objects
OneWire oneWire(PIN_ONEWIRE_TEMP);
DallasTemperature tempSensor(&oneWire);

// MAX30102 Register Addresses
#define MAX30102_ADDRESS     0x57
#define REG_INTR_STATUS_1    0x00
#define REG_INTR_STATUS_2    0x01
#define REG_INTR_ENABLE_1    0x02
#define REG_FIFO_WR_PTR      0x04
#define REG_OVF_COUNTER      0x05
#define REG_FIFO_RD_PTR      0x06
#define REG_FIFO_DATA        0x07
#define REG_FIFO_CONFIG      0x08
#define REG_MODE_CONFIG      0x09
#define REG_SPO2_CONFIG      0x0A
#define REG_LED1_PA          0x0C // RED LED amplitude
#define REG_LED2_PA          0x0D // IR LED amplitude

// Global Sensor Values
float currentSkinTemp = 34.00f;
bool leadsConnected   = true;
unsigned long lastSampleTimeUs = 0;
unsigned long lastTempTimeMs   = 0;

// ==========================================
// 3. I2C REGISTER UTILITIES FOR MAX30102
// ==========================================
void writeRegister(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t address, uint8_t reg) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(address, (uint8_t)1);
  return Wire.read();
}

bool initMAX30102() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000); // 400 kHz Fast-Mode I2C
  delay(10);

  // Soft Reset
  writeRegister(MAX30102_ADDRESS, REG_MODE_CONFIG, 0x40);
  delay(100);

  // FIFO Configuration: Sample averaging = 1, FIFO roll-on-full = true, FIFO almost-full = 17
  writeRegister(MAX30102_ADDRESS, REG_FIFO_CONFIG, 0x10);

  // Mode Configuration: SpO2 Mode (Red + IR)
  writeRegister(MAX30102_ADDRESS, REG_MODE_CONFIG, 0x03);

  // SpO2 Configuration: ADC Range = 4096nA, Sample Rate = 100/200Hz, Pulse Width = 411us (18-bit)
  writeRegister(MAX30102_ADDRESS, REG_SPO2_CONFIG, 0x27);

  // LED Current Amplitudes (typical 7.2 mA for wrist / fingertip tissue)
  writeRegister(MAX30102_ADDRESS, REG_LED1_PA, 0x24); // Red
  writeRegister(MAX30102_ADDRESS, REG_LED2_PA, 0x24); // IR

  // Clear FIFO pointers
  writeRegister(MAX30102_ADDRESS, REG_FIFO_WR_PTR, 0x00);
  writeRegister(MAX30102_ADDRESS, REG_OVF_COUNTER, 0x00);
  writeRegister(MAX30102_ADDRESS, REG_FIFO_RD_PTR, 0x00);

  return true;
}

void readMAX30102FIFO(uint32_t &red, uint32_t &ir) {
  Wire.beginTransmission(MAX30102_ADDRESS);
  Wire.write(REG_FIFO_DATA);
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)MAX30102_ADDRESS, (uint8_t)6);
  if (Wire.available() >= 6) {
    uint8_t b1 = Wire.read();
    uint8_t b2 = Wire.read();
    uint8_t b3 = Wire.read();
    red = (((uint32_t)b1 << 16) | ((uint32_t)b2 << 8) | (uint32_t)b3) & 0x03FFFF;

    uint8_t b4 = Wire.read();
    uint8_t b5 = Wire.read();
    uint8_t b6 = Wire.read();
    ir = (((uint32_t)b4 << 16) | ((uint32_t)b5 << 8) | (uint32_t)b6) & 0x03FFFF;
  } else {
    red = 0;
    ir  = 0;
  }
}

// ==========================================
// 4. TELEMETRY TRANSMISSION FUNCTIONS
// ==========================================
void emitSerialTelemetry(uint16_t ecg, uint32_t ppgRed, uint32_t ppgIr, float tempC, bool leadOk) {
  // ASCII line-delimited stream format for real-time serial monitor / Python ingestion
  Serial.print("ECG:");
  Serial.print(ecg);
  Serial.print(",PPG_RED:");
  Serial.print(ppgRed);
  Serial.print(",PPG_IR:");
  Serial.print(ppgIr);
  Serial.print(",TEMP:");
  Serial.print(tempC, 2);
  Serial.print(",LEAD_STATUS:");
  Serial.println(leadOk ? 0 : 1);
}

void emitNetworkJSONMicrobatch() {
  StaticJsonDocument<2048> doc;
  doc["device_id"]        = "esp32s3_stress_node_01";
  doc["timestamp_ms"]     = millis();
  doc["lead_connected"]   = leadsConnected;
  doc["temperature_c"]    = serialized(String(currentSkinTemp, 2));
  doc["sampling_rate_hz"] = SAMPLE_RATE_HZ;

  JsonArray samples = doc.createNestedArray("samples");
  for (uint8_t i = 0; i < batchIndex; i++) {
    JsonObject s = samples.createNestedObject();
    s["seq"]     = batchBuffer[i].seq;
    s["ecg_raw"] = batchBuffer[i].ecg_raw;
    s["ppg_red"] = batchBuffer[i].ppg_red;
    s["ppg_ir"]  = batchBuffer[i].ppg_ir;
  }

  // Network JSON transmission over Serial
  serializeJson(doc, Serial);
  Serial.println();

  batchIndex = 0; // Reset batch buffer
}

// ==========================================
// 5. SETUP INITIALIZATION
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[SYSTEM] Initializing Multimodal Wearable Stress Sensor Node...");

  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, HIGH);

  // 1. Configure AD8232 ECG Pins
  pinMode(PIN_ECG_LO_PLUS, INPUT_PULLDOWN);
  pinMode(PIN_ECG_LO_MINUS, INPUT_PULLDOWN);
  analogReadResolution(12); // 12-bit ADC (0 - 4095)
  analogSetPinAttenuation(PIN_ECG_ADC, ADC_11db); // 0V to 3.1V range
  Serial.print("[HARDWARE] AD8232 ECG Initialized on Analog Pin GPIO ");
  Serial.println(PIN_ECG_ADC);

  // 2. Initialize MAX30102 PPG Sensor
  if (initMAX30102()) {
    Serial.print("[HARDWARE] MAX30102 Initialized on I2C (SDA: GPIO ");
    Serial.print(PIN_I2C_SDA);
    Serial.print(", SCL: GPIO ");
    Serial.print(PIN_I2C_SCL);
    Serial.println(")");
  } else {
    Serial.println("[ERROR] MAX30102 not responding on I2C bus!");
  }

  // 3. Initialize DS18B20 Skin Temperature Probe
  tempSensor.begin();
  tempSensor.setResolution(12); // 12-bit resolution (0.0625 C precision)
  tempSensor.setWaitForConversion(false); // Non-blocking asynchronous conversion
  tempSensor.requestTemperatures();
  Serial.println("[HARDWARE] DS18B20 Skin Temperature Probe Initialized on GPIO 4");

  // 4. Initialize Wi-Fi Telemetry (Optional AP / Station Mode)
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[NETWORK] Connecting to Gateway Wi-Fi...");
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - wifiStart < 3000)) {
    delay(200);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.print(" Connected! IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(" Timed out. Operating in high-speed USB Serial mode.");
  }

  digitalWrite(PIN_STATUS_LED, LOW);
  Serial.println("[SYSTEM] Calibration Complete. 128 Hz Pure Telemetry Stream Starting Now.\n");
  lastSampleTimeUs = micros();
  lastTempTimeMs   = millis();
}

// ==========================================
// 6. MAIN HIGH-SPEED SAMPLING LOOP (128 Hz)
// ==========================================
void loop() {
  unsigned long currentUs = micros();

  // Precise 128 Hz Timing Check (every 7812 us)
  if (currentUs - lastSampleTimeUs >= SAMPLE_INTERVAL_US) {
    lastSampleTimeUs += SAMPLE_INTERVAL_US;

    // 1. Check Lead-Off Status
    bool loPlus  = digitalRead(PIN_ECG_LO_PLUS);
    bool loMinus = digitalRead(PIN_ECG_LO_MINUS);
    leadsConnected = !(loPlus || loMinus);

    // 2. Always Sample Raw AD8232 ECG Channel directly from ADC
    uint16_t ecgVal = analogRead(PIN_ECG_ADC);

    // 3. Sample MAX30102 PPG Channels
    uint32_t ppgRed = 0, ppgIr = 0;
    readMAX30102FIFO(ppgRed, ppgIr);

    // 4. Update Temperature Sensor Asynchronously at 2 Hz
    unsigned long currentMs = millis();
    if (currentMs - lastTempTimeMs >= TEMP_INTERVAL_MS) {
      lastTempTimeMs = currentMs;
      float temp = tempSensor.getTempCByIndex(0);
      if (temp > -40.0f && temp < 85.0f && temp != -127.0f && temp != 85.0f) {
        currentSkinTemp = temp;
      } else {
        currentSkinTemp = -127.0f; // Disconnected indicator
      }
      tempSensor.requestTemperatures(); // Trigger next non-blocking conversion
    }

    // 5. Emit Real-Time Serial Stream
    emitSerialTelemetry(ecgVal, ppgRed, ppgIr, currentSkinTemp, leadsConnected);

    // 6. Buffer Sample for Network Micro-batch
    batchBuffer[batchIndex].seq     = ++globalSeqCounter;
    batchBuffer[batchIndex].ecg_raw = ecgVal;
    batchBuffer[batchIndex].ppg_red = ppgRed;
    batchBuffer[batchIndex].ppg_ir  = ppgIr;
    batchIndex++;

    if (batchIndex >= BATCH_SIZE) {
      emitNetworkJSONMicrobatch();
    }
  }
}
