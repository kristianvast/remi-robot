#include <WiFi.h>
#include <ESPmDNS.h>
#include <ArduinoOTA.h>
#include "esp_camera.h"
#include "esp_http_server.h"

#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#endif
#ifndef HOME_WIFI_SSID
#define HOME_WIFI_SSID ""
#define HOME_WIFI_PASSWORD ""
#endif

#define HOSTNAME "elegoo-car"
#define BRIDGE_PORT 100
#define MAX_MISSED_BEATS 3

#if CONFIG_IDF_TARGET_ESP32S3
#define UNO_RX 3
#define UNO_TX 40

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13
#else
#define UNO_RX 33
#define UNO_TX 4
#define LED_PIN 13

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM 15
#define XCLK_GPIO_NUM 27
#define SIOD_GPIO_NUM 22
#define SIOC_GPIO_NUM 23
#define Y9_GPIO_NUM 19
#define Y8_GPIO_NUM 36
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 39
#define Y5_GPIO_NUM 5
#define Y4_GPIO_NUM 34
#define Y3_GPIO_NUM 35
#define Y2_GPIO_NUM 32
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 26
#define PCLK_GPIO_NUM 21
#endif

WiFiServer bridge(BRIDGE_PORT);
WiFiClient client;
String fromClient;
String fromUno;
bool gotBeat = false;
uint8_t missedBeats = 0;
uint32_t lastBeat = 0;

static esp_err_t indexHandler(httpd_req_t *req) {
  static const char page[] =
      "<!doctype html><meta name=viewport content='width=device-width'><title>ELEGOO car</title>"
      "<body style='margin:0;background:#000'><img id=v style='width:100%'>"
      "<script>v.src=location.protocol+'//'+location.hostname+':81/stream'</script>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, page, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t captureHandler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// Uptime, last reset reason, Wi-Fi signal and free heap: tells a reboot from a Wi-Fi drop.
static esp_err_t statusHandler(httpd_req_t *req) {
  char body[160];
  snprintf(body, sizeof body,
           "{\"uptime_s\":%lu,\"reset_reason\":%d,\"rssi\":%d,\"free_heap\":%u,\"free_psram\":%u}",
           (unsigned long)(millis() / 1000), (int)esp_reset_reason(), (int)WiFi.RSSI(),
           (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, body, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t streamHandler(httpd_req_t *req) {
  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  char part[96];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;
    size_t len = snprintf(part, sizeof part,
                          "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                          (unsigned)fb->len);
    esp_err_t res = httpd_resp_send_chunk(req, part, len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) return res;
  }
}

static void registerUri(httpd_handle_t server, const char *uri, esp_err_t (*handler)(httpd_req_t *)) {
  httpd_uri_t u = {};
  u.uri = uri;
  u.method = HTTP_GET;
  u.handler = handler;
  httpd_register_uri_handler(server, &u);
}

static void startHttp() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  httpd_handle_t web = NULL;
  httpd_handle_t stream = NULL;
  if (httpd_start(&web, &cfg) == ESP_OK) {
    registerUri(web, "/", indexHandler);
    registerUri(web, "/capture", captureHandler);
    registerUri(web, "/status", statusHandler);
  }
  cfg.server_port = 81;
  cfg.ctrl_port += 1;
  if (httpd_start(&stream, &cfg) == ESP_OK) registerUri(stream, "/stream", streamHandler);
}

static bool startCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 10000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;
  c.jpeg_quality = 12;
  c.fb_count = psramFound() ? 2 : 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

static void startWifi() {
  uint64_t mac = ESP.getEfuseMac();
  char apName[32];
  snprintf(apName, sizeof apName, "ELEGOO-%04X%08X", (uint16_t)(mac >> 32), (uint32_t)mac);
  WiFi.setHostname(HOSTNAME);
  WiFi.mode(WIFI_AP_STA);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  WiFi.softAP(apName, "", 9);
  Serial.printf("fallback AP %s at %s\n", apName, WiFi.softAPIP().toString().c_str());
  if (strlen(HOME_WIFI_SSID) == 0) return;
  WiFi.setAutoReconnect(true);
  WiFi.begin(HOME_WIFI_SSID, HOME_WIFI_PASSWORD);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) delay(250);
  if (WiFi.status() == WL_CONNECTED) {
    // The fallback AP shares the one radio and modem sleep delays every packet: both stall the camera stream.
    WiFi.softAPdisconnect(true);
    WiFi.setSleep(false);
    Serial.printf("joined %s as %s\n", HOME_WIFI_SSID, WiFi.localIP().toString().c_str());
  } else
    Serial.printf("could not join %s yet; retrying in background\n", HOME_WIFI_SSID);
}

static void stopCar() {
  Serial2.print("{\"N\":100}");
}

static void dropClient(const char *why) {
  client.stop();
  stopCar();
  Serial.printf("[client dropped: %s]\n", why);
}

static void serviceBridge() {
  if (!client.connected()) {
    if (client) dropClient("disconnected");
    WiFiClient incoming = bridge.accept();
    if (incoming) {
      client = incoming;
      client.setNoDelay(true);
      fromClient = "";
      fromUno = "";
      gotBeat = true;
      missedBeats = 0;
      lastBeat = millis();
      Serial.printf("[client %s]\n", client.remoteIP().toString().c_str());
    }
    while (Serial2.available()) Serial2.read();
    return;
  }

  while (client.available()) {
    char ch = client.read();
    if (ch == '{') fromClient = "";
    if (ch != ' ') fromClient += ch;
    if (ch == '}') {
      if (fromClient == "{Heartbeat}") gotBeat = true;
      else if (fromClient[0] == '{') Serial2.print(fromClient);
      fromClient = "";
    } else if (fromClient.length() > 200) {
      fromClient = "";
    }
  }

  while (Serial2.available()) {
    char ch = Serial2.read();
    fromUno += ch;
    if (ch == '}') {
      client.print(fromUno);
      fromUno = "";
    } else if (fromUno.length() > 200) {
      fromUno = "";
    }
  }

  if (millis() - lastBeat >= 1000) {
    lastBeat = millis();
    client.print("{Heartbeat}");
    missedBeats = gotBeat ? 0 : missedBeats + 1;
    gotBeat = false;
    if (missedBeats > MAX_MISSED_BEATS) dropClient("heartbeat lost");
  }
}

static void updateLed() {
#ifdef LED_PIN
  static uint32_t last = 0;
  static bool on = false;
  if (client.connected()) {
    digitalWrite(LED_PIN, LOW);
  } else if (millis() - last > 300) {
    last = millis();
    on = !on;
    digitalWrite(LED_PIN, on);
  }
#endif
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1, UNO_RX, UNO_TX);
#ifdef LED_PIN
  pinMode(LED_PIN, OUTPUT);
#endif
  startCamera();
  startWifi();
  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    MDNS.addService("elegoo", "tcp", BRIDGE_PORT);
  }
  ArduinoOTA.setHostname(HOSTNAME);
  ArduinoOTA.setMdnsEnabled(false);
  ArduinoOTA.onStart(stopCar);
  ArduinoOTA.begin();
  startHttp();
  bridge.begin();
  bridge.setNoDelay(true);
  Serial.println("ready: http://" HOSTNAME ".local/  stream :81/stream  control tcp :100");
}

void loop() {
  ArduinoOTA.handle();
  serviceBridge();
  updateLed();
  delay(1);
}
