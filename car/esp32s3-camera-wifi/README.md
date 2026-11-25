# ESP32-S3 camera firmware

This is ELEGOO's 2023 ESP32-S3 camera firmware with station-mode WiFi support.
It keeps the ELEGOO access point available while also joining a configured
WiFi network. The original serial bridge to the car's Arduino remains on TCP
port 100.

## Configure

Create a local `wifi_secrets.h` beside the sketch, using
`ESP32_CameraServer_S3_STA/wifi_secrets.example.h` as a template. Set
`HOME_WIFI_SSID` and `HOME_WIFI_PASSWORD`. The local secrets file is ignored by
Git.

## Build and flash

Use Arduino-ESP32 core 2.0.14 and select **ESP32S3 Dev Module** with:

- USB CDC On Boot: Enabled
- Flash Size: 8MB
- Partition Scheme: 8M with SPIFFS
- PSRAM: OPI PSRAM

Compile the sketch folder with the matching board settings. On Linux, the camera
module usually appears as `/dev/ttyACM0`. Upload at 115200 baud if the default
high upload speed is unreliable.

The camera web page is served on port 80. Driving commands use TCP port 100
and require `{Heartbeat}` messages about once per second.

## Controls

`../tools/car_wifi_teleop.py [ip]` provides keyboard control from the repo
root. Use `car_teleop.py [serial-port]` for USB serial control of the Arduino.
