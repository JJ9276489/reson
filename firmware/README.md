# Firmware

This folder contains both the legacy Arduino sketch and the modular C++ PlatformIO firmware.

## Available firmware targets

1. Arduino sketch (legacy):
   - Path: `firmware/esp32_emg_stream/esp32_emg_stream.ino`
2. PlatformIO C++ (recommended for future IMU work):
   - Path: `firmware/esp32_emg_stream_cpp/`

Both targets use the same stream contract:
- Output: `t raw env` (space-delimited integers)
- Baud: `230400`
- Target sample rate: `~250 Hz`

Example line:

```text
12345 2089 37
```

## Flash (Arduino IDE path)

1. Open `firmware/esp32_emg_stream/esp32_emg_stream.ino` in Arduino IDE.
2. Select ESP32 board and serial port.
3. Upload.
4. Open serial monitor (or `pyserial-miniterm`) at `230400`.

## Flash (PlatformIO path)

```bash
cd /Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream_cpp
pio run -t upload
pio device monitor -b 230400
```

## Verify stream from terminal

```bash
pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw
```

If no data appears, press ESP32 reset once.

## Important

If you change the serial row format, update parser/tests/docs together:
- `src/reson/parser.py`
- tests for parser/pipeline
- `README.md` and `AGENTS.md`
