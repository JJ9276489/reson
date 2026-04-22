# Firmware

This folder contains both the legacy Arduino sketch and the modular C++ PlatformIO firmware.

## Available Firmware Targets

1. Arduino sketch (legacy):
   - Path: `firmware/esp32_emg_stream/esp32_emg_stream.ino`
2. PlatformIO C++ firmware (recommended for future IMU work):
   - Path: `firmware/esp32_emg_stream_cpp/`

Both targets use the same stream contract:

- Output: `t raw env` as space-delimited integers.
- Baud: `230400`.
- Target sample rate: approximately `250 Hz`.

Example line:

```text
12345 2089 37
```

## Wiring

Current prototype wiring:

| AD8232 pin | ESP32 pin |
| --- | --- |
| `3.3V` | `3V3` |
| `GND` | `GND` |
| `OUTPUT` | `GPIO34` |

`LO+` and `LO-` are not used by current firmware.

## Flash With Arduino IDE

1. Open `firmware/esp32_emg_stream/esp32_emg_stream.ino` in Arduino IDE.
2. Select ESP32 board and serial port.
3. Upload.
4. Open serial monitor or `pyserial-miniterm` at `230400` baud.

## Flash With PlatformIO

From repo root:

```bash
python -m pip install -r requirements-firmware.txt
make firmware-upload
make firmware-monitor
```

Equivalent direct commands:

```bash
cd firmware/esp32_emg_stream_cpp
pio run -t upload
pio device monitor -b 230400
```

If your board needs explicit port selection:

```bash
cd firmware/esp32_emg_stream_cpp
pio run -t upload --upload-port /dev/cu.usbserial-XXXX
pio device monitor -b 230400 --port /dev/cu.usbserial-XXXX
```

## Verify Stream From Terminal

```bash
pyserial-miniterm /dev/cu.usbserial-XXXX 230400 --raw
```

If no data appears, press ESP32 reset once.

## Protocol Change Rule

If the serial row format changes, update parser, tests, and docs together:

- `src/reson/parser.py`
- parser/pipeline tests under `tests/`
- `README.md`
- `docs/data_schema.md`
