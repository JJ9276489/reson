# Firmware

This folder contains the ESP32 sketch used by Reson.

## Sketch

- Path: `firmware/esp32_emg_stream/esp32_emg_stream.ino`
- Output contract: `t raw env` (space-delimited integers)
- Baud: `230400`
- Target sample rate: `~250 Hz`

Example line:

```text
12345 2089 37
```

## Flash

1. Open `firmware/esp32_emg_stream/esp32_emg_stream.ino` in Arduino IDE.
2. Select your ESP32 board and serial port.
3. Upload the sketch.
4. Open serial monitor (or use `pyserial-miniterm`) at `230400`.

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
