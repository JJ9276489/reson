# ESP32 C++ Firmware (PlatformIO)

This is the modular C++ version of the Reson ESP32 stream firmware.

## Why this exists

- Keeps serial protocol stable while preparing for IMU integration.
- Uses module boundaries (`main.cpp`, `emg_streamer.cpp/.h`) instead of a monolithic `.ino`.
- Is reproducible and CI-friendly with PlatformIO.

## Output contract

- Baud: `230400`
- Sample target: `~250 Hz`
- Line format: `t raw env` (space-delimited integers)

Example:

```text
12345 2089 37
```

## Build / upload

```bash
cd /Users/jeraldyuan/dev/reson/firmware/esp32_emg_stream_cpp
pio run
pio run -t upload
pio device monitor -b 230400
```

If your board needs explicit port selection:

```bash
pio run -t upload --upload-port /dev/cu.usbserial-XXXX
pio device monitor -b 230400 --port /dev/cu.usbserial-XXXX
```

## Notes

- The legacy Arduino sketch remains at:
  - `firmware/esp32_emg_stream/esp32_emg_stream.ino`
- Both paths currently preserve the same serial protocol expected by Python parser.
