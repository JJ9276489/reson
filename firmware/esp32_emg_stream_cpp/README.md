# ESP32 C++ Firmware (PlatformIO)

This is the modular C++ version of the Reson ESP32 stream firmware.

## Why This Exists

- Keeps the serial protocol stable while preparing for IMU integration.
- Uses module boundaries (`main.cpp`, `emg_streamer.cpp/.h`) instead of a monolithic `.ino`.
- Is more reproducible than a local Arduino IDE-only workflow.

## Output Contract

- Baud: `230400`.
- Sample target: approximately `250 Hz`.
- Line format: `t raw env` as space-delimited integers.

Example:

```text
12345 2089 37
```

## Build / Upload

From this directory:

```bash
pio run
pio run -t upload
pio device monitor -b 230400
```

From repo root:

```bash
make firmware-upload
make firmware-monitor
```

If your board needs explicit port selection:

```bash
pio run -t upload --upload-port /dev/cu.usbserial-XXXX
pio device monitor -b 230400 --port /dev/cu.usbserial-XXXX
```

## Notes

- The legacy Arduino sketch remains at `firmware/esp32_emg_stream/esp32_emg_stream.ino`.
- Both firmware paths currently preserve the same serial protocol expected by the Python parser.
- If the protocol changes, update parser, tests, and docs in the same change.
