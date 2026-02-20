.PHONY: firmware-install firmware-upload firmware-monitor

firmware-install:
	python -m pip install -r requirements-firmware.txt

firmware-upload:
	cd firmware/esp32_emg_stream_cpp && pio run -t upload

firmware-monitor:
	cd firmware/esp32_emg_stream_cpp && pio device monitor -b 230400
