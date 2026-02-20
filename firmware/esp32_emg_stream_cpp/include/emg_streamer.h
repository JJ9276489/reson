#pragma once

#include <Arduino.h>

struct EmgSampleLine {
  uint32_t t_ms;
  int raw;
  int env;
};

class EmgStreamer {
 public:
  explicit EmgStreamer(int emg_pin);

  void begin();
  bool tick();

 private:
  int emg_pin_;
  uint32_t dt_ms_;
  uint32_t next_sample_ms_;

  float baseline_;
  float env_;
  float noise_floor_;

  // Same tuning as the original .ino.
  const float baseline_alpha_ = 0.002f;
  const float env_alpha_ = 0.10f;
  const float noise_alpha_ = 0.01f;

  EmgSampleLine read_sample_line(uint32_t t_ms);
  void emit_sample_line(const EmgSampleLine& line) const;
};
