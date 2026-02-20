#include "emg_streamer.h"

namespace {
constexpr int kWarmupMs = 800;
constexpr int kWarmupReadDelayMs = 2;
constexpr float kInitialNoiseFloor = 10.0f;
}  // namespace

EmgStreamer::EmgStreamer(int emg_pin)
    : emg_pin_(emg_pin),
      dt_ms_(1000 / SAMPLE_HZ),
      next_sample_ms_(0),
      baseline_(0.0f),
      env_(0.0f),
      noise_floor_(kInitialNoiseFloor) {}

void EmgStreamer::begin() {
  pinMode(emg_pin_, INPUT);
  Serial.begin(SERIAL_BAUD);
  delay(200);

  long sum = 0;
  int n = 0;
  const uint32_t t0 = millis();
  while (millis() - t0 < kWarmupMs) {
    sum += analogRead(emg_pin_);
    n++;
    delay(kWarmupReadDelayMs);
  }
  baseline_ = (n > 0) ? static_cast<float>(sum) / static_cast<float>(n) : 0.0f;
  next_sample_ms_ = millis();
}

bool EmgStreamer::tick() {
  const uint32_t now = millis();
  if (now < next_sample_ms_) {
    return false;
  }

  // Keep cadence stable and avoid long-term drift from per-loop latency.
  next_sample_ms_ += dt_ms_;
  if (next_sample_ms_ < now) {
    next_sample_ms_ = now + dt_ms_;
  }

  const EmgSampleLine line = read_sample_line(now);
  emit_sample_line(line);
  return true;
}

EmgSampleLine EmgStreamer::read_sample_line(uint32_t t_ms) {
  const int raw = analogRead(emg_pin_);
  const float x = static_cast<float>(raw);
  const float diff = x - baseline_;
  const float abs_diff = (diff >= 0.0f) ? diff : -diff;

  env_ = (1.0f - env_alpha_) * env_ + env_alpha_ * abs_diff;

  // Keep baseline/noise floor tracking (same as legacy .ino behavior).
  if (env_ < noise_floor_ * 4.0f) {
    baseline_ = (1.0f - baseline_alpha_) * baseline_ + baseline_alpha_ * x;
    noise_floor_ = (1.0f - noise_alpha_) * noise_floor_ + noise_alpha_ * env_;
  }

  EmgSampleLine out{};
  out.t_ms = t_ms;
  out.raw = raw;
  out.env = static_cast<int>(env_);
  return out;
}

void EmgStreamer::emit_sample_line(const EmgSampleLine& line) const {
  // Contract: "t raw env"
  Serial.print(line.t_ms);
  Serial.print(" ");
  Serial.print(line.raw);
  Serial.print(" ");
  Serial.println(line.env);
}
