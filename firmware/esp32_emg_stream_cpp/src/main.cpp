#include <Arduino.h>

#include "emg_streamer.h"

namespace {
constexpr int kEmgPin = 34;
EmgStreamer g_streamer(kEmgPin);
}  // namespace

void setup() { g_streamer.begin(); }

void loop() {
  g_streamer.tick();
  // Keep loop cooperative; ready for future IMU work.
  delay(1);
}
