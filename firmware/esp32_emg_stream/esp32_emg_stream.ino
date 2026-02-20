const int emgPin = 34;

const int SAMPLE_HZ = 250;
const int DT_MS = 1000 / SAMPLE_HZ;

float baseline = 0.0f;
float env = 0.0f;
float noiseFloor = 10.0f;

float baselineAlpha = 0.002f;
float envAlpha = 0.10f;
float noiseAlpha = 0.01f;

void setup() {
  Serial.begin(230400);
  delay(200);

  long sum = 0;
  int n = 0;
  uint32_t t0 = millis();
  while (millis() - t0 < 800) {
    sum += analogRead(emgPin);
    n++;
    delay(2);
  }
  baseline = (n > 0) ? (float)sum / n : 0.0f;
}

void loop() {
  uint32_t t = millis();

  int raw = analogRead(emgPin);
  float x = (float)raw;

  float diff = x - baseline;
  float absDiff = diff >= 0 ? diff : -diff;

  env = (1.0f - envAlpha) * env + envAlpha * absDiff;

  // keep baseline/noiseFloor tracking (still useful)
  if (env < noiseFloor * 4.0f) {
    baseline = (1.0f - baselineAlpha) * baseline + baselineAlpha * x;
    noiseFloor = (1.0f - noiseAlpha) * noiseFloor + noiseAlpha * env;
  }

  // t raw env
  Serial.print(t);
  Serial.print(" ");
  Serial.print(raw);
  Serial.print(" ");
  Serial.println((int)env);

  delay(DT_MS);
}
