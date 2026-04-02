#include <Arduino.h>

/*
  FlexiForce / piezoresistive breathing sensor (Arduino UNO)
  
  SIMPLIFIED: Arduino only reads sensor and sends raw ADC value.
  ALL signal processing (filtering, BPM, I:E ratio, tidal volume,
  variability, apnea) is done in Python on the PC side.
  
  Output: one integer per line at 20 Hz
*/

const int PIN             = A0;
const int BAUD            = 9600;
const unsigned long Ts_ms = 50;   // 20 Hz

unsigned long lastSample = 0;

void setup() {
  Serial.begin(BAUD);
  pinMode(PIN, INPUT);
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample < Ts_ms) return;
  lastSample = now;
  Serial.println(analogRead(PIN));
}