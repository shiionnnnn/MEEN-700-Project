const int trigObject = 42;
const int echoObject = 40;
const int trigTable = 12;
const int echoTable = 11;

const unsigned long ECHO_TIMEOUT_US = 30000UL;
const float SPEED_OF_SOUND_CM_PER_US = 0.0343f;

const int INTER_SENSOR_DELAY_MS = 60;
const int LOOP_DELAY_MS = 50;

float readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, ECHO_TIMEOUT_US);

  if (duration == 0) return -1.0;

  float distance_cm = (duration * SPEED_OF_SOUND_CM_PER_US) / 2.0;

  if (distance_cm < 1.0 || distance_cm > 400.0) return -1.0;

  return distance_cm;
}

void setup() {
  Serial.begin(9600);

  pinMode(trigObject, OUTPUT);
  pinMode(echoObject, INPUT);
  pinMode(trigTable, OUTPUT);
  pinMode(echoTable, INPUT);

  digitalWrite(trigObject, LOW);
  digitalWrite(trigTable, LOW);

  delay(500);

  Serial.println("arduino_ms,object_raw_cm,table_raw_cm");
}

void loop() {
  float objectDist = readUltrasonic(trigObject, echoObject);
  delay(INTER_SENSOR_DELAY_MS);

  float tableDist = readUltrasonic(trigTable, echoTable);
  delay(INTER_SENSOR_DELAY_MS);

  Serial.print(millis());
  Serial.print(",");
  Serial.print(objectDist, 2);
  Serial.print(",");
  Serial.println(tableDist, 2);

  delay(LOOP_DELAY_MS);
}