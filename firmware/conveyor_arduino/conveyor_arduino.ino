/*
 * UltraFabric-Vision - Conveyor stepper driver (Arduino Uno/Nano)
 * ---------------------------------------------------------------
 * Receives text commands over Serial (9600) from the ESP32 Wi-Fi board and
 * drives the conveyor stepper. Adds two features over the original firmware:
 *
 *   1. DEFECT stop  - a "DEFECT" command halts the belt immediately (the
 *                     vision backend sends this the instant a defect is seen).
 *   2. Auto batches - given a cloth (batch) length and the total line length,
 *                     the belt advances exactly one cloth length, pauses so the
 *                     batch can be recorded, then continues to the next batch.
 *
 * All distances are handled in millimetres and converted to steps using the
 * mechanical calibration below. Position is reported back so the ESP32 / UI can
 * show where along the batch the belt is.
 *
 * Serial protocol (ESP32 -> Arduino), one command per line:
 *   START            run continuously (manual jog)
 *   STOP             stop (manual)
 *   DEFECT           stop (reason: defect detected by vision system)
 *   RESET            zero the position counter
 *   SPEED <mm/min>   set conveyor linear speed
 *   CLOTH <mm>       set one batch (cloth) length
 *   LINE  <mm>       set total line length -> number of batches = LINE / CLOTH
 *   AUTO             start the automatic batch run
 *
 * Status replies (Arduino -> ESP32):
 *   ARDUINO READY | ACK ... | SPEED_RPM <n> | POS <mm> |
 *   AUTO_START <total> | BATCH <n>/<total> | BATCH_DONE <n> |
 *   RUN_DONE | STOPPED <reason> | ERR <msg>
 */
#include <Stepper.h>

// ---------------- Motor / mechanics ----------------
const int STEPS_PER_REV = 200;          // full steps per revolution (NEMA-17 = 200)
#define IN1 8
#define IN2 9
#define IN3 10
#define IN4 11
Stepper motor(STEPS_PER_REV, IN1, IN2, IN3, IN4);

// CALIBRATION - linear fabric travel per motor revolution = drive-roller
// circumference (pi * roller diameter). MEASURE THIS on your rig: command one
// revolution and measure how far the belt moved. Example: a 40 mm roller gives
// pi * 40 = 125.7 mm/rev. Every distance depends on this value.
const float MM_PER_REV   = 125.7;       // <-- SET FOR YOUR ROLLER
const float STEPS_PER_MM = STEPS_PER_REV / MM_PER_REV;

const int STEP_DIR = -1;                 // feed direction (as in original sketch)
const unsigned long DWELL_MS = 1500;     // pause between batches (recording gap)

// ---------------- Runtime state ----------------
enum Mode { IDLE, MANUAL, AUTO };
Mode mode = IDLE;

long  clothSteps   = 0;     // steps in one batch (one cloth length)
int   totalBatches = 0;     // LINE / CLOTH
int   batchIndex   = 0;     // current batch (1-based)
long  batchRemain  = 0;     // steps left in the current batch
long  posSteps     = 0;     // absolute steps since RESET
long  lastPosReport = 0;
unsigned long dwellUntil = 0;

void reportPos() {
  // Report position (mm) at most every ~5 mm of travel to avoid flooding serial.
  if (labs(posSteps - lastPosReport) >= (long)(5 * STEPS_PER_MM)) {
    Serial.print("POS ");
    Serial.println((long)(posSteps / STEPS_PER_MM));
    lastPosReport = posSteps;
  }
}

void stopMotor(const char* reason) {
  mode = IDLE;
  batchRemain = 0;
  Serial.print("STOPPED ");
  Serial.println(reason);
}

void setSpeedMmPerMin(float mmMin) {
  long rpm = (long)(mmMin / MM_PER_REV + 0.5);   // RPM = (mm/min) / (mm/rev)
  if (rpm < 1) rpm = 1;                          // Stepper lib needs >= 1 RPM
  motor.setSpeed(rpm);
  Serial.print("SPEED_RPM ");
  Serial.println(rpm);
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "START") {
    mode = MANUAL;
    Serial.println("ACK START");
  } else if (cmd == "STOP") {
    stopMotor("manual");
  } else if (cmd == "DEFECT") {              // <-- stop because a defect was detected
    stopMotor("defect");
  } else if (cmd == "RESET") {
    posSteps = 0; lastPosReport = 0;
    Serial.println("ACK RESET");
  } else if (cmd.startsWith("SPEED ")) {
    setSpeedMmPerMin(cmd.substring(6).toFloat());
  } else if (cmd.startsWith("CLOTH ")) {
    clothSteps = (long)(cmd.substring(6).toFloat() * STEPS_PER_MM + 0.5);
    Serial.print("ACK CLOTH "); Serial.println(clothSteps);
  } else if (cmd.startsWith("LINE ")) {
    long lineSteps = (long)(cmd.substring(5).toFloat() * STEPS_PER_MM + 0.5);
    totalBatches = (clothSteps > 0) ? (int)(lineSteps / clothSteps) : 0;
    Serial.print("ACK BATCHES "); Serial.println(totalBatches);
  } else if (cmd == "AUTO") {
    if (clothSteps > 0 && totalBatches > 0) {
      mode = AUTO;
      batchIndex = 1;
      batchRemain = clothSteps;
      dwellUntil = 0;
      Serial.print("AUTO_START "); Serial.println(totalBatches);
      Serial.print("BATCH 1/"); Serial.println(totalBatches);
    } else {
      Serial.println("ERR need CLOTH and LINE before AUTO");
    }
  } else {
    Serial.print("ERR unknown ");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(9600);
  motor.setSpeed(60);              // default RPM; override with SPEED command
  Serial.println("ARDUINO READY");
}

void loop() {
  // A command is checked once per step, so STOP/DEFECT halts within one step.
  if (Serial.available()) {
    handleCommand(Serial.readStringUntil('\n'));
  }

  if (mode == MANUAL) {
    motor.step(STEP_DIR); posSteps++; reportPos();
  }
  else if (mode == AUTO) {
    if (millis() < dwellUntil) return;             // pausing between batches

    if (batchRemain > 0) {
      motor.step(STEP_DIR); posSteps++; batchRemain--; reportPos();
    } else {
      Serial.print("BATCH_DONE "); Serial.println(batchIndex);
      if (batchIndex >= totalBatches) {
        mode = IDLE;
        Serial.println("RUN_DONE");
      } else {
        batchIndex++;
        batchRemain = clothSteps;
        dwellUntil = millis() + DWELL_MS;          // recording gap
        Serial.print("BATCH "); Serial.print(batchIndex);
        Serial.print("/"); Serial.println(totalBatches);
      }
    }
  }
}
