/*
 * UltraFabric-Vision - Conveyor stepper driver (Arduino Uno/Nano)
 * ---------------------------------------------------------------
 * Receives text commands over Serial (9600) from the ESP32 Wi-Fi board and
 * drives the conveyor stepper. Features:
 *
 *   1. DEFECT stop  - a "DEFECT" command halts the belt immediately (the
 *                     vision backend sends this the instant a defect is seen).
 *   2. Auto batches - given a cloth (batch) length and total line length, the
 *                     belt advances one cloth length, pauses so the batch can be
 *                     recorded, then continues for LINE/CLOTH batches.
 *   3. Calibration  - mm-per-revolution is settable at runtime (CAL) and stored
 *                     in EEPROM, so you calibrate from the dashboard ONCE and it
 *                     survives power cycles. JOG moves an exact number of
 *                     revolutions so you can measure the belt travel.
 *
 * Serial protocol (ESP32 -> Arduino), one command per line:
 *   START            run continuously (manual jog)
 *   STOP             stop (manual)
 *   DEFECT           stop (reason: defect detected by vision system)
 *   RESET            zero the position counter
 *   SPEED <mm/min>   set conveyor linear speed
 *   CLOTH <mm>       set one batch (cloth) length
 *   LINE  <mm>       set total line length -> batches = LINE / CLOTH
 *   AUTO             start the automatic batch run
 *   JOG <revs>       move exactly <revs> revolutions then stop (for calibration)
 *   CAL <mm/rev>     set + persist mm-per-revolution calibration
 *   GETCAL           report the stored calibration
 *
 * Status replies (Arduino -> ESP32):
 *   ARDUINO READY | ACK ... | SPEED_RPM <n> | POS <mm> | CAL <mm/rev> |
 *   JOG_DONE | AUTO_START <total> | BATCH <n>/<total> | BATCH_DONE <n> |
 *   RUN_DONE | STOPPED <reason> | ERR <msg>
 */
#include <Stepper.h>
#include <EEPROM.h>

// ---------------- Motor / mechanics ----------------
const int STEPS_PER_REV = 200;          // full steps per revolution (NEMA-17 = 200)
#define IN1 8
#define IN2 9
#define IN3 10
#define IN4 11
Stepper motor(STEPS_PER_REV, IN1, IN2, IN3, IN4);

const int STEP_DIR = -1;                 // feed direction (as in original sketch)
const unsigned long DWELL_MS = 1500;     // pause between batches (recording gap)

// mm-per-revolution = drive-roller circumference. Default is a starting guess;
// the real value is calibrated from the dashboard and stored in EEPROM.
const float DEFAULT_MM_PER_REV = 125.7;  // pi * ~40 mm roller
const int   EE_ADDR  = 0;                // EEPROM location for the calibration
const long  EE_MAGIC = 0x554643;         // "UFC" - marks a valid stored value

float mmPerRev   = DEFAULT_MM_PER_REV;
float stepsPerMm = STEPS_PER_REV / DEFAULT_MM_PER_REV;

// ---------------- Runtime state ----------------
enum Mode { IDLE, MANUAL, AUTO, JOG };
Mode mode = IDLE;

long  clothSteps   = 0;     // steps in one batch (one cloth length)
int   totalBatches = 0;     // LINE / CLOTH
int   batchIndex   = 0;     // current batch (1-based)
long  batchRemain  = 0;     // steps left in the current batch
long  jogRemain    = 0;     // steps left in a JOG move
long  posSteps     = 0;     // absolute steps since RESET
long  lastPosReport = 0;
unsigned long dwellUntil = 0;

struct CalStore { long magic; float mmPerRev; };

void applyCal(float v) {
  if (v < 1.0) v = 1.0;                   // guard against nonsense
  mmPerRev = v;
  stepsPerMm = STEPS_PER_REV / mmPerRev;
}

void loadCal() {
  CalStore c;
  EEPROM.get(EE_ADDR, c);
  if (c.magic == EE_MAGIC && c.mmPerRev > 1.0 && c.mmPerRev < 100000.0) {
    applyCal(c.mmPerRev);
  }
}

void saveCal(float v) {
  applyCal(v);
  CalStore c = { EE_MAGIC, mmPerRev };
  EEPROM.put(EE_ADDR, c);
  Serial.print("CAL "); Serial.println(mmPerRev, 3);
}

void reportPos() {
  if (labs(posSteps - lastPosReport) >= (long)(5 * stepsPerMm)) {
    Serial.print("POS ");
    Serial.println((long)(posSteps / stepsPerMm));
    lastPosReport = posSteps;
  }
}

void stopMotor(const char* reason) {
  mode = IDLE;
  batchRemain = 0;
  jogRemain = 0;
  Serial.print("STOPPED ");
  Serial.println(reason);
}

void setSpeedMmPerMin(float mmMin) {
  long rpm = (long)(mmMin / mmPerRev + 0.5);     // RPM = (mm/min) / (mm/rev)
  if (rpm < 1) rpm = 1;                          // Stepper lib needs >= 1 RPM
  motor.setSpeed(rpm);
  Serial.print("SPEED_RPM "); Serial.println(rpm);
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd == "START") {
    mode = MANUAL; Serial.println("ACK START");
  } else if (cmd == "STOP") {
    stopMotor("manual");
  } else if (cmd == "DEFECT") {              // <-- stop because a defect was detected
    stopMotor("defect");
  } else if (cmd == "RESET") {
    posSteps = 0; lastPosReport = 0; Serial.println("ACK RESET");
  } else if (cmd == "GETCAL") {
    Serial.print("CAL "); Serial.println(mmPerRev, 3);
  } else if (cmd.startsWith("CAL ")) {
    saveCal(cmd.substring(4).toFloat());
  } else if (cmd.startsWith("JOG ")) {
    float revs = cmd.substring(4).toFloat();
    if (revs <= 0) revs = 1;
    jogRemain = (long)(revs * STEPS_PER_REV + 0.5);
    mode = JOG;
    Serial.print("ACK JOG "); Serial.println(revs, 2);
  } else if (cmd.startsWith("SPEED ")) {
    setSpeedMmPerMin(cmd.substring(6).toFloat());
  } else if (cmd.startsWith("CLOTH ")) {
    clothSteps = (long)(cmd.substring(6).toFloat() * stepsPerMm + 0.5);
    Serial.print("ACK CLOTH "); Serial.println(clothSteps);
  } else if (cmd.startsWith("LINE ")) {
    long lineSteps = (long)(cmd.substring(5).toFloat() * stepsPerMm + 0.5);
    totalBatches = (clothSteps > 0) ? (int)(lineSteps / clothSteps) : 0;
    Serial.print("ACK BATCHES "); Serial.println(totalBatches);
  } else if (cmd == "AUTO") {
    if (clothSteps > 0 && totalBatches > 0) {
      mode = AUTO; batchIndex = 1; batchRemain = clothSteps; dwellUntil = 0;
      Serial.print("AUTO_START "); Serial.println(totalBatches);
      Serial.print("BATCH 1/"); Serial.println(totalBatches);
    } else {
      Serial.println("ERR need CLOTH and LINE before AUTO");
    }
  } else {
    Serial.print("ERR unknown "); Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(9600);
  loadCal();                       // restore calibration from EEPROM (if any)
  motor.setSpeed(60);              // default RPM; override with SPEED command
  Serial.println("ARDUINO READY");
  Serial.print("CAL "); Serial.println(mmPerRev, 3);
}

void loop() {
  // A command is checked once per step, so STOP/DEFECT halts within one step.
  if (Serial.available()) {
    handleCommand(Serial.readStringUntil('\n'));
  }

  if (mode == MANUAL) {
    motor.step(STEP_DIR); posSteps++; reportPos();
  }
  else if (mode == JOG) {
    if (jogRemain > 0) {
      motor.step(STEP_DIR); posSteps++; jogRemain--; reportPos();
    } else {
      mode = IDLE; Serial.println("JOG_DONE");
    }
  }
  else if (mode == AUTO) {
    if (millis() < dwellUntil) return;             // pausing between batches
    if (batchRemain > 0) {
      motor.step(STEP_DIR); posSteps++; batchRemain--; reportPos();
    } else {
      Serial.print("BATCH_DONE "); Serial.println(batchIndex);
      if (batchIndex >= totalBatches) {
        mode = IDLE; Serial.println("RUN_DONE");
      } else {
        batchIndex++; batchRemain = clothSteps;
        dwellUntil = millis() + DWELL_MS;
        Serial.print("BATCH "); Serial.print(batchIndex);
        Serial.print("/"); Serial.println(totalBatches);
      }
    }
  }
}
