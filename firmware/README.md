# Conveyor firmware (assembly-line side)

Two boards drive the fabric conveyor and connect it to the UltraFabric-Vision
backend:

| Board | Sketch | Role |
|-------|--------|------|
| ESP32 | [`conveyor_esp32/conveyor_esp32.ino`](conveyor_esp32/conveyor_esp32.ino) | Wi-Fi control page + HTTP API; relays commands to the Arduino |
| Arduino Uno/Nano | [`conveyor_arduino/conveyor_arduino.ino`](conveyor_arduino/conveyor_arduino.ino) | Drives the stepper; auto-batch stepping; defect stop |

```
 Vision backend ──HTTP /defect──▶ ESP32 ──Serial2 (9600)──▶ Arduino ──▶ stepper/conveyor
 Dashboard/phone ─HTTP /,/auto──▶ ESP32                     (RX16/TX17)
```

## Wiring
- **ESP32 → Arduino:** ESP32 `GPIO17 (TX)` → Arduino `RX (D0)`, ESP32 `GPIO16 (RX)` ← Arduino `TX (D1)`, common **GND**. (Both directions are used now — the Arduino reports batch/position status back.)
- **Arduino → stepper driver:** `IN1..IN4` = pins `8,9,10,11` (unchanged).

> Serial note: the Arduino uses its hardware `Serial` (pins D0/D1) to talk to the
> ESP32, so unplug that link while flashing the Arudino over USB, then reconnect.

## Calibrate before first use (important)
Distances only work if the Arduino knows how far the belt moves per motor turn.
In `conveyor_arduino.ino` set:

```cpp
const float MM_PER_REV = 125.7;  // = pi * drive-roller diameter (mm)
```

Measure it: mark the belt, command exactly one revolution, measure travel in mm.
Everything (cloth length, line length, speed) is derived from this value.

## Feature 1 — Motor stops on defect
The backend calls `GET http://<esp32-ip>/defect` the instant a defect is seen on
the live stream; the ESP32 forwards `DEFECT` and the Arduino halts within one
step. Enable it by pointing the backend at the board:

```powershell
$env:UFV_ESP32_URL = "http://192.168.1.50"   # the IP the ESP32 prints on boot
# optional: $env:UFV_ESP32_COOLDOWN = "5"     # min seconds between stop signals
```

Then start the backend as usual. Without `UFV_ESP32_URL` the hook is a no-op, so
the software runs fine with no board attached. (Wiring is in
[`../line_control.py`](../line_control.py), called from the `/ws/stream` path.)

## Feature 2 — Auto batch recording
On the ESP32 page (`http://<esp32-ip>/`) enter **cloth length**, **line length**
and **motor speed**, then **START AUTO**. The belt advances one cloth length,
pauses ~1.5 s so that batch can be recorded, then continues — for
`floor(line ÷ cloth)` batches. Position and batch number are shown live.

Same thing over HTTP:
```
GET /config?cloth=1.0&line=5.0&speed=2.0   # metres, metres, m/min
GET /auto
```

## Serial protocol (ESP32 ⇆ Arduino, 9600)
Commands to the Arduino: `START`, `STOP`, `DEFECT`, `RESET`,
`SPEED <mm/min>`, `CLOTH <mm>`, `LINE <mm>`, `AUTO`.
Replies from the Arduino: `POS <mm>`, `AUTO_START <total>`, `BATCH <n>/<total>`,
`BATCH_DONE <n>`, `RUN_DONE`, `STOPPED <reason>`, `ACK ...`, `ERR <msg>`.

Set your Wi-Fi SSID/password at the top of `conveyor_esp32.ino`.
