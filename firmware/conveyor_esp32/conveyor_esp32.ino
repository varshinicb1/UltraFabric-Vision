/*
 * UltraFabric-Vision - Conveyor Wi-Fi controller (ESP32)
 * ------------------------------------------------------
 * Web + HTTP API front-end for the Arduino stepper driver
 * (see conveyor_arduino.ino). Talks to the Arduino over Serial2 (9600, pins
 * RX=16 / TX=17) and to the network over HTTP.
 *
 * Adds over the original firmware:
 *   1. GET /defect  - the vision backend calls this the moment a defect is
 *                     detected; the ESP32 forwards DEFECT to the Arduino so the
 *                     conveyor stops. (Set UFV_ESP32_URL on the backend to this
 *                     board's IP.)
 *   2. Auto batch recording - a config form (cloth length, line length, motor
 *                     speed) drives the Arduino's automatic batch mode, so the
 *                     belt advances one cloth length at a time.
 *
 * HTTP API:
 *   GET /                 control page (buttons + config form + live status)
 *   GET /start            manual run
 *   GET /stop             manual stop
 *   GET /defect           stop (reason: defect)  <- called by the backend
 *   GET /auto             start the automatic batch run
 *   GET /config?cloth=<m>&line=<m>&speed=<m_per_min>   set batch parameters
 *   GET /status           JSON: {status, reason, batch, total, pos_m}
 *
 * Config values on the wire to the Arduino are millimetres / mm-per-min; the UI
 * takes metres and m/min and this sketch converts.
 */
#include <WiFi.h>
#include <WebServer.h>

const char* ssid     = "vivo";
const char* password = "vivo1234";

WebServer server(80);

// ---- Status parsed from the Arduino over Serial2 ----
String lineStatus = "idle";     // idle | running | stopped | done
String stopReason = "";         // manual | defect
int    curBatch = 0, totBatch = 0;
long   posMm = 0;

void sendToMotor(const String& s) { Serial2.println(s); }

// ---------------- Web page ----------------
String page() {
  return String(R"HTML(
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:sans-serif;text-align:center;background:#0f172a;color:#e2e8f0;margin:0;padding:16px}
 h2{margin:8px}
 button{width:150px;height:56px;font-size:20px;margin:8px;border:0;border-radius:10px;color:#fff;cursor:pointer}
 .start{background:#16a34a}.stop{background:#dc2626}.auto{background:#2563eb}
 input{width:90px;font-size:18px;padding:6px;margin:4px;border-radius:8px;border:1px solid #334155;background:#1e293b;color:#e2e8f0}
 .card{background:#1e293b;border-radius:12px;padding:14px;max-width:420px;margin:12px auto}
 #st{font-size:20px;font-weight:bold}
 label{display:inline-block;width:150px;text-align:right;margin-right:6px}
</style></head><body>
<h2>UltraFabric Conveyor Control</h2>

<div class="card">
  <button class="start" onclick="g('/start')">START</button>
  <button class="stop"  onclick="g('/stop')">STOP</button>
</div>

<div class="card">
  <h3>Auto batch recording</h3>
  <div><label>Cloth length (m)</label><input id="cloth" type="number" step="0.1" value="1.0"></div>
  <div><label>Line length (m)</label><input id="line" type="number" step="0.1" value="5.0"></div>
  <div><label>Motor speed (m/min)</label><input id="speed" type="number" step="0.1" value="2.0"></div>
  <button class="auto" onclick="startAuto()">START AUTO</button>
</div>

<div class="card">
  <div id="st">status: --</div>
  <div id="info"></div>
</div>

<script>
function g(u){return fetch(u).then(()=>refresh());}
function startAuto(){
  var c=document.getElementById('cloth').value,
      l=document.getElementById('line').value,
      s=document.getElementById('speed').value;
  fetch('/config?cloth='+c+'&line='+l+'&speed='+s)
    .then(()=>fetch('/auto')).then(()=>refresh());
}
function refresh(){
  fetch('/status').then(r=>r.json()).then(d=>{
    var t='status: '+d.status+(d.reason?(' ('+d.reason+')'):'');
    document.getElementById('st').textContent=t;
    document.getElementById('st').style.color =
      d.status=='stopped'&&d.reason=='defect' ? '#f87171'
      : d.status=='running' ? '#4ade80' : '#e2e8f0';
    document.getElementById('info').textContent =
      'batch '+d.batch+'/'+d.total+'   |   position '+d.pos_m.toFixed(2)+' m';
  });
}
setInterval(refresh,1000); refresh();
</script>
</body></html>
)HTML");
}

// ---------------- Handlers ----------------
void handleRoot()  { server.send(200, "text/html", page()); }

void handleStart() { sendToMotor("START"); lineStatus="running"; stopReason=""; server.send(200,"text/plain","ok"); }
void handleStop()  { sendToMotor("STOP");  server.send(200,"text/plain","ok"); }
void handleDefect(){ sendToMotor("DEFECT"); lineStatus="stopped"; stopReason="defect"; server.send(200,"text/plain","stopped: defect"); }
void handleAuto()  { sendToMotor("AUTO");  lineStatus="running"; stopReason=""; server.send(200,"text/plain","auto started"); }

void handleConfig() {
  // UI sends metres and m/min; the Arduino wants mm and mm/min.
  if (server.hasArg("speed")) sendToMotor("SPEED " + String(server.arg("speed").toFloat() * 1000.0, 1));
  if (server.hasArg("cloth")) sendToMotor("CLOTH " + String(server.arg("cloth").toFloat() * 1000.0, 1));
  if (server.hasArg("line"))  sendToMotor("LINE "  + String(server.arg("line").toFloat()  * 1000.0, 1));
  server.send(200, "text/plain", "configured");
}

void handleStatus() {
  String j = "{\"status\":\"" + lineStatus + "\",\"reason\":\"" + stopReason +
             "\",\"batch\":" + String(curBatch) + ",\"total\":" + String(totBatch) +
             ",\"pos_m\":" + String(posMm / 1000.0, 2) + "}";
  server.sendHeader("Access-Control-Allow-Origin", "*");   // allow the dashboard to poll
  server.send(200, "application/json", j);
}

// Parse status lines coming back from the Arduino.
void readMotorSerial() {
  while (Serial2.available()) {
    String l = Serial2.readStringUntil('\n');
    l.trim();
    if (l.startsWith("POS ")) {
      posMm = l.substring(4).toInt();
    } else if (l.startsWith("BATCH ")) {           // "BATCH 2/10"
      int slash = l.indexOf('/');
      if (slash > 6) {
        curBatch = l.substring(6, slash).toInt();
        totBatch = l.substring(slash + 1).toInt();
        lineStatus = "running";
      }
    } else if (l.startsWith("AUTO_START ")) {
      totBatch = l.substring(11).toInt();
      lineStatus = "running";
    } else if (l.startsWith("STOPPED ")) {
      lineStatus = "stopped";
      stopReason = l.substring(8);
    } else if (l == "RUN_DONE") {
      lineStatus = "done";
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600, SERIAL_8N1, 16, 17);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  Serial.print("ESP32 IP (set UFV_ESP32_URL=http://this): ");
  Serial.println(WiFi.localIP());

  server.on("/",       handleRoot);
  server.on("/start",  handleStart);
  server.on("/stop",   handleStop);
  server.on("/defect", handleDefect);
  server.on("/auto",   handleAuto);
  server.on("/config", handleConfig);
  server.on("/status", handleStatus);
  server.begin();
}

void loop() {
  server.handleClient();
  readMotorSerial();
}
