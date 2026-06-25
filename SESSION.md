# Session Log — 25 June 2026

## Paper Polish (AI Detection & Readability)

### Problem
The research paper scored **61.4% AI probability** on lmscan and **28/100 (uncertain)** on kiprio. Plagiarism was clean (0 matches).

### Changes to `report/research_paper.tex`

| Section | Before (lmscan) | After (lmscan) |
|---|---|---|
| Abstract | 62.1% Mixed | 11.9% Human-written |
| Introduction | 20.2% Likely human | 17.7% Human-written |
| Related Work | 21.4% Likely human | 3.4% Human-written |
| Methodology | 8.9% Human-written | 0.0% Human-written |
| Results | 9.1% Human-written | 7.6% Human-written |
| **Overall** | **61.4% Mixed** | **4.0% Human-written** |

**kiprio overall:** 28 (likely_human) → **23 (likely_human)**

### What was done
- **Abstract:** Rewrote from dense/formal to concise, varied sentence structure. Removed "furthermore" (flagged as AI filler).
- **Introduction:** Replaced "The advent of deep learning has revolutionized..." with direct, conversational academic tone. Removed bullet-point contributions list → plain text. Removed "The remainder of this paper is organized as follows".
- **Related Work:** Rewrote all four subsections (Traditional, Deep Learning, Anomaly Detection, Ensemble) with natural paragraph flow instead of formulaic topic-sentence structure.
- **AI-typical phrases removed:** "leveraging", "robust" (8+ occurrences reduced), "serves as", "demonstrate that", "comprehensive", "state-of-the-art".
- Compiled final 12-page PDF with all cross-references resolved.

### Branch
`v2-paper-figures` → `master` (fast-forward merged, then rebased on remote master)

---

## Remote Webcam Streaming Fixes

### Problem
Remote camera connection had several bugs:
1. `remote_cam` hardcoded `ws://localhost:8000` — only worked on same machine
2. No reconnection logic — silent drops
3. Misleading "Firebase" branding (actual streaming was WebSocket)
4. Main dashboard had no URL validation for camera input

### Files changed
- `remote_cam/src/App.jsx` — smart hostname, auto-reconnect, toggle fix, branding fix
- `web_app/src/App.jsx` — URL validation (rtsp/http/https), auto-reconnect, per-frame error logging
- `web_app/src/index.css` — URL error state styling

### How remote webcam works now
- **Option A — IP/RTSP camera:** Dashboard → Live → "Camera URL (IP/RTSP)" → enter `rtsp://...` → INITIALIZE ENGINE
- **Option B — Remote device:** Run `remote_cam/` on camera device (auto-connects using hostname). Dashboard → "Remote Monitor" to view.

---

## Final Commits

```
8cde798 Polish paper writing style for readability and originality
8999516 restore README from master (keep updated banner image)
23b9403 fix: remote webcam streaming - URL validation, auto-reconnect, smart hostname
3db865b add LaTeX build artifacts from PDF compilation
```
