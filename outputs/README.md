# Output Directory

Processed results from the crime detection pipeline are saved here automatically.

### What is saved here:
1. **Annotated Videos**: E.g., `outputs/detected_<video_name>.mp4`
   - Bounding boxes with DeepSORT Track IDs.
   - Live threat probability score bars per individual.
   - Warning banners (`CRIME / SUSPICIOUS BEHAVIOR DETECTED`) when threat score exceeds threshold.
   - Real-time stats HUD (FPS, active tracks, device, threat count).

2. **Incident Summary Logs**: E.g., `outputs/<video_name>_alerts.json`
   - JSON report detailing suspicious incidents, start/end frames, suspect track IDs, and peak threat probabilities.
