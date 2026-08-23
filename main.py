from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import cv2
import time
from people_counter import PeopleCounter

app = FastAPI(
    title="Gate People Counter"
)

counter = PeopleCounter()

# -----------------------------------------
# Configuration Model
# -----------------------------------------
class ConfigUpdate(BaseModel):
    detection_mode: str = None
    line_orientation: str = None
    line_position: float = None

# -----------------------------------------
# Home Page UI
# -----------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Advanced human detection, tracking, and occupancy monitoring dashboard powered by OpenCV and FastAPI.">
        <title>HAWKEYE - Advanced Gate Watch Dashboard</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0b0f19;
                --card-bg: rgba(20, 26, 43, 0.6);
                --card-border: rgba(255, 255, 255, 0.06);
                --text-color: #f1f5f9;
                --text-muted: #94a3b8;
                --primary: #00f0ff;
                --primary-glow: rgba(0, 240, 255, 0.3);
                --success: #00ff87;
                --success-glow: rgba(0, 255, 135, 0.25);
                --danger: #ff3860;
                --danger-glow: rgba(255, 56, 96, 0.25);
                --font-display: 'Rajdhani', sans-serif;
                --font-body: 'Inter', sans-serif;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: var(--font-body);
                background-color: var(--bg-color);
                background-image: 
                    radial-gradient(at 0% 0%, rgba(0, 114, 255, 0.1) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(255, 56, 96, 0.05) 0px, transparent 50%);
                color: var(--text-color);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
            }

            header {
                border-bottom: 1px solid var(--card-border);
                padding: 1.5rem 2rem;
                backdrop-filter: blur(10px);
                background-color: rgba(11, 15, 25, 0.7);
                position: sticky;
                top: 0;
                z-index: 10;
            }

            .header-content {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .logo-group {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .logo-indicator {
                width: 12px;
                height: 12px;
                background-color: var(--success);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--success);
                animation: pulse 1.8s infinite;
            }

            h1 {
                font-family: var(--font-display);
                font-size: 1.8rem;
                font-weight: 700;
                letter-spacing: 2px;
                background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .system-status {
                font-family: var(--font-display);
                font-size: 0.95rem;
                letter-spacing: 1px;
                color: var(--text-muted);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .system-status span {
                color: var(--primary);
                font-weight: 600;
            }

            main {
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 1.5rem;
                width: 100%;
                display: grid;
                grid-template-columns: 1.7fr 1fr;
                gap: 2rem;
                flex-grow: 1;
            }

            @media (max-width: 1024px) {
                main {
                    grid-template-columns: 1fr;
                }
            }

            .card {
                background-color: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                backdrop-filter: blur(16px);
                padding: 1.75rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }

            .stream-card {
                align-items: center;
                justify-content: center;
                overflow: hidden;
                position: relative;
            }

            .stream-wrapper {
                position: relative;
                width: 100%;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.05);
                aspect-ratio: 16/9;
                background-color: #000;
            }

            .stream-wrapper img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }

            .stream-overlay {
                position: absolute;
                top: 1rem;
                left: 1rem;
                background: rgba(11, 15, 25, 0.8);
                border: 1px solid var(--card-border);
                padding: 0.4rem 0.8rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-family: var(--font-display);
                text-transform: uppercase;
                letter-spacing: 1px;
                display: flex;
                align-items: center;
                gap: 0.4rem;
                z-index: 2;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1.25rem;
            }

            .kpi-card {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 12px;
                padding: 1.25rem;
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .kpi-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
            }

            .kpi-card.entry::before { background-color: var(--success); }
            .kpi-card.exit::before { background-color: var(--danger); }
            .kpi-card.occupancy::before { background-color: var(--primary); }

            .kpi-card.entry:hover {
                box-shadow: 0 5px 20px var(--success-glow);
                transform: translateY(-2px);
            }
            .kpi-card.exit:hover {
                box-shadow: 0 5px 20px var(--danger-glow);
                transform: translateY(-2px);
            }
            .kpi-card.occupancy:hover {
                box-shadow: 0 5px 20px var(--primary-glow);
                transform: translateY(-2px);
            }

            .kpi-label {
                font-family: var(--font-display);
                font-size: 0.9rem;
                letter-spacing: 1px;
                color: var(--text-muted);
                margin-bottom: 0.5rem;
            }

            .kpi-value {
                font-family: var(--font-display);
                font-size: 2.8rem;
                font-weight: 700;
                line-height: 1;
            }

            .kpi-card.entry .kpi-value { color: var(--success); }
            .kpi-card.exit .kpi-value { color: var(--danger); }
            .kpi-card.occupancy .kpi-value { color: var(--primary); }

            .control-group {
                display: flex;
                flex-direction: column;
                gap: 1.25rem;
            }

            .section-title {
                font-family: var(--font-display);
                font-size: 1.2rem;
                font-weight: 600;
                letter-spacing: 1px;
                color: var(--text-color);
                border-bottom: 1px solid var(--card-border);
                padding-bottom: 0.5rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .control-item {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            .control-label {
                font-size: 0.85rem;
                color: var(--text-muted);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .btn-tabs {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                background-color: rgba(0, 0, 0, 0.25);
                border: 1px solid var(--card-border);
                padding: 3px;
                border-radius: 8px;
            }

            .btn-tabs.two-cols {
                grid-template-columns: repeat(2, 1fr);
            }

            .tab-btn {
                background: none;
                border: none;
                color: var(--text-muted);
                font-family: var(--font-display);
                font-size: 0.9rem;
                font-weight: 600;
                padding: 0.6rem 0;
                cursor: pointer;
                border-radius: 6px;
                transition: all 0.2s ease;
                letter-spacing: 0.5px;
            }

            .tab-btn:hover {
                color: var(--text-color);
            }

            .tab-btn.active {
                background-color: var(--primary);
                color: #000;
                box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
            }

            /* Slider */
            .slider-wrapper {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .range-slider {
                -webkit-appearance: none;
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.1);
                outline: none;
                transition: background 0.3s;
            }

            .range-slider::-webkit-slider-runnable-track {
                width: 100%;
                height: 6px;
                cursor: pointer;
            }

            .range-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: var(--primary);
                cursor: pointer;
                box-shadow: 0 0 10px var(--primary);
                margin-top: -6px;
                transition: transform 0.1s;
            }

            .range-slider::-webkit-slider-thumb:hover {
                transform: scale(1.2);
            }

            .slider-val {
                font-family: var(--font-display);
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--primary);
                min-width: 3rem;
                text-align: right;
            }

            .btn-reset {
                background: linear-gradient(135deg, #ff3860 0%, #d61b40 100%);
                border: none;
                color: white;
                font-family: var(--font-display);
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 0.8rem 0;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 56, 96, 0.2);
                text-transform: uppercase;
                margin-top: 1rem;
            }

            .btn-reset:hover {
                box-shadow: 0 4px 20px rgba(255, 56, 96, 0.45);
                filter: brightness(1.1);
                transform: translateY(-1px);
            }

            .btn-reset:active {
                transform: translateY(1px);
            }

            .system-meta {
                margin-top: auto;
                background: rgba(0, 0, 0, 0.15);
                border-radius: 10px;
                padding: 1rem;
                border: 1px solid var(--card-border);
                font-size: 0.8rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            .meta-row {
                display: flex;
                justify-content: space-between;
                color: var(--text-muted);
            }

            .meta-row span:last-child {
                color: var(--text-color);
                font-weight: 600;
            }

            @keyframes pulse {
                0% { transform: scale(0.9); opacity: 0.6; }
                50% { transform: scale(1.15); opacity: 1; }
                100% { transform: scale(0.9); opacity: 0.6; }
            }

            footer {
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                font-size: 0.85rem;
                border-top: 1px solid var(--card-border);
                margin-top: auto;
            }

            footer a {
                color: var(--primary);
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="header-content">
                <div class="logo-group">
                    <div class="logo-indicator"></div>
                    <h1>HAWKEYE GATEWATCH</h1>
                </div>
                <div class="system-status">
                    STATUS: <span id="sys-status">ACTIVE</span>
                </div>
            </div>
        </header>

        <main>
            <!-- Video Stream Section -->
            <div class="card stream-card">
                <div class="stream-overlay">
                    <div style="width: 8px; height: 8px; background-color: var(--danger); border-radius: 50%; animation: pulse 1s infinite;"></div>
                    Camera Live Feed
                </div>
                <div class="stream-wrapper">
                    <img id="video-feed" src="/video" alt="Live Video Feed">
                </div>
                
                <div style="width: 100%;">
                    <div class="kpi-grid">
                        <div class="kpi-card entry">
                            <div class="kpi-label">TOTAL ENTRIES</div>
                            <div class="kpi-value" id="count-entry">0</div>
                        </div>
                        <div class="kpi-card exit">
                            <div class="kpi-label">TOTAL EXITS</div>
                            <div class="kpi-value" id="count-exit">0</div>
                        </div>
                        <div class="kpi-card occupancy">
                            <div class="kpi-label">CURRENT INSIDE</div>
                            <div class="kpi-value" id="count-inside">0</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Controls Section -->
            <div class="card control-group">
                <div class="section-title">
                    CONFIGURATION
                </div>

                <!-- Detection Mode -->
                <div class="control-item">
                    <div class="control-label">Detection Mode</div>
                    <div class="btn-tabs" id="mode-tabs">
                        <button class="tab-btn" onclick="updateConfig({detection_mode: 'motion'})" id="mode-motion">Motion</button>
                        <button class="tab-btn" onclick="updateConfig({detection_mode: 'hog'})" id="mode-hog">HOG</button>
                        <button class="tab-btn active" onclick="updateConfig({detection_mode: 'hybrid'})" id="mode-hybrid">Hybrid</button>
                    </div>
                </div>

                <!-- Gate Orientation -->
                <div class="control-item">
                    <div class="control-label">Gate Orientation</div>
                    <div class="btn-tabs two-cols" id="orientation-tabs">
                        <button class="tab-btn active" onclick="updateConfig({line_orientation: 'vertical'})" id="orient-vertical">Vertical</button>
                        <button class="tab-btn" onclick="updateConfig({line_orientation: 'horizontal'})" id="orient-horizontal">Horizontal</button>
                    </div>
                </div>

                <!-- Gate Position -->
                <div class="control-item">
                    <div class="control-label">Gate Line Position</div>
                    <div class="slider-wrapper">
                        <input type="range" class="range-slider" min="10" max="90" value="50" id="position-slider" oninput="onSliderChange(this.value)">
                        <span class="slider-val" id="position-val">50%</span>
                    </div>
                </div>

                <!-- Reset Counter -->
                <button class="btn-reset" onclick="resetCounter()">Reset System Counters</button>

                <!-- Metadata/Info -->
                <div class="system-meta">
                    <div class="meta-row">
                        <span>Active Direction (Entry)</span>
                        <span id="meta-entry-dir">-</span>
                    </div>
                    <div class="meta-row">
                        <span>Active Direction (Exit)</span>
                        <span id="meta-exit-dir">-</span>
                    </div>
                    <div class="meta-row">
                        <span>FPS (Estimated)</span>
                        <span>30 FPS</span>
                    </div>
                    <div class="meta-row">
                        <span>Team Info</span>
                        <span>Team Hawkeye (SIH)</span>
                    </div>
                </div>
            </div>
        </main>

        <footer>
            <p>HAWKEYE Gate Occupancy & Human Tracker &copy; 2026</p>
        </footer>

        <script>
            let currentConfig = {
                detection_mode: 'hybrid',
                line_orientation: 'vertical',
                line_position: 0.5
            };

            // Slider delay timer
            let sliderTimer = null;

            function onSliderChange(val) {
                document.getElementById("position-val").innerText = val + "%";
                
                // Debounce the slider change POST request
                clearTimeout(sliderTimer);
                sliderTimer = setTimeout(() => {
                    updateConfig({ line_position: val / 100.0 });
                }, 100);
            }

            async function updateConfig(newValues) {
                try {
                    const response = await fetch("/config", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(newValues)
                    });
                    const result = await response.json();
                    if (result.status === "success") {
                        updateUIConfig(result.config);
                    }
                } catch (err) {
                    console.error("Failed to update config:", err);
                }
            }

            function updateUIConfig(config) {
                currentConfig = config;

                // Update Mode Tabs
                document.querySelectorAll("#mode-tabs .tab-btn").forEach(btn => btn.classList.remove("active"));
                const activeModeBtn = document.getElementById("mode-" + config.detection_mode);
                if (activeModeBtn) activeModeBtn.classList.add("active");

                // Update Orientation Tabs
                document.querySelectorAll("#orientation-tabs .tab-btn").forEach(btn => btn.classList.remove("active"));
                const activeOrientBtn = document.getElementById("orient-" + config.line_orientation);
                if (activeOrientBtn) activeOrientBtn.classList.add("active");

                // Update Slider position
                const pct = Math.round(config.line_position * 100);
                document.getElementById("position-slider").value = pct;
                document.getElementById("position-val").innerText = pct + "%";
            }

            async function resetCounter() {
                try {
                    const response = await fetch("/reset", {
                        method: "POST"
                    });
                    const result = await response.json();
                    console.log(result.message);
                    updateStats();
                } catch (err) {
                    console.error("Failed to reset counters:", err);
                }
            }

            async function updateStats() {
                try {
                    const response = await fetch("/stats");
                    const data = await response.json();

                    // Update Counts
                    document.getElementById("count-entry").innerText = data.entry_count;
                    document.getElementById("count-exit").innerText = data.exit_count;
                    document.getElementById("count-inside").innerText = data.current_inside;

                    // Update directions
                    document.getElementById("meta-entry-dir").innerText = data.entry_direction ? data.entry_direction.replace(/_/g, ' ') : "Not established";
                    document.getElementById("meta-exit-dir").innerText = data.exit_direction ? data.exit_direction.replace(/_/g, ' ') : "Not established";

                    // Update config if changed remotely or initially
                    if (data.detection_mode !== currentConfig.detection_mode ||
                        data.line_orientation !== currentConfig.line_orientation ||
                        Math.abs(data.line_position - currentConfig.line_position) > 0.01) {
                        updateUIConfig(data);
                    }
                } catch(error) {
                    console.log("Stats fetch error:", error);
                    document.getElementById("sys-status").innerText = "DISCONNECTED";
                    document.getElementById("sys-status").style.color = "var(--danger)";
                }
            }

            // Poll stats
            setInterval(updateStats, 300);
            updateStats();
        </script>
    </body>
    </html>
    """

# -----------------------------------------
# Configuration Update API
# -----------------------------------------
@app.post("/config")
def update_config(config: ConfigUpdate):
    counter.set_config(
        mode=config.detection_mode,
        orientation=config.line_orientation,
        position=config.line_position
    )
    return {
        "status": "success",
        "config": counter.get_stats()
    }

# -----------------------------------------
# Statistics API
# -----------------------------------------
@app.get("/stats")
def stats():
    return counter.get_stats()

# -----------------------------------------
# Video Stream Response
# -----------------------------------------
def generate_frames():
    while True:
        frame = counter.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# -----------------------------------------
# Reset Counters API
# -----------------------------------------
@app.post("/reset")
def reset():
    counter.reset()
    return {
        "message": "Counter reset successfully"
    }