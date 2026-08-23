from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

import cv2
import time

from people_counter import PeopleCounter


app = FastAPI(
    title="Gate People Counter"
)


counter = PeopleCounter()


# -----------------------------------------
# Home page
# -----------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Gate People Counter</title>

        <style>

            body {
                font-family: Arial;
                background: #111;
                color: white;
                text-align: center;
            }

            .container {
                width: 90%;
                max-width: 1000px;
                margin: auto;
            }

            img {
                width: 100%;
                max-width: 900px;
                border: 2px solid white;
            }

            .stats {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin: 20px;
            }

            .box {
                padding: 20px;
                background: #222;
                border-radius: 10px;
                min-width: 150px;
            }

            .number {
                font-size: 40px;
                font-weight: bold;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>Gate People Counter</h1>

            <img src="/video">

            <div class="stats">

                <div class="box">

                    <h3>ENTRY</h3>

                    <div
                        class="number"
                        id="entry"
                    >
                        0
                    </div>

                </div>


                <div class="box">

                    <h3>EXIT</h3>

                    <div
                        class="number"
                        id="exit"
                    >
                        0
                    </div>

                </div>


                <div class="box">

                    <h3>INSIDE</h3>

                    <div
                        class="number"
                        id="inside"
                    >
                        0
                    </div>

                </div>

            </div>


            <p id="direction">
                Direction not established
            </p>

        </div>


        <script>

            async function updateStats() {

                try {

                    const response =
                        await fetch("/stats");

                    const data =
                        await response.json();

                    document.getElementById(
                        "entry"
                    ).innerText =
                        data.entry_count;


                    document.getElementById(
                        "exit"
                    ).innerText =
                        data.exit_count;


                    document.getElementById(
                        "inside"
                    ).innerText =
                        data.current_inside;


                    document.getElementById(
                        "direction"
                    ).innerText =
                        "Entry: "
                        + data.entry_direction
                        + " | Exit: "
                        + data.exit_direction;

                }

                catch(error) {

                    console.log(error);

                }

            }


            setInterval(
                updateStats,
                500
            );

            updateStats();

        </script>

    </body>

    </html>
    """


# -----------------------------------------
# Statistics API
# -----------------------------------------

@app.get("/stats")
def stats():

    return counter.get_stats()


# -----------------------------------------
# Video stream
# -----------------------------------------

def generate_frames():

    while True:

        frame = counter.get_frame()

        if frame is None:

            time.sleep(0.01)

            continue

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

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
# Reset counter
# -----------------------------------------

@app.post("/reset")
def reset():

    counter.reset()

    return {
        "message": "Counter reset successfully"
    }