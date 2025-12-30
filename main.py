import cv2
import threading
import time
import socket
from flask import Flask, Response, render_template
from flask_sock import Sock
from flaredantic import FlareTunnel, FlareConfig
from demo_detector import detect, check_person_fall
from queue import Queue, Empty
import webbrowser
from discordbot import send_text, start_async_loop, start_bot, send_alert

# =====================================================
# Flask setup
# =====================================================
app = Flask(__name__)
sock = Sock(app)

# =====================================================
# Shared state
# =====================================================
frame_lock = threading.Lock()
latest_frame = None

clients_lock = threading.Lock()
client_queues = set()  # each client gets its own queue

# =====================================================
# Fall detection state
# =====================================================
first_fall_time = None
alert_sent = False
was_fallen_last_frame = False

# =====================================================
# Performance tuning
# =====================================================
FRAME_SIZE = 320
CAP_FPS = 15
PROCESS_EVERY_N_FRAMES = 2

# =====================================================
# Camera / Detection Thread
# =====================================================
def camera_loop():
    global latest_frame, first_fall_time, alert_sent, was_fallen_last_frame

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        # Skip some frames for detection
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            with frame_lock:
                latest_frame = frame
            continue

        # ---------------- Detection ----------------
        try:
            actually_fallen, info, annotated, keypoints = detect(frame)
        except Exception as e:
            print("Detect error:", e)
            continue

        # ---------------- Fall Event ----------------
        if actually_fallen and not was_fallen_last_frame:
            now = time.time()
            if first_fall_time is None:
                first_fall_time = now
                alert_sent = False
                print("🟡 First fall detected")
            else:
                elapsed = now - first_fall_time
                if elapsed < 7:
                    print("⚪ Fall too soon, ignored")
                elif 2.5 <= elapsed <= 15 and not alert_sent:
                    print("🔴 ALERT: second fall in window")
                    send_alert(
                        "⚠️ FALL DETECTED (confirmed twice)",
                        frame
                    )
                    with clients_lock:
                        for q in client_queues:
                            q.put("FALLDETECTED")
                    alert_sent = True
                elif elapsed > 15:
                    print("🔁 Window expired, new first fall")
                    first_fall_time = now
                    alert_sent = False

        was_fallen_last_frame = actually_fallen

        # ---------------- Share frame ----------------
        with frame_lock:
            latest_frame = annotated if annotated is not None else frame

        time.sleep(0.005)

# =====================================================
# MJPEG Streaming
# =====================================================
@app.route("/video_feed")
def video_feed():
    def gen():
        last_sent_time = 0
        while True:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)
                continue

            # Limit FPS
            now = time.time()
            if now - last_sent_time < 1 / 15:
                time.sleep(0.01)
                continue
            last_sent_time = now

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =====================================================
# WebSocket for Alerts
# =====================================================
@sock.route('/ws')
def ws(ws):
    q = Queue()
    with clients_lock:
        client_queues.add(q)

    try:
        while True:
            try:
                msg = q.get(timeout=1)  # blocking ensures reliable alert delivery
                ws.send(msg)
                time.sleep(0.01)  # prevent busy loop
            except Empty:
                continue
    finally:
        with clients_lock:
            client_queues.remove(q)

# =====================================================
# Index
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")

# =====================================================
# Flare Tunnel
# =====================================================
def start_tunnel():
    global public_url
    config = FlareConfig(port=5000)
    with FlareTunnel(config) as tunnel:
        public_url = tunnel.tunnel_url
        print("🌐 Public URL:", public_url)
        send_text(public_url)
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
# If your server runs on a specific port, set it here
        port = 5000  # change if needed
        url = f"http://{local_ip}:{port}"
        webbrowser.open(url)
        while True:
            time.sleep(1)

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    threading.Thread(target=start_async_loop, daemon=True).start()
    threading.Thread(target=start_bot, daemon=True).start()
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=start_tunnel, daemon=True).start()

    print("🚀 Server running")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
