import discord
import asyncio
import cv2
from io import BytesIO
import os

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "cv2/data/haarcascade_frontalface_default.xml")
cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ================= CONFIG =================
BOT_TOKEN = "Yourtoken"

with open("userids.txt", "r") as f:
    CHANNEL_IDS = [int(x.strip()) for x in f if x.strip()]

# ================= DISCORD =================
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

loop = None

# ================= FACE BLUR =================
def blur_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)
    return frame

# ================= SEND =================
async def send_text(text):
    for cid in CHANNEL_IDS:
        ch = client.get_channel(cid)
        if ch:
            await ch.send(text)
            await asyncio.sleep(0.2)

async def send_frame(frame):
    frame = blur_faces(frame.copy())

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        await send_text("⚠️ Image encode failed")
        return

    file = discord.File(
        fp=BytesIO(buf.tobytes()),
        filename="fall.jpg"
    )

    for cid in CHANNEL_IDS:
        ch = client.get_channel(cid)
        if ch:
            await ch.send(file=file)
            await asyncio.sleep(0.2)

async def send_frame_and_text(frame, text):
    await send_text(text)
    await send_frame(frame)

# ================= THREAD-SAFE ENTRY =================
def send_alert_from_detector(frame, text):
    if loop is None:
        print("❌ Discord not ready")
        return

    asyncio.run_coroutine_threadsafe(
        send_frame_and_text(frame, text),
        loop
    )

# ================= EVENTS =================
@client.event
async def on_ready():
    global loop
    loop = asyncio.get_running_loop()
    print("✅ Discord bot ready")


@client.event
async def on_message(message):
    # ignore bot's own messages
    if message.author.bot:
        return

    if message.content.strip().upper() == "SERVERIDS":
        if message.guild:
            server_id = message.guild.id
            print(f"📌 Server ID: {server_id}")
            await message.channel.send(f"Server ID: `{server_id}`")
        else:
            await message.channel.send("❌ This command only works in a server.")

# ================= START =================
def run_bot():
    client.run(BOT_TOKEN)

if __name__ == "__main__":
    run_bot()
