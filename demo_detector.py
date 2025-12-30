import cv2
from ultralytics import YOLO
import mediapipe as mp
import math

# ---------------- PARAMETERS -----------------
MODEL_PATH = "yolo12n.pt"
CLASSES_PATH = "classes.txt"

FRAME_SIZE = 320
MIN_PERSON_AREA_RATIO = 0.05
KEYPOINT_VIS_TH = 0.35
BODY_VIS_FRACTION = 0.35
FALL_RATIO_THRESHOLD = 1.35  # h/w
FALL_ANGLE_THRESHOLD = 45    # degrees, optional tilt check

# ---------------- LOAD MODEL -----------------
model = YOLO(MODEL_PATH)
model.fuse()  # CPU speedup
with open(CLASSES_PATH, "r") as f:
    classnames = f.read().splitlines()

# ---------------- MEDIAPIPE -----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)
TOTAL_LANDMARKS = 33

# ---------------- HELPER FUNCTIONS -----------------
def body_visibility_ok(keypoints, bbox):
    if not keypoints:
        return False
    x1, y1, x2, y2 = bbox
    inside = 0
    for x, y, v in keypoints:
        if v < KEYPOINT_VIS_TH:
            continue
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside += 1
    return (inside / TOTAL_LANDMARKS) >= BODY_VIS_FRACTION

def fall_angle(keypoints):
    """
    Optional: compute tilt angle of body using shoulders and hips.
    Returns angle in degrees between vertical and line connecting shoulders/hips.
    """
    if len(keypoints) < 33:
        return 0
    # shoulder midpoint
    lx, ly, lv = keypoints[11]
    rx, ry, rv = keypoints[12]
    if lv < KEYPOINT_VIS_TH or rv < KEYPOINT_VIS_TH:
        return 0
    mid_shoulder = ((lx+rx)/2, (ly+ry)/2)
    # hip midpoint
    lx, ly, lv = keypoints[23]
    rx, ry, rv = keypoints[24]
    if lv < KEYPOINT_VIS_TH or rv < KEYPOINT_VIS_TH:
        return 0
    mid_hip = ((lx+rx)/2, (ly+ry)/2)
    dx = mid_hip[0] - mid_shoulder[0]
    dy = mid_hip[1] - mid_shoulder[1]
    if dy == 0:
        return 90
    angle = abs(math.degrees(math.atan(dx/dy)))
    return angle

# ---------------- FALL CHECK (AGGRESSIVE) -----------------
def check_person_fall(info, frame, keypoints=None):
    """
    Aggressive fall detection: anyone prone triggers fall.
    Ignores objects to reduce false negatives.
    """
    if "person" not in info:
        return False

    fallen_any = False
    for p in info["person"]:
        ratio = p["height"] / (p["width"] + 1e-6)
        angle = 0
        if keypoints:
            angle = fall_angle(keypoints)

        if ratio < FALL_RATIO_THRESHOLD or angle > FALL_ANGLE_THRESHOLD:
            fallen_any = True
            cv2.putText(frame, "FALL!", (p["x1"], p["y1"] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "standing", (p["x1"], p["y1"] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return fallen_any

# ---------------- MAIN DETECT FUNCTION -----------------
def detect(frame, conf_threshold=0.6):
    info = {}
    new_frame = frame.copy()

    # YOLO detection
    results = model(frame, imgsz=FRAME_SIZE, verbose=False)

    # MediaPipe pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb)
    keypoints = []
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks.landmark:
            keypoints.append((
                int(lm.x * frame.shape[1]),
                int(lm.y * frame.shape[0]),
                lm.visibility
            ))

    # Process YOLO results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = classnames[cls]

            if conf < conf_threshold:
                continue
            if name not in ("person", "bed", "sofa", "chair"):
                continue

            w, h = x2 - x1, y2 - y1

            if name == "person":
                area_ratio = (w*h) / (frame.shape[0]*frame.shape[1])
                if area_ratio < MIN_PERSON_AREA_RATIO:
                    continue
                if not body_visibility_ok(keypoints, (x1, y1, x2, y2)):
                    cv2.putText(new_frame, "partial body", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    continue

            # Store info
            info.setdefault(name, []).append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": w, "height": h})

            # Draw bbox
            color = (0,255,0) if name != "person" else (255,255,0)
            cv2.rectangle(new_frame, (x1,y1), (x2,y2), color, 2)

    # Aggressive fall check
    actually_fallen = check_person_fall(info, new_frame, keypoints)

    return actually_fallen, info, new_frame, keypoints

# ---------------- TEST -----------------
if __name__ == "__main__":
    frame = cv2.imread("hq722.jpg")
    fallen, info, annotated, keypoints = detect(frame)
    print("Fall detected:", fallen)
    print("Detection info:", info)
    print("Keypoints:", keypoints)
    cv2.imshow("Annotated", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
