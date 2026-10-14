import cv2
import numpy as np
import datetime
import time
import os
from collections import deque
import easyocr
import torch
import torchvision.transforms as transforms
from torchvision.models import detection

# --- CONFIG ---
os.makedirs('violations', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SPEED_LIMIT = 75  # km/h
MIN_CONFIDENCE = 0.5
TRACKING_LENGTH = 30
PIXELS_PER_METER = 10  # calibrate this for your camera
FPS = 25  # set to your camera's FPS

GREEN = (0, 255, 0)
RED = (0, 0, 255)
PURPLE = (255, 0, 255)
BLUE = (255, 0, 0)

# Initialize EasyOCR
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"EasyOCR init error: {e}")
    reader = None

# Helmet detection model (NOTE: COCO pretrained model won't reliably detect helmets)
def load_helmet_detection_model():
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

try:
    helmet_model = load_helmet_detection_model()
except Exception as e:
    print(f"Helmet model load error: {e}")
    helmet_model = None

# COCO class names (for the model above)
COCO_INSTANCE_CATEGORY_NAMES = [
    'background','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A','N/A',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup',
    'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog',
    'pizza','donut','cake','chair','couch','potted plant','bed','N/A','dining table','N/A','N/A',
    'toilet','N/A','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
    'sink','refrigerator','N/A','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Traffic signal simulation
class TrafficSignal:
    def __init__(self):
        self.state = "red"
        self.last_change = time.time()
        self.cycle_time = 20

    def update(self):
        now = time.time()
        if (now - self.last_change) > self.cycle_time:
            self.last_change = now
            if self.state == "red":
                self.state = "green"
            elif self.state == "green":
                self.state = "yellow"
            else:
                self.state = "red"
        return self.state

traffic_signal = TrafficSignal()

# YOLO init
if not os.path.exists('yolov3-tiny.weights') or not os.path.exists('yolov3-tiny.cfg') or not os.path.exists('coco.names'):
    print("Missing YOLO files. Place yolov3-tiny.weights, yolov3-tiny.cfg and coco.names in working dir.")
    exit()

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
layer_names = net.getLayerNames()
out_layers = net.getUnconnectedOutLayers()
if out_layers.ndim == 2:
    output_layers = [layer_names[i[0] - 1] for i in out_layers]
else:
    output_layers = [layer_names[i - 1] for i in out_layers]

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Accept both common variants
vehicle_classes = {'car', 'motorbike', 'motorcycle', 'bus', 'truck', 'bicycle'}
person_class = 'person'

# Tracking and bookkeeping
trackers = {}            # obj_id -> tracker
tracked_objects = {}     # obj_id -> metadata dict (positions, type, etc.)
violations_log = []
frame_count = 0
obj_counter = 0  # unique id counter

# Define signal area polygon
signal_area = [(800, 100), (1100, 100), (1100, 300), (800, 300)]

# Utility functions
def calculate_speed(prev_pos, curr_pos, fps, ppm):
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    pixel_distance = np.sqrt(dx*dx + dy*dy)
    meters = pixel_distance / ppm
    # time between frames = 1/fps
    speed_m_s = meters * fps
    speed_kmh = speed_m_s * 3.6
    return speed_kmh

def save_violation_evidence(obj_id, violation_type, frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"violations/{violation_type}_{obj_id}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def in_signal_area(x, y):
    n = len(signal_area)
    inside = False
    p1x, p1y = signal_area[0]
    for i in range(n+1):
        p2x, p2y = signal_area[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def detect_license_plate(vehicle_roi):
    if reader is None:
        return "UNKNOWN"
    try:
        h = vehicle_roi.shape[0]
        # crop lower third of vehicle ROI (typical plate position)
        crop = vehicle_roi[int(h*0.6):, :]
        if crop.size == 0:
            crop = vehicle_roi
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)
        for r in results:
            text = r[1].upper().replace(" ", "")
            if 6 <= len(text) <= 12 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                return text
        return "UNKNOWN"
    except Exception as e:
        print(f"License plate detection error: {e}")
        return "UNKNOWN"

def detect_helmet(person_roi):
    # NOTE: pretrained COCO model doesn't include 'helmet' reliably.
    if helmet_model is None:
        return False
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        inp = transform(person_roi).unsqueeze(0)
        with torch.no_grad():
            preds = helmet_model(inp)
        for score, label in zip(preds[0]['scores'], preds[0]['labels']):
            if score > 0.5:
                name = COCO_INSTANCE_CATEGORY_NAMES[label]
                if 'helmet' in name.lower() or 'hat' in name.lower():
                    return True
        # fallback: return False (better to require dedicated model)
        return False
    except Exception as e:
        print(f"Helmet detection error: {e}")
        return False

# Basic IoU for simple duplicate detection (to avoid creating trackers for already tracked objects)
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    if (boxAArea + boxBArea - interArea) == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

# Start capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Starting traffic violation detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    frame_count += 1
    height, width = frame.shape[:2]

    right_panel = np.zeros((height, 400, 3), dtype=np.uint8) + 240

    # Update traffic signal
    signal_state = traffic_signal.update()

    # Draw signal area
    cv2.polylines(frame, [np.array(signal_area, dtype=np.int32)], True, (0,255,255), 2)
    cv2.putText(frame, f"Signal: {signal_state}", (signal_area[0][0], signal_area[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Detection every N frames
    if frame_count % 5 == 0:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                if confidence > MIN_CONFIDENCE:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if idxs is not None and len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]] if class_ids[i] < len(classes) else "N/A"
                if label not in vehicle_classes and label != person_class:
                    continue

                # Avoid creating trackers that overlap existing tracked objects (simple IoU threshold)
                overlap = False
                for tid, meta in tracked_objects.items():
                    if 'bbox' in meta:
                        if iou(meta['bbox'], (x,y,w,h)) > 0.5:
                            overlap = True
                            break
                if overlap:
                    continue

                # Create tracker (handle cv2 version differences)
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError:
                    tracker = cv2.TrackerKCF.create()

                tracker.init(frame, (x, y, w, h))
                obj_id = f"{label}_{obj_counter}"
                obj_counter += 1
                trackers[obj_id] = tracker

                # initial metadata
                roi = frame[max(0,y):y+h, max(0,x):x+w]
                license_plate = "N/A"
                helmet_detected = True
                if label != person_class:
                    license_plate = detect_license_plate(roi)
                if label in ('motorcycle','motorbike'):
                    helmet_detected = detect_helmet(roi)

                tracked_objects[obj_id] = {
                    'positions': deque(maxlen=TRACKING_LENGTH),
                    'type': label,
                    'speed': 0.0,
                    'license_plate': license_plate,
                    'helmet_detected': helmet_detected,
                    'violations': [],
                    'in_signal_area': False,
                    'bbox': (x,y,w,h)
                }

    # Update trackers
    to_remove = []
    for obj_id, tracker in list(trackers.items()):
        success, box = tracker.update(frame)
        if not success:
            to_remove.append(obj_id)
            continue

        x, y, w, h = [int(v) for v in box]
        cx = x + w//2
        cy = y + h//2

        meta = tracked_objects.get(obj_id)
        if meta is None:
            continue

        meta['positions'].append((cx, cy))
        meta['bbox'] = (x,y,w,h)

        if len(meta['positions']) > 1:
            sp = calculate_speed(meta['positions'][-2], meta['positions'][-1], FPS, PIXELS_PER_METER)
            meta['speed'] = sp

        # Signal area check
        if meta['type'] in vehicle_classes and signal_state == "red" and in_signal_area(cx, cy):
            if not meta.get('in_signal_area', False):
                meta['in_signal_area'] = True
                if 'signal_jump' not in meta['violations']:
                    meta['violations'].append('signal_jump')
                    ev = save_violation_evidence(obj_id, 'signal_jump', frame)
                    violations_log.append({'type':'Signal Jump','object_id':obj_id,'timestamp':datetime.datetime.now().strftime("%H:%M:%S"),'evidence':ev})
        else:
            meta['in_signal_area'] = False

        has_violation = False
        # Speed violation
        if meta['speed'] > SPEED_LIMIT and 'speed' not in meta['violations']:
            meta['violations'].append('speed')
            ev = save_violation_evidence(obj_id, 'speed', frame)
            violations_log.append({'type':'Speed Violation','object_id':obj_id,'speed':meta['speed'],'timestamp':datetime.datetime.now().strftime("%H:%M:%S"),'evidence':ev})
            has_violation = True

        # Helmet violation
        if meta['type'] in ('motorcycle','motorbike') and not meta['helmet_detected'] and 'no_helmet' not in meta['violations']:
            meta['violations'].append('no_helmet')
            ev = save_violation_evidence(obj_id, 'no_helmet', frame)
            violations_log.append({'type':'No Helmet','object_id':obj_id,'timestamp':datetime.datetime.now().strftime("%H:%M:%S"),'evidence':ev})
            has_violation = True

        if meta['in_signal_area']:
            has_violation = True

        color = RED if has_violation else (PURPLE if meta['type']==person_class else GREEN)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

        label_text = f"{meta['type']} {meta['speed']:.1f}km/h"
        if meta['type'] != person_class:
            label_text += f" {meta.get('license_plate','N/A')}"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for rid in to_remove:
        trackers.pop(rid, None)
        tracked_objects.pop(rid, None)

    # Right panel (violations)
    y_offset = 30
    cv2.putText(right_panel, "VIOLATIONS LOG", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    y_offset += 30
    for v in violations_log[-8:]:
        txt = f"{v['timestamp']} - {v['type']}"
        if 'speed' in v:
            txt += f" ({v['speed']:.1f}km/h)"
        cv2.putText(right_panel, txt, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        y_offset += 25

    # Legend
    y_offset = height - 140
    cv2.putText(right_panel, "LEGEND:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    y_offset += 25
    cv2.putText(right_panel, "Green Box: Normal", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    y_offset += 20
    cv2.putText(right_panel, "Red Box: Violation", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    y_offset += 20
    cv2.putText(right_panel, "Purple: Person", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PURPLE, 1)

    combined = np.hstack((frame, right_panel))
    cv2.imshow('Traffic Violation Detection', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Detected {len(violations_log)} violations. Evidence saved to 'violations/'")
