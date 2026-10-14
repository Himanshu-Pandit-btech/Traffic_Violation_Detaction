# traffic_system_with_ai_lpr.py 
import cv2
import numpy as np
import datetime
import time
import os
from collections import deque
import easyocr
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

# --- CONFIGURATION ---
CONFIG = {
    'SCREEN_WIDTH': 1920, 'SCREEN_HEIGHT': 1080, 'SPEED_LIMIT_KMH': 30,
    'PIXELS_PER_METER': 20.0, # !!! IMPORTANT: YOU MUST UPDATE THIS VALUE AFTER CALIBRATION !!!
    'STOP_LINE_Y': 550,      # !!! IMPORTANT: ADJUST THIS Y-COORDINATE FOR YOUR VIDEO'S STOP LINE !!!
    'DETECTION_CONFIDENCE_THRESHOLD': 0.5,
    'CONFIRM_STATIONARY_FRAMES': 15, 'ZONE_EXPIRATION_SECONDS': 10,
    'VIOLATION_GRACE_PERIOD_FRAMES': 15, 'MIN_RIDER_AREA': 5000,
    'RIDER_BIKE_IOU_THRESHOLD': 0.2, 'HELMET_HEAD_IOU_THRESHOLD': 0.5,
    'LOG_FILE': 'violations_log.txt'
}
CONFIG['SPEED_LIMIT_MS'] = CONFIG['SPEED_LIMIT_KMH'] / 3.6

# Colors & Fonts
COLORS = {
    'NORMAL': (0, 255, 0), 'VIOLATION': (0, 0, 255), 'PERSON': (255, 0, 255),
    'HELMET': (255, 165, 0), 'TWOWHEELER': (0, 165, 255), 'PANEL_BG': (240, 240, 240),
    'TEXT': (0, 0, 0), 'STOP_LINE': (0, 255, 255),
    'LIGHT_RED': (0, 0, 255), 'LIGHT_YELLOW': (0, 255, 255), 'LIGHT_GREEN': (0, 255, 0),
    'LIGHT_UNKNOWN': (128, 128, 128)
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- DATA STRUCTURES (Global) ---
tracked_objects: Dict[int, Dict[str, Any]] = {}
violations_log: List[str] = []
stationary_zones: List[Dict[str, Any]] = []

# --- HELPER FUNCTIONS ---
def get_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]: 
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

def get_area(box: Tuple[int, int, int, int]) -> int: 
    return (box[2] - box[0]) * (box[3] - box[1])

def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = get_area(boxA); boxBArea = get_area(boxB)
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def calculate_speed_ms(positions: deque, fps: float) -> float:
    if len(positions) < 5 or fps == 0: return 0.0
    SPEED_WINDOW = min(len(positions), 10)
    relevant_pos = list(positions)[-SPEED_WINDOW:]
    start_point, end_point = relevant_pos[0], relevant_pos[-1]
    pixel_dist = np.linalg.norm(np.array(start_point) - np.array(end_point))
    time_s = len(relevant_pos) / fps
    if time_s == 0: return 0.0
    meter_dist = pixel_dist / CONFIG['PIXELS_PER_METER']
    return meter_dist / time_s

def detect_traffic_light_color(light_roi: np.ndarray) -> str:
    if light_roi is None or light_roi.size == 0 or light_roi.shape[0] < 3: return "Unknown"
    h, w, _ = light_roi.shape; hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
    section_height = h // 3
    top_section = hsv[0:section_height, :]; middle_section = hsv[section_height:2*section_height, :]; bottom_section = hsv[2*section_height:3*section_height, :]
    top_brightness = np.mean(top_section[:,:,2]); middle_brightness = np.mean(middle_section[:,:,2]); bottom_brightness = np.mean(bottom_section[:,:,2])
    brightnesses = {'Red': top_brightness, 'Yellow': middle_brightness, 'Green': bottom_brightness}
    brightest_light = max(brightnesses, key=brightnesses.get)
    if brightnesses[brightest_light] < 100: return "Unknown"
    return brightest_light

def detect_license_plate_ai(vehicle_roi: np.ndarray, lpr_model: YOLO, reader: easyocr.Reader) -> str:
    """Detects license plates using a dedicated LPR YOLO model and then OCR."""
    if reader is None or lpr_model is None or vehicle_roi is None or vehicle_roi.size == 0:
        return "No Plate Detected"
    
    try:
        lpr_results = lpr_model(vehicle_roi, verbose=False)
        if len(lpr_results[0].boxes) == 0: return "No Plate Detected"
        
        plate_box = lpr_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = plate_box
        cropped_plate = vehicle_roi[y1:y2, x1:x2]

        if cropped_plate.size == 0: return "No Plate Detected"

        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        ocr_result = reader.readtext(gray_plate, detail=0, paragraph=False, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        if ocr_result:
            plate_text = "".join(ocr_result).upper().replace(" ", "").replace("-", "")
            return plate_text if 4 < len(plate_text) < 11 else "Reading..."
        
        return "Reading..."
    except Exception:
        return "No Plate Detected"

def save_violation_evidence(frame: np.ndarray, v_type_folder: str, obj_id: int, plate: str = "No Plate Detected"):
    timestamp = datetime.datetime.now().strftime("%Ym%d_%H%M%S_%f")
    v_dir = os.path.join('Violations', v_type_folder); os.makedirs(v_dir, exist_ok=True)
    filepath = os.path.join(v_dir, f"{v_type_folder}_{obj_id}_{timestamp}.jpg")
    cv2.putText(frame, f"Violation: {v_type_folder}", (10, 30), FONT, 1, COLORS['VIOLATION'], 2); cv2.putText(frame, f"License: {plate}", (10, 70), FONT, 1, COLORS['VIOLATION'], 2)
    cv2.imwrite(filepath, frame)

def log_to_file(message: str):
    with open(CONFIG['LOG_FILE'], 'a') as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        
# --- CORE LOGIC FUNCTIONS ---
def update_trackers(results: list, general_model, video_fps: float):
    global stationary_zones, tracked_objects
    current_time = time.time()
    current_frame_ids = set()
    
    if results[0].boxes.id is not None:
        boxes, class_ids, tracker_ids, confs = (d.cpu().numpy() for d in [results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id, results[0].boxes.conf])
        for box, cls_id, trk_id, conf in zip(boxes, class_ids, tracker_ids, confs):
            if conf < CONFIG['DETECTION_CONFIDENCE_THRESHOLD']: continue
            if general_model.names[int(cls_id)] == 'traffic light': continue
            box = box.astype(int); current_frame_ids.add(trk_id)
            if trk_id not in tracked_objects:
                tracked_objects[trk_id] = {'positions': deque(maxlen=int(video_fps*2)), 'violations': set(), 'plate': 'No Plate Detected', 'speed': 0.0, 'stationary_frames': 0, 'frames_seen': 0, 'has_helmet': False}
            trk_obj = tracked_objects[trk_id]; cls_name = general_model.names[int(cls_id)].capitalize()
            trk_obj.update({'box': tuple(box), 'cls': cls_name, 'frames_seen': trk_obj.get('frames_seen', 0) + 1, 'has_helmet': False})
            trk_obj['positions'].append(get_center(box))
            is_in_zone = any(zone['cls'] == trk_obj['cls'] and iou(box, zone['box']) > 0.7 for zone in stationary_zones)
            if is_in_zone: trk_obj['speed'] = 0.0; trk_obj['stationary_frames'] = CONFIG['CONFIRM_STATIONARY_FRAMES'] + 1
            else:
                calculated_speed = calculate_speed_ms(trk_obj['positions'], video_fps)
                if calculated_speed < 0.2: trk_obj['stationary_frames'] += 1
                else: trk_obj['stationary_frames'] = 0
                trk_obj['speed'] = calculated_speed
            if trk_obj['stationary_frames'] > CONFIG['CONFIRM_STATIONARY_FRAMES']:
                trk_obj['speed'] = 0.0
                if not is_in_zone: stationary_zones.append({'box': box, 'cls': trk_obj['cls'], 'last_seen': current_time})
    lost_ids = set(tracked_objects.keys()) - current_frame_ids
    for obj_id in lost_ids: del tracked_objects[obj_id]

def check_for_violations(frame: np.ndarray, helmet_boxes: List[np.ndarray], lpr_model: YOLO, reader: easyocr.Reader, traffic_light_state: str):
    global tracked_objects, violations_log
    
    persons = [(obj_id, obj) for obj_id, obj in tracked_objects.items() if obj.get('cls') == 'Person']
    for person_id, person_obj in persons:
        person_box = person_obj['box']; head_area = (person_box[0], person_box[1], person_box[2], person_box[1] + int((person_box[3] - person_box[1]) / 2.5))
        for h_box in helmet_boxes:
            if iou(head_area, tuple(h_box)) > CONFIG['HELMET_HEAD_IOU_THRESHOLD']: person_obj['has_helmet'] = True; break

    for obj_id, obj in tracked_objects.items():
        if obj['frames_seen'] < CONFIG['VIOLATION_GRACE_PERIOD_FRAMES']: continue
        cls_name = obj.get('cls')

        if cls_name in ['Car', 'Bus', 'Truck', 'Motorcycle']:
            # Read license plate ONCE for any vehicle that is tracked long enough
            if obj['plate'] == 'No Plate Detected':
                obj['plate'] = detect_license_plate_ai(frame[obj['box'][1]:obj['box'][3], obj['box'][0]:obj['box'][2]], lpr_model, reader)

            if obj['speed'] > CONFIG['SPEED_LIMIT_MS'] and 'speeding' not in obj['violations']:
                obj['violations'].add('speeding'); save_violation_evidence(frame.copy(), 'Overspeeding', obj_id, obj['plate'])
                time_str = time.strftime('%H:%M:%S'); plate_info = f" ({obj['plate']})" if obj['plate'] != "No Plate Detected" else ""
                log_message = f"VIOLATION: Overspeeding by {cls_name}{plate_info} at {time_str}"; print(log_message); log_to_file(log_message)
                violations_log.append(f"{time_str} - Speeding: {cls_name}{plate_info}")
            if traffic_light_state == 'Red' and 'signal_jump' not in obj['violations']:
                if len(obj['positions']) > 1 and obj['positions'][-2][1] < CONFIG['STOP_LINE_Y'] and obj['positions'][-1][1] >= CONFIG['STOP_LINE_Y'] and obj['speed'] > 1.0:
                    obj['violations'].add('signal_jump'); save_violation_evidence(frame.copy(), 'Signal_Jump', obj_id, obj['plate'])
                    time_str = time.strftime('%H:%M:%S'); plate_info = f" ({obj['plate']})" if obj['plate'] != "No Plate Detected" else ""
                    log_message = f"VIOLATION: Signal Jump by {cls_name}{plate_info} at {time_str}"; print(log_message); log_to_file(log_message)
                    violations_log.append(f"{time_str} - Signal Jump: {cls_name}{plate_info}")
        elif cls_name == 'Person':
            if 'no_helmet' in obj['violations']: continue
            moving_motorcycles = [bike for bike in tracked_objects.values() if bike.get('cls') == 'Motorcycle' and bike['speed'] > 0.5]
            closest_bike = next((bike for bike in moving_motorcycles if np.linalg.norm(np.array(get_center(obj['box'])) - np.array(get_center(bike['box']))) < 150), None)
            if closest_bike and iou(obj['box'], closest_bike['box']) > CONFIG['RIDER_BIKE_IOU_THRESHOLD'] and get_area(obj['box']) > CONFIG['MIN_RIDER_AREA'] and get_center(obj['box'])[1] < get_center(closest_bike['box'])[1]:
                if not obj['has_helmet']:
                    obj['violations'].add('no_helmet'); save_violation_evidence(frame.copy(), 'No_Helmet', obj_id)
                    time_str = time.strftime('%H:%M:%S'); log_message = f"VIOLATION: No Helmet by Person (Rider ID: {obj_id}) at {time_str}"; print(log_message); log_to_file(log_message)
                    violations_log.append(f"{time_str} - No Helmet Violation")

def draw_overlays(frame: np.ndarray, traffic_lights: List[Dict]):
    global tracked_objects
    cv2.line(frame, (0, CONFIG['STOP_LINE_Y']), (frame.shape[1], CONFIG['STOP_LINE_Y']), COLORS['STOP_LINE'], 2)
    for obj_id, obj in tracked_objects.items():
        box, cls_name = obj['box'], obj.get('cls', '')
        if cls_name == 'Traffic light': continue
        has_violation = len(obj['violations']) > 0; color = COLORS['VIOLATION'] if has_violation else COLORS.get(cls_name.upper(), COLORS['NORMAL'])
        if not has_violation:
            if cls_name == 'Motorcycle': color = COLORS['TWOWHEELER']
            elif cls_name == 'Person': color = COLORS['PERSON']
        
        label = f"{cls_name} {obj['speed']:.1f} m/s"
        if obj.get('cls') in ['Car', 'Motorcycle'] and obj.get('plate') and obj['plate'] not in ["No Plate Detected", "Reading..."]:
            label += f" | {obj['plate']}"
        elif cls_name == 'Person':
            label = cls_name
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), FONT, 0.6, color, 2)

    for light in traffic_lights:
        box, color_name = light['box'], light['color']
        color = COLORS.get(f"LIGHT_{color_name.upper()}", COLORS['LIGHT_UNKNOWN'])
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2); cv2.putText(frame, f"Light: {color_name}", (box[0], box[1] - 10), FONT, 0.7, color, 2)

def create_info_panel(height: int) -> np.ndarray:
    global violations_log
    panel = np.zeros((height, 500, 3), dtype=np.uint8); panel[:] = COLORS['PANEL_BG']
    y = 40; cv2.putText(panel, "VIOLATIONS LOG", (10, y), FONT, 1, COLORS['TEXT'], 2); y += 40
    for log in violations_log[-15:]: cv2.putText(panel, log, (10, y), FONT, 0.6, COLORS['TEXT'], 1); y += 30;
    y = height - 200; cv2.putText(panel, "LEGEND", (10, y), FONT, 1, COLORS['TEXT'], 2); y += 40
    legend_items = {"Violation": COLORS['VIOLATION'], "Normal": COLORS['NORMAL'], "Two-Wheeler": COLORS['TWOWHEELER'], "Person": COLORS['PERSON'], "Helmet": COLORS['HELMET']}
    for name, color in legend_items.items():
        cv2.rectangle(panel, (10, y - 15), (40, y + 5), color, -1); cv2.putText(panel, name, (55, y), FONT, 0.7, COLORS['TEXT'], 2); y += 35
    return panel

# --- MAIN APPLICATION ---
def main():
    global stationary_zones, tracked_objects, violations_log, general_model, helmet_model, lpr_model
    script_dir = os.path.dirname(os.path.abspath(__file__)); os.chdir(script_dir)
    os.makedirs('Violations', exist_ok=True); print("Choose input source: 1. Webcam, 2. Video file"); choice = input("Enter your choice (1 or 2): ")
    cap = None
    if choice == "1": cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    elif choice == "2":
        video_path = input("Enter path to video file: "); 
        if not os.path.exists(video_path): print("File not found!"); return
        cap = cv2.VideoCapture(video_path)
    else: print("Invalid choice."); return
    if not cap.isOpened(): print("Cannot open video source."); return
    
    print("Loading models..."); general_model = YOLO('yolov8n.pt'); helmet_model = YOLO('yolov8n-helmet.pt'); lpr_model = YOLO('lpr_yolov8n.pt')
    try: reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e: print(f"EasyOCR Error: {e}"); reader = None
    
    calibration_points = []; 
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(calibration_points) < 2: calibration_points.append((x, y)); print(f"Point {len(calibration_points)} captured: ({x}, {y})")
            if len(calibration_points) == 2:
                dist_pixels = np.linalg.norm(np.array(calibration_points[0]) - np.array(calibration_points[1]))
                print("-" * 30); print(f"Calibration Distance: {dist_pixels:.1f} pixels"); print("Formula: PIXELS_PER_METER = (Pixel distance) / (Real-world length in meters)"); print("-" * 30)
                calibration_points.clear()
    
    WINDOW_NAME = 'Traffic Violation Detection'; cv2.namedWindow(WINDOW_NAME); cv2.setMouseCallback(WINDOW_NAME, mouse_callback); print("Starting detection...")

    paused = False; traffic_light_state = "Unknown"
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if choice == '2': cap.set(cv2.CAP_PROP_POS_FRAMES, 0); tracked_objects = {}; stationary_zones = []; violations_log.clear(); continue
                else: break
            
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            general_results = general_model.track(frame, persist=True, verbose=False, classes=[0, 2, 3, 5, 7, 9])
            helmet_results = helmet_model(frame, verbose=False)
            helmet_boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in helmet_results[0].boxes if b.conf[0] > 0.5]
            
            traffic_lights = []
            if general_results[0].boxes.id is not None:
                for box, cls_id, conf in zip(general_results[0].boxes.xyxy.cpu().numpy().astype(int), general_results[0].boxes.cls.cpu().numpy().astype(int), general_results[0].boxes.conf.cpu().numpy()):
                    if general_model.names[cls_id] == 'traffic light' and conf > CONFIG['DETECTION_CONFIDENCE_THRESHOLD']:
                        light_roi = frame[box[1]:box[3], box[0]:box[2]]; color = detect_traffic_light_color(light_roi)
                        traffic_lights.append({'box': box, 'color': color})
            
            if traffic_lights: traffic_light_state = traffic_lights[0]['color']

            update_trackers(general_results, general_model, video_fps)
            check_for_violations(frame, helmet_boxes, lpr_model, reader, traffic_light_state)
            draw_overlays(frame, traffic_lights)
            
        height, _ = frame.shape[:2]
        info_panel = create_info_panel(height); cv2.putText(frame, f"FPS: {video_fps:.1f}", (10, 30), FONT, 0.8, (0, 255, 0), 2)
        combined_frame = np.hstack((info_panel, frame)); ch, cw = combined_frame.shape[:2]
        scale = min(CONFIG['SCREEN_WIDTH'] / cw, CONFIG['SCREEN_HEIGHT'] / ch, 1.0)
        if scale < 1.0: display_frame = cv2.resize(combined_frame, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
        else: display_frame = combined_frame
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused

    cap.release(); cv2.destroyAllWindows(); print("Detection stopped.")

if __name__ == "__main__":
    main()