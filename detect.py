import RPi.GPIO as GPIO
import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, db
import os
import logging
from typing import List, Dict, Tuple

# ===================================================================
# === 1. SETUP
# ===================================================================

# This section sets up logging, creates a directory for captured images, and initializes the Firebase connection.
# - Logging is configured to provide informative, timestamped output for easy debugging.
# - The 'captured_images' directory is created to store temporary image files and detected images.
# - The Firebase connection is initialized using a service account key.
# - A mapping of food labels to unique IDs is created for consistent database entries.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
IMAGE_DIR = "captured_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# === Firebase Setup ===
previous_labels = set()
try:
    cred = credentials.Certificate("smart-fridge-fd12e-firebase-adminsdk-fbsvc-340297fc62.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smart-fridge-fd12e-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")

# === Generate base label and id mapping ===
ordered_base_labels = [
    "apel", "bayam", "brokoli", "buncis", "ayam", "sapi", "jagung", "jamur", "jeruk",
    "kangkung", "kacangpanjang", "kentang", "kubis", "labusiam", "lemon", "naga", "nanas", "pear",
    "sawihijau", "sawiputih", "tahu", "tauge", "tempe", "terong", "wortel"
]
label_id_map = {name: idx + 1 for idx, name in enumerate(ordered_base_labels)}


# ===================================================================
# === 2. IMPROVED OBJECT DETECTOR CLASS
# ===================================================================

class ImprovedObjectDetector:
    """
    Encapsulates the entire object detection pipeline using a TensorFlow Lite model.
    The class handles model loading, image preprocessing, inference,
    Non-Maximum Suppression (NMS) for filtering detections, and visualization.
    """
    def __init__(self, model_path: str, label_path: str):
        self.labels = self._load_labels(label_path)
        self.interpreter = self._load_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Loaded {len(self.labels)} labels from {label_path}")

    def _load_labels(self, label_path: str) -> Dict[int, str]:
        """Loads human-readable labels from a text file."""
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        return {i: label for i, label in enumerate(labels)}

    def _load_model(self, model_path: str) -> tflite.Interpreter:
        """Loads the TensorFlow Lite model and allocates tensors."""
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def _get_color(self, label_id: int) -> Tuple[int, int, int]:
        """Generates a consistent, random color for a given label ID for visualization."""
        np.random.seed(label_id * 42)
        return tuple(int(c) for c in np.random.randint(50, 255, 3))

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes and pads an image to match the model's input dimensions while
        preserving the aspect ratio. This prevents distortion and improves detection accuracy.
        """
        original_height, original_width = image.shape[:2]
        scale = min(self.input_width / original_width, self.input_height / original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_height, self.input_width, 3), 128, dtype=np.uint8)
        y_offset = (self.input_height - new_height) // 2
        x_offset = (self.input_width - new_width) // 2
        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        return padded, scale, (x_offset, y_offset)

    def _apply_numpy_nms(self, detections: np.ndarray, iou_threshold: float, score_threshold: float) -> List[Tuple]:
        """
        Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes
        based on their confidence score and Intersection over Union (IoU).
        """
        boxes_index, score_index, class_id_index = slice(1, 5), 5, 6
        valid_detections = detections[detections[:, score_index] > score_threshold]
        if valid_detections.shape[0] == 0: return []
        boxes, scores, class_ids = valid_detections[:, boxes_index], valid_detections[:, score_index], valid_detections[:, class_id_index].astype(int)
        final_detections = []
        for class_id in np.unique(class_ids):
            class_indices = np.where(class_ids == class_id)[0]
            if len(class_indices) == 0: continue
            class_boxes, class_scores = boxes[class_indices], scores[class_indices]
            order = class_scores.argsort()[::-1]
            y1, x1, y2, x2 = class_boxes[:, 0], class_boxes[:, 1], class_boxes[:, 2], class_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            keep_indices = []
            while order.size > 0:
                i = order[0]
                keep_indices.append(i)
                if len(order) == 1: break
                xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
                xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
                w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
                intersection = w * h
                iou = intersection / (areas[i] + areas[order[1:]] - intersection)
                inds_to_keep = np.where(iou <= iou_threshold)[0]
                order = order[inds_to_keep + 1]
            for idx in keep_indices:
                final_detections.append((class_boxes[idx], class_scores[idx], class_id))
        final_detections.sort(key=lambda x: x[1], reverse=True)
        return final_detections

    def _postprocess_detections(self, detections: List[Tuple], original_shape: Tuple[int, int], scale: float, offset: Tuple[int, int]) -> List[Dict]:
        """
        Converts normalized bounding box coordinates from the model's output back to the
        original image's pixel dimensions.
        """
        original_height, original_width = original_shape
        x_offset, y_offset = offset
        processed_detections = []
        for box, score, class_id in detections:
            ymin, xmin, ymax, xmax = box
            xmin, ymin = (xmin - x_offset) / scale, (ymin - y_offset) / scale
            xmax, ymax = (xmax - x_offset) / scale, (ymax - y_offset) / scale
            xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
            xmax, ymax = min(original_width - 1, int(xmax)), min(original_height - 1, int(ymax))
            if xmax <= xmin or ymax <= ymin: continue
            processed_detections.append({'bbox': [xmin, ymin, xmax, ymax], 'score': float(score), 'class_id': int(class_id), 'label': self.labels.get(class_id, 'unknown')})
        return processed_detections

    def _visualize_detections(self, image: np.ndarray, detections: List[Dict], output_path: str = None):
        """Draws bounding boxes and labels on the image and saves the result to a file."""
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            score, label, class_id = det['score'], det['label'], det['class_id']
            color = self._get_color(class_id)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label_text = f"{label}: {int(score * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            padding = 5
            cv2.rectangle(image, (x_min, y_min), (x_min + text_width + padding * 2, y_min + text_height + padding * 2), color, -1)
            cv2.putText(image, label_text, (x_min + padding, y_min + text_height + padding), font, font_scale, (0, 0, 0), font_thickness)
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Saved result to {output_path}")

    def run_detection(self, image_path: str, score_threshold: float = 0.2, iou_threshold: float = 0.4) -> List[Dict]:
        """
        Main method to run the complete detection process on an image file.
        Returns a list of detected objects with their details.
        """
        original = cv2.imread(image_path)
        if original is None: raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        preprocessed, scale, offset = self._preprocess_image(img_rgb)
        input_tensor = np.expand_dims(preprocessed, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        raw_detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        filtered_detections = self._apply_numpy_nms(raw_detections, iou_threshold, score_threshold)
        final_detections = self._postprocess_detections(filtered_detections, original.shape[:2], scale, offset)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(IMAGE_DIR, f"{base_name}_detected.jpg")
        self._visualize_detections(original, final_detections, result_path)
        return final_detections


# ===================================================================
# === 3. FIREBASE FUNCTION
# ===================================================================

def send_to_firebase(current_labels: Dict[str, int]):
    """
    Synchronizes the detected food items with the Firebase Realtime Database.
    It handles additions, updates, and deletions of items efficiently in a single batch update.
    """
    global previous_labels
    now_iso = datetime.now().isoformat()
    # Grace period simulates a new item having a full freshness period.
    grace_period_time = datetime.now() - timedelta(hours=23, minutes=50)
    grace_period_time_iso = grace_period_time.isoformat()

    root_ref = db.reference("objects")
    current_base_labels = set()
    update_payload = {}

    # Prepare updates and additions
    for full_label, count in current_labels.items():
        parts = full_label.split('_')
        if len(parts) != 2: continue
        
        base_name, status_suffix = parts
        current_base_labels.add(base_name)
        
        path = base_name
        existing_data = root_ref.child(path).get()
        
        is_fresh_value = 1 if status_suffix == "segar" else 0
        status_value = "is_fresh" if status_suffix == "segar" else "is_rotten"
        image_ref = db.reference(f"images_references/{base_name}/image_url")
        image_url = image_ref.get() or "https://via.placeholder.com/150"
        nama_item_capitalized = base_name.capitalize()

        if existing_data is None:
            logger.info(f"➕ Preparing to ADD: {base_name}")
            update_payload[path] = {
                "id": label_id_map.get(base_name, 0), "nama_item": nama_item_capitalized,
                "entry_time": grace_period_time_iso, "last_updated": now_iso, "frequency": 1,
                "count": count, "status": status_value, "is_fresh": is_fresh_value,
                "image_url": image_url
            }
        else:
            logger.info(f"🔄 Preparing to UPDATE: {base_name}")
            update_payload[f'{path}/last_updated'] = now_iso
            update_payload[f'{path}/count'] = count
            update_payload[f'{path}/status'] = status_value
            update_payload[f'{path}/is_fresh'] = is_fresh_value
            update_payload[f'{path}/image_url'] = image_url
            update_payload[f'{path}/nama_item'] = nama_item_capitalized
            
            if base_name not in previous_labels:
                update_payload[f'{path}/entry_time'] = now_iso
                update_payload[f'{path}/frequency'] = existing_data.get("frequency", 0) + 1

    # Prepare deletions
    labels_to_remove = previous_labels - current_base_labels
    for label_to_remove in labels_to_remove:
        logger.info(f"➖ Preparing to DELETE: {label_to_remove}")
        # Setting a path to None is the Firebase command for deletion.
        update_payload[label_to_remove] = None

    # Execute all changes in one atomic operation.
    if update_payload:
        logger.info("🚀 Sending all changes to Firebase...")
        root_ref.update(update_payload)
        logger.info("✅ Database synchronized.")
    
    # Update the global state for the next run.
    previous_labels = current_base_labels

# ===================================================================
# === 4. HARDWARE AND CORE LOGIC
# ===================================================================

# This section sets up the GPIO pins for the relay and magnetic sensor.
# It also initializes the camera and the object detector.

# Pin definitions for Raspberry Pi GPIO
RELAY_PIN, SENSOR_PIN = 26, 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Camera initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.critical("Could not open camera. Exiting.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(2)  # Allow camera to warm up

# Initialize the object detection model
try:
    detector = ImprovedObjectDetector(
        model_path="trained-model/efficientdet-lite1.tflite",
        label_path="trained-model/labelmap.txt"
    )
except Exception as e:
    logger.critical(f"Failed to initialize detector. Exiting. Error: {e}")
    exit()

def initialize_previous_labels_from_firebase():
    """
    Fetches the initial state of items from the Firebase database to prime the system.
    This ensures the program starts with an accurate list of existing items.
    """
    logger.info("Initializing state from Firebase...")
    try:
        root_ref = db.reference("objects")
        existing_objects = root_ref.get()
        if existing_objects:
            initial_labels = set(existing_objects.keys())
            logger.info(f"Found initial items in Firebase: {initial_labels}")
            return initial_labels
        return set()
    except Exception as e:
        logger.error(f"Could not initialize state from Firebase: {e}")
        return set()

previous_labels = initialize_previous_labels_from_firebase()
image_counter = 1

def capture_and_process_image() -> Dict[str, int]:
    """
    Captures an image from the camera, runs object detection on it, and
    returns a dictionary of detected labels and their counts.
    """
    global image_counter
    logger.info("📸 Clearing camera buffer...")
    for _ in range(3): cap.read() # Read and discard a few frames to clear the buffer
    ret, frame = cap.read()
    if ret:
        filename = os.path.join(IMAGE_DIR, f"cap_img_{image_counter}.jpg")
        cv2.imwrite(filename, frame)
        logger.info(f"📷 Image captured: {filename}")
        image_counter += 1
        detections = detector.run_detection(image_path=filename)
        label_counts = {}
        for det in detections:
            label = det['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"Detection complete. Found: {label_counts}")
        return label_counts
    logger.error("❌ Failed to capture valid image.")
    return {}

# ===================================================================
# === 5. MAIN EXECUTION LOOP
# ===================================================================

# This is the main program loop. It continuously monitors the magnetic sensor
# and triggers the capture and synchronization process when the fridge door opens.

try:
    logger.info("📟 Monitoring sensor... Press Ctrl+C to stop.")
    last_state = GPIO.HIGH
    while True:
        state = GPIO.input(SENSOR_PIN)
        if last_state == GPIO.HIGH and state == GPIO.LOW:
            start_time = time.time()  # --- TIMER START ---
            
            logger.info("🔔 Magnet detected! Triggering capture...")
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            time.sleep(2)  # Wait for light to turn on and stabilize
            
            label_counts = capture_and_process_image()
            
            # Send data to Firebase if there are new detections or if a previous state needs clearing.
            if label_counts or previous_labels:
                send_to_firebase(label_counts)
            else:
                logger.info("No objects detected and no previous state to clear.")
            
            GPIO.output(RELAY_PIN, GPIO.LOW)
            
            end_time = time.time()  # --- TIMER END ---
            duration = end_time - start_time
            logger.info(f"✔️ Cycle complete. Total time: {duration:.2f} seconds.")

        last_state = state
        time.sleep(0.1)  # Small delay to prevent high CPU usage
except KeyboardInterrupt:
    logger.info("🛑 Program interrupted by user.")
finally:
    # Clean up all resources when the program exits.
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("✅ Cleaned up GPIO and camera.")