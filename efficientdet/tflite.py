import os
import numpy as np
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter

# === Configuration ===
MODEL_PATH = "models/efficientdet-lite0.tflite"  # path to your TFLite model
IMAGE_FOLDER = "images"  # folder with input images
CONF_THRESHOLD = 0.5  # confidence threshold to show detection

# === Label Map (replace or expand as needed) ===
LABELS = {
    1: 'apel_busuk', 2: 'apel_segar', 3: 'ayam_busuk', 4: 'ayam_segar',
    5: 'bayam_busuk', 6: 'bayam_segar', 7: 'brokoli_busuk', 8: 'brokoli_segar',
    9: 'buncis_busuk', 10: 'buncis_segar', 11: 'jagung_busuk', 12: 'jagung_segar',
    13: 'jamur_busuk', 14: 'jamur_segar', 15: 'jeruk_busuk', 16: 'jeruk_segar',
    17: 'kacangpanjang_segar', 18: 'kangkung_busuk', 19: 'kangkung_segar',
    20: 'kentang_busuk', 21: 'kentang_segar', 22: 'kubis_busuk', 23: 'kubis_segar',
    24: 'labusiam_busuk', 25: 'labusiam_segar', 26: 'lemon_busuk', 27: 'lemon_segar',
    28: 'naga_busuk', 29: 'naga_segar', 30: 'nanas_busuk', 31: 'nanas_segar',
    32: 'pear_busuk', 33: 'pear_segar', 34: 'sapi_busuk', 35: 'sapi_segar',
    36: 'sawihijau_busuk', 37: 'sawihijau_segar', 38: 'sawiputih_busuk',
    39: 'sawiputih_segar', 40: 'tahu_segar', 41: 'tauge_busuk', 42: 'tauge_segar',
    43: 'tempe_busuk', 44: 'tempe_segar', 45: 'terong_busuk', 46: 'terong_segar',
    47: 'wortel_busuk', 48: 'wortel_segar'
}

# === Load TFLite Model ===
print("Loading model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # [1, height, width, 3]
print("Model loaded successfully.")

# === Inference Function ===
def run_inference(image_path):
    print(f"Processing {image_path}")
    
    image = Image.open(image_path).convert('RGB').resize((input_shape[2], input_shape[1]))
    input_data = np.expand_dims(np.array(image), axis=0)

    # Normalize if model requires float input
    if input_details[0]['dtype'] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # [N, 4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]    # [N]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]     # [N]

    # Draw results
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    detected = False

    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(classes[i])
            label = LABELS.get(class_id, f"class_{class_id}")
            score = scores[i]

            # Coordinates
            left = int(xmin * image.width)
            top = int(ymin * image.height)
            right = int(xmax * image.width)
            bottom = int(ymax * image.height)

            # Draw box and label
            draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
            draw.text((left, top), f"{label} ({score:.2f})", fill="red")
            print(f"  → {label} ({score:.2f})")
            detected = True

    # Save annotated result
    output_path = image_path.replace(".jpg", "_detected.jpg").replace(".png", "_detected.png")
    image_draw.save(output_path)

    if detected:
        print(f"Detection result saved to {output_path}")
    else:
        print("No confident detections found.")

# === Run Inference on All Images ===
if not os.path.exists(IMAGE_FOLDER):
    print(f"Image folder not found: {IMAGE_FOLDER}")
    exit(1)

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    print("No images found in folder.")
else:
    for img_file in image_files:
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        run_inference(img_path)
