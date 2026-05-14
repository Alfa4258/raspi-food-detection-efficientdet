# Raspberry Pi Food Detection

Raspberry Pi pipeline for detecting food items (fresh vs rotten) with an EfficientDet Lite TFLite model, capturing images from a USB camera, and syncing counts/status to Firebase Realtime Database.

## What it does

- Watches a magnetic door sensor and turns on a relay-controlled light when the door opens.
- Captures an image, runs EfficientDet Lite detection, and saves an annotated result in `captured_images/`.
- Aggregates detected labels and updates Firebase with counts, freshness status, and timestamps.

## Project layout

- `detect.py` - Main Raspberry Pi runtime (GPIO, camera capture, TFLite inference, Firebase sync).
- `trained-model/efficientdet-lite1.tflite` - TFLite model.
- `trained-model/labelmap.txt` - Label list used by the model.
- `environment-raspi.txt` - Python dependencies for Raspberry Pi.
- `efficientdet/` - Model training/inference utilities and reference code.

## Hardware

- Raspberry Pi with GPIO access
- USB camera (or Pi camera with OpenCV support)
- Magnetic door sensor (connected to GPIO pin 16 by default)
- Relay module for light control (connected to GPIO pin 26 by default)

## Firebase setup

This project expects a Firebase Admin SDK service account JSON in the repo root:

- `smart-fridge-fd12e-firebase-adminsdk-fbsvc-340297fc62.json`

The realtime database URL is configured in `detect.py`. Ensure the service account has read/write access to your database.

## Setup (Raspberry Pi)

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r environment-raspi.txt
```

3. Make sure `tflite-runtime` is installed for your device (the file path in `environment-raspi.txt` is device-specific).
4. Place the Firebase service account JSON in the repo root (see Firebase setup above).
5. Connect the camera, relay, and sensor to the GPIO pins listed below (or adjust in `detect.py`).

## GPIO pins

Configured in `detect.py`:

- Relay: GPIO 26
- Sensor: GPIO 16

## Run

```bash
python detect.py
```

The script will log sensor events, capture images, and update Firebase. Press `Ctrl+C` to stop.

## Notes

- Detected images are saved to `captured_images/` with bounding boxes.
- The code uses a grace period for new items when writing timestamps to Firebase.
- If you change label names, update `labelmap.txt` and the base label mapping in `detect.py`.
