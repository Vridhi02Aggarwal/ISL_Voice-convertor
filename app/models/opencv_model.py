


import cv2
import numpy as np

def process_frame(frame):
    """
    Detects hand region and returns cropped hand image + frame with bounding box.
    Designed for EfficientNetB0 input (224x224).
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    # Define skin color range (tuned for ISL dataset, can adjust if needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Mask skin regions
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological cleanup
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get largest contour (likely the hand)
        cnt = max(contours, key=cv2.contourArea)

        # Ignore very small contours (noise)
        if cv2.contourArea(cnt) < 5000:
            return frame, None

        # Bounding box around hand
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw rectangle on frame for visualization
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop hand region safely
        hand_img = frame[y:y + h, x:x + w]

        if hand_img.size == 0:
            return frame, None

        # Resize cropped hand to EfficientNetB0 input size
        hand_img = cv2.resize(hand_img, (224, 224))

        return frame, hand_img

    # No hand detected
    return frame, None
