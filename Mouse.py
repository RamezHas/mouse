import cv2
import HTModule as htm
import pyautogui
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Initialize hand detector
detector = htm.handDetector(maxHands=1, detectionCon=0.8)

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Parameters
click_dist = 15  # Distance between fingers to trigger click
smooth_factor = 7  # Smoothing factor for cursor movement
prev_x, prev_y = 0, 0  # Previous cursor positions

while True:
    # Read frame
    success, img = cap.read()

    # Flip image horizontally and detect hands
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=True)
    lmList = detector.FindPosition(img, draw=False)

    if len(lmList) != 0:
        # Get landmarks for index (8) and thumb (4) fingertips
        x1, y1 = lmList[8][1], lmList[8][2]  # Index
        x2, y2 = lmList[4][1], lmList[4][2]  # Thumb

        # Draw circles
        cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)  # Index
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)  # Thumb
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Midpoint
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

        # Calculate distance between fingers
        distance = np.hypot(x2 - x1, y2 - y1)

        # Convert hand coordinates to screen coordinates
        mouse_x = np.interp(x1, [100, 540], [0, screen_w])
        mouse_y = np.interp(y1, [50, 380], [0, screen_h])

        # Smooth movement
        curr_x = prev_x + (mouse_x - prev_x) / smooth_factor
        curr_y = prev_y + (mouse_y - prev_y) / smooth_factor

        # Move mouse
        try:
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
        except Exception as e:
            print("Mouse control error:", e)

        # Click if fingers are close
        if distance < click_dist:
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
            try:
                pyautogui.click()
                pyautogui.sleep(0.2)
            except Exception as e:
                print("Click error:", e)

        # Display distance
        cv2.putText(img, f"Distance: {int(distance)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Virtual Mouse", img)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

