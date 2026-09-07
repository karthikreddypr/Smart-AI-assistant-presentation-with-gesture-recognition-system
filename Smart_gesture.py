import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
from collections import deque

from voice_assistant import check_voice_command

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

def voice_loop():
    while True:
        check_voice_command()

voice_thread = threading.Thread(target=voice_loop, daemon=True)
voice_thread.start()

cap = cv2.VideoCapture(0)
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

PREVIEW_W = 320
PREVIEW_H = 240
WIN_NAME  = "Gesture Controller"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, PREVIEW_W, PREVIEW_H)
cv2.moveWindow(WIN_NAME, SCREEN_W - PREVIEW_W - 10, 10)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)

last_action_time = 0
delay = 1.0

SWIPE_WINDOW    = 0.4
SWIPE_THRESHOLD = 0.18
MIN_SWIPE_SPEED = 0.3
position_history = deque(maxlen=30)
swipe_label      = ""
swipe_label_time = 0

PINCH_ON_THRESH  = 0.10  
PINCH_OFF_THRESH = 0.13   
PINCH_DRAW_THRESH = 0.22 
PINCH_HOLD_REQUIRED = 0.5
pinch_start_time = None
drawing_active   = False
mouse_is_down    = False
prev_sx          = None   
prev_sy          = None   



def fingers_up(lm):
    tips = [4, 8, 12, 16, 20]
    f = [lm[tips[0]].x < lm[tips[0] - 1].x]
    for i in range(1, 5):
        f.append(lm[tips[i]].y < lm[tips[i] - 2].y)
    return f

def pinch_distance(lm):
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    return (dx*dx + dy*dy) ** 0.5

def detect_swipe(history, now):
    recent = [(t, x) for t, x in history if now - t <= SWIPE_WINDOW]
    if len(recent) < 5:
        return None
    displacement = recent[-1][1] - recent[0][1]
    elapsed      = recent[-1][0] - recent[0][0]
    if elapsed == 0:
        return None
    speed = abs(displacement) / elapsed
    if abs(displacement) >= SWIPE_THRESHOLD and speed >= MIN_SWIPE_SPEED:
        return "right" if displacement > 0 else "left"
    return None

def hand_to_screen(lm):
    sx = int(np.interp(lm[8].x, [0.05, 0.95], [0, SCREEN_W - PREVIEW_W - 20]))
    sy = int(np.interp(lm[8].y, [0.05, 0.95], [0, SCREEN_H]))
    return sx, sy

def release_mouse():
    global mouse_is_down, prev_sx, prev_sy, drawing_active, pinch_start_time
    if mouse_is_down:
        pyautogui.mouseUp(button='left')
        mouse_is_down = False
    prev_sx          = None
    prev_sy          = None
    drawing_active   = False
    pinch_start_time = None

def draw_laser_dot(img, cx, cy):
    cv2.circle(img, (cx, cy), 14, (0,   0,  60),  -1)
    cv2.circle(img, (cx, cy),  9, (0,   0, 220),  -1)
    cv2.circle(img, (cx, cy),  3, (220, 220, 255), -1)

def hud(img, text, color, y=28):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thick+1)
    cv2.putText(img, text, (x,   y),   font, scale, color,   thick)


while True:
    ret, img = cap.read()
    if not ret:
        break

    img     = cv2.flip(img, 1)
    preview = cv2.resize(img, (PREVIEW_W, PREVIEW_H))
    ph, pw  = preview.shape[:2]

    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    now     = time.time()

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(preview, hand, mp_hands.HAND_CONNECTIONS)

        lm           = hand.landmark
        finger_state = fingers_up(lm)
        p_dist       = pinch_distance(lm)

        ix = int(lm[8].x * pw)
        iy = int(lm[8].y * ph)

        position_history.append((now, lm[0].x))

        all_up  = all(finger_state)

        if mouse_is_down:
            is_pinch = p_dist < PINCH_DRAW_THRESH  
        elif drawing_active or pinch_start_time is not None:
            is_pinch = p_dist < PINCH_OFF_THRESH
        else:
            is_pinch = p_dist < PINCH_ON_THRESH

        all_down = not any(finger_state)
        sx, sy   = hand_to_screen(lm)

        cv2.putText(preview, f"p:{p_dist:.2f}",
                    (pw - 75, ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 255, 0) if is_pinch else (160, 160, 160), 1)

        if all_up:
            release_mouse()
            pyautogui.moveTo(sx, sy, duration=0)
            draw_laser_dot(preview, ix, iy)
            hud(preview, "LASER", (0, 0, 255))

        elif is_pinch:
            if pinch_start_time is None:
                pinch_start_time = now

            hold_elapsed = now - pinch_start_time

            if not drawing_active:
              
                angle = int(360 * hold_elapsed / PINCH_HOLD_REQUIRED)
                cv2.ellipse(preview, (ix, iy), (16, 16), -90, 0, angle,
                            (0, 255, 255), 2)
                hud(preview, "HOLD...", (0, 255, 255))
                
                pyautogui.moveTo(sx, sy, duration=0)
                prev_sx = sx
                prev_sy = sy

                if hold_elapsed >= PINCH_HOLD_REQUIRED:
                    drawing_active = True
                    pyautogui.mouseDown(button='left')
                    mouse_is_down  = True

            else:
                pyautogui.moveTo(sx, sy, duration=0)

                cv2.circle(preview, (ix, iy), 10, (0, 0, 255), -1)
                cv2.circle(preview, (ix, iy),  4, (255, 255, 255), -1)
                hud(preview, "DRAWING", (0, 80, 255))

        else:
            if not mouse_is_down: 
                release_mouse()

            swipe = detect_swipe(position_history, now)

            if swipe and now - last_action_time > delay:
                if swipe == "right":
                    pyautogui.press("right")
                    swipe_label      = ">> NEXT"
                    swipe_label_time = now + 1.0
                else:
                    pyautogui.press("left")
                    swipe_label      = "<< BACK"
                    swipe_label_time = now + 1.0
                last_action_time = now
                position_history.clear()

            elif finger_state == [False, True, False, False, False]:
                hud(preview, "NEXT ->", (0, 255, 0))
                if now - last_action_time > delay:
                    pyautogui.press("right")
                    last_action_time = now

            elif finger_state == [True, False, False, False, False]:
                hud(preview, "<- PREV", (0, 255, 255))
                if now - last_action_time > delay:
                    pyautogui.press("left")
                    last_action_time = now

    else:
        release_mouse()
        position_history.clear()

    if now < swipe_label_time:
        hud(preview, swipe_label, (0, 200, 255))

    mini = ["OpenHand=Laser", "Pinch=Draw on PPT",
            "Index=Next", "Thumb=Prev"]
    for i, line in enumerate(mini):
        cv2.putText(preview, line,
                    (4, ph - len(mini)*13 + i*13 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

    cv2.imshow(WIN_NAME, preview)
    if cv2.waitKey(1) & 0xFF == 27:
        release_mouse()
        break

cap.release()
cv2.destroyAllWindows()