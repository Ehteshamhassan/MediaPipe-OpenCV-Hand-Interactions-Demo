import cv2
import mediapipe as mp
import numpy as np
import random

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Create Bricks (Added 1 Extra Column to the Right) ---
def create_bricks():
    bricks = []
    rows, cols = 4, 6  # Expanded to 6 columns
    brick_width, brick_height = 45, 18
    start_x, start_y = 132, 40  
    
    for r in range(rows):
        for c in range(cols):
            x1 = start_x + c * (brick_width + 6)
            y1 = start_y + r * (brick_height + 6)
            x2 = x1 + brick_width
            y2 = y1 + brick_height
            
            rand_val = random.random()
            if rand_val < 0.25:
                # Hard Brick (3 Hits - Gray)
                b_type = "hard"
                color = (120, 120, 120)
                hp = 3
            elif rand_val < 0.60:
                # Normal Brick (2 Hits - Orange)
                b_type = "normal"
                color = (0, 165, 255)
                hp = 2
            else:
                # Easy Brick (1 Hit - Bright Green)
                b_type = "easy"
                color = (0, 255, 0)
                hp = 1
                
            bricks.append({
                "rect": [x1, y1, x2, y2],
                "type": b_type,
                "color": color,
                "hp": hp,
                "max_hp": hp
            })
    return bricks

# Game State
bricks = create_bricks()
lives = 3
game_over = False
win_state = False

# Balls (Supports Multi-Ball)
balls = [{"pos": np.array([260.0, 300.0]), "vel": np.array([3.0, -5.0])}]
ball_radius = 12
gravity = 0.3
bounce_speed = 16

# Paddle State
paddle_x = 320
paddle_width = 110  # Dynamic width
paddle_y_fixed = 420

# Laser & Item States
powerups = [] # Falling power items
lasers = []   # Shots fired
laser_timer = 0

# Track Hand Presence
hand_detected = False

# --- Helper Functions ---
def check_brick_collision(ball_pos, ball_radius, brick_rect):
    x1, y1, x2, y2 = brick_rect
    closest_x = np.clip(ball_pos[0], x1, x2)
    closest_y = np.clip(ball_pos[1], y1, y2)
    dist = np.linalg.norm(ball_pos - np.array([closest_x, closest_y]))
    return dist < ball_radius

def spawn_power_item(x, y):
    """Spawns both beneficial power-ups and risky power-downs with unique colors."""
    items = [
        # Power-Ups
        {"type": "2X", "color": (255, 0, 255)},        # Magenta
        {"type": "LASER", "color": (0, 255, 255)},     # Cyan
        {"type": "GUNPOWDER", "color": (0, 140, 255)}, # Dark Orange
        {"type": "P+", "color": (0, 255, 0)},         # Bright Green
        # Power-Downs
        {"type": "P-", "color": (0, 0, 255)},         # Pure Red
        {"type": "S+", "color": (0, 100, 255)},       # Light Red/Orange (Fast Ball)
        {"type": "S-", "color": (255, 191, 0)}        # Deep Sky Blue (Slow Ball)
    ]
    chosen = random.choice(items)
    return {"pos": np.array([float(x), float(y)]), "type": chosen["type"], "color": chosen["color"]}

def trigger_gunpowder(bricks_list):
    """Destroys a cluster of random bricks on gunpowder catch."""
    to_destroy = min(5, len(bricks_list))
    for _ in range(to_destroy):
        if bricks_list:
            bricks_list.pop(random.randint(0, len(bricks_list) - 1))

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, _ = image.shape
        image = cv2.flip(image, 1)
        paddle_y_fixed = h - 40

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- Track Hand X Position ---
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            target_x = int(wrist.x * w)
            paddle_x = int(paddle_x + (target_x - paddle_x) * 0.4)
        else:
            hand_detected = False

        # Keep paddle within screen boundaries
        paddle_x = np.clip(paddle_x, paddle_width // 2, w - paddle_width // 2)
        paddle_p1 = (paddle_x - paddle_width // 2, paddle_y_fixed)
        paddle_p2 = (paddle_x + paddle_width // 2, paddle_y_fixed)

        # --- Game Logic ---
        if not game_over and not win_state:
            # --- Update Balls (Backwards Loop) ---
            for i in range(len(balls) - 1, -1, -1):
                ball = balls[i]
                ball["vel"][1] += gravity
                ball["pos"] += ball["vel"]

                # Wall Collisions
                if ball["pos"][0] + ball_radius > w or ball["pos"][0] - ball_radius < 0:
                    ball["pos"][0] = np.clip(ball["pos"][0], ball_radius, w - ball_radius)
                    ball["vel"][0] *= -1
                if ball["pos"][1] - ball_radius < 0:
                    ball["pos"][1] = ball_radius
                    ball["vel"][1] *= -1

                # Bottom Out
                if ball["pos"][1] - ball_radius > h:
                    balls.pop(i)
                    continue

                # Paddle Collision
                if (paddle_p1[0] <= ball["pos"][0] <= paddle_p2[0] and 
                    abs(ball["pos"][1] - paddle_y_fixed) < ball_radius and ball["vel"][1] > 0):
                    
                    offset = (ball["pos"][0] - paddle_x) / (paddle_width / 2)
                    ball["vel"][0] = offset * 6.0
                    ball["vel"][1] = -bounce_speed
                    ball["pos"][1] = paddle_y_fixed - ball_radius

                # Brick Collision
                for brick in bricks[:]:
                    if check_brick_collision(ball["pos"], ball_radius, brick["rect"]):
                        ball["vel"][1] *= -1
                        brick["hp"] -= 1
                        
                        if brick["hp"] <= 0:
                            bricks.remove(brick)
                            # Power Item Drop Rate: 1 out of 7 (~14.2%)
                            if random.random() < (1.0 / 7.0):
                                cx = (brick["rect"][0] + brick["rect"][2]) // 2
                                powerups.append(spawn_power_item(cx, brick["rect"][3]))
                        break

            # --- Check Ball Loss / Deduct Life ---
            if len(balls) == 0:
                lives -= 1
                if lives <= 0:
                    game_over = True
                else:
                    # Respawn 1 ball
                    balls.append({"pos": np.array([float(paddle_x), float(paddle_y_fixed - 30)]), "vel": np.array([2.0, -8.0])})

            # --- Update Power-Ups & Power-Downs ---
            for i in range(len(powerups) - 1, -1, -1):
                p = powerups[i]
                p["pos"][1] += 3.5  # Fall speed
                
                # Catch Item with Paddle
                if (paddle_p1[0] <= p["pos"][0] <= paddle_p2[0] and 
                    abs(p["pos"][1] - paddle_y_fixed) < 15):
                    
                    p_type = p["type"]
                    if p_type == "2X":
                        new_balls = []
                        for b in balls:
                            new_balls.append({"pos": b["pos"].copy(), "vel": np.array([-b["vel"][0], b["vel"][1]])})
                        balls.extend(new_balls)
                    elif p_type == "LASER":
                        laser_timer = 120
                    elif p_type == "GUNPOWDER":
                        trigger_gunpowder(bricks)
                    elif p_type == "P+":
                        paddle_width = min(220, paddle_width + 40)  # Expand paddle
                    elif p_type == "P-":
                        paddle_width = max(60, paddle_width - 30)   # Shrink paddle
                    elif p_type == "S+":
                        for b in balls:
                            b["vel"] *= 1.25                        # Speed up ball
                    elif p_type == "S-":
                        for b in balls:
                            b["vel"] *= 0.75                        # Slow down ball
                        
                    powerups.pop(i)
                elif p["pos"][1] > h:
                    powerups.pop(i)

            # --- Handle Lasers ---
            if laser_timer > 0:
                laser_timer -= 1
                if laser_timer % 15 == 0:
                    lasers.append(np.array([float(paddle_p1[0] + 10), float(paddle_y_fixed - 10)]))
                    lasers.append(np.array([float(paddle_p2[0] - 10), float(paddle_y_fixed - 10)]))

            for i in range(len(lasers) - 1, -1, -1):
                l = lasers[i]
                l[1] -= 10
                cv2.line(image, (int(l[0]), int(l[1])), (int(l[0]), int(l[1] - 12)), (0, 255, 255), 3)
                
                hit = False
                for brick in bricks[:]:
                    if (brick["rect"][0] <= l[0] <= brick["rect"][2] and 
                        brick["rect"][1] <= l[1] <= brick["rect"][3]):
                        brick["hp"] -= 1
                        if brick["hp"] <= 0:
                            bricks.remove(brick)
                        hit = True
                        break
                if hit or l[1] < 0:
                    lasers.pop(i)

            # Win Condition
            if len(bricks) == 0:
                win_state = True

        # --- Visual Rendering ---
        # Draw Bricks
        for brick in bricks:
            x1, y1, x2, y2 = brick["rect"]
            base_col = list(brick["color"])
            hp_factor = brick["hp"] / brick["max_hp"]
            display_color = (int(base_col[0] * hp_factor), int(base_col[1] * hp_factor), int(base_col[2] * hp_factor))
            
            cv2.rectangle(image, (x1, y1), (x2, y2), display_color, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Draw Paddle with Dynamic Color
        if laser_timer > 0:
            paddle_color = (0, 255, 255)  # Cyan when Laser active
        elif hand_detected:
            paddle_color = (0, 255, 0)    # Green when Hand detected
        else:
            paddle_color = (0, 0, 255)    # Red when No Hand detected

        cv2.line(image, paddle_p1, paddle_p2, paddle_color, 10)

        # Draw Balls
        for ball in balls:
            cv2.circle(image, (int(ball["pos"][0]), int(ball["pos"][1])), ball_radius, (0, 0, 255), -1)

        # Draw Falling Power-Ups & Power-Downs
        for p in powerups:
            px, py = int(p["pos"][0]), int(p["pos"][1])
            cv2.circle(image, (px, py), 13, p["color"], -1)
            cv2.circle(image, (px, py), 13, (255, 255, 255), 1)  # White border
            cv2.putText(image, p["type"], (px - 10, py + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # --- Draw HUD ---
        heart_symbol = "<3 "
        hearts_text = "LIVES: " + (heart_symbol * max(0, lives))
        cv2.putText(image, hearts_text, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if laser_timer > 0:
            cv2.putText(image, "LASER ACTIVE!", (w - 180, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # End Game Overlays
        if game_over:
            cv2.rectangle(image, (w // 6, h // 3), (5 * w // 6, 2 * h // 3), (0, 0, 0), -1)
            cv2.rectangle(image, (w // 6, h // 3), (5 * w // 6, 2 * h // 3), (0, 0, 255), 3)
            cv2.putText(image, "GAME OVER", (w // 4 + 30, h // 2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(image, "Press 'r' to Restart", (w // 4 + 15, h // 2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif win_state:
            cv2.rectangle(image, (w // 6, h // 3), (5 * w // 6, 2 * h // 3), (0, 0, 0), -1)
            cv2.rectangle(image, (w // 6, h // 3), (5 * w // 6, 2 * h // 3), (0, 255, 255), 3)
            cv2.putText(image, "STAGE CLEARED!", (w // 4 + 10, h // 2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(image, "Press 'r' to Restart", (w // 4 + 15, h // 2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Target Practice', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            bricks = create_bricks()
            lives = 3
            paddle_width = 110
            game_over = False
            win_state = False
            balls = [{"pos": np.array([260.0, 300.0]), "vel": np.array([3.0, -5.0])}]
            powerups.clear()
            lasers.clear()

cap.release()
cv2.destroyAllWindows()
