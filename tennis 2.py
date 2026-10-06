import cv2
import mediapipe as mp
import numpy as np
import random

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- Game Variables ---
WINNING_SCORE = 12  # Score limit to win the game

# Ball properties
ball_pos = np.array([480.0, 50.0])  # Start on right side
ball_vel = np.array([-random.uniform(1, 3), 5.0]) # Start moving left
ball_radius = 20
gravity = 0.8
bounce_speed = 20 # Speed on bounce

# Paddle properties
paddle_R_p1 = np.array([0, 0])
paddle_R_p2 = np.array([1, 1])
paddle_L_p1 = np.array([0, 0])
paddle_L_p2 = np.array([1, 1])
hand_R_detected = False
hand_L_detected = False

# Game state
score_L = 0 # Left Player (P1) score
score_R = 0 # Right Player (P2) score
game_over = False
winner_text = ""
serve_side = 1 # -1 for left, 1 for right. Start on right.

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Webcam opened. First to 12 points wins!")
print("P1 (Blue) on Left, P2 (Green) on Right.")
print("Press 'r' to reset at any time.")
print("Press 'q' to quit.")

# --- Helper Function for Collision ---
def check_paddle_collision(ball_pos, ball_vel, ball_radius, p1, p2, bounce_speed):
    line_vec = p2 - p1
    point_vec = ball_pos - p1
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return None, None

    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)
    
    closest_point = p1 + t * line_vec
    distance = np.linalg.norm(ball_pos - closest_point)
    
    if distance < ball_radius:
        normal_vec = line_vec[1], -line_vec[0]
        normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-6)
        
        if normal[1] > 0:
            normal = -normal
            
        new_vel = normal * bounce_speed
        new_pos = closest_point + normal * (ball_radius + 1)
        return new_pos, new_vel
    
    return None, None

# --- Main Program ---
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, _ = image.shape
        image = cv2.flip(image, 1) # Flip for selfie-view
        
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- Hand Interaction Logic ---
        hand_R_detected = False
        hand_L_detected = False
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                
                index_mcp_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_mcp_lm = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                
                p1_knuckle = np.array([int(index_mcp_lm.x * w), int(index_mcp_lm.y * h)])
                p2_knuckle = np.array([int(pinky_mcp_lm.x * w), int(pinky_mcp_lm.y * h)])
                
                v = p2_knuckle - p1_knuckle
                v_norm = v / (np.linalg.norm(v) + 1e-6) 
                length = np.linalg.norm(v)
                p1 = p1_knuckle - v_norm * (length / 2)
                p2 = p2_knuckle + v_norm * (length / 2)

                if handedness == "Right" and (index_mcp_lm.x * w) > (w // 2):
                    paddle_R_p1, paddle_R_p2 = p1, p2
                    hand_R_detected = True
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 8)
                    
                elif handedness == "Left" and (index_mcp_lm.x * w) < (w // 2):
                    paddle_L_p1, paddle_L_p2 = p1, p2
                    hand_L_detected = True
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 8)

        # Check if any hand is present
        any_hand_detected = hand_R_detected or hand_L_detected

        # --- Game Logic (Only run if active and hands detected) ---
        if not game_over and any_hand_detected:
            ball_vel[1] += gravity
            ball_pos += ball_vel

            # --- Wall Bouncing ---
            if ball_pos[0] + ball_radius > w or ball_pos[0] - ball_radius < 0:
                ball_pos[0] = np.clip(ball_pos[0], ball_radius, w - ball_radius)
                ball_vel[0] *= -1
            if ball_pos[1] - ball_radius < 0:
                ball_pos[1] = ball_radius
                ball_vel[1] *= -1

            # --- Solid Net Collision ---
            net_rigid_y_start = h // 2
            if ball_pos[1] + ball_radius > net_rigid_y_start:
                dist_to_net = ball_pos[0] - (w // 2)
                if abs(dist_to_net) < ball_radius:
                    if dist_to_net > 0 and ball_vel[0] < 0: 
                        ball_pos[0] = (w // 2) + ball_radius
                        ball_vel[0] *= -1
                    elif dist_to_net < 0 and ball_vel[0] > 0: 
                        ball_pos[0] = (w // 2) - ball_radius
                        ball_vel[0] *= -1

            # --- Scoring Logic ---
            if ball_pos[1] - ball_radius > h:
                if ball_pos[0] < w // 2:
                    score_R += 1 # Fell on left, P2 scores
                else:
                    score_L += 1 # Fell on right, P1 scores
                
                # Check for winner
                if score_L >= WINNING_SCORE:
                    game_over = True
                    winner_text = "P1 (BLUE) WINS!"
                elif score_R >= WINNING_SCORE:
                    game_over = True
                    winner_text = "P2 (GREEN) WINS!"
                else:
                    # Alternate serve side
                    serve_side *= -1
                    start_x = (w // 2) + (w // 4) * serve_side 
                    vel_x = (serve_side * -1) * random.uniform(1, 3) 
                    
                    ball_pos = np.array([float(start_x), 50.0])
                    ball_vel = np.array([vel_x, 5.0])
                
            # --- Paddle Collision Check ---
            new_pos, new_vel = (None, None)
            if hand_R_detected:
                new_pos, new_vel = check_paddle_collision(ball_pos, ball_vel, ball_radius, paddle_R_p1, paddle_R_p2, bounce_speed)
            if new_pos is None and hand_L_detected:
                new_pos, new_vel = check_paddle_collision(ball_pos, ball_vel, ball_radius, paddle_L_p1, paddle_L_p2, bounce_speed)

            if new_pos is not None:
                ball_pos = new_pos
                ball_vel = new_vel
        
        # --- Draw Game Elements ---
        # Top half: light guide line; Bottom half: solid thick net
        cv2.line(image, (w // 2, 0), (w // 2, h // 2), (150, 150, 150), 2)
        cv2.line(image, (w // 2, h // 2), (w // 2, h), (255, 255, 255), 6)
        
        # Draw Ball
        cv2.circle(image, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, (0, 0, 255), -1)
        
        # Draw Scores
        cv2.putText(image, f'P1: {score_L}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, f'P2: {score_R}', (w - 180, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        # --- Draw Pause Banner if No Hands Detected ---
        if not any_hand_detected and not game_over:
            cv2.rectangle(image, (w // 8, h // 2 - 40), (7 * w // 8, h // 2 + 40), (0, 0, 0), -1)
            cv2.rectangle(image, (w // 8, h // 2 - 40), (7 * w // 8, h // 2 + 40), (0, 0, 255), 2)
            cv2.putText(image, "HAND NOT DETECTED - PAUSED", (w // 8 + 15, h // 2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # --- Draw Game Over Screen ---
        if game_over:
            cv2.rectangle(image, (w // 6, h // 3), (5 * w // 6, 2 * h // 3), (0, 0, 0), -1)
            cv2.rectangle(image, (w // 6, h // 3), (5 * w // 6, 2 * h // 3), (0, 255, 255), 3)
            
            cv2.putText(image, winner_text, (w // 4, h // 2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, "Press 'r' to Restart", (w // 4 + 10, h // 2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Display ---
        cv2.imshow('Hand Paddle Game', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        
        # --- Restart Game ---
        if key == ord('r'):
            score_L = 0
            score_R = 0
            game_over = False
            winner_text = ""
            serve_side = 1 # Reset serve to the right
            start_x = (w // 2) + (w // 4) * serve_side 
            vel_x = (serve_side * -1) * random.uniform(1, 3) 
            
            ball_pos = np.array([float(start_x), 50.0])
            ball_vel = np.array([vel_x, 5.0])
        
cap.release()
cv2.destroyAllWindows()
