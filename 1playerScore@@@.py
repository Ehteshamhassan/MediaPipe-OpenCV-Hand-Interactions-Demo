import cv2
import mediapipe as mp
import numpy as np
import random

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- Game Variables ---
ball_pos = np.array([480.0, 50.0])  
ball_vel = np.array([-random.uniform(1, 3), 5.0]) 
ball_radius = 20
gravity = 0.8
bounce_speed = 20 

# --- Paddle properties for BOTH hands (using Fists) ---
paddle_R_p1 = np.array([0, 0])
paddle_R_p2 = np.array([1, 1])
paddle_L_p1 = np.array([0, 0])
paddle_L_p2 = np.array([1, 1])
hand_R_detected = False
hand_L_detected = False

# Game state
score_L = 0 # Left Player (P1) score
score_R = 0 # Right Player (P2) score
serve_side = 1                            #Start on right.

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()
print("Webcam opened. P1 (Blue) on Left, P2 (Green) on Right.")
print("Press 'r' at any time to reset the score to 0-0.")
print("Press 'q' to quit.")

# --for Collision ---
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
        # --- COLLISION! ---
        normal_vec = line_vec[1], -line_vec[0]
        normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-6)
        
        if normal[1] > 0:
            normal = -normal
            
        new_vel = normal * bounce_speed
        new_pos = closest_point + normal * (ball_radius + 1)
        
        return new_pos, new_vel
    
    return None, None # No collision

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
            print("Ignoring empty camera frame.")
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
                
                # --- landmarks for the fist ---
                index_mcp_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                pinky_mcp_lm = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                
                # --- detect hand ---
                p1_knuckle = np.array([int(index_mcp_lm.x * w), int(index_mcp_lm.y * h)])
                p2_knuckle = np.array([int(pinky_mcp_lm.x * w), int(pinky_mcp_lm.y * h)])
                
                v = p2_knuckle - p1_knuckle
                v_norm = v / (np.linalg.norm(v) + 1e-6) 
                length = np.linalg.norm(v)
                p1 = p1_knuckle - v_norm * (length / 2)
                p2 = p2_knuckle + v_norm * (length / 2)

                if handedness == "Right" and (index_mcp_lm.x * w) > (w // 2):
                    paddle_R_p1 = p1
                    paddle_R_p2 = p2
                    hand_R_detected = True
                    paddle_color = (0, 255, 0) # Green
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), paddle_color, 8)
                    
                elif handedness == "Left" and (index_mcp_lm.x * w) < (w // 2):
                    paddle_L_p1 = p1
                    paddle_L_p2 = p2
                    hand_L_detected = True
                    paddle_color = (255, 0, 0) # Blue
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), paddle_color, 8)

    
        # --- Game Logic ---
        ball_vel[1] += gravity
        ball_pos += ball_vel

        # --- Wall Bouncing ---
        if ball_pos[0] + ball_radius > w or ball_pos[0] - ball_radius < 0:
            ball_pos[0] = np.clip(ball_pos[0], ball_radius, w - ball_radius)
            ball_vel[0] *= -1
        if ball_pos[1] - ball_radius < 0:
            ball_pos[1] = ball_radius
            ball_vel[1] *= -1

        # --- Rigid Net Collision Logic ---
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
                score_R += 1 # Fell on left, P2 (Right) scores
            else:
                score_L += 1 # Fell on right, P1 (Left) scores
            
            serve_side *= -1 # Flip side
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
        
        # split net
        cv2.line(image, (w // 2, 0), (w // 2, h // 2), (150, 150, 150), 2)
        cv2.line(image, (w // 2, h // 2), (w // 2, h), (255, 255, 255), 4)
        
        # Ball
        cv2.circle(image, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, (0, 0, 255), -1)
        
        # scores
        cv2.putText(image, f'P1: {score_L}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, f'P2: {score_R}', (w - 180, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)



        cv2.imshow('Hand Paddle Game', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        
        # ---restart ---
        if key == ord('r'):
            score_L = 0
            score_R = 0
            serve_side = 1 
            start_x = (w // 2) + (w // 4) * serve_side 
            vel_x = (serve_side * -1) * random.uniform(1, 3) 
            
            ball_pos = np.array([float(start_x), 50.0])
            ball_vel = np.array([vel_x, 5.0])
        
cap.release()
cv2.destroyAllWindows()
