import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize OpenCV
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Get frame dimensions 
success, temp_frame = cap.read()
if not success:
    print("Error: Cannot read a frame from the camera")
    cap.release()
    exit()
    
height, width, _ = temp_frame.shape
print(f"Camera feed initialized: {width}x{height}")

# --- Ball Physics  ---
ball_x = width // 2
ball_y = height // 4
ball_vx = 0
ball_vy = 0
ball_radius = 15
gravity = 0.8        
damping = 1       
finger_thickness = 10 

# --- collision  ---
def line_circle_collision(cx, cy, cr, x1, y1, x2, y2, damping_factor, current_vx, current_vy):
    line_vec_x = x2 - x1
    line_vec_y = y2 - y1

    circle_to_line_start_x = cx - x1
    circle_to_line_start_y = cy - y1

    dot_product = circle_to_line_start_x * line_vec_x + circle_to_line_start_y * line_vec_y
    len_sq = line_vec_x * line_vec_x + line_vec_y * line_vec_y
    
    t = dot_product / len_sq if len_sq != 0 else 0

    t = max(0, min(1, t))

    closest_x = x1 + t * line_vec_x
    closest_y = y1 + t * line_vec_y

    dist_x = cx - closest_x
    dist_y = cy - closest_y
    distance = math.hypot(dist_x, dist_y)

    # Check if a collision occurred (distance < radius)
    if distance < cr + finger_thickness / 2: 
        
        # Calculate penetration depth
        penetration_depth = (cr + finger_thickness / 2) - distance

        if distance == 0: 
            return cx, cy, current_vx, current_vy, True 

        # Normalize the collision normal vector (from closest point to circle center)
        normal_x = dist_x / distance
        normal_y = dist_y / distance

        # Move the ball out of collision by the penetration depth along the normal
        new_cx = cx + normal_x * penetration_depth
        new_cy = cy + normal_y * penetration_depth

        # Calculate the projection of the ball's current velocity onto the normal
        dot_product_vel_normal = current_vx * normal_x + current_vy * normal_y

        if dot_product_vel_normal < 0:
            # Calculate the reflected velocity
            reflected_vx = current_vx - 2 * dot_product_vel_normal * normal_x
            reflected_vy = current_vy - 2 * dot_product_vel_normal * normal_y

            # Apply damping
            new_vx = reflected_vx * damping_factor
            new_vy = reflected_vy * damping_factor
        else:
            new_vx = current_vx
            new_vy = current_vy

        return new_cx, new_cy, new_vx, new_vy, True
    
    return cx, cy, current_vx, current_vy, False # No collision

# --- Main Application Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # --- Physics---
    ball_vy += gravity
    ball_x += ball_vx
    ball_y += ball_vy

    # --- Wall Collision---
    
    # Floor collision
    if ball_y + ball_radius > height:
        ball_y = height - ball_radius
        ball_vy = -ball_vy * damping
        ball_vx *= damping # Applied friction
        
    # Ceiling collision
    if ball_y - ball_radius < 0:
        ball_y = ball_radius
        ball_vy = -ball_vy * damping
        
    # Right wall collision
    if ball_x + ball_radius > width:
        ball_x = width - ball_radius
        ball_vx = -ball_vx * damping
        
    # Left wall collision
    if ball_x - ball_radius < 0:
        ball_x = ball_radius
        ball_vx = -ball_vx * damping


    # --- Hand Tracking and Fingers Collision ---
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    finger_collided = False 

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # hand landmarks
        # Fingertips
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        
        #landmark to pixel point
        def to_pixel(landmark):
            return (int(landmark.x * width), int(landmark.y * height))

        # segments
        all_hand_segments = []

        # Thumb segments
        all_hand_segments.append((to_pixel(thumb_mcp), to_pixel(thumb_ip)))
        all_hand_segments.append((to_pixel(thumb_ip), to_pixel(thumb_tip)))

        # Index finger segments
        all_hand_segments.append((to_pixel(index_mcp), to_pixel(index_pip)))
        all_hand_segments.append((to_pixel(index_pip), to_pixel(index_dip)))
        all_hand_segments.append((to_pixel(index_dip), to_pixel(index_tip)))

        # Middle finger segments
        all_hand_segments.append((to_pixel(middle_mcp), to_pixel(middle_pip)))
        all_hand_segments.append((to_pixel(middle_pip), to_pixel(middle_dip)))
        all_hand_segments.append((to_pixel(middle_dip), to_pixel(middle_tip)))

        # Ring finger segments
        all_hand_segments.append((to_pixel(ring_mcp), to_pixel(ring_pip)))
        all_hand_segments.append((to_pixel(ring_pip), to_pixel(ring_dip)))
        all_hand_segments.append((to_pixel(ring_dip), to_pixel(ring_tip)))

        # Pinky finger segments
        all_hand_segments.append((to_pixel(pinky_mcp), to_pixel(pinky_pip)))
        all_hand_segments.append((to_pixel(pinky_pip), to_pixel(pinky_dip)))
        all_hand_segments.append((to_pixel(pinky_dip), to_pixel(pinky_tip)))

        # ---Palm Segments---
        all_hand_segments.append((to_pixel(index_mcp), to_pixel(middle_mcp)))
        all_hand_segments.append((to_pixel(middle_mcp), to_pixel(ring_mcp)))
        all_hand_segments.append((to_pixel(ring_mcp), to_pixel(pinky_mcp)))
        
        # Connect 
        all_hand_segments.append((to_pixel(pinky_mcp), to_pixel(wrist)))
        
        # Connect wrist to thumb 
        all_hand_segments.append((to_pixel(wrist), to_pixel(thumb_mcp)))


        for p1, p2 in all_hand_segments:
            # Draw the hand segment as a line
            cv2.line(image, p1, p2, (0, 255, 0), finger_thickness) # Green line
            
            # Check for collision with this segment
            ball_x, ball_y, ball_vx, ball_vy, hit = line_circle_collision(
                ball_x, ball_y, ball_radius, p1[0], p1[1], p2[0], p2[1], damping, ball_vx, ball_vy
            )
            if hit:
                finger_collided = True

    # ---  Drawing ---
    
    #ball
    ball_color = (0, 0, 255) 
    if finger_collided:
        ball_color = (0, 255, 255) # touching 

    cv2.circle(image, (int(ball_x), int(ball_y)), ball_radius, ball_color, cv2.FILLED)

    # ---final image ---
    cv2.imshow('Hand Physics Ball - Press ESC to quit', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# --- Cleanups---
print("Exiting...")
hands.close()
cap.release()
cv2.destroyAllWindows()

