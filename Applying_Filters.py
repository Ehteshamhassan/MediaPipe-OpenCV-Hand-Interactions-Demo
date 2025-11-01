import cv2
import mediapipe as mp
import math
import numpy as np

# --- Initialize ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # For drawing styles

# --- Function to rotate a point around a center ---
def rotate_point(point, center, angle_rad):

    cx, cy = center
    x, y = point
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_new = cx + (x - cx) * cos_a - (y - cy) * sin_a
    y_new = cy + (x - cx) * sin_a + (y - cy) * cos_a
    return int(x_new), int(y_new)

cap = cv2.VideoCapture(0)
def main():
    # --- MediaPipe Hands and Pose ---
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands, \
        mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            # --Read Frame ---
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # ---Flip and Convert to RGB ---
            frame = cv2.flip(frame, 1) # Flip for selfie view
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -- Pose ---
            image_rgb.flags.writeable = False
            pose_results = pose.process(image_rgb)

            # --- Hands ---
            hands_results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            # -- Frame for Drawing ---
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_h, image_w, _ = image_bgr.shape

            # ---  Create blank layers ---
            square_mask = np.zeros((image_h, image_w), dtype=np.uint8)
            skeletal_overlay = np.zeros_like(image_bgr)
            pts = None 

            # --- Logic for Hand Control (1 or 2 Hands) ---
            if hands_results.multi_hand_landmarks:
                num_hands = len(hands_results.multi_hand_landmarks)

                if num_hands == 2:
                    # 2 HAND LOGIC
                    hand_landmarks_0 = hands_results.multi_hand_landmarks[0]
                    hand_landmarks_1 = hands_results.multi_hand_landmarks[1]
                    index_tip_0 = hand_landmarks_0.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_tip_1 = hand_landmarks_1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_0 = int(index_tip_0.x * image_w); y_0 = int(index_tip_0.y * image_h)
                    x_1 = int(index_tip_1.x * image_w); y_1 = int(index_tip_1.y * image_h)
                    mid_x = int((x_0 + x_1) / 2); mid_y = int((y_0 + y_1) / 2)
                    distance = math.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
                    angle_rad = math.atan2(y_1 - y_0, x_1 - x_0)
                    side_length = int(distance * 1.5)

                elif num_hands == 1:
                    # ONE HAND LOGIC
                    hand_landmarks = hands_results.multi_hand_landmarks[0]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x_0 = int(thumb_tip.x * image_w); y_0 = int(thumb_tip.y * image_h)
                    x_1 = int(index_tip.x * image_w); y_1 = int(index_tip.y * image_h)
                    mid_x = int((x_0 + x_1) / 2); mid_y = int((y_0 + y_1) / 2)
                    distance = math.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
                    angle_rad = math.atan2(y_1 - y_0, x_1 - x_0)
                    side_length = int(distance)

                # --- Common Calculation for Hands ---
                midpoint = (mid_x, mid_y)
                half_side = side_length // 2
                min_square_size = 50
                if side_length < min_square_size:
                    side_length = min_square_size
                    half_side = min_square_size // 2

                corners_unrotated = [
                    (mid_x - half_side, mid_y - half_side),
                    (mid_x + half_side, mid_y - half_side),
                    (mid_x + half_side, mid_y + half_side),
                    (mid_x - half_side, mid_y + half_side)
                ]
                rotated_corners = [rotate_point(p, midpoint, angle_rad) for p in corners_unrotated]

                # Draw the square 
                pts = np.array(rotated_corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(square_mask, [pts], 255) # Fill mask 

            #negative image
            inverted_image = cv2.bitwise_not(image_bgr)

            # squared negative
            negative_inside_square = cv2.bitwise_and(inverted_image, inverted_image, mask=square_mask)


            skeleton_inside_square = cv2.bitwise_and(skeletal_overlay, skeletal_overlay, mask=square_mask)

  
            combined_filter_inside_square = cv2.add(negative_inside_square, skeleton_inside_square)

            # outside square
            background_outside_square = cv2.bitwise_and(image_bgr, image_bgr, mask=cv2.bitwise_not(square_mask))

            #  Combine
            image_bgr = cv2.add(background_outside_square, combined_filter_inside_square)

            # Square Outline ---
            if pts is not None:
                cv2.polylines(image_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)



            # --- Display the frame ---
            cv2.imshow('Adaptive Skeletal+Negative Filter', image_bgr)

            # ---  Quit Condition ---
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # --- Release Resources ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
