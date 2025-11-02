# MediaPipe-OpenCV-Hand-Interactions-Demo
A Python CV project using OpenCV &amp; MediaPipe for real-time Human-Computer Interaction (HCI). Developed a hand-controlled game (e.g., bouncing ball) that maps the 21-point hand model to game controls. Integrated dynamic visual filters that react to hand gestures. Demonstrates expertise in real-time tracking and AR-like overlays.

........................1.Hand-Tracking Physics Ball

This is an interactive physics simulation built with Python, OpenCV, and MediaPipe. It uses your webcam to track your hand in real-time, allowing you to "juggle" and interact with a virtual ball that obeys gravity.

Features

Real-Time Hand Tracking:   Uses Google's MediaPipe to detect and track a 21-point hand skeleton.
Simple Physics Engine:     The ball is affected by gravity, momentum, and damping (friction).
Boundary Collisions:       The ball realistically bounces off all four edges of the "world" (the camera feed).
Advanced Hand Collision:   The entire hand (all fingers and palm) is modeled as a set of line segments.

A custom line-circle collision function detects when the ball touches any part of the hand.
The ball bounces off the hand segments with realistic reflection and damping.
Visual Feedback:The hand skeleton is drawn on the screen in green.
The ball turns yellow upon contact with the hand.

How It Works...
    Webcam Capture:   OpenCV is used to capture the video feed from the default webcam.
    Hand Processing:  The image is flipped (for a "mirror" effect) and converted to RGB for MediaPipe. MediaPipe's hands.process() function finds the 21 landmarks of the hand in the frame.
    Hand Skeleton:    The script defines a list of "segments" (e.g., index_mcp to index_pip, index_pip to index_dip) that connect the 21 landmarks to form a complete hand skeleton.
    Physics Loop:      On every frame, the ball's position is updated by its velocity, and its vertical velocity is increased by gravity.
    Collision Detection:
                Walls: The script checks if the ball's position exceeds the width or height of the screen and reverses its velocity if it does.
                Hand:  The script iterates through every segment in the hand skeleton and performs a line_circle_collision check between that segment and the ball.
   Collision Response:
                If a collision is detected, the line_circle_collision function calculates the collision normal, moves the ball out of the overlapping position, and computes a new reflected velocity vector, which is then applied to the ball.
Here is a complete, well-formatted README.md file for your GitHub project, based on the Python script you provided.

.................2. Hand-Controlled Adaptive Video Filter

This is an interactive Python project that uses OpenCV and MediaPipe to create a "magic window" or "lens" effect through your webcam. It tracks your hand(s) to draw a dynamic square on the screen. The video feed inside this square is inverted (like a photo negative), while the area outside remains normal.

The position, size, and rotation of the square are controlled by your hand gestures.

Features

Real-Time Hand Tracking: Uses MediaPipe to track up to two hands simultaneously.
    Dual-Control Modes:
        Two-Hand Control: The square's position, size, and rotation are determined by the location of your two index fingertips.
        One-Hand Control: Uses a "pinch" gesture (thumb and index finger) to control the square.
    Dynamic "Lens" Effect:
        A photo-negative (inverted) filter is applied in real-time.
        This filter is selectively applied only inside the bounds of the hand-controlled square.
    Rotational Control: The square rotates to match the angle of your fingers, thanks to a custom rotate_point function.
    Numpy Masking: Uses numpy and cv2.bitwise_and operations to efficiently combine the filtered and non-filtered parts of the image.
    
How It Works..............
    Initialization: The script captures the webcam feed and initializes both MediaPipe Hands and Pose modules.
    Hand Detection: On each frame, the script processes the image to find hand landmarks.
    Control Logic:
        If 2 hands are detected: It calculates the midpoint, distance, and angle between the two index fingertips.
        If 1 hand is detected: It calculates the midpoint, distance, and angle between the thumb and index fingertips.
    Square Definition: Based on the control logic, it defines the four corners of a square. The square is centered at the midpoint, its size is proportional to the distance between the fingers, and it's rotated by the calculated angle.
    Masking: It creates a black-and-white mask (square_mask) by "filling in" the area of the rotated square.
    Selective Filtering:
        An inverted (negative) version of the entire webcam frame is created.
        cv2.bitwise_and is used with the square_mask to get only the negative-image part inside the square.
        cv2.bitwise_and is used with the inverted mask (cv2.bitwise_not(square_mask)) to get the original image part outside the square.

Final Composite: The "inside" and "outside" parts are added together, creating the final frame where only the box is filtered.

Display: The final image is shown in a window titled Adaptive Skeletal+Negative Filter.

(Note: While the script initializes MediaPipe Pose and names the window 'Skeletal+Negative Filter', the current code version only implements the Negative filter inside the box.)

............................3.Hand-Tracking Two-Player Paddle Game

This is a two-player "tennis" or "volleyball" style game built with Python, OpenCV, and MediaPipe. Two players use their fists as paddles in front of a single webcam to hit a virtual ball over a net. The game features a full scoring system, physics, and real-time hand detection.

How to Play
    Two Players, One Camera: Stand side-by-side in front of your webcam.
        Player 1 (Left): Controls the Blue paddle on the left side of the screen.
        Player 2 (Right): Controls the Green paddle on the right side of the screen.
        
Make a Fist: The game doesn't track your open hand. Make a fist to activate your paddle. The paddle is drawn between your index and pinky knuckles.

Objective: Hit the red ball over the net.
    Scoring:
        If the ball hits the floor on your opponent's side, you get a point.
        If you miss the ball and it hits the floor on your side, your opponent gets a point.
        After a point is scored, the ball will automatically serve from the side of the player who just scored.
    The Net: The net is split into two parts. The thin upper part is just for show. The thick, white lower part is a rigid wall—the ball will bounce off it!
    Controls:
        q: Quit the game.
        r: Reset the score to 0-0.
Features
    Two-Player Mode: Tracks two hands simultaneously, assigning one to Player 1 (Left) and one to Player 2 (Right).
    Fist-Based Paddles: Uses the index and pinky finger knuckles (MCP) to create a stable, line-based paddle.
    Physics Engine: The ball is affected by gravity, momentum, and bounces realistically off walls and paddles.
    Vector Collision: A custom check_paddle_collision function uses NumPy to calculate precise, vector-based collisions between the circular ball and the line-segment paddles.
    Full Game Loop: Includes a scoring system (score_L, score_R), a rigid net, and an alternating serve system after each point.
    Visual Feedback:
        Paddles are color-coded (Blue for P1, Green for P2).
        The score is displayed clearly at the top of the screen.
How It Works
    Hand Detection: MediaPipe Hands is configured to detect up to two hands (max_num_hands=2).
    Handedness: The script checks the handedness (Left/Right) and the x position of the detected hand to correctly assign it to P1 or P2.
    Paddle Creation: For a detected hand, it gets the pixel coordinates of the index and pinky knuckles. A line is then drawn between these two points to act as the paddle.
    Collision Logic:
        Walls/Net: Simple boundary checks.
        Paddles: The script calculates the closest point on the paddle's line segment to the ball's center. If the distance is less than the ball's radius, a collision is registered.
    Collision Response: When the ball hits a paddle, its velocity isn't just reflected. It's given a new, fixed upward velocity (bounce_speed) along the paddle's normal vector, ensuring a consistent and playable "pop" upward.

Requirements.....
You will need the following Python libraries:
    OpenCV: opencv-python
    MediaPipe: mediapipe
    NumPy: numpy
