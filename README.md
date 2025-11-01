# MediaPipe-OpenCV-Hand-Interactions-Demo
A Python CV project using OpenCV &amp; MediaPipe for real-time Human-Computer Interaction (HCI). Developed a hand-controlled game (e.g., bouncing ball) that maps the 21-point hand model to game controls. Integrated dynamic visual filters that react to hand gestures. Demonstrates expertise in real-time tracking and AR-like overlays.

Hand-Tracking Physics Ball

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

    Collision Response: If a collision is detected, the line_circle_collision function calculates the collision normal, moves the ball out of the overlapping position, and computes a new reflected velocity vector, which is then applied to the ball.
