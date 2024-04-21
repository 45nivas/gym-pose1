from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# Generator function for Squat video frames
def gen_squat_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        squat_counter = 0
        stage = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(color)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage = "down"
                if angle < 100 and stage == 'down':
                    stage = "up"
                    squat_counter += 1

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display squat counter
                cv2.putText(frame, f'Squats: {squat_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw progress bar
            progress = 100 - angle if angle < 100 else 0
            cv2.rectangle(frame, (10, 100), (110, 150), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 100), (10 + int(progress), 150), (255, 0, 0), -1)

            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Generator function for Bicep Curls video frames for both arms
def gen_bicep_frames():
    # Open video capture using webcam
    cap = cv2.VideoCapture(0)
    # Initialize mediapipe pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Initialize counters and stages for both arms
        left_curl_counter = 0
        right_curl_counter = 0
        left_stage = None
        right_stage = None

        while cap.isOpened():
            # Capture frame from the video source
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame to mirror image (optional, may need to be removed if causing confusion)
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame with mediapipe pose
            results = pose.process(rgb_frame)

            # If landmarks are detected, process them
            if results.pose_landmarks:
                # Extract landmark data
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for the shoulders, elbows, and wrists of both arms
                # Since the frame is mirrored, we switch left and right arms here:
                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angles for both arms
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Check for reps on the left arm
                if left_arm_angle <= 30 and left_stage == 'down':
                    left_curl_counter += 1
                    left_stage = 'up'
                    print(f"Left arm rep completed! Total left reps: {left_curl_counter}")

                if left_arm_angle >= 160:
                    left_stage = 'down'

                # Check for reps on the right arm
                if right_arm_angle <= 30 and right_stage == 'down':
                    right_curl_counter += 1
                    right_stage = 'up'
                    print(f"Right arm rep completed! Total right reps: {right_curl_counter}")

                if right_arm_angle >= 160:
                    right_stage = 'down'

                # Draw landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Display curl count for both arms
                cv2.putText(frame, f'Left Curls: {left_curl_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f'Right Curls: {right_curl_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Convert frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()

            # Yield the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video capture
    cap.release()

# Generator function for Push-up video frames
def gen_pushup_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        pushup_counter = 0
        stage = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(color)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    pushup_counter += 1

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display push-up counter
                cv2.putText(frame, f'Push-ups: {pushup_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw progress bar
            progress = 100 - angle if angle < 100 else 0
            cv2.rectangle(frame, (10, 100), (110, 150), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 100), (10 + int(progress), 150), (255, 0, 0), -1)

            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_hammer_curl_frames():
    # Open video capture using webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize mediapipe pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Initialize counters and stages for both arms
        left_curl_counter = 0
        right_curl_counter = 0
        left_stage = None
        right_stage = None

        while cap.isOpened():
            # Capture frame from the video source
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame to mirror image
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame with mediapipe pose
            results = pose.process(rgb_frame)

            # If landmarks are detected, process them
            if results.pose_landmarks:
                # Extract landmark data
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for the shoulders, elbows, and wrists of both arms
                right_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                left_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angles for both arms
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Define the hammer curl angle limits
                curl_angle_threshold = 90

                # Check for reps on the left arm
                if left_arm_angle <= curl_angle_threshold and left_stage == 'down':
                    left_curl_counter += 1
                    left_stage = 'up'
                    print(f"Left hammer curl rep completed! Total left reps: {left_curl_counter}")

                if left_arm_angle >= 160:
                    left_stage = 'down'

                # Check for reps on the right arm
                if right_arm_angle <= curl_angle_threshold and right_stage == 'down':
                    right_curl_counter += 1
                    right_stage = 'up'
                    print(f"Right hammer curl rep completed! Total right reps: {right_curl_counter}")

                if right_arm_angle >= 160:
                    right_stage = 'down'

                # Draw landmarks and connections on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display hammer curl count for both arms
                cv2.putText(frame, f'Left Hammer Curls: {left_curl_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f'Right Hammer Curls: {right_curl_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Convert frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()

            # Yield the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video capture
    cap.release()

def gen_side_raises_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        side_raises_counter = 0
        stage = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(color)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get left and right shoulder and elbow landmarks
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                # Calculate angles for left and right side raises
                left_angle = calculate_angle(left_shoulder, left_elbow, (left_elbow[0], 0))
                right_angle = calculate_angle(right_shoulder, right_elbow, (right_elbow[0], 0))

                # Check for left side raise stage transitions
                if left_angle < 45:
                    stage = "up"
                if left_angle > 150 and stage == "up":
                    stage = "down"
                    side_raises_counter += 1

                # Check for right side raise stage transitions
                if right_angle < 45:
                    stage = "up"
                if right_angle > 150 and stage == "up":
                    stage = "down"
                    side_raises_counter += 1

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display side raises counter
                cv2.putText(frame, f'Shoulder press reps: {side_raises_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Convert the frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Release video capture
        cap.release()


# Flask Routes
@app.route('/')
def index():
    # The login page
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def handle_login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if the entered credentials are correct
    if username == 'nivas' and password == '12345678':
        # Render afterlogin.html if the login is successful
        return render_template('afterlogin.html')
    else:
        # Return an error message on the index.html page if login fails
        return render_template('index.html', error='Invalid username or password')

@app.route('/workout', methods=['GET', 'POST'])
def workout_choice():
    return render_template('workout.html')

@app.route('/squat')
def squat():
    return render_template('squat.html')

@app.route('/squat_feed')
def squat_feed():
    return Response(gen_squat_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pushup')
def pushup():
    return render_template('pushup.html')

@app.route('/pushup_feed')
def pushup_feed():
    return Response(gen_pushup_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bicep')
def bicep():
    return render_template('bicep.html')

@app.route('/bicep_feed')
def bicep_feed():
    return Response(gen_bicep_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hammer_curl')
def hammer_curl():
    return render_template('hammer_curl.html')

@app.route('/hammer_feed')
def hammer_curl_feed():
    return Response(gen_hammer_curl_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/side_raises')
def side_raises():
    return render_template('side_raises.html')


@app.route('/sideraises_feed')
def side_raises_feed():
    # Stream the video frames using gen_side_raises_frames()
    return Response(gen_side_raises_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

