from function import *
import cv2
import os
import numpy as np
from time import sleep

# Actions, number of sequences, and sequence length need to be defined
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # Add your actions here
no_sequences = 30  # Example: 30 sequences per action
sequence_length = 30  # Example: 30 frames per sequence

# Define the data path where you want to save the .npy files
DATA_PATH = os.path.join('data')  # Update this with the actual path

# Create the necessary folders for each action
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass  # If directory already exists, ignore the error

# Initialize the MediaPipe hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through actions (A, B, C, etc.)
    for action in actions:
        # Loop through sequences (0, 1, 2, etc.)
        for sequence in range(no_sequences):
            # Loop through frames in each sequence
            for frame_num in range(sequence_length):

                # Read image from the file
                image_path = os.path.join('Image', action, '{}.png'.format(sequence))
                frame = cv2.imread(image_path)

                if frame is None:
                    print(f"Frame not found at: {image_path}")
                    continue

                # Convert the BGR image to RGB for MediaPipe processing
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks on the image
                draw_styled_landmarks(image, results)

                # Display the image with a wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Extract keypoints from the results
                keypoints = extract_keypoints(results)

                # Save the keypoints as .npy files
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
