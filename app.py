from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import cv2
import numpy as np
import mediapipe as mp

# Load model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define actions from A to Z
actions = [chr(i) for i in range(ord('A'), ord('Z')+1)]  # This generates ['A', 'B', 'C', ..., 'Z']

# Define colors for visualization
colors = [(245,117,16) for _ in range(len(actions))]  # One color per action

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if prob > threshold:
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8 

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
mp_hands = mp.solutions.hands
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 0, 0), 2)
        image, results = mediapipe_detection(cropframe, hands)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action_index = np.argmax(res)
                action_label = actions[action_index]
                print(action_label)
                predictions.append(action_index)
                
                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == action_index: 
                    if res[action_index] > threshold: 
                        if len(sentence) > 0: 
                            if action_label != sentence[-1]:
                                sentence.append(action_label)
                                accuracy.append(f"{res[action_index]*100:.2f}")
                        else:
                            sentence.append(action_label)
                            accuracy.append(f"{res[action_index]*100:.2f}") 

                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

                # Viz probabilities
                frame = prob_viz(res, actions, frame, colors, threshold)
        except Exception as e:
            # print(e)
            pass

        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -"+' '.join(sentence)+''.join(accuracy), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
