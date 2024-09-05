import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Suppress TensorFlow oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define actions, no_sequences, and sequence_length according to your project
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
no_sequences = 30  # Example: 30 sequences per action
sequence_length = 30  # Example: 30 frames per sequence

# Specify the path to where your data is stored (you can update this to your actual data path)
DATA_PATH = os.path.join('data')  # Make sure this is the correct path to your .npy files

# Create a label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Load data from .npy files
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            try:
                # Load the .npy file
                res = np.load(npy_path)
                window.append(res)
            except FileNotFoundError:
                print(f"File not found: {npy_path}")
                continue  # Continue with the next frame if file is missing

        # Ensure that all windows (sequences) have the correct length
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Sequence length mismatch for action {action} and sequence {sequence}")

# Convert sequences and labels into arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Ensure the data is in the correct shape
print(f"Data shape: X: {X.shape}, y: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Output neurons = number of actions

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

# Print model summary
model.summary()

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the trained model weights
model.save('model.h5')

print("Model training completed and saved.")
