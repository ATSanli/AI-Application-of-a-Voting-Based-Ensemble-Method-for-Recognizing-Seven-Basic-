import cv2
import dlib
import numpy as np, h5py
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import h5py
import os
from keras.models import load_model
import matplotlib.pyplot as plt

# Change the current working directory to the directory containing the saved model       
os.chdir("C:\\Users\\ATS\\Desktop\\EMODETECT")

# Set the file path of the saved model
model_path = "fer2013_CNN_06062023withouthypeR_params.h5"

# Load the saved model 
model = load_model(model_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Map emotion labels to human-readable strings
emotion_map = {0: 'Sinirli', 1: 'Tiksinme', 2: 'Korku', 3: 'Mutlu', 4: 'Uzgun', 5: 'Saskinlik', 6: 'Normal'}

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('EmoDetectionCNN_Ver2_.mp4', fourcc, 20.0, (640,480))

# Initialize a list to store the detected emotions
emotion_list = []

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # For each detected face, detect the emotion and overlay it on the frame
    for face in faces:
        # Extract the face region as a numpy array
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_roi = gray[y1:y2, x1:x2]

        # Resize the face region to match the input size of the model
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the pixel values
        face_roi = face_roi / 255.0

        # Reshape the face region to match the input shape of the model
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Predict the emotion using the model
        emotion_probabilities = model.predict(face_roi)
        emotion_index = np.argmax(emotion_probabilities)
        emotion_label = emotion_map[emotion_index]

        # Add the detected emotion to the list
        emotion_list.append(emotion_label)

        # Overlay the emotion label on the frame
        cv2.putText(frame, emotion_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    # Write the frame to the video
    out.write(frame)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and video writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()


## -- -- -- -- -- -- -- -- Showing the occurrences of each detected emotion

# Count the occurrences of each detected emotion
emotion_counts = {emotion: emotion_list.count(emotion) for emotion in emotion_map.values()}

# Create lists for emotion labels and counts
emotions = list(emotion_counts.keys())
counts = list(emotion_counts.values())

# Create a bar plot of the detected emotions
plt.bar(emotions, counts)
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.title('Detected Emotions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
