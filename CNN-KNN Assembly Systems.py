import cv2
import dlib
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import matplotlib.pyplot as plt

# Load the FER2013 dataset
data = pd.read_csv('YOUR_DATASET.csv')

# Extract the features and labels
X = []
y = []
for i in range(len(data)):
    pixels = [int(p) for p in data['pixels'][i].split()]
    image = np.array(pixels, dtype=np.uint8).reshape(48, 48)  # Reshape to 48x48
    X.append(image.flatten())  # Flatten the image to a 1D array
    y.append(data['emotion'][i])
X = np.array(X)
y = np.array(y)

# Perform PCA ( Principal Component Analysis) for dimensionality reduction
pca = PCA(n_components=2304)  # Reduced to 2304 features
X = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train, y_train)

# Load the CNN model
cnn_model = load_model('YOUR_MODEL.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Map emotion labels to human-readable strings
emotion_map = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Initialize a list to store the detected emotions
emotion_list = []

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # For each detected face, detect the emotion using both models and apply voting
    for face in faces:
        # Extract the face region as a numpy array
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_roi = gray[y1:y2, x1:x2]

        # Resize the face region to match the input size of the models
        face_roi_cnn = cv2.resize(face_roi, (48, 48))
        face_roi_knn = cv2.resize(face_roi, (48, 48)).flatten()

        # Normalize the pixel values
        face_roi_cnn = face_roi_cnn / 255.0
        face_roi_knn = face_roi_knn / 255.0

        # Reshape the face region for the CNN model
        face_roi_cnn = np.reshape(face_roi_cnn, (1, 48, 48, 1))

        # Perform prediction using both models
        emotion_probabilities_cnn = cnn_model.predict(face_roi_cnn)
        emotion_index_cnn = np.argmax(emotion_probabilities_cnn)
        emotion_label_cnn = emotion_map[emotion_index_cnn]

        emotion_label_knn = knn_model.predict([face_roi_knn])[0]

        # Apply voting to determine the final emotion label
        votes = [emotion_label_cnn, emotion_label_knn]
        final_emotion_label = max(set(votes), key=votes.count)

        # Add the detected emotion to the list
        emotion_list.append(final_emotion_label)

        # Overlay the emotion label on the frame
        cv2.putText(frame, str(final_emotion_label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

## -- -- -- -- -- -- -- -- Showing the occurrences of each detected emotion graphically

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
