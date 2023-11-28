import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report

# Load the testing dataset
data = pd.read_csv('your_CSV_File.csv')

# Extract the features and labels
X_test = []
y_test = []
for i in range(len(data)):
    X_test.append([int(p) for p in data['pixels'][i].split()])
    y_test.append(data['emotion'][i])
X_test = np.array(X_test)
y_test = np.array(y_test)

# Preprocess the testing data
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1) / 255.0

# Load the trained model
model = load_model('Your_Model_File.h5')

# Perform prediction on the testing data
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Convert labels to emotion names
emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
y_test_emotions = [emotions[label] for label in y_test]
y_pred_emotions = [emotions[label] for label in y_pred]

# Print classification report
print(classification_report(y_test_emotions, y_pred_emotions))
