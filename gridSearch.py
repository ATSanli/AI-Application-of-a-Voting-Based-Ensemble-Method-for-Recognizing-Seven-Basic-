import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier

# Define augmentation functions
def random_crop(image, crop_size):
    height, width = image.shape[:2]
    dy, dx = crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[y:(y+dy), x:(x+dx)]

# Load the FER2013 dataset
data = pd.read_csv('ATS_FER_DB_2023.csv')

# Extract the features and labels
X = []
y = []
for i in range(len(data)):
    X.append([int(p) for p in data['pixels'][i].split()])
    y.append(data['emotion'][i])
X = np.array(X)
y = np.array(y)

# Convert labels to integers
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

# Preprocess the data
X = X.reshape(X.shape[0], 48, 48, 1) / 255.0
y_categorical = np_utils.to_categorical(y_int)

# Split the data into training and testing sets (using integer labels)
X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.1, random_state=42)

# Function to create the CNN model
def create_model(learning_rate=0.001, dropout_rate=0.2, conv_filters=128):
    model = Sequential()
    model.add(Conv2D(conv_filters, (3, 3), activation='relu', input_shape=X.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(conv_filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the Keras classifier model
keras_classifier = KerasClassifier(build_fn=create_model, epochs=100, batch_size=64, verbose=1)

# Define the hyperparameters for grid search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.2, 0.3, 0.4],
    'conv_filters': [64, 128, 256]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, scoring='accuracy', cv=StratifiedKFold(n_splits=3))

# Fit the GridSearchCV object to the data
grid_result = grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_)

# Save the best model after the grid search has finished
best_model = grid_result.best_estimator_.model
best_model.save('best_model_ATS_FER_DB_2023.h5')

# Evaluate the best model on the testing data
test_loss, test_accuracy = best_model.evaluate(X_test, np_utils.to_categorical(y_test))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


// The phase where the parameters tried to find best and suitable one by system.
// Basically we made grid on these parameters:  'learning_rate': [0.001, 0.01, 0.1], 'dropout_rate': [0.2, 0.3, 0.4], 'conv_filters': [64, 128, 256]


