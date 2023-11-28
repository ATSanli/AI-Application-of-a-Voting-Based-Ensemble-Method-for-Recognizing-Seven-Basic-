import os
import pandas as pd
from PIL import Image

# Path to your ATS_FER_DB-2023 dataset
dataset_path = 'YOUR_DataBase_PATH'

# Define the emotions mapping
emotion_mapping = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

# Create lists to store data
emotions = []
pixels = []
usages = []

# Loop through the folders in the dataset path
for folder_name in emotion_mapping.keys():
    folder_path = os.path.join(dataset_path, folder_name)
    
    # Get the label for the current emotion
    label = emotion_mapping[folder_name]
    
    # Loop through the images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png")):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            pixel_values = ' '.join(map(str, list(image.getdata())))
            
            # Append data to lists
            emotions.append(label)
            pixels.append(pixel_values)
            usages.append('Training')  # Assuming these are training images

# Create a DataFrame
df = pd.DataFrame({'emotion': emotions, 'pixels': pixels, 'Usage': usages})

# Save to CSV
df.to_csv('FER2013_CK+_KDEF_2.csv', index=False)
