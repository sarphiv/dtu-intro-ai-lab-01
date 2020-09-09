import numpy as np
import cv2
from tensorflow.keras.applications import mobilenet_v2
import os
import time
import asyncio


#Images path
images_path = "images"

#Set up neural network
model = mobilenet_v2.MobileNetV2(weights='imagenet')


def check_images_available(images_path):
    #Path must exist, path must lead to a directory, directory must have files
    return os.path.exists(images_path) and os.path.isdir(images_path) and len(next(os.walk(images_path))[2]) > 0

def classify_image(file_url):
    #Retrieve image
    frame = cv2.imread(file_url, cv2.IMREAD_COLOR)

    #Get smallest dimension
    min_dim = np.min(frame.shape[0:2])

    #If width is smallest dimension
    if frame.shape[1] < frame.shape[0]:
        width_start = 0
        width_end = min_dim
        height_start = int(np.floor((frame.shape[0] - min_dim) / 2))
        height_end = int(np.floor((frame.shape[0] + min_dim) / 2))
    else:
        width_start = int(np.floor((frame.shape[1] - min_dim) / 2))
        width_end = int(np.floor((frame.shape[1] + min_dim) / 2))
        height_start = 0
        height_end = min_dim

    #Crop into centered square based on smallest dimension
    # Scale into correct size
    frame = cv2.resize(frame[height_start:height_end, width_start:width_end, :], (224, 224))

    #Classify
    processed_image = mobilenet_v2.preprocess_input(np.expand_dims(frame, axis=0))
    predictions = model.predict(processed_image)
    labels = mobilenet_v2.decode_predictions(predictions)

    return labels


def classify_all(images_path):
    files = next(os.walk(images_path))[2]

    return [(file, classify_image(f"{images_path}/{file}")) for file in files]


def print_classifications(image_labels):
    for (filename, labels) in image_labels:
        print(f"\"{filename}\": \"{labels[0][0][1]}\" with certainty {labels[0][0][2]}")

    print("--------------------------------")



async def main():
    #If images are available, classify
    if check_images_available(images_path):
        time_modified = os.path.getmtime(images_path)
        #Always run at start even if no modifications happened since last
        print_classifications(classify_all(images_path))
    #Else, notify nothing is available
    else:
        print(f"No images found in folder \"{images_path}\"")
        time_modified = time.time()


    while True:
        #If images are not available or no changes have happened, wait
        if not check_images_available(images_path) or time_modified == os.path.getmtime(images_path):
            await asyncio.sleep(2)
        #Else, classify
        else:
            print_classifications(classify_all(images_path))
            time_modified = os.path.getmtime(images_path)



asyncio.run(main())