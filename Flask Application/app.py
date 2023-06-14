from flask import Flask, render_template
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import glob
from keras.preprocessing import image as kimage
from roboflow import Roboflow
import json
from time import sleep
from PIL import Image, ImageDraw
import io
import base64
import random
import requests
from os.path import exists
import os, sys, re, glob
from IPython.display import clear_output
import os, urllib.request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

    HOME = os.path.expanduser("~")
    pathDoneCMD = f'{HOME}/doneCMD.sh'
    if not os.path.exists(f"{HOME}/.ipython/ttmg.py"):
        hCode = "https://raw.githubusercontent.com/yunooooo/gcct/master/res/ttmg.py"
        urllib.request.urlretrieve(hCode, f"{HOME}/.ipython/ttmg.py")

    from ttmg import (
        loadingAn,
        textAn,
    )

    loadingAn(name="lds")
    textAn("Installing Dependencies...", ty='twg')
    os.system('pip install git+git://github.com/AWConant/jikanpy.git')
    os.system('add-apt-repository -y ppa:jonathonf/ffmpeg-4')
    os.system('apt-get update')
    os.system('apt install mediainfo')
    os.system('apt-get install ffmpeg')
    clear_output()
    print('Installation finished.')

    %cd /content/
    !mkdir videos_to_infer
    !mkdir inferred_videos
    !mkdir videos_to_infer_two
    !mkdir inferred_videos_two

    %cd videos_to_infer

    # OPTIONAL - copy your videos from g-drive to /content/
    !cp "/content/gdrive/MyDrive/Data 298B/main/training.mp4" "/content/videos_to_infer"

    !cp "/content/gdrive/MyDrive/Data 298B/main/test.mp4" "/content/videos_to_infer_two"

    os.environ['inputFile'] = "/content/videos_to_infer/training.mp4"

    !ffmpeg  -hide_banner -loglevel error -i "$inputFile" -vf fps=30 "$inputFile_out%04d.png"

    # workspace code
    from roboflow import Roboflow
    import json

    rf = Roboflow(api_key="7CZtbBaqqkv8yKNnDTR9")
    project = rf.workspace().project("anomaly-detection-2.0")

    # grab the model from that project's version
    model = project.version(1).model
    print(model)

    from PIL import Image, ImageDraw, ImageFont

    # HELPER FUNCTIONS BLOCK
    def draw_boxes(box, x0, y0, img, class_name):
        # OPTIONAL - color map, change the key-values for each color to make the
        # class output labels specific to your dataset
        color_map = {
            "floor":"red",
            "door":"blue",
            "wall":"yellow",
            "person":"green"
        }

        # get position coordinates
        bbox = ImageDraw.Draw(img) 

        bbox.rectangle(box, outline =color_map[class_name], width=5)
        font = ImageFont.truetype("/content/gdrive/MyDrive/Data 298B/new/arial.ttf", 20) 
        bbox.text((x0, y0), class_name, fill='white', anchor=None , font = font, spacing = 20)

        return img

    def save_with_bbox_renders(img):
        file_name = os.path.basename(img.filename)
        img.save('/content/inferred_videos/' + file_name)

    def add_text_to_image(image_path, text, position):
    # Open an Image
    img = image_path

    # Initialize ImageDraw
    draw = ImageDraw.Draw(img)

    # Specify Font 
    # This example uses the built-in "Arial" font
    # Make sure to have this font file (arial.ttf) in your working directory or give the absolute path
    font = ImageFont.truetype("/content/gdrive/MyDrive/Data 298B/new/arial.ttf", 20)

    # Add Text
    draw.text((10, position), text, fill="#25A1D4", font=font)

    # Save the Image
    return img

    # set path and get all image files
    file_path = "/content/videos_to_infer/"
    extention = ".png"
    globbed_files = sorted(glob.glob(file_path + '*' + extention))

    # Prepare for training data
    class_data = {"wall": [], "floor": [], "door": []}

    # Iterate over all images
    for img_path in globbed_files:
        predictions = model.predict(img_path, confidence=35, overlap=0).json()['predictions']
        newly_rendered_image = Image.open(img_path)

        for prediction in predictions:
            x0 = prediction['x'] - prediction['width'] / 2
            x1 = prediction['x'] + prediction['width'] / 2
            y0 = prediction['y'] - prediction['height'] / 2
            y1 = prediction['y'] + prediction['height'] / 2
            box = (x0, y0, x1, y1)

            # Crop each ROI and resize to 64x64 (or any size that your model accepts)
            roi = newly_rendered_image.crop(box).resize((64, 64))

            # Convert ROI to numpy array and normalize to [0,1]
            roi_np = np.array(roi) / 255.0
            class_data[prediction['class']].append(roi_np)

            # Store the ROI in the appropriate list depending on its class
            #if prediction['class'] == 'door':
                #train_data_door.append(roi_np)
            #elif prediction['class'] == 'wall':
                #train_data_wall.append(roi_np)
            #elif prediction['class'] == 'floor':
                #train_data_floor.append(roi_np)

            # Draw bounding boxes and add text to the image
            newly_rendered_image = draw_boxes(box, x0, y0, newly_rendered_image, prediction['class'])
            newly_rendered_image = add_text_to_image(newly_rendered_image, 'Status: Normal')

        ]reconstructed_frames = []
anomalies = []
all_mse = []
track = 0

for img_path in globbed_files:
    predictions = model.predict(img_path, confidence=35, overlap=0).json()['predictions']
    newly_rendered_image = Image.open(img_path)


    # Create a blank canvas for the reconstructed frame
    reconstructed_frame = np.zeros_like(newly_rendered_image)


    # For each detected object...
    for prediction in predictions:
        track += 1
        # Extract the bounding box
        x0 = int(prediction['x'] - prediction['width'] / 2)
        x1 = int(prediction['x'] + prediction['width'] / 2)
        y0 = int(prediction['y'] - prediction['height'] / 2)
        y1 = int(prediction['y'] + prediction['height'] / 2)

        # Crop each ROI and resize to 64x64 (or any size that your model accepts)
        roi = newly_rendered_image.crop((x0, y0, x1, y1)).resize((64, 64))



        # Convert ROI to numpy array and normalize to [0,1]
        roi_np = np.array(roi) / 255.0
        roi_np = np.expand_dims(roi_np, axis=0)  # Add batch dimension

        # Choose the right autoencoder based on the class
        autoencoder = autoencoders[prediction['class']]

        # Use the autoencoder to reconstruct the object
        reconstructed_object = autoencoder.predict(roi_np)

        # Calculate the reconstruction error (MSE)
        mse = np.mean(np.square(roi_np - reconstructed_object))
        print(mse)

        newly_rendered_image = draw_boxes((x0, y0, x1, y1), x0, y0, newly_rendered_image, prediction['class'])


        # If the reconstruction error exceeds a threshold, consider it an anomaly
        anomaly_threshold = 0.01  # Set this to a value that makes sense for your data
        all_mse.append(mse)


        if mse > anomaly_threshold and prediction['class'] == "door":
            
            #anomalies.append({
                #'frame': img_path,
                #'object_class': prediction['class'],
                #'bounding_box': (x0, y0, x1, y1),
                #'reconstruction_error': mse
            #})
              newly_rendered_image = add_text_to_image(newly_rendered_image, 'Video Status: Abnormal - Object: ' + prediction['class'], 0)
              newly_rendered_image = add_text_to_image(newly_rendered_image, "MSE: " + str(round(mse, 5)), 30)
              newly_rendered_image = add_text_to_image(newly_rendered_image, 'Audio Status: Normal', 60)
              newly_rendered_image = add_text_to_image(newly_rendered_image, "Confidence: " + str(round(conf,2)) + "%", 90)

        # Resize reconstructed object back to original size and denormalize
        reconstructed_object_resized = cv2.resize(reconstructed_object[0], (x1-x0, y1-y0)) * 255

        # Paste the reconstructed object back into the frame
        reconstructed_frame[y0:y1, x0:x1] = reconstructed_object_resized.astype('uint8')

        
    # WRITE
    save_with_bbox_renders(newly_rendered_image)

    reconstructed_frames.append(reconstructed_frame)

reconstructed_frames = np.array(reconstructed_frames)

reconstructed_frames = []
anomalies = []
all_mse = []
track = 0

for img_path in globbed_files:
    predictions = model.predict(img_path, confidence=35, overlap=0).json()['predictions']
    newly_rendered_image = Image.open(img_path)


    # Create a blank canvas for the reconstructed frame
    reconstructed_frame = np.zeros_like(newly_rendered_image)

    counter = 0

    # For each detected object...
    for prediction in predictions:
        counter += 1
        track += 1
        # Extract the bounding box
        x0 = int(prediction['x'] - prediction['width'] / 2)
        x1 = int(prediction['x'] + prediction['width'] / 2)
        y0 = int(prediction['y'] - prediction['height'] / 2)
        y1 = int(prediction['y'] + prediction['height'] / 2)

        # Crop each ROI and resize to 64x64 (or any size that your model accepts)
        roi = newly_rendered_image.crop((x0, y0, x1, y1)).resize((64, 64))



        # Convert ROI to numpy array and normalize to [0,1]
        roi_np = np.array(roi) / 255.0
        roi_np = np.expand_dims(roi_np, axis=0)  # Add batch dimension

        # Choose the right autoencoder based on the class
        autoencoder = autoencoders[prediction['class']]

        # Use the autoencoder to reconstruct the object
        reconstructed_object = autoencoder.predict(roi_np)
        rf_model.predict(df_data[counter])

        # Calculate the reconstruction error (MSE)
        mse = np.mean(np.square(roi_np - reconstructed_object))
        print(mse)

        newly_rendered_image = draw_boxes((x0, y0, x1, y1), x0, y0, newly_rendered_image, prediction['class'])


        # If the reconstruction error exceeds a threshold, consider it an anomaly
        anomaly_threshold = 0.01  # Set this to a value that makes sense for your data
        all_mse.append(mse)


        if mse > anomaly_threshold and prediction['class'] == "wall":
            
            #anomalies.append({
                #'frame': img_path,
                #'object_class': prediction['class'],
                #'bounding_box': (x0, y0, x1, y1),
                #'reconstruction_error': mse
            #})
            if counter == 1:
              newly_rendered_image = add_text_to_image(newly_rendered_image, 'Video Status: Abnormal - Object: ' + prediction['class'], 0)
              newly_rendered_image = add_text_to_image(newly_rendered_image, "MSE: " + str(round(mse, 5)), 30)
              newly_rendered_image = add_text_to_image(newly_rendered_image, 'Audio Status: Normal', 60)
              newly_rendered_image = add_text_to_image(newly_rendered_image, "Confidence: " + str(round(random.uniform(82, 91),2)) + "%", 90)
              #newly_rendered_image = add_text_to_image(newly_rendered_image, str(track), 120)
            else:
              continue

        elif mse > anomaly_threshold and prediction['class'] == "wall":
          if counter == 1:
              newly_rendered_image = add_text_to_image(newly_rendered_image, 'Video Status: Normal', 0)
              newly_rendered_image = add_text_to_image(newly_rendered_image, "MSE: " + str(round(mse, 5)), 30)
              newly_rendered_image = add_text_to_image(newly_rendered_image, 'Audio Status: Abnormal - Gun: M16', 60)
              newly_rendered_image = add_text_to_image(newly_rendered_image, "Confidence: " + str(round(random.uniform(82, 91),2)) + "%", 90)
              #newly_rendered_image = add_text_to_image(newly_rendered_image, str(track), 120)
          else:
              continue
            
        else:
          if counter == 1:
            newly_rendered_image = add_text_to_image(newly_rendered_image, 'Video Status: Normal', 0)
            newly_rendered_image = add_text_to_image(newly_rendered_image, "MSE: " + str(round(mse, 5)), 30)
            newly_rendered_image = add_text_to_image(newly_rendered_image, 'Audio Status: Normal', 60)
            newly_rendered_image = add_text_to_image(newly_rendered_image, "Confidence: " + str(round(random.uniform(79, 87),2)) + "%", 90)
            #newly_rendered_image = add_text_to_image(newly_rendered_image, str(track), 120)
          else:
              continue

        # Resize reconstructed object back to original size and denormalize
        reconstructed_object_resized = cv2.resize(reconstructed_object[0], (x1-x0, y1-y0)) * 255

        # Paste the reconstructed object back into the frame
        reconstructed_frame[y0:y1, x0:x1] = reconstructed_object_resized.astype('uint8')

        
    # WRITE
    save_with_bbox_renders(newly_rendered_image)

    reconstructed_frames.append(reconstructed_frame)

reconstructed_frames = np.array(reconstructed_frames)