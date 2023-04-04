from flask import Flask
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import cv2
import pandas as pd
from flask import Flask, request, jsonify
import base64
from PIL import Image
from torch.autograd import Variable
import pickle
import io


app = Flask(__name__)


# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
model_state_dict = torch.load('best_model/model01.pt')

model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 3)

model_ft.load_state_dict(model_state_dict)

model_ft.eval()

toTensor = transforms.ToTensor()

@app.route('/isblur', methods=['POST'])       
def is_base64():
    try:

        # receiving the data from request
        image_data = request.files['image'].read()

        # Convert the image data to a numpy array
        nparr = np.fromstring(image_data, np.uint8)

        # Decode the numpy array to an image using cv2
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale using cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces and crop the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            faces = img[y:y + h, x:x + w]

        # convert image to tensor
        image_tensor = toTensor(faces)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)

        #feeding into the model to get output 
        output = model_ft(input)
        output = output.cpu()
        index = output.data.numpy().argmax()
        
        #classifying the output 
        if index == 0 or index == 1:
            pred = "Blur"
        else:
            pred = "Sharp"


        # Return the prediction as a JSON response
        return jsonify({'prediction': pred,
                        'message': "successful"})
    except:
        return jsonify({'prediction': "error",
                        'message': "unsuccessful"})
                
            
@app.route('/predict', methods=['POST'])
def predict():
    # Get the base64-encoded image from the request
    class_names = ['defocused_blurred','motion_blurred','sharp']
    image_data = request.files['image'].read()
    image_b64 = request.json['image']

    # Decode the image from base64
    image_data = base64.b64decode(image_b64)
    image = np.array(Image.open(io.BytesIO(image_data)))

    toTensor = transforms.ToTensor()
    image_tensor = toTensor(image)
    image_tensor = image_tensor.float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)

    output = model_ft(input)
    output = output.cpu()
    index = output.data.numpy().argmax()
    
    if index == 0 or index == 1:
        pred = "Blur"
    else:
        pred = "Sharp"


    # Return the prediction as a JSON response
    return jsonify({'prediction': pred,
                    'message': "successful"})