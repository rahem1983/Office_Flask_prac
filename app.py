from flask import Flask
import numpy as np
import cv2
import pandas as pd
app = Flask(__name__)

@app.route("/")       
def hello_world():
    conda = 23 + 3 
    return {{conda}}
                
            