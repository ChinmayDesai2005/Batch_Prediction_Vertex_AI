#decodeBase64
#pass to model as image
#cv gbr2rgb
import cv2 as cv
import base64
import numpy as np

def preprocess(input_base64):

    img = base64.b64decode(input_base64); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    source = cv.imdecode(npimg, 1)
    source = cv.cvtColor(source, cv.COLOR_BGR2RGB)

    return (np.array([source]) / 255)
