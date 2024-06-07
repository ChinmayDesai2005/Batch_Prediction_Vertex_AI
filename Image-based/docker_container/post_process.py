import numpy as np

def postprocess(prediction):
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    return class_names[np.argmax(prediction)]