from potato_detector.python.predict import *
from PIL import Image, ImageDraw,ImageOps
from tensorflow.keras.models import load_model
import numpy as np

img_cls_width, img_cls_height = 160, 160

# Load the model
model = load_model('weights/keras_model.h5')

def classification(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)[0]
    return prediction

def detect_and_classification(image_file, threshold=.5):

    boxes = hb_detector(image_file)
        
    # Initialize good and bad counts as 0
    good = 0
    bad = 0

    image = Image.open(image_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        if (boxes[i]['tagName'] == 'potato') and (boxes[i]['probability'] >= threshold):
            imgcopy = image.copy()
            width, height = image.size
            left = round(boxes[i]['boundingBox']['left']*width)
            top = round(boxes[i]['boundingBox']['top']*height)
            wb = round(boxes[i]['boundingBox']['width']*width)
            hb = round(boxes[i]['boundingBox']['height']*height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            if left >= width:
                left = width -1
            if top >= height:
                top = height -1

            # pre-procesamiento para la clasificación
            imgcopy = imgcopy.crop((left, top, left+wb, top+hb))

            # clasificación
            result_cls = classification(imgcopy)
            if np.argmax(result_cls) == 0:
                draw.rectangle(((left, top), (left+wb, top+hb)), outline="green", width = 10)
                #draw.text((left, top), "Sano", fill="black")
                good += 1
            else:
                draw.rectangle(((left, top), (left+wb, top+hb)), outline="red", width = 10)
                #draw.text((left, top), "Malo", fill="black")
                bad += 1

            

    quantity=good + bad
    quality=good / (good+bad + 1e-8) * 100
    quality=round(quality, 3)
    return image, quantity, quality


if __name__ == "__main__":
    detect_and_classification("./readme_pictures/DPV.png")
