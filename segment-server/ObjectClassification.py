import sys
import base64
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

def enhance(frame):
    path = "protobufs/EDSR_x4.pb"
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(path)
    sr.setModel("edsr", 2)
    result = sr.upsample(frame)
    return result

def segment():
    #Load model
    model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

    image = cv2.imread("temp.img")

    if image is None:
        print('Could not open or find the image')
        sys.exit()

    #Resize image and preprocess
    image = cv2.resize(image, (540, 540), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    #Get results
    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int)

    height, width, _ = image.shape

    #Crop out each area inside each bounding box
    images = []
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.5:
            box = output_dict['detection_boxes'][i]
            ymin, xmin, ymax, xmax = box

            ymin = int(ymin * height)
            ymax = int(ymax * height)
            xmin = int(xmin * width)
            xmax = int(xmax * width)

            crop = image[ymin:ymax, xmin:xmax]
            crop = enhance(crop)
            images.append(crop)

    images = np.array(images, dtype=object)

    #Convert to base 64
    output_images_base64 = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode() for img in images]
    print(json.dumps(output_images_base64))

segment()
