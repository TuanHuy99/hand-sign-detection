# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import cv2
import os
from ultralytics import YOLO
from matplotlib import pyplot as plt

# Loading classification model
cls_model = load_model("weights/vggmodel.h5", compile=False)
# Loading yolo v8 model
yolo_model = YOLO('weights/best.pt')

# Get labels
labels = os.listdir("C:/Users/vutua/Workspace/Đồ án/Thu-hand/data_gesture_hand/data_copy")

# Preparing and pre-processing the image
def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape

# Predicting function
def predict_result(img):
    all_labels = []
    list_label = []
    results = yolo_model.predict(img)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            # print('BOX: ', box.xyxy[0])
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            print(x1, y1, x2, y2)
            cropped_image = img[y1:y2, x1:x2]
            cropped_image = cv2.resize(cropped_image, (128, 128))
            cropped_image = np.array([cropped_image])
            pred = cls_model.predict(cropped_image)
            rs = np.argmax(pred[0], axis=-1)
            all_labels.append({
                'cls': labels[rs],
                'position': (x1, y1, x2, y2)
                })
            list_label.append(labels[rs])
            # img = cv2.rectangle(img, (x1, y1), (x2,y2), color=(0, 255, 0), thickness=2)
    
    return all_labels, list_label

