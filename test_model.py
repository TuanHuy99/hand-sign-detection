# # Importing required libs
# from keras.models import load_model
# from keras.utils import img_to_array
# import numpy as np
# from PIL import Image

# # Loading model
# model = load_model("weights/digit_model.h5")


# # Preparing and pre-processing the image
# def preprocess_img(img_path):
#     op_img = Image.open(img_path)
#     img_resize = op_img.resize((224, 224))
#     img2arr = img_to_array(img_resize) / 255.0
#     img_reshape = img2arr.reshape(1, 224, 224, 3)
#     return img_reshape


# # Predicting function
# def predict_result(predict):
#     pred = model.predict(predict)
#     return np.argmax(pred[0], axis=-1)

# from ultralytics import YOLO
import cv2
import numpy as np
import os

labels = os.listdir("C:/Users/vutua/Workspace/Đồ án/Thu-hand/data_gesture_hand/data_copy")
print("Labels:", labels)
# model = YOLO('weights/best.pt')
original_image = cv2.imread("2-hand-test.jpg")
print(original_image.shape)
img = cv2.resize(original_image, (128, 128))
print(img.shape)
imgs = np.array([img])
# results = model.predict(img)
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     for box in boxes:
#         print('BOX: ', box.xyxy[0])

        
from keras.models import load_model
model = load_model('weights/vggmodel.h5', compile=False)
pred = model.predict(imgs)
rs = np.argmax(pred[0], axis=-1)
print(labels[rs])
