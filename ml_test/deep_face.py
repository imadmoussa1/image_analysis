from deepface import DeepFace
from deepface.commons import functions
import matplotlib.pyplot as plt
import cv2

img_path = "test_data/n_1.jpg"

backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

detected_and_aligned_face = DeepFace.detectFace(img_path, detector_backend='mtcnn')
plt.imshow(detected_and_aligned_face)
plt.show()

# # face detection
# detected_face = functions.detect_face(img=img_path, detector_backend='mtcnn')
# # face alignment
# aligned_face = functions.align_face(img=detected_face, detector_backend='mtcnn')

# face verification
obj = DeepFace.verify("test_data/y_1.jpg", "test_data/y_3.jpg", detector_backend='mtcnn')
print("-----------------------------------")
print(obj)
# face recognition
df = DeepFace.find(img_path="test_data/n_1.jpg", db_path="test_data", detector_backend='mtcnn', enforce_detection=False)
print("-----------------------------------")
print(df)
# facial analysis
# demography = DeepFace.analyze("test_data/n_2.jpg", detector_backend='mtcnn')
# print("-----------------------------------")
# print(demography)
