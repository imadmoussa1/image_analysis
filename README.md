# image_analysis
Project to Analysis image using different computer vision Algorithm. to able to get all the info from an image

## Model we will use in this project
1) Detect objects
2) Image Caption
3) Detect image properties
4) Assign general image attributes
5) Face detection
6) Face recognition (ex: Celebrity recognition)
7) OCR
8) Reverse Image search

## System
A micro service system, each service will do a specific task.
we are using:
1) GRPC
2) docker
3) Tensorflow serving
4) Python
5) Mongodb

## Store Image
Save image and meta data in mongo DB

Send Image via grpc https://stackoverflow.com/questions/62171037/grpc-python-sending-image-meta-data

## GRPC
https://grpc.io/docs/languages/python/quickstart/

## Options
1) Multi-task learning
2) Many models


### Detect object
1) https://github.com/hunglc007/tensorflow-yolov4-tflite
1) https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/detect.py
2) https://github.com/tensorflow/models/tree/master/research/object_detection


### Image Caption
1) Mutli Task : https://www.ijcai.org/Proceedings/2018/0168.pdf (paper), https://github.com/andyweizhao/Multitask_Image_Captioning (code)
2) https://www.tensorflow.org/tutorials/text/image_captioning

### Face recognition
- [x] https://github.com/serengil/deepface
- [ ] https://github.com/deepinsight/insightface (ArcFace with LResNet100E-IR)
- [ ] https://github.com/ageitgey/face_recognition
