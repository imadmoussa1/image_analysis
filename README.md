# Image analysis
Project to Analysis images using different computer vision Algorithm. to able to get all the info from an image

## Model we will use in this project
1) [Object Detection](#Object-Detection)
2) [Image Captioning](#Image-Captioning)
5) [Face Recognition](#Face-recognition)
6) [Face Clustering](Face-Clustering)
7) [OCR](#OCR)
8) Reverse Image search
3) [Detect image properties](#Detect-image-properties)
1) [Assign general image attributes](#Assign-general-image-attributes)

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
- [GRPC](https://grpc.io/docs/languages/python/quickstart/)

## Options
1) Multi-task learning
2) Many models


## Models
### Object Detection
1) https://github.com/hunglc007/tensorflow-yolov4-tflite
1) https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/detect.py
2) https://github.com/tensorflow/models/tree/master/research/object_detection



### Image Captioning
1) Mutli Task : https://www.ijcai.org/Proceedings/2018/0168.pdf (paper), https://github.com/andyweizhao/Multitask_Image_Captioning (code)
2) https://www.tensorflow.org/tutorials/text/image_captioning

### Face recognition
- [x] [Deepface](https://github.com/serengil/deepface)
- [ ] [Insightface](https://github.com/deepinsight/insightface) (ArcFace with LResNet100E-IR)
- [ ] [face_recognition](https://github.com/ageitgey/face_recognition)

### Face Clustering
- [x] [dlib](http://dlib.net/)

### OCR

### Detect image properties

### Assign general image attributes

## Dataset

### MS-Celeb-1M
- [MS-Celeb-1M](https://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97/tech&hit=1&filelist=1)
- [VGGFace2](http://zeus.robots.ox.ac.uk/vgg_face2/)