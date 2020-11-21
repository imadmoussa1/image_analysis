# Image analysis
Project to Analysis images using different computer vision Algorithm. to able to get all the info from an image.
And save them to a DB

## Targets
1) [Object Detection](#Object-Detection)
2) [Image Captioning](#Image-Captioning)
6) [Face Clustering](Face-Clustering)
5) [Face Recognition](#Face-recognition)
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

### Face Clustering
- [x] [dlib](http://dlib.net/)
#### Ref:
  - [dlib tutorial and example](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/)

### Face recognition
- [x] [Deepface](https://github.com/serengil/deepface)
- [ ] [Insightface](https://github.com/deepinsight/insightface) (ArcFace with LResNet100E-IR)
- [ ] [face_recognition](https://github.com/ageitgey/face_recognition)
#### Ref:
  - [deepface tutorial](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/)

### OCR

### Detect image properties

### Assign general image attributes

## Datasets

### Test Data set
based on Celebrity in places we chosen images of this celebrities:
  - [x] Taylor hill
  - [x] Zac Efron
  - [x] Zayn Malek
  - [x] random 1
  - [x] random 2
To test the pertained model before using them in our system,

### Public Data Set
- [ ] [MS-Celeb-1M](https://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97/tech&hit=1&filelist=1)
- [ ] [VGGFace2](http://zeus.robots.ox.ac.uk/vgg_face2/)
- [x] [Celebrity In Places](http://www.robots.ox.ac.uk/~vgg/data/celebrity_in_places/)
- [ ] [Celebrity Together](http://www.robots.ox.ac.uk/~vgg/data/celebrity_together/)