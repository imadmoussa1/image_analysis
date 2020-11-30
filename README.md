# Image analysis
Project to Analysis images using different computer vision Algorithm. to able to get all the info from an image.
And save them to a DB
[Article: Computer vision projects](https://www.analyticsvidhya.com/blog/2020/09/18-open-source-computer-vision-projects-beginners/)

## Targets
1) [X] [Object Detection](#Object-Detection)
2) [ ] [Deep Labelling for Semantic Image Segmentation](#Deep-Labelling-for-Semantic-Image-Segmentation)
2) [ ] [Image Captioning](#Image-Captioning)
6) [X] [Face Clustering](Face-Clustering)
5) [X] [Face Recognition](#Face-recognition)
7) [X] [OCR](#OCR)
3) [ ] [Detect image properties](#Detect-image-properties)
1) [X] [Detect Labels](#Assign-general-image-attributes)
8) [X] [Landmark Detection](#Landmark-Detection)
9) [ ] [Pose Estimation](#pose-estimation)
8) [ ] Reverse Image search

## System
A micro service system, each service will do a specific task.
we are using:
1) GRPC
2) docker
3) TFX
4) Python
5) elastic

### Store Image
Save image and meta data in mongo DB

Send Image via grpc https://stackoverflow.com/questions/62171037/grpc-python-sending-image-meta-data

### GRPC
- [GRPC](https://grpc.io/docs/languages/python/quickstart/)

## Options
1) [ ] Multi-task learning
2) [X] Different Pre-trained models

## Production
## Machine Learning Pipelines
- [ ] TFX
- [ ] Apache Beam

## Orchestrations
- [ ] Apache Airflow
- [ ] KubeFlow

## Models
All pertained model should be downloaded and unzipped in the `pretrained_model` folder

### Object Detection
[details](objects/README.md)
- [ ] [YOLOv4](https://github.com/AlexeyAB/darknet)
- [ ] [tensorflow yolov4](https://github.com/hunglc007/tensorflow-yolov4-tflite)
- [X] [Object Detection API TF2](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [ ] [DERT](https://arxiv.org/pdf/2005.12872.pdf)

### Deep Labelling for Semantic Image Segmentation
- [X] [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [ ] [SOLO](https://github.com/WXinlong/SOLO)

### Image Captioning
1) Mutli Task : https://www.ijcai.org/Proceedings/2018/0168.pdf (paper), https://github.com/andyweizhao/Multitask_Image_Captioning (code)
2) https://www.tensorflow.org/tutorials/text/image_captioning

### Face Clustering
- [x] [dlib](http://dlib.net/)
- [ ] [face_recognition](https://github.com/ageitgey/face_recognition)
#### Ref:
  - [dlib tutorial and example](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/)

### Face recognition
- [x] [Deepface](https://github.com/serengil/deepface)
- [ ] [Insightface](https://github.com/deepinsight/insightface) (ArcFace with LResNet100E-IR)
#### Ref:
  - [deepface tutorial](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/)

### OCR
- [ ] [EAST](https://github.com/argman/EAST)
- [X] [Keras OCR](https://pypi.org/project/keras-ocr)
- [ ] [Tesseract](https://pypi.org/project/pytesseract)
- [ ] [Textractor](https://github.com/danwald/pytextractor)
- [ ] [PSNET](https://pypi.org/project/psenet-text-detector)

### Detect image properties
https://github.com/tensorflow/models/tree/master/research/delf#delg

### Detect Labels
Using the object detection with Model trained on OID v4 data set

### Landmark Detection
No dataset with name: Landmark are only ID
[details](Landmarks/README.md)
- [X] [DELF](https://github.com/tensorflow/models/tree/master/research/delf)

### Pose Estimation

## Datasets

### Test Data set
based on Celebrity in places we chosen images of this celebrities, also image from social media to generate a
small data set, To test the pertained model before using them in our system.

### Public Data Set
- [MS-Celeb-1M](https://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97/tech&hit=1&filelist=1)
- [VGGFace2](http://zeus.robots.ox.ac.uk/vgg_face2/)
- [Celebrity In Places](http://www.robots.ox.ac.uk/~vgg/data/celebrity_in_places/)
- [Celebrity Together](http://www.robots.ox.ac.uk/~vgg/data/celebrity_together/)
- [COCO Dataset](https://cocodataset.org/#home)
- [Open Images OID](https://storage.googleapis.com/openimages/web/index.html)
- [The Oxford Buildings Dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/s)