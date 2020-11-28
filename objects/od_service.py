import os
import pathlib

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_model(model_name):
  base_url = 'pretrained_model/'+model_name
  model_dir = pathlib.Path(base_url)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]
  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    num_detections = int(output_dict.pop('num_detections'))
    need_detection_key = ['detection_classes','detection_boxes','detection_masks','detection_scores']
    output_dict = {key: output_dict[key][0, :num_detections].numpy()
                  for key in need_detection_key}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(tf.convert_to_tensor(output_dict['detection_masks']), output_dict['detection_boxes'], image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
  else:
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  # display(Image.fromarray(image_np))
  img = Image.fromarray(image_np)
  # save a image using extension
  model_folder = "new_data/"+model_name
  if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
  img_name = model_folder+"/"+ os.path.basename(image_path)
  img.save(img_name)
  # img.show()

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map/mscoco_complete_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_data')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

model_name = 'efficientdet_d5_coco17_tpu-32'
detection_model = load_model(model_name)
print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes
for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)


model_name = "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8"
masking_model = load_model(model_name)
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map/mscoco_complete_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
masking_model.signatures['serving_default'].output_shapes
for image_path in TEST_IMAGE_PATHS:
  show_inference(masking_model, image_path)


model_name = 'ssd_mobilenet_v2_oid_v4_2018_12_12'
detection_model = load_model(model_name)
PATH_TO_LABELS = 'label_map/oid_v4_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
for image_path in TEST_IMAGE_PATHS:
  show_inference(detection_model, image_path)