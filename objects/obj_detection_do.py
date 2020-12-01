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

import apache_beam as beam


class DetectionClasses(beam.DoFn):
  def __init__(self, model_name, label_file_path):
    super(DetectionClasses, self).__init__()
    self.category_index = label_map_util.create_category_index_from_labelmap(label_file_path, use_display_name=True)
    self.model_name = model_name
    self.model = None

  def load_model(self):
    base_url = 'pretrained_model/' + self.model_name
    model_dir = pathlib.Path(base_url)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model

  def run_inference_for_single_image(self, model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    if 'detection_masks' in output_dict:
      num_detections = int(output_dict.pop('num_detections'))
      need_detection_key = ['detection_classes', 'detection_boxes', 'detection_masks', 'detection_scores']
      output_dict = {key: output_dict[key][0, :num_detections].numpy() for key in need_detection_key}
      output_dict['num_detections'] = num_detections
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(tf.convert_to_tensor(output_dict['detection_masks']), output_dict['detection_boxes'], image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
      output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    else:
      num_detections = int(output_dict.pop('num_detections'))
      output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
      output_dict['num_detections'] = num_detections
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

  def show_inference(self, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = self.run_inference_for_single_image(self.model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        self.category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    image_labels = set()
    for i in range(len(output_dict['detection_classes'])):
      detection_classes = output_dict['detection_classes'][i]
      detection_scores = output_dict['detection_scores'][i]
      category_class = self.category_index[detection_classes]['name']
      if detection_scores > 0.4:
        image_labels.add(category_class)
        # print(category_class, detection_scores)

    # display(Image.fromarray(image_np))
    img = Image.fromarray(image_np)
    # save a image using extension
    model_folder = "new_data/"+self.model_name
    image_name = os.path.basename(image_path)
    if not os.path.isdir(model_folder):
      os.makedirs(model_folder)
    img_path = model_folder+"/" + image_name
    img.save(img_path)
    # print(image_name, image_labels)
    image_name = image_name.replace(".jpg", "")
    if image_labels:
      return [(image_name, list(image_labels))]
    # img.show()

  def process(self, element):
    if not self.model:
      self.model = self.load_model()
    return self.show_inference(element)
