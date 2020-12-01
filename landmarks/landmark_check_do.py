from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from delf import box_io
from delf import utils
from delf import detector

import apache_beam as beam


class LandmarkCheck(beam.DoFn):
  def __init__(self):
    super(LandmarkCheck, self).__init__()
    self.detector_thresh = 0.8
    self.detector_fn = None

  def _FilterBoxesByScore(self, boxes, scores, class_indices, score_threshold):
    selected_boxes = []
    selected_scores = []
    selected_class_indices = []
    for i, box in enumerate(boxes):
      if scores[i] >= score_threshold:
        selected_boxes.append(box)
        selected_scores.append(scores[i])
        selected_class_indices.append(class_indices[i])

    return np.array(selected_boxes), np.array(selected_scores), np.array(selected_class_indices)

  def _PlotBoxesAndSaveImage(self, image, boxes, output_path):
    _BOX_EDGE_COLORS = ['r', 'y', 'b', 'm', 'k', 'g', 'c', 'w']
    height = image.shape[0]
    width = image.shape[1]

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i, box in enumerate(boxes):
      scaled_box = [
          box[0] * height, box[1] * width, box[2] * height, box[3] * width
      ]
      rect = patches.Rectangle([scaled_box[1], scaled_box[0]],
                               scaled_box[3] - scaled_box[1],
                               scaled_box[2] - scaled_box[0],
                               linewidth=3,
                               edgecolor=_BOX_EDGE_COLORS[i % len(_BOX_EDGE_COLORS)],
                               facecolor='none')
      ax.add_patch(rect)

    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

  def check(self, image_path):
    output_dir = "new_data/frcnn_boxes"
    output_viz_dir = 'new_data/frcnn_boxes_vi'
    # Extension/suffix of produced files.
    _BOX_EXT = '.boxes'
    _VIZ_SUFFIX = '_viz.jpg'

   # Create output directories if necessary.
    if not tf.io.gfile.exists(output_dir):
      tf.io.gfile.makedirs(output_dir)
    if output_viz_dir and not tf.io.gfile.exists(output_viz_dir):
      tf.io.gfile.makedirs(output_viz_dir)

    # If descriptor already exists, skip its computation.
    base_boxes_filename, _ = os.path.splitext(os.path.basename(image_path))
    out_boxes_filename = base_boxes_filename + _BOX_EXT
    out_boxes_fullpath = os.path.join(output_dir, out_boxes_filename)
    if tf.io.gfile.exists(out_boxes_fullpath):
      print(f'Skipping {image_path}')

    im = np.expand_dims(np.array(utils.RgbLoader(image_path)), 0)
    # Extract and save boxes.
    (boxes_out, scores_out, class_indices_out) = self.detector_fn(im)
    (selected_boxes, selected_scores,
     selected_class_indices) = self._FilterBoxesByScore(boxes_out[0],
                                                    scores_out[0],
                                                    class_indices_out[0],
                                                    self.detector_thresh)
    box_io.WriteToFile(out_boxes_fullpath, selected_boxes, selected_scores, selected_class_indices)
    if output_viz_dir:
      out_viz_filename = base_boxes_filename + _VIZ_SUFFIX
      out_viz_fullpath = os.path.join(output_viz_dir, out_viz_filename)
      self._PlotBoxesAndSaveImage(im[0], selected_boxes, out_viz_fullpath)
    image_name = os.path.basename(image_path).replace(".jpg", "")
    if len(selected_scores) > 0:
      # print(selected_scores[0])
      return [(image_name, selected_scores[0])]

  def process(self, element):
    detector_path = 'pretrained_model/d2r_mnetssd_20190411'
    if not self.detector_fn:
      self.detector_fn = detector.MakeDetector(detector_path)
    return self.check(element)
