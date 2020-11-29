"""Extracts bounding boxes from a list of images, saving them to files.
The images must be in JPG format. The program checks if boxes already
exist, and skips computation for those.
"""

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

cmd_args = None

# Extension/suffix of produced files.
_BOX_EXT = '.boxes'
_VIZ_SUFFIX = '_viz.jpg'

# Used for plotting boxes.
_BOX_EDGE_COLORS = ['r', 'y', 'b', 'm', 'k', 'g', 'c', 'w']

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.
  Args:
    list_path: Path to list of images, one image path per line.
  Returns:
    image_paths: List of image paths.
  """
  with tf.io.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def _FilterBoxesByScore(boxes, scores, class_indices, score_threshold):
  """Filter boxes based on detection scores.
  Boxes with detection score >= score_threshold are returned.
  Args:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.
    score_threshold: Float detection score threshold to use.
  Returns:
    selected_boxes: selected `boxes`.
    selected_scores: selected `scores`.
    selected_class_indices: selected `class_indices`.
  """
  selected_boxes = []
  selected_scores = []
  selected_class_indices = []
  for i, box in enumerate(boxes):
    if scores[i] >= score_threshold:
      selected_boxes.append(box)
      selected_scores.append(scores[i])
      selected_class_indices.append(class_indices[i])

  return np.array(selected_boxes), np.array(selected_scores), np.array(selected_class_indices)


def _PlotBoxesAndSaveImage(image, boxes, output_path):
  """Plot boxes on image and save to output path.
  Args:
    image: Numpy array containing image.
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    output_path: String containing output path.
  """
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
                             edgecolor=_BOX_EDGE_COLORS[i %
                                                        len(_BOX_EDGE_COLORS)],
                             facecolor='none')
    ax.add_patch(rect)

  ax.axis('off')
  plt.savefig(output_path, bbox_inches='tight')
  plt.close(fig)


def main(unused_argv):
  # Read list of images.
  print('Reading list of images...')
  image_paths = _ReadImageList(list_images_path)
  num_images = len(image_paths)
  print(f'done! Found {num_images} images')

  # Create output directories if necessary.
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  if output_viz_dir and not tf.io.gfile.exists(output_viz_dir):
    tf.io.gfile.makedirs(output_viz_dir)

  detector_fn = detector.MakeDetector(detector_path)

  start = time.time()
  for i, image_path in enumerate(image_paths):
    # Report progress once in a while.
    if i == 0:
      print('Starting to detect objects in images...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print(
          f'Processing image {i} out of {num_images}, last '
          f'{_STATUS_CHECK_ITERATIONS} images took {elapsed} seconds'
      )
      start = time.time()

    # If descriptor already exists, skip its computation.
    base_boxes_filename, _ = os.path.splitext(os.path.basename(image_path))
    out_boxes_filename = base_boxes_filename + _BOX_EXT
    out_boxes_fullpath = os.path.join(output_dir, out_boxes_filename)
    if tf.io.gfile.exists(out_boxes_fullpath):
      print(f'Skipping {image_path}')
      continue

    im = np.expand_dims(np.array(utils.RgbLoader(image_paths[i])), 0)

    # Extract and save boxes.
    (boxes_out, scores_out, class_indices_out) = detector_fn(im)
    (selected_boxes, selected_scores,
     selected_class_indices) = _FilterBoxesByScore(boxes_out[0],
                                                   scores_out[0],
                                                   class_indices_out[0],
                                                   detector_thresh)
    print(class_indices_out[0])
    box_io.WriteToFile(out_boxes_fullpath, selected_boxes, selected_scores, selected_class_indices)
    if output_viz_dir:
      out_viz_filename = base_boxes_filename + _VIZ_SUFFIX
      out_viz_fullpath = os.path.join(output_viz_dir, out_viz_filename)
      _PlotBoxesAndSaveImage(im[0], selected_boxes, out_viz_fullpath)

if __name__ == '__main__':
  detector_path = 'pretrained_model/d2r_frcnn_20190411'
  detector_thresh = 0.8
  list_images_path = "landmarks/list_images.txt"
  output_dir = "new_data/oxford5k_boxes"
  output_viz_dir = 'new_data/oxford5k_boxes_vi'
  app.run(main=main)
