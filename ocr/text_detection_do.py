import os
import matplotlib.pyplot as plt

import keras_ocr
import apache_beam as beam


class TextDetection(beam.DoFn):

  def __init__(self):
    super(TextDetection, self).__init__()
    self.pipeline = None

  def process(self, image_path):
    if not self.pipeline:
      self.pipeline = keras_ocr.pipeline.Pipeline()

    images = [keras_ocr.tools.read(url) for url in [image_path]]
    text = []
    image_name = os.path.basename(image_path).replace(".jpg", "")
    prediction_groups = self.pipeline.recognize(images)
    for prediction in prediction_groups:
      for word, box in prediction:
        print(word)
        text.append(word)
    return [(image_name, text)]
