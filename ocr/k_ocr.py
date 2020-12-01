import matplotlib.pyplot as plt

import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()
# Get a set of three example images
images = [keras_ocr.tools.read(url) for url in ["t_2.jpg", "t_1.jpg", 'n_1.jpg']]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)
for prediction in prediction_groups:
  for word, box in prediction:
    print(word)
# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
  keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
