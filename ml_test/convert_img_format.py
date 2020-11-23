import cv2
import glob
import os

faces_folder_path = "test_data"

# change file format
for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
  print("Processing file: {}".format(f))
  # Load .png image
  image = cv2.imread(f)
  new_file_name = f.replace("png", "jpg")
  # # Save .jpg image
  cv2.imwrite(new_file_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  os.remove(f)


for f in glob.glob(os.path.join(faces_folder_path, "*.jpeg")):
  print("Processing file: {}".format(f))
  # Load .png image
  image = cv2.imread(f)
  new_file_name = f.replace("jpeg", "jpg")
  # # Save .jpg image
  cv2.imwrite(new_file_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  os.remove(f)

# Change High resolution
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
  image = cv2.imread(f)
  dimensions = image.shape
  print(dimensions)
  if image.shape[1] > 1024:
    scale_percent = 20  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
