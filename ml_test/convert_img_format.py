import cv2
import glob
import os

faces_folder_path = "test_data"

for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
  print("Processing file: {}".format(f))
  # Load .png image
  image = cv2.imread(f)
  new_file_name = f.replace("png", "jpg")
  # # Save .jpg image
  cv2.imwrite(new_file_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  os.remove(f)
