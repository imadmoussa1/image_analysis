# python3 -m pipelines --input list_images.txt --output /path/to/write/counts
import json

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from objects.obj_detection_do import DetectionClasses
from landmarks.landmark_check_do import LandmarkCheck
from ocr.text_detection_do import TextDetection

class MyOptions(PipelineOptions):

  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument('--input',
                        help='Input for the pipeline',
                        default='./data/')
    # parser.add_argument('--output',
    #                     help='Output for the pipeline',
    #                     default='./output/')


def flaten_dict_list(element):
#   print(element[1])
  labels_flat = sum(element[1]["labels"], [])
  landmark_flat = element[1]["landmark"]
  text_flat = sum(element[1]["text"], [])
  return (element[0], {'labels': labels_flat,'text': text_flat, "landmark": landmark_flat})


options = PipelineOptions()
p = beam.Pipeline(options=options)

my_options = options.view_as(MyOptions)

d5_model = DetectionClasses('efficientdet_d5_coco17_tpu-32', 'label_map/mscoco_complete_label_map.pbtxt')
# mask_model = DetectionClasses("mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8", "label_map/mscoco_complete_label_map.pbtxt")
oid_model = DetectionClasses("ssd_mobilenet_v2_oid_v4_2018_12_12", "label_map/oid_v4_label_map.pbtxt")
landmark_check = LandmarkCheck()
text_detection = TextDetection()

read_image = (
    p |
    beam.io.ReadFromText(my_options.input)
)

d5_pcollection = (
    read_image
    | "d5" >> beam.ParDo(d5_model)
)

# mask_pcollection = (
#     read_image
#     | "mask" >> beam.ParDo(mask_model)
# )

oid_pcollection = (
    read_image
    | "oid" >> beam.ParDo(oid_model)
)

merged_labels = (
    # (d5_pcollection, mask_pcollection, oid_pcollection)
    (d5_pcollection, oid_pcollection)
    | 'MergedPColl' >> beam.Flatten()
)

# face_check = (
#     merged_labels
#     | beam.GroupByKey()
#     | beam.CombineValues(lambda values: list(set(sum(values, []))))
#     | "face_detection" >> beam.ParDo(landmark_check)
# )

landmark_pcollection = (
    read_image
    | "landmark" >> beam.ParDo(text_detection)
)

text_pcollection = (
    read_image
    | "text" >> beam.ParDo(landmark_check)
)

last = (
    ({'labels': merged_labels, 'text': text_pcollection,'landmark': landmark_pcollection})
    | beam.CoGroupByKey()
    | beam.Map(flaten_dict_list)
    | beam.Map(print)
)

# merged = (
#     p
#     | beam.Create([
#         ('ny', {'labels': [['person']], 'landmark': [0.9]}),
#         ('paris_1', {'labels': [['Tower'], ['person', 'car', 'bus', 'truck']], 'landmark': [0.9]}),
#         ('n_2', {'labels': [['Human face'], ['person']], 'landmark': []}),
#         ('y_1', {'labels': [['Woman', 'Clothing', 'Human face', 'Dress'], ['cup', 'cell phone', 'person', 'bottle']], 'landmark': []})
#     ])
# )

# beam.io.WriteToText("a.json")
p.run()
