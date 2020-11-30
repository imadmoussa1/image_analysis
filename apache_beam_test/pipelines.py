# python3 -m pipelines --input list_images.txt --output /path/to/write/counts
import json

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from objects.b_od import ObjectBox


class MyOptions(PipelineOptions):

  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument('--input',
                        help='Input for the pipeline',
                        default='./data/')
    # parser.add_argument('--output',
    #                     help='Output for the pipeline',
    #                     default='./output/')


# class Split(beam.DoFn):
#   def process(self, element):
#     print(element)
#     return [{'image_name': str(element)}]

options = PipelineOptions()
p = beam.Pipeline(options=options)

my_options = options.view_as(MyOptions)

d5_model = ObjectBox('efficientdet_d5_coco17_tpu-32', 'label_map/mscoco_complete_label_map.pbtxt')
oid_model = ObjectBox("mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8", "label_map/mscoco_complete_label_map.pbtxt")

read_image = (
    p |
    beam.io.ReadFromText(my_options.input)
)

d5_pcollection = (
    read_image
    | "d5" >> beam.ParDo(d5_model)
    # | 'd5 output' >> beam.Map(json.dumps)
    # | beam.Map(print)
)

oid_pcollection = (
    read_image
    | "oid" >> beam.ParDo(oid_model)
    # | 'oid output' >> beam.Map(json.dumps)
    # | beam.Map(print)
)

merged = (
    (d5_pcollection, oid_pcollection)
    | 'MergedPColl' >> beam.Flatten()
)

# merged = p | beam.Create([
#     ('The-Eiffel-Tower-paris.jpg', ['truck', 'motorcycle', 'bus', 'person', 'car']),
#     ('The-Eiffel-Tower-paris.jpg', ['person', 'car', 'bus']),
#     ('ny.jpg', ['person']),
#     ('y_2.jpg', ['handbag', 'person', 'tv']),
#     ('y_2.jpg', ['chair', 'person']),
#     ('n_1.jpg', ['cell phone', 'person', 'clock', 'handbag']),
#     ('n_1.jpg', ['handbag', 'potted plant', 'clock', 'tie', 'person', 'cell phone'])
# ])

join = (
    merged
    | beam.GroupByKey()
    | beam.CombineValues(lambda values: list(set(sum(values, []))))
    | beam.Map(print)
)

# beam.io.WriteToText("a.json")
p.run()
