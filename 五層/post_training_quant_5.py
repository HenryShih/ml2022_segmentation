import cv2
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import tensorflow as tf

from pathlib import Path


def representative_dataset():
  for data in Path('../../Testing_Data_for_Qualification/').glob('*.jpg'):
    img = cv2.imread(str(data))
    img = np.expand_dims(img,0)
    img = img.astype(np.float32)
    yield [img]


converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = 'depth5.pb',
    input_arrays = ['Placeholder'],
    input_shapes = {'Placeholder':[ 1, 1080, 1920, 3]},
    output_arrays = ['ArgMax'],
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()
open('quantized_depth5.tflite', 'wb').write(tflite_quant_model)