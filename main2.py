#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################################################################
# Programı çalıştırma komutu : python3 main2.py imshow  #
# Programı çalıştırma ve çıktıyı alma komutu: python3 main2.py imwrite #
#############################################################################################

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
from packaging import version

from collections import defaultdict
from io import StringIO
from PIL import Image

# Nesne Tanıma importları
from utils import label_map_util
from utils import visualization_utils as vis_util

# csv dosyalarını çağırma
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])

# kullanılacak video
source_video = 'input_video2.mp4'
cap = cv2.VideoCapture(source_video)


# Değişkenler
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

total_passed_vehicle = 0  # araç sayımında kullanılıyor

# "SSD with Mobilnet" Modeli kullanıldı
# Tensorflow üzerinden indirilmesi
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Frozen detection graph yolu tanımlaması. Nesne tanıma için kullanılan asıl model
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Her kutucuğa doğru etiket verilmesi için gereken değişkenler.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Hafızaya Tenserflow modeli atma
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef() 
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Label maps kategorileri bulundurur. Convulution network 5 tahmin ederse bu map ile bunun Airplane olduğu bilinir.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


# Tanımlama
def object_detection_function(command):
    total_passed_vehicle = 0
    speed = '--------'
    direction = '--------'
    size = '--------'
    color = '--------'

    if(command=="imwrite"):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(source_video.split(".")[0]+'_output.avi', fourcc, fps, (width, height))

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess: 

            # detection_graph için giriş ve çıkış tensorları
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Her bir kutucuk nesne tanımlandığı zaman resmin bir kısmını ifade eder.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Her score tanımlanan nesne için ihtimal oranını temsil eder.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Giriş videosundan alınan her bir frame için
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('Video sonu...')
                    break

                input_frame = frame

                # Boyut ayarlaması. Girilen videonun belirli bir kalıpta olması için
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Asıl tanımlama kısmı
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                # Tanımlama sonucunun görselleştirilmesi
                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )

                total_passed_vehicle = total_passed_vehicle + counter

                # Bulunan değerlerin video üzerine işlenmesi
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Arac Sayisi: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # Bir araç Region of Interest (ROI) çizgisini geçtiğinde çizginin yeşil olması
                
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (1270, 200), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, 200), (1270, 200), (0, 0, 0xFF), 5)

                # Verilerin Video üzerine işlenmesi
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'ROI Cizgisi',
                    (545, 190),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'En Son Gecen Arac Bilgisi',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    '-Hareket Yonu: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-HIZ(km/h): ' + str(speed).split(".")[0],
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Renk: ' + color,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Arac Boyutu/Tip: ' + size,
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                if(command=="imshow"):
                    cv2.imshow('vehicle detection', input_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                elif(command=="imwrite"):
                    output_movie.write(input_frame)
                    print("Frame yazdiriliyor...")

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, color, direction, speed) = \
                            csv_line.split(',')
                        writer.writerows([csv_line.split(',')])
            cap.release()
            cv2.destroyAllWindows()


import argparse

parser = argparse.ArgumentParser(description='Tensorflow ile Araç Sayma')
parser.add_argument("command",
                    metavar="<command>",
                    help="'imshow' or 'imwrite'")
args = parser.parse_args()
object_detection_function(args.command)		
