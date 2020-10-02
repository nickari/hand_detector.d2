import argparse
import cv2

import logging
import os
import time
from collections import OrderedDict
from tqdm import tqdm
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Boxes, BoxMode
import random

cfg = get_cfg()
cfg.merge_from_file("faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
cfg.MODEL.WEIGHTS = 'models/model_0529999.pth' # add model weight here
# cfg.MODEL.WEIGHTS = 'models/model_0974999.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 0.5 , set the testing threshold for this model

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

if rval:
    predictor = DefaultPredictor(cfg)


while rval:
    
    im = frame
    start = time.time()
    outputs = predictor(im)
    end = time.time()
    print('======================')
    print(f"inference took {end - start} seconds")
    print('======================')
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    z = outputs["instances"].pred_boxes
    s = outputs["instances"].scores
    z_conv = BoxMode.convert(z.tensor, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS)

    for idx, box in enumerate(z_conv):
        print(f"Detection #{idx}: {box}")
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        print(s[idx].item())
        text = "%s (%s)" % (idx, round(s[idx].item(), 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
