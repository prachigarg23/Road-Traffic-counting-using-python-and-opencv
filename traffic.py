import os
import logging
import logging.handlers
import random
import numpy as np
import sys
import skvideo.io
import cv2
import matplotlib.pyplot as plt

import myutils #this is a python file containing support functions

# a few problematic interactions between the python bindings and OpenCL
# disabling OpenCL support in run-time to avoid problems
cv2.ocl.setUseOpenCL(False)
random.seed(123)

from pipeline import (
PipelineRunner,
ContourDetection,
Visualizer,
CsvWriter,
VehicleCounter
)

Image_dir = "./out"
Video_source = "Road_traffic_video_for_object_recognition.mp4"
Shape = (720,1280)  #height*width
Exit_pts = np.array([
                    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
                    [[0, 400], [645, 400], [645, 0], [0, 0]]
                    ])

def train_bg_subtractor(bg_object, cap, num=500):
    '''
    BG subtractor needs to process some number of frames to start giving results
    '''
    print("Training BG subtractor! ")
    i = 0
    for frame in cap:
        bg_object.apply(frame, None, 0.001)
        i += 1
        if i>= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creating an exit mask from points where we will be counting out vehicles
    base = np.zeros(Shape + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, Exit_pts, (255,255,255))[:,:,0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True)

    # processing pipeline for programming convenience
    pipeline = PipelineRunner(pipeline=[
    ContourDetection(bg_subtractor=bg_subtractor, save_image=True, image_dir=Image_dir),
    # we use y_weight == 2.0 because traffic are moving vertically on video
    # use x_weight == 2.0 for horizontal
    VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
    Visualizer(image_dir=Image_dir),
    CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    #setting up image source
    cap = skvideo.io.vreader(Video_source)

    # skip 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)
    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("frame capture failed, stopping...")
            break

        # real frame number
        _frame_number += 1
        #skipping every 2nd frame to speed up processing
        if _frame_number % 2!=0:
            continue
        frame_number += 1

        pipeline.set_context({
        'frame': frame,
        'frame_number': frame_number
        })
        pipeline.run()


# we check if the out image directory exists or not and call main function
if __name__ == "__main__":
    log = myutils.init_logging()

    if not os.path.exists(Image_dir):
        log.debug("Creating image directory '%s'....", Image_dir)
        os.makedirs(Image_dir)
    main()
