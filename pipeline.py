import os
import logging
import csv
import numpy as np
import cv2
import utils

DIVIDER_COLOUR = (255,255,0)
BOUNDING_BOX_COLOUR = (255,0,0)
CENTROID_COLOUR = (0,0,255)
CAR_COLOURS = [(0,0,255)]
EXIT_COLOR = (66,183,42)

class PipelineRunner(object):

    #constructor
    def __init__(self, pipeline=None, log_level=logging.DEBUG):
        self.pipeline = pipeline or []
        self.context = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()

    def set_context(self, data):
        self.context = data

    def add(self, processor):
        if not isinstance(processor, PipelineProcessor):
            raise Exception('Processor should be an isinstance of PipelineProcessor.')
            processor.log.setLevel(self.log_level)
            self.pipeline.append(processor)

    def remove(self, name):
        for i, p in enumerate(self.Pipeline):
            if p.__class__.__name__ == name:
                del self.pipeline[i]
                return True
        return False

    def set_log_level(self):
        for p in self.pipeline:
            p.log.setLevel(self.log_level)

    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        self.log.debug("Frame #%d processed.", self.context['frame_number'])
        return self.context

class PipelineProcessor(object):
        '''
        Base class for processors.
        '''

        def __init__(self):
            self.log = logging.getLogger(self.__class__.__name__)

class ContourDetection(PipelineProcessor):
    '''
    This class provides functions to get foreground masks, and use them to extract objects that would be vehicles
    '''
    def __init__(self, bg_subtractor, min_contour_width=35, min_contour_height=35, save_image=False, image_dir='images'):
        super(ContourDetection, self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.save_image = save_image
        self.image_dir = image_dir

    def filter_mask(self, img, a=None):
        '''
        These filters are based on visual observations
        '''
        #defining the structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

        #fill any small holes by closing
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #remove noise by opening
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)

        return dilation


    def detect_vehicles(self, fg_mask, context):
        '''
        1. We get the bounding rectangles of each foreground object using contour information.
        2. These rectangles are filtered using the min height and weight.
        3. The valid contours along with their centers are returned.
        '''
        matches = []

        #finding only external contours
        #using Teh-Chin chain approximation algorithm (faster)
        im, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # filering the contours by width, height
        for (i, contour) in enumerate(contours):
            (x,y,w,h) = cv2.boundingRect(contour)
            contour_valid = (w>=self.min_contour_width) and (h>=self.min_contour_height)

            if not contour_valid:
                continue

            #if valid contour, get the center of the bounding box
            centroid = utils.get_centroid(x, y, w, h)
            matches.append(((x,y,w,h), centroid))
        return matches



    def __call__(self, context):
        '''
        This function does the following:
        1. Get all frames and frame numbers
        2. Apply background subtractor to get the foreground mask
        3. Threshold and apply filters to foreground mask
        4. Save frame
        5. Get the objects that are valid and add them to context['objects']
        '''
        frame = context['frame'].copy()
        frame_number = context['frame_number']

        # applying the background subtractor to the frame to get the foreground mask
        fg_mask = self.bg_subtractor.apply(frame, None, 0.001)
        # threshold images to get white foreground on blask background
        fg_mask[fg_mask<240] = 0
        # apply morphological filters to remove noise and fill holes, making foreground more consistent
        fg_mask = self.filter_mask(fg_mask, frame_number)

        if self.save_image:
            utils.save_frame(fg_mask, self.image_dir + "/mask_%04d.png" % frame_number, flip=False)

        context['objects'] = self.detect_vehicles(fg_mask, context)
        context['fg_mask'] = fg_mask
        return context

class VehicleCounter(PipelineProcessor):
    '''

    '''
    def __init__(self, exit_masks = [], path_size = 10, max_dst = 30, x_weight = 1.0, y_weight = 1.0):
        super(VehicleCounter, self).__init__()  #for inheriting other class functions

        self.exit_masks = exit_masks
        self.vehicle_count = 0
        self.path_size = path_size
        self.paths = []
        self.max_dst = max_dst
        self.x_weight = x_weight
        self.y_weight = y_weight

    def check_exit(self, point):
        # Here we check if the passed point lies in the exit zone or not
        for exit_mask in self.exit_masks:
            try:
                if exit_mask[point[1]][point[0]] == 255:
                    return True
            except:
                return True
            return False

    def __call__(self, context):
        objects = context['objects']
        context['exit_masks'] = self.exit_masks
        context['paths'] = self.paths
        context['vehicle_count'] = self.vehicle_count

        # We start following the algorithm
        # If there are no objects detected, return
        if not objects:
            return context

        # extract points
        points = np.array(objects)[:, 0:2]
        points = points.tolist()

        # add new points if paths is empty
        if not self.paths:
            for point in points:
                self.paths.append([point])

        else:
            # link each new point with an old path based on minimum distance between the points and the path points
            new_paths = []
            for path in self.paths:
                min_ = 999999
                match_ = None
                for p in points:
                    if len(path) == 1:
                        # distance from last point in the path to current
                        d = utils.distance(p[0], path[-1][0])
                    else:
                        # predict new point using last 2 points in path
                        # find min distance between predicted point and current point
                        xn = 2 * path[-1][0][0] - path[-2][0][0]
                        yn = 2 * path[-1][0][1] - path[-2][0][1]
                        d = utils.distance(p[0], (xn, yn), x_weight=self.x_weight, y_weight=self.y_weight)

                    if d < min_:
                        min_ = d
                        match_ = p
                # if a valid match has been found, remove it from the points list and add it to the paths
                if match_ and min_ <= self.max_dst:
                    points.remove(match_)
                    path.append(match_)
                    new_paths.append(path)


                # do not remove the path if current frame has no match for the path
                if match_ is None:
                    new_paths.append(path)

            self.paths = new_paths

            # add remaining points as new paths
            if len(points):
                for p in points:
                    # don't add those that should already be counted, i.e that are in the exit zones
                    if self.check_exit(p[1]):
                        continue
                    self.paths.append([p])

        # save only last N points in each path
        for i, _ in enumerate(self.paths):
            self.paths[i] = self.paths[i][self.path_size * -1:]

        # count vehicles and drop counted paths:
        new_paths = []
        for i, path in enumerate(self.paths):
            d = path[-2:]

            if (
                # need at list two points to count
                len(d) >= 2 and
                # prev point not in exit zone
                not self.check_exit(d[0][1]) and
                # current point in exit zone
                self.check_exit(d[1][1]) and
                # path len is bigger then min
                self.path_size <= len(path)
            ):
                self.vehicle_count += 1
            else:
                # prevent linking with path that already in exit zone
                add = True
                for p in path:
                    if self.check_exit(p[1]):
                        add = False
                        break
                if add:
                    new_paths.append(path)

        self.paths = new_paths

        context['paths'] = self.paths
        context['objects'] = objects
        context['vehicle_count'] = self.vehicle_count

        self.log.debug('#VEHICLES FOUND: %s' % self.vehicle_count)

        return context


class CsvWriter(PipelineProcessor):

    def __init__(self, path, name, start_time=0, fps=15):
        super(CsvWriter, self).__init__()

        self.fp = open(os.path.join(path, name), 'w')
        self.writer = csv.DictWriter(self.fp, fieldnames=['time', 'vehicles'])
        self.writer.writeheader()
        self.start_time = start_time
        self.fps = fps
        self.path = path
        self.name = name
        self.prev = None

    def __call__(self, context):
        frame_number = context['frame_number']
        count = _count = context['vehicle_count']

        if self.prev:
            _count = count - self.prev

        time = ((self.start_time + int(frame_number / self.fps)) * 100
                + int(100.0 / self.fps) * (frame_number % self.fps))
        self.writer.writerow({'time': time, 'vehicles': _count})
        self.prev = count

        return context


class Visualizer(PipelineProcessor):

    def __init__(self, save_image=True, image_dir='images'):
        super(Visualizer, self).__init__()

        self.save_image = save_image
        self.image_dir = image_dir

    def check_exit(self, point, exit_masks=[]):
        for exit_mask in exit_masks:
            if exit_mask[point[1]][point[0]] == 255:
                return True
        return False

    def draw_paths(self, img, paths):
        if not img.any():
            return

        for i, path in enumerate(paths):
            path = np.array(path)[:, 1].tolist()
            for point in path:
                cv2.circle(img, point, 2, CAR_COLOURS[0], -1)
                cv2.polylines(img, [np.int32(path)], False, CAR_COLOURS[0], 1)

        return img

    def draw_boxes(self, img, paths, exit_masks=[]):
        for (i, match) in enumerate(paths):

            contour, centroid = match[-1][:2]
            if self.check_exit(centroid, exit_masks):
                continue

            x, y, w, h = contour

            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1),
                          BOUNDING_BOX_COLOUR, 1)
            cv2.circle(img, centroid, 2, CENTROID_COLOUR, -1)

        return img

    def draw_ui(self, img, vehicle_count, exit_masks=[]):

        # this just add green mask with opacity to the image
        for exit_mask in exit_masks:
            _img = np.zeros(img.shape, img.dtype)
            _img[:, :] = EXIT_COLOR
            mask = cv2.bitwise_and(_img, _img, mask=exit_mask)
            cv2.addWeighted(mask, 1, img, 1, 0, img)

        # drawing top block with counts
        cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, ("Vehicles passed: {total} ".format(total=vehicle_count)), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return img

    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']
        paths = context['paths']
        exit_masks = context['exit_masks']
        vehicle_count = context['vehicle_count']

        frame = self.draw_ui(frame, vehicle_count, exit_masks)
        frame = self.draw_paths(frame, paths)
        frame = self.draw_boxes(frame, paths, exit_masks)

        utils.save_frame(frame, self.image_dir +
                         "/processed_%04d.png" % frame_number)

        return context
