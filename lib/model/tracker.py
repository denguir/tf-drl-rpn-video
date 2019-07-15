
import numpy as np 
import cv2
from model.sort import Sort
from model.config import cfg
from model.pyflow import pyflow
from model.nms_wrapper import nms

class Tracker(object):
    def __init__(self):
        self.n_classes = cfg.NBR_CLASSES
        # bboxes obtained after last detection process:
        self.det_bboxes = [[] for _ in range(self.n_classes)]
        # bboxes obtained after last tracking process:
        self.track_bboxes = [[] for _ in range(self.n_classes)]
        # color tracking:
        self.track_colors = [{} for _ in range(self.n_classes)]
        self.score = [{} for _ in range(self.n_classes)]
        self.track_memory = []
        self.trackers = self.init_trackers()

    def init_trackers(self, max_age=4, min_hits=3):
        trackers = []
        for _ in range(self.n_classes):
            tracker_cls = Sort(max_age=max_age, min_hits=min_hits)
            trackers.append(tracker_cls)
        return trackers
    
    def update(self, bboxes):
        """ bboxes is supposed to have shape (n_classes, n_objects, 5)
        5 being 4 coords + score"""
        for cls, tracker in enumerate(self.trackers):
            tracked_obj = tracker.update(bboxes[cls])
            self.track_bboxes[cls] = tracked_obj
            id_obj = tracked_obj[:,4]
            for id in id_obj:
                if id not in self.track_colors[cls].keys():
                    self.track_colors[cls][id] =  np.random.uniform(0, 1, size=(1,3))

    def clean(self, track_bboxes):
        all_boxes = np.copy(track_bboxes)
        # convert last column (id) into score
        for i in range(len(all_boxes)):
            all_boxes[i] = np.float32(all_boxes[i])
            if len(all_boxes[i]) > 0:
                all_boxes[i][:,4] = 0.71 # put score = 0.71 for tracked obj
        return all_boxes

    def produce_prev_boxes(self):
        all_boxes = [[] for _ in range(cfg.NBR_CLASSES)]
        for j in range(1, cfg.NBR_CLASSES):
            cls_dets = np.vstack((self.track_memory[0][j], self.track_memory[1][j], 
                                    self.track_memory[2][j]))
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j] = cls_dets
        return all_boxes

class FlowTracker(object):
    def __init__(self, frame):
        self.n_classes = cfg.NBR_CLASSES
        # bboxes obtained after last detection process:
        self.det_bboxes = [[] for _ in range(self.n_classes)]
        # bboxes obtained after last tracking process:
        self.track_bboxes = [[] for _ in range(self.n_classes)]
        self.prev_frame = frame
        self.track_memory = []

    def produce_prev_boxes(self):
        all_boxes = [[] for _ in range(cfg.NBR_CLASSES)]
        for j in range(1, cfg.NBR_CLASSES):
            cls_dets = np.vstack((self.track_memory[0][j], self.track_memory[1][j], 
                                    self.track_memory[2][j]))
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j] = cls_dets
        return all_boxes

    
    def predict(self, new_frame, bboxes):
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        # new_frame = new_frame.astype(float) / 255.
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        maxY, maxX = new_frame.shape[:2]
        # vx, vy, _ = pyflow.coarse2fine_flow(
        #             self.prev_frame, new_frame, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        #             nSORIterations, colType)
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame,new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vx, vy = flow[..., 0], flow[..., 1]
        track_bboxes = self.track(bboxes, vx, vy, maxX, maxY)
        self.prev_frame = new_frame
        return track_bboxes

    def track(self, all_bboxes, vx, vy, maxX, maxY):
        track_bboxes = np.copy(all_bboxes)
        for cls, bboxes in enumerate(track_bboxes):
            for i, coords in enumerate(bboxes):
                vx1 = vx[int(coords[1]), int(coords[0])]
                vx2 = vx[int(coords[3]), int(coords[2])]
                vy1 = vy[int(coords[1]), int(coords[0])]
                vy2 = vy[int(coords[3]), int(coords[2])]

                x1 = coords[0] + vx1
                y1 = coords[1] + vy1
                x2 = coords[2] + vx2
                y2 = coords[3] + vy2
                score = 0.71

                if ((x1 < 0 or x1 > maxX-1) or 
                    (y1 < 0 or y1 > maxY-1) or 
                    (x2 < 0 or x2 > maxX-1) or 
                    (y2 < 0 or y2 > maxY-1)):
                    np.delete(track_bboxes[cls], i, 0)
                else:
                    track_bboxes[cls][i][0] = x1
                    track_bboxes[cls][i][1] = y1
                    track_bboxes[cls][i][2] = x2
                    track_bboxes[cls][i][3] = y2
                    track_bboxes[cls][i][4] = score

        return track_bboxes

