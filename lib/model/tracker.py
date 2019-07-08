
from model.sort import Sort
from model.config import cfg
import numpy as np


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
        self.trackers = self.init_trackers()

    def init_trackers(self, max_age=1, min_hits=2):
        trackers = []
        for _ in range(self.n_classes):
            tracker_cls = Sort(max_age=max_age, min_hits=min_hits)
            trackers.append(tracker_cls)
        return trackers
    
    def update(self, bboxes):
        """ bboxes is supposed to have shape (n_classes, n_objects, 5)
        5 being 4 coords + score"""
        for cls, tracker in enumerate(self.trackers):
            scores = bboxes[cls][:,4]
            tracked_obj = tracker.update(bboxes[cls])
            self.track_bboxes[cls] = tracked_obj
            id_obj = tracked_obj[:,4]
            for id in id_obj:
                if id not in self.track_colors[cls].keys():
                    self.track_colors[cls][id] =  np.random.uniform(0, 1, size=(1,3))
