# --------------------------------------------------------
# Tensorflow drl-RPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
from time import sleep

from utils.timer import Timer
from utils.blob import im_list_to_blob
from utils.statcoll import StatCollector

from model.config import cfg, get_output_dir, cfg_from_list
from model.nms_wrapper import nms
from model.factory import run_drl_rpn, print_timings, get_image_blob, track_objects
from model.tracker import FlowTracker

from skimage.transform import resize as resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def im_detect(sess, net, im, timers, im_idx=None, nbr_gts=None):

  # Setup image blob
  blobs = {}
  blobs['data'], im_scales, blobs['im_shape_orig'] = get_image_blob(im)
  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]])

  # Run drl-RPN
  all_boxes, timers, stats \
    = run_drl_rpn(sess, net, blobs, timers, 'test', cfg.DRL_RPN_TEST.BETA,
                  im_idx, nbr_gts)

  return all_boxes, timers, stats

def im_predict(net):
  track_boxes = net.tracker.track_bboxes
  all_boxes = np.copy(track_boxes)
  # convert last column (id) into score
  for i in range(len(all_boxes)):
    all_boxes[i] = np.float32(all_boxes[i])
    if len(all_boxes[i]) > 0:
      all_boxes[i][:,4] = 0.71 # put score = 0.71 for tracked obj
  return all_boxes

def merge_detection(pred_boxes, track_boxes):
  all_boxes = [[] for _ in range(cfg.NBR_CLASSES)]
  for j in range(1, cfg.NBR_CLASSES):
    cls_dets = np.vstack((pred_boxes[j], track_boxes[j]))
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    all_boxes[j] = cls_dets
  return all_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.00):
  """Test a drl-RPN network on an image database."""

  # Set numpy's random seed
  np.random.seed(cfg.RNG_SEED)

  nbr_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  # all_boxes = [[[] for _ in range(nbr_images)] for _ in range(cfg.NBR_CLASSES)]
  all_boxes = [[[] for _ in range(cfg.NBR_CLASSES)] for _ in range(nbr_images)]
  obj_ids = [[[] for _ in range(cfg.NBR_CLASSES)] for _ in range(nbr_images)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  _t_drl_rpn = {'init': Timer(), 'fulltraj': Timer(),
                'upd-obs-vol': Timer(), 'upd-seq': Timer(), 'upd-rl': Timer(),
                'action-rl': Timer(), 'coll-traj': Timer()}
  avg_traj = 0.0
  avg_frac = 0.0

  # Create StatCollector (tracks various drl-RPN test statistics)
  stat_strings = ['#fix/img', 'exploration']
  sc = StatCollector(nbr_images, stat_strings, False)

  # Try getting gt-info if available
  try:
    gt_roidb = imdb.gt_roidb()
  except:
    gt_roidb = None

  # Visualize search trajectories?
  do_visualize = cfg.DRL_RPN_TEST.DO_VISUALIZE

  # Can be convenient to run from some other image, especially if visualizing,
  # but having nbr_ims_eval = nbr_images and start_idx = 0 --> regular testing!
  nbr_ims_eval = nbr_images
  start_idx = 0 
  end_idx = start_idx + nbr_ims_eval

  flow_tracker = FlowTracker(0)
  # Test drl-RPN on the test images
  for i in range(start_idx, end_idx):

    # Need to know image index if performing visualizations
    if do_visualize: im_idx = i
    else: im_idx = None

    # Try extracting gt-info for diagnostics (possible for voc 2007)
    if gt_roidb is None:
      nbr_gts = None
    else:
      try:
        nbr_gts = gt_roidb[i]['boxes'].shape[0]
      except:
        # if pickle transforms pascal voc keys in bytes
        nbr_gts = gt_roidb[i][b'boxes'].shape[0]

    # Detect!
    im = cv2.imread(imdb.image_path_at(i))
    
    _t['im_detect'].tic()

    if i < 5 or i%3==0:
      # run drl rpn
       
      prev_pred, _t_drl_rpn, stats = im_detect(sess, net, im, _t_drl_rpn,
                                                  im_idx, nbr_gts)
      all_boxes[i] = prev_pred
      # flow_tracker.prev_frame = im.astype(float)/255.
      # flow_tracker.prev_frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      # flow_tracker.track_memory.append(all_boxes[i])
      net.tracker.update(all_boxes[i])
      # net.tracker.track_memory.append(all_boxes[i])
      # Update and print some stats
      sc.update(0, stats)
      sc.print_stats(False)
    else:
      # run object tracker
      # det_boxes, _t_drl_rpn, stats = im_detect(sess, net, im, _t_drl_rpn,
      #                                             im_idx, nbr_gts)
      # print(pred_boxes)
      # prev_pred = net.tracker.produce_prev_boxes()
      # track_boxes = net.tracker.clean(prev_pred) # impose a lower score to old boxes
      # prev_pred = flow_tracker.produce_prev_boxes()
      # track_boxes = flow_tracker.predict(im, prev_pred)

      # net.tracker.track_memory.pop(0)
      # net.tracker.track_memory.append(det_boxes)
      # flow_tracker.track_memory.pop(0)
      # flow_tracker.track_memory.append(det_boxes)


      # track_boxes = im_predict(net)
      # print(track_boxes)
      # all_boxes[i] = merge_detection(det_boxes, track_boxes)
      # all_boxes[i] = merge_detection(det_boxes, track_boxes)
      all_boxes[i] = im_predict(net)
      net.tracker.update(all_boxes[i])
      # Update and print some stats
      # sc.update(0, stats)
      # sc.print_stats(False)
      # all_boxes[i] = im_predict(net)
      # net.tracker.update(all_boxes[i])
      #all_boxes[i] = [[] for _ in range(cfg.NBR_CLASSES)]

    _t['im_detect'].toc()

    _t['misc'].tic()

    if do_visualize:
      print('visualize_test')
      visualize(im, im_idx, all_boxes[i])
      

    # skip j = 0, because it's the background class
    # cls_dets_all = []
    # for j in range(1, cfg.NBR_CLASSES):
    #   inds = np.where(scores[:, j] > thresh)[0]
    #   cls_scores = scores[inds, j]
    #   cls_boxes = boxes[inds, j*4:(j+1)*4]
    #   cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
    #   keep = nms(cls_dets, cfg.TEST.NMS)
    #   cls_dets = cls_dets[keep, :]
    #   all_boxes[j][i] = cls_dets
    # #   for jj in range(cls_dets.shape[0]):
    # #     crop = np.squeeze(cls_dets[jj, :])
    # #     cls_dets_all.append(crop)
    # # cls_dets_all = np.array(cls_dets_all)

    # # Limit to max_per_image detections *over all classes*
    # if max_per_image > 0:
    #   image_scores = np.hstack([all_boxes[j][i][:, -1]
    #                 for j in range(1, cfg.NBR_CLASSES)])
    #   if len(image_scores) > max_per_image:
    #     image_thresh = np.sort(image_scores)[-max_per_image]
    #     for j in range(1, cfg.NBR_CLASSES):
    #       keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
    #       all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    
    # print("#############################")
    # print("ALL BOXES")
    # print(len(all_boxes), len(all_boxes[0]), len(all_boxes[0][15]), len(all_boxes[0][15][0]))
    # print("#############################")


    print('\nim_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, nbr_images, _t['im_detect'].average_time,
            _t['misc'].average_time))
    #print_timings(_t_drl_rpn) # uncomment for some timing details!
  
  all_boxes = list(map(list, zip(*all_boxes))) # transpose all_boxes to match imdb.evaluate_detections

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir, start_idx, end_idx)

def visualize(im, im_idx, boxes, show_text=True):

  # Setup image blob
  blob = {}
  blob['data'], im_scales, blob['im_shape_orig'] = get_image_blob(im)
  im_blob = blob['data']
  im_shape = blob['im_shape_orig']
  blob['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]])
  im_info = blob['im_info']

  # Make sure image in right range
  im = im_blob[0, :, :, :]
  im -= np.min(im)
  im /= np.max(im)
  im = resize(im, (im_shape[0], im_shape[1]), order=1, mode='reflect')

  # BGR --> RGB
  im = im[...,::-1]

  # Produce final detections post-NMS
  cls_dets, names_and_coords = produce_trusted_boxes(boxes, thresh=0.70)

  # Show image
  fig, ax = plt.subplots(1)
  ax.imshow(im)

# Draw all detection boxes
  for j in range(len(names_and_coords)):

    coords = names_and_coords[j]['coords']
    score = names_and_coords[j]['score']
    name = names_and_coords[j]['class_name']
    color = names_and_coords[j]['color']
    cls_det = cls_dets[j]

    # Show object category + confidence
    if show_text:
      ax.text(coords[0], coords[1], name + " " + str(score),
                weight='bold', color='black', fontsize=8,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='white', pad=-0.1))

    # Show detection bounding boxes
    rect = patches.Rectangle((cls_det[0], cls_det[1]), cls_det[2] - cls_det[0],
                              cls_det[3] - cls_det[1],
                              linewidth=7, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

  # Final save / close of figure
  im_name = 'im' + str(im_idx + 1) + '.jpg' 
  plt.savefig('img-out-test/' + im_name)
  plt.close()

  # Display success message
  print("Saved image " + im_name + "!\n")
  

def color_from_id(list_ids, seed=0):
  colors = []
  for id in list_ids:
    np.random.seed(int(id)) # to guarantee color consistency
    color_id = np.random.uniform(0, 1, size=(1,3))
    colors.append(color_id)
  return colors


def produce_trusted_boxes(boxes, thresh=0.80):
  class_names = ['bg',' aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car',
                 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'moto', 'person',
                 'plant', 'sheep', 'sofa', 'train', 'tv']
  
  cls_dets_all = []
  names_and_coords = []
  for j in range(1, cfg.NBR_CLASSES):
    cls_dets = boxes[j]
    cls_scores = boxes[j][:,4]
    keep = cls_scores > thresh
    cls_dets = cls_dets[keep]
    name = class_names[j]

    n_objects = cls_dets.shape[0]
    for jj in range(n_objects):
      crop = np.squeeze(cls_dets[jj, :])
      cls_dets_all.append(crop[:4])
      coords = [crop[0], crop[1]]
      names_and_coords.append({'coords': coords,
                               'score': round(crop[4], 2),
                               'class_name': name,
                               'color': np.array([0,0,0]),
                               })
  return cls_dets_all, names_and_coords
