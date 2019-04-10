from roi_data_layer.layer import RoIDataLayer
from model.train_val import get_training_roidb, train_net
from datasets.factory import get_imdb
from model.config import cfg
import datasets.imdb
import cv2
import numpy as np

def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':
    imdb, roidb = combined_roidb('paris_train')
    data_layer = RoIDataLayer(roidb, cfg.NBR_CLASSES) 
    seqBlobs = data_layer.forward() # 3 keys: data, gt_boxes, im_info
    for fnum in range(cfg.TRAIN.SEQ_LENGTH):
      blobs = {'data': seqBlobs['data'][fnum,:,:,:],
              'gt_boxes': seqBlobs['gt_boxes'][fnum],
              'im_info': seqBlobs['im_info']}
      blobs['data'] = np.expand_dims(blobs['data'], axis=0)
      print(blobs['data'].shape)
      img = blobs['data'][0,:,:,:]
      cv2.imwrite('img%d.jpg'%fnum, img)
      cv2.imshow('img%d'%fnum, img)
      cv2.waitKey(0)

