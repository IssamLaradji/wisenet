import torch
from torch.utils import data
# import torch.optim as optim
import datetime
import random
import timeit
import time
import test
import misc as ms
from losses import losses
import torchvision.transforms as transforms
import torch.nn.functional as F
import datasets, models
import copy
# from models import icarl as icarl_model
import pandas as pd 
import torch
import os
import os.path as osp
import torch.nn as nn
import datetime
import pytz
import yaml
import argparse
import ann_utils as au
from metrics import metrics
from addons.samplers import RandomN, FirstN
import numpy as np
import GPUtil
import threading
from threading import Thread
from train_utils import train_da
############ Debug: Large Scale
def da_debug(main_dict, reset=0, 
          n_train=100,
          n_val=-1,
          num_workers=1,
          multiprocess=True,
          borgy_running=False):

  save_fname = "/debug/da_train"
  loader_dict = train_da.get_loaders(main_dict,
                            n_train=n_train,  
                            n_val = n_val,
                            num_workers=num_workers,          
                            val_indices=[8, 3, 12])

  model = ms.load_model(main_dict, 
                        fname="%s_latest" % save_fname, 
                        reset=reset)  

  fit_and_validate(main_dict, model,
                  loader_dict["train_src_loader"], 
                  loader_dict["train_tgt_loader"],
                  loader_dict["val_tgt_loader"], 
                  multiprocess=multiprocess,
                  save_fname=save_fname,
                  vis_flag=False)




def fit_and_validate(main_dict, model,
                     train_src_loader, 
                     train_tgt_loader,
                     val_loader, 
                     multiprocess=False,
                     save_fname=None,
                     vis_flag=False):
       
    if 1:
      batch_src = ms.get_batch(train_src_loader.dataset, [8])
      batch_tgt = ms.get_batch(val_loader.dataset, [8])

      # ms.images(batch_src["images"], batch_src["labels"].long(), denorm=1)

      # ms.images(batch_src["images"], model.predict(batch_src, method="annList_seg"), denorm=1,win="pred")
      # ms.images(batch_src["images"], denorm=1)
      # ms.images(batch_src["maskClasses"])
      import ipdb; ipdb.set_trace()  # breakpoint f93b7a6d //

      val(main_dict, model, batch_tgt)
      val(main_dict, model, train_src_loader.dataset)

      fit(main_dict, model, (batch_src, batch_tgt), n_iters=5)
      model.predict(batch_src, method="annList_seg")
      
      ms.images(batch_src["images"],
                     au.annList2mask(batch_src["annList"])["mask"], 
                     denorm=1)

      ms.images(batch_src["images"],
                     model.predict(batch_src, method="maskClasses"), 
                     denorm=1)
      ms.images(batch_src["images"],
                     batch_src["points"], 
                     enlarge=1,
                     denorm=1)
  
    model.eval()
    val_thread = None

    for e in range(1000):
      # Validation
      if multiprocess:
        val_thread = Thread(target=train_da.valEpoch, 
                            args=(main_dict, 
                                  copy.deepcopy(model), 
                                  val_loader, 
                                  model.history["val"], 
                                  save_fname))
        val_thread.start()
      else:
        train_da.valEpoch(main_dict, copy.deepcopy(model), val_loader, 
                          model.history["val"], save_fname)

      # Training
      train_da.fitEpoch(main_dict, model, 
               train_src_loader, train_tgt_loader)
      
      ms.save_model(main_dict,  
                    fname="%s_latest" % save_fname,
                    model=model)

      # Wait for validation      
      if val_thread is not None:
        val_thread.join()

      # # Visualize
      if vis_flag:
        vis(main_dict, model, val_loader, win_name="val")
        vis(main_dict, model, train_src_loader, win_name="train", n_images=3)




def fit(main_dict, model, batch, n_iters=1):
  loss_function = main_dict["loss_dict"][main_dict["loss_name"]]
  for i in range(n_iters):
    loss = model.step(batch, 
                    loss_function=loss_function)
    print("%d - loss: %.3f" % (i, loss))

@torch.no_grad()
def val(main_dict, model, batch):
  metric_name = main_dict["metric_name"]

  metric_class = main_dict["metric_dict"][metric_name]
  metric_object = metric_class()

  if isinstance(batch, dict):
    
    metric_object.addBatch(model, batch)

    score_dict = metric_object.compute_score_dict()

    print("%s: %.3f" % (metric_name, score_dict["score"]))

  else:
    dataset = batch

    n_batches = len(dataset)
    for i in range(n_batches):
      batch = ms.get_batch(dataset, [i])
      print("%d/%d - validating" % (i+1, n_batches))

      metric_object.addBatch(model, batch)

    score_dict = metric_object.compute_score_dict()

  return score_dict

@torch.no_grad()
def vis(main_dict, model, data_loader, win_name, n_images=None):
  # Validate
  for i, batch in enumerate(data_loader):  
    pred_annList = model.predict(batch, method="annList_seg")

    if n_images is not None and i > n_images:
      break

    gt_mask = au.annList2mask(batch["annList"])["mask"]
    pred_mask = au.annList2mask(pred_annList)["mask"]

    if pred_mask is None:
      pred_mask = torch.zeros(gt_mask.shape).long()

    vis_image = torch.cat([batch["images"], batch["images"]], 
                        dim=2)
    vis_mask = torch.cat([pred_mask, gt_mask], dim=0)
    ms.images(vis_image, 
                vis_mask,
                    denorm=1,  win="%s_gt_pred_%d"%(win_name, i))

