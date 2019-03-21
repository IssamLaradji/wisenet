import misc as ms
import torch
from addons.pycocotools import mask as maskUtils
from addons import pycocotools
from addons.pycocotools.cocoeval import COCOeval 
from addons.pycocotools.coco import COCO 
import numpy as np
import os 
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def qualitative(main_dict):
  #ud.debug_sheep(main_dict)
  # model_name = main_dict["model_name"]
  # loss_dict = main_dict["loss_dict"]
  # loss_name = main_dict["loss_name"]

  qualitative_dict = main_dict["qualitative_dict"]
  qualitative_name = main_dict["qualitative_name"]
  qualitative_function = qualitative_dict[qualitative_name]

  ms.print_welcome(main_dict)
  train_set, val_set = ms.load_trainval(main_dict)
  # batch = ms.get_batch(val_set, indices=[1]) 
  # model = ms.load_best_model(main_dict)
  model = ms.load_latest_model(main_dict)
  #batch = ms.get_batch(train_set, indices=np.arange(100))
  # model, opt, _ = ms.init_model_and_opt(main_dict)

  test_qualitative(model, val_set, qualitative_function)

@torch.no_grad()
def test_qualitative(model, test_set, qualitative_function, history=None,
                     n_images=5):
    if history is not None:
        print("Model from training for %d epochs." % history["epoch"])
        
    qualitative_name = qualitative_function.__name__
    n_batches = len(test_set)
    link = ms.get_ngrok()
    for i in range(n_batches):
        batch = ms.get_batch(test_set, [i])
        print("%d/%d - %s - %s" % (i, n_batches, qualitative_name, link))

        if n_images is not None and i > n_images:
            break
        
        qualitative_function(model, batch, image_id=i)
        