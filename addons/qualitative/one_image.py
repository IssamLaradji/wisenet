import qualitative
import torch
from torch.utils import data
import torch.optim as optim
import datetime
import random
import timeit
import time
import test
import misc as ms
from losses import losses
import torchvision.transforms as transforms

import datasets, models
# from models import icarl as icarl_model
import pandas as pd 
import torch
import os
import os.path as osp
import datetime
import pytz
import yaml
import argparse

def one_image(main_dict):

  model_name = main_dict["model_name"]
  loss_dict = main_dict["loss_dict"]
  loss_name = main_dict["loss_name"]

  qualitative_dict = main_dict["qualitative_dict"]
  qualitative_name = main_dict["qualitative_name"]
  qualitative_function = qualitative_dict[qualitative_name]
  
  ms.print_welcome(main_dict)
  train_set, val_set = ms.load_trainval(main_dict)
  batch = ms.get_batch(val_set, indices=[1]) 

  model, opt, _ = ms.init_model_and_opt(main_dict)
  import ipdb; ipdb.set_trace()  # breakpoint 3fd81923 //

  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                opt=opt, epochs=20, visualize=True)



  
  qualitative_function(model, batch, image_id=0)