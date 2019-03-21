import torch
# import torch.optim as optim
import datetime
import misc as ms
import copy
import torch.nn.functional as F
# from models import icarl as icarl_model
import pandas as pd 
from addons.samplers import RandomN, FirstN
import numpy as np
import GPUtil
import threading
from threading import Thread
import datasets
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from torch.backends import cudnn
from threading import Thread
############ Train: Large Scale
from train_utils.train import fit, val
import os 

root = "/mnt/home/issam/Summaries/experiments/lifelong/"
def train(main_dict, reset=0, 
          n_train=100,
          n_val=-1,
          num_workers=1,
          multiprocess=True,
          borgy_running=False,
          debug_mode=0,
          summary_mode=0):
  if summary_mode:
    return summary(main_dict)

  cudnn.benchmark = False
  n_train = 10
  loader_dict = get_loaders(main_dict,
                            n_train=n_train,  
                            num_workers=num_workers,          
                            val_indices=None)
  if borgy_running:
    save = True
    multiprocess = True
  else:
    save = False
    multiprocess = False
  # model = ms.load_model(main_dict, fname=None)  

  if debug_mode:
    debug(main_dict,  loader_dict["train_loader"])
  else:
    train_eval(main_dict, loader_dict["train_loader"], save=save, multiprocess=multiprocess)


def summary(main_dict):
  fname = "%s_%s" % (main_dict["model_name"],
                      main_dict["dataset_name"])
  path = (root + "/%s/" % fname)

  fname = path + "history_val.yaml"
  if not os.path.exists(fname):
    print(None)
  else:
    history_val = ms.load_yaml(fname)
    # for s in summary[-10:]:
    #   print(s)
    print(fname) 
    print("python main.py -e %s -m sanity -s 1" % main_dict["exp_name"])
    import ipdb; ipdb.set_trace()  # breakpoint b282a875 //

    # print(summary["time_updated"])
    print(history_val["scoreList"][-1])
    print()


def debug(main_dict, train_loader):
  fname = "%s_%s" % (main_dict["model_name"],
                      main_dict["dataset_name"])
  path = (root + "/%s/" % fname)
  import ipdb; ipdb.set_trace()  # breakpoint a4a68bc9 //

  summary = ms.load_yaml(path + "summary.yaml")
  for s in summary[-10:]:
    print(s)
  import ipdb; ipdb.set_trace()  # breakpoint 34cedbd6 //
  batch = ms.get_batch(train_loader.dataset, [100] )
  model = ms.create_model(main_dict) 

  fit(main_dict, model, batch, n_iters=500)
  #model = ms.load_model_only(main_dict["path_save"] + "model.pth")
  # model.step(batch)
  # print("Model trained for %d iters" % model.history["iters"])
  # history = model.history
  # sanity_score = ("%d - loss: %.3f  - Best %s (%d): %.3f - Curr : %.3f" % 
  #         (history["iters"], history["train"][-1]["loss"] ,
  #          history["metric_name"], 
  #          history["val"]["best_score_dict"]["iters"],
  #          history["val"]["best_score_dict"]["score"],
  #          history["val"]["scoreList"][-1]["score"]))
  # print(sanity_score)
  # valEpoch(main_dict, model, 
  #            train_loader, model.history, verbose=0)
  valEpoch(main_dict, model, 
              val_loader, model.history, verbose=0)
  for i, batch in enumerate(train_loader): 
    # if 1:
    #   pred_annList = model.predict(batch, method="annList")                                                                                        
    #   gt_annList = batch["annList"]   
    #   eval_prec_recall.evaluate_annList(gt_annList[:2], gt_annList)
    #   _, _, H, W = torch.cat(batch["meta"]["shape"])
    #   pred_annList[1:2]
    #   gt_annList[:1]
    #   ms.images(F.interpolate(batch["images"], size=(H, W)), 
    #                      annList=pred_annList)

    import ipdb; ipdb.set_trace()  # breakpoint 36805a08 //

    model.visualize(batch)    
    # val(main_dict, model, batch)




def train_eval(main_dict, train_loader, save, multiprocess):
  sanity_list = []
  fname = "%s_%s" % (main_dict["model_name"],
                      main_dict["dataset_name"])
  path = (root + "/%s/" % fname)
  model = ms.load_model(main_dict, fname=None)  
  n_epochs = 1001
  verbose = 0
  sanity_dict = {}
  for e in range(n_epochs):
    # Validate
    if multiprocess:
      val_thread = Thread(target=valEpoch, 
                                args=(main_dict, 
                                      copy.deepcopy(model), 
                                      train_loader,  
                                      model.history,
                                      None,
                                      verbose))
      val_thread.start()
    else:
      valEpoch(main_dict, model, 
              train_loader, model.history, None, verbose)

    # Training
    fitEpoch(main_dict, model, train_loader, verbose=0)

    if multiprocess:
      val_thread.join()
    assert len(model.history["val"]["scoreList"]) == e + 1

    # # Validate
    # valEpoch(main_dict, model, 
    #          train_loader, model.history, verbose=0)
    
    if e % 50 == 0 and save:
      ms.save_model_only(main_dict["path_save"] + "model.pth", model)

    history = model.history
    sanity_score = ("%d/%d- loss: %.3f  - Best %s (%d): %.3f - Curr : %.3f - time: %s" % 
            (e, n_epochs, history["train"][-1]["loss"] ,
             history["metric_name"], 
             history["val"]["best_score_dict"]["iters"],
             history["val"]["best_score_dict"]["score"],
             history["val"]["scoreList"][-1]["score"],
             ms.time_to_montreal()))

    print(sanity_score)
    sanity_list += [sanity_score]

    # # visualize
    for i, batch in enumerate(train_loader):
      figure1 = model.visualize(batch, method="pred", return_image=True)
      figure2 = model.visualize(batch, method="gt", return_image=True)
      figure = np.concatenate([figure1, figure2], axis=2)

      figure = ms.resize_image(ms.f2l(figure[0]), min_dim=300, max_dim=500, padding=True)
      ms.imsave(path + "%s.png"%i, figure)

    print("images saved in: %s" % path)

    # sanity_dict["scoreList"] = sanity_list
    # sanity_dict["val_scoreList"] = history["val"]["scoreList"]
    ms.save_yaml(path + "history_val.yaml" , history["val"])


   



def fitEpoch(main_dict, model, train_loader, verbose=1):
  n_batches = len(train_loader)
  history = model.history

  for i, batch in enumerate(train_loader):
    
    loss = model.step(batch)

    GPUs = GPUtil.getGPUs()

    if len(GPUs) != 0:
      gpu = GPUs[0]
      gpu_load = gpu.load*100
      gpu_mem = gpu.memoryUtil*100
    if verbose:
      print("%d/%d - seen: %d - load: %d%% - mem:%d%%" \
            "  - loss: %.3f - Best %s (%d): %.3f" % 
            (history["iters"], n_batches,
              len(history["seen_image_ids"]), 
              gpu_load, gpu_mem, 
              loss, 
             history["metric_name"], 
             history["val"]["best_score_dict"]["iters"],
             history["val"]["best_score_dict"]["score"]))
      
    history["train"] += [{"loss":loss, "iter":history["iters"]}]
    history["iters"] += 1
    history["seen_image_ids"].add(batch["meta"]["image_id"][0])
    # history["seen_image_ids"].add(batch["meta"]["image_id"][0])



@torch.no_grad()
def valEpoch(main_dict, model, val_loader,  
             history, save_fname=None, verbose=1):
  # Validate

  n_batches_val = len(val_loader)
  history_val = history["val"]
  iters = history["iters"]

  metric_class = main_dict["metric_dict"][main_dict["metric_name"]]
  metric_object = metric_class()

  for i, batch in enumerate(val_loader):
    metric_object.addBatch(model, batch)
    if verbose:
      print("%d/%d - validating - Best %s: %.3f" % (i+1, 
                  n_batches_val, 
                  main_dict["metric_name"], 
                  history_val["best_score_dict"]["score"]))

  score_dict = metric_object.compute_score_dict()
  score_dict["iters"] = iters

  history_val["scoreList"] += [score_dict]
  # print("best_flag", score_dict["best_flag"])

  if metric_object.is_best_score_dict(history_val["best_score_dict"]):
    score_dict["path"] = main_dict["path_save"] + "%s_best" % save_fname
    history_val["best_score_dict"] = score_dict
    
    if save_fname is not None:
      ms.save_model(main_dict, 
                    fname="%s_best" % save_fname,
                    model=model)


def get_loaders(main_dict, n_train,
                     num_workers=1, val_indices=None):
  # load datasets

  dataset_name = main_dict["dataset_name"]

  train_set =  ms.load_dataset(main_dict, 
                               dataset_name=dataset_name, split="val")

 
  ###############
  # Samplers
  sampler_train = FirstN(train_set, 
        indices=np.random.choice(len(train_set),
               n_train, 
               replace=False) )

  # loaders
  train_loader = torch.utils.data.DataLoader(
              train_set, 
              collate_fn=datasets.base_dataset.collate_fn_0_4,
               batch_size=1, sampler=sampler_train,
               shuffle=False, num_workers=num_workers, pin_memory=True)

 
  print("\n%s (model)- %s (loss) - %s" % 
        (main_dict["model_name"],
             main_dict["loss_name"],
                                                            dataset_name))
  return {"train_loader":train_loader}

def update_summary(main_dict, model):
  lossList = pd.DataFrame(model.history["train"])
  if len(lossList) != 0:
      loss = np.array(lossList["loss"])
  else:
      loss = np.array([-1]) 

  scoreList = pd.DataFrame(model.history["val"]["scoreList"])
  if len(scoreList) != 0:

      scores = np.array(scoreList["score"])
  else:
      scores = np.array([-1]) 

  best_score_dict = model.history["val"]["best_score_dict"]
  best_score = float(best_score_dict["score"])
  best_iters = model.history["val"]["best_score_dict"]["iters"]
  metric_name = main_dict["metric_name"]

  if "per_class" not in best_score_dict:
    per_class_str = -1
  else:
    per_class = best_score_dict["per_class"]
    categoryList = model.history["categoryList"] 
    per_class_str = ["%s-%.3f"% (s, p) for s, p in zip(categoryList, per_class)]
  model.history["summary"]["time_updated"] = ms.time_to_montreal()
  model.history["summary"]["best_iters"] = "%d - %d" % (best_iters, model.history["iters"])
  model.history["summary"].update(
      {"best_%s" % (metric_name): "%.3f" % best_score, 
              "best_score": "%.3f" % best_score,
              "best_score_per_class": per_class_str,
              "metric_name":main_dict["metric_name"],
              "iters":model.history["iters"],
              "loss":"%.3f" % loss[-1],
              "loss_max":"%.3f" % loss.max(),
              "loss_min":"%.3f" % loss.min(),
              "loss_mean":"%.3f" % loss.mean(),
              "score_min":"%.3f" % scores.min(),
              "score_max":"%.3f" % scores.max(),
              "score_mean":"%.3f" % scores.mean(),
              "model_name":main_dict["model_name"],
              "config_name":main_dict["config_name"],
              "borgy_running":main_dict["borgy_running"]})


############### debug ###############


# @torch.no_grad()
# def val(main_dict, model, batch):
#   metric_name = main_dict["metric_name"]

#   metric_class = main_dict["metric_dict"][metric_name]
#   metric_object = metric_class()

#   if isinstance(batch, dict):
    
#     metric_object.addBatch(model, batch)

#     score_dict = metric_object.compute_score_dict()

#     print("%s: %.3f" % (metric_name, score_dict["score"]))

#   else:
#     dataset = batch

#     n_batches = len(dataset)
#     for i in range(n_batches):
#       batch = ms.get_batch(dataset, [i])
#       print("%d/%d - validating" % (i+1, n_batches))

#       metric_object.addBatch(model, batch)

#     score_dict = metric_object.compute_score_dict()

#   return score_dict
