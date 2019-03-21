import torch
# import torch.optim as optim

import misc as ms
import copy
import os
# from models import icarl as icarl_model
import pandas as pd 
from addons.samplers import RandomN, FirstN
import numpy as np
import GPUtil
from threading import Thread
import datasets
from torch.backends import cudnn
from addons.pycocotools import eval_funcs
############ Train: Large Scale
def train(main_dict, reset=0, 
          n_train=100,
          n_val=-1,
          num_workers=1,
          multiprocess=True,
          borgy_running=False,
          debug_mode=0,
          summary_mode=0,
          qualitative_mode=0,
          visualize_mode=0):

  if summary_mode:
    return summary(main_dict)

  if borgy_running:
    save_fname = "train"
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    n_train=-1
    multiprocess = False
    num_workers = 0
    n_val=-1

  else:
    # torch.backends.cudnn.enabled = False
    n_train=-1
    multiprocess = multiprocess
    save_fname = "/tmp/train"
    reset = 1
    num_workers = 0


  loader_dict = get_loaders(main_dict,
                            n_train=n_train,  
                            n_val = n_val,
                            num_workers=num_workers,          
                            val_indices=None)


  if visualize_mode:
    visualize(main_dict, 
              loader_dict["train_loader"], 
              loader_dict["val_loader"] )

  elif debug_mode:
    debug(main_dict,
                    loader_dict["train_loader"], 
                    loader_dict["val_loader"], 
                    multiprocess=multiprocess,
                    save_fname=save_fname)

  else:
    model = ms.load_model(main_dict, 
                          fname="%s_latest" % save_fname, 
                          reset=reset)  

    model.history["summary"]["n_train"] = len(loader_dict["train_loader"].dataset)
    model.history["summary"]["n_val"] = len(loader_dict["val_loader"].dataset)

    model.history["summary"]["train_set"] = type(loader_dict["train_loader"].dataset).__name__
    model.history["summary"]["val_set"] = type(loader_dict["val_loader"].dataset).__name__

    fit_and_validate(main_dict, model,
                    loader_dict["train_loader"], 
                    loader_dict["val_loader"], 
                    multiprocess=multiprocess,
                    save_fname=save_fname)


# @torch.jit.script
# def get_preds(model, val_loader):
#   annList = []
#   for batch in val_loader:
#     annList += model.predict(batch, "annList")
    

def effect_of_k(main_dict, model, val_loader):
    ### How does K effect
    results = {}
    for n_bg_seeds in np.arange(1, 15):
      model.n_bg_seeds = n_bg_seeds
      results[n_bg_seeds] = valEpoch(main_dict, model, val_loader,  
             model.history, save_fname=None)

    n_bg_seeds = None
    model.n_bg_seeds = n_bg_seeds
    results[n_bg_seeds] = valEpoch(main_dict, model, val_loader,  
             model.history, save_fname=None)

    return results

def debug(main_dict,
         train_loader, 
         val_loader,  
         multiprocess=True,
         save_fname=None):
    # history = ms.load_model(main_dict, fname="train_latest", 
    #                         history_only=True)
    model = ms.create_model(main_dict) 
    import ipdb; ipdb.set_trace()  # breakpoint 738c19f9 //

    results = effect_of_k(main_dict, model, val_loader)

    
    # batch = ms.get_batch(val_loader.dataset, [100])

    import ipdb; ipdb.set_trace()  # breakpoint bb22a0ee //


    # model.n_bg_seeds = 1; model.visualize(batch)
    valEpoch(main_dict, model, val_loader,  
             model.history, save_fname=None)
    # history = ms.load_json()
    import ipdb; ipdb.set_trace()  # breakpoint 81b76b00 //
    # gt_ann_tmp = []
    # for i, batch in enumerate(val_loader):
    #   print(i)
    #   gt_ann_tmp += batch["annList"]
    # # pred_annList = ms.load_pkl(base + "pred_annList.pkl")
    # # gt_annList = ms.load_json('/mnt/datasets/public/issam/kitti//annotations/val_gt_annList.json')
    # eval_funcs.evaluate_annList(pred_annList=pred_annList,
    #                                           gt_annList=gt_annList["annotations"],
    #                                           iouType="segm",
    #                                           iouThr_list=(0.5,),
    #                                           ap=1)
    # au.get_perSizeResults(gt_annList, pred_annList)

    import ipdb; ipdb.set_trace()  # breakpoint a9efe681 //
    
    # model = ms.load_model(main_dict, 
    #                           fname="train_best", 
    #                           reset=0)  

    # model = ms.create_model(main_dict) 
    val(main_dict, model, batch)
    valEpoch(main_dict, model, val_loader,  
             model.history, save_fname=None)
    

    batch = ms.get_batch(val_loader.dataset, [100])
    import ipdb; ipdb.set_trace()  # breakpoint 65447497 //

    annList = model.predict(batch, method="annList") 
    model.visualize(batch)
    fit(main_dict, model, batch, n_iters=500)
    val(main_dict, model, batch)
    fit(main_dict, model, batch, n_iters=500)
    
    import ipdb; ipdb.set_trace()  # breakpoint f33edd08 //
    val_set =  ms.load_dataset(main_dict, 
                             dataset_name="PascalOriginal", split="val")
    batch = ms.get_batch(val_set, [201])
    annList = model.predict(batch, method="annList")
    fit(main_dict, model, batch, n_iters=500)
    fitEpoch(main_dict, model, train_loader)

    valEpoch(main_dict, model, val_loader,  
             model.history, save_fname=None)
    

    ms.images(batch["images"], 
              au.bbox2mask(model.predict(batch, method="bbox").bbox, batch["images"].shape,mode="xyxy"),  
              denorm=2, 
              win="pred")      

    ms.images(batch["images"],
              au.bbox2mask(batch["targets"][0].bbox, batch["images"].shape, mode="xyxy") , 
              denorm=2,
              win="gt")
    # result = images.shape
    # result = overlay_boxes(result, top_predictions)
    # result = overlay_mask(result, top_predictions)

def summary(main_dict):
    history = ms.load_model(main_dict, fname="train_latest", history_only=True)
    key = "%s - %s" % (main_dict["dataset_name"], main_dict["model_name"])
    print(key)
    print("python main.py -e %s -m train -s 1" % main_dict["exp_name"])
    if "best_score" not in history["summary"]:
      print("None")
    else:
      best_score = history["summary"]["best_score"]
     
      if "time_updated" not in history["summary"]:
        history["summary"]["time_updated"] = -1
      print("%d - seen: %d/%d - time updated: %s" %( history["iters"], len(history["seen_image_ids"]),
            history["summary"]["n_train"], history["summary"]["time_updated"])) 

      print("{:s} ({:d}): {:s}".format(
        main_dict["metric_name"],
                                             history["val"]["scoreList"][-1]["iters"],
                                             
                                             best_score))
      stats = pd.DataFrame()
      per_class = pd.Series(history["val"]["scoreList"][0]['0.5_all_per_class'])
      stats["iters_%d" % history["val"]["scoreList"][0]["iters"]] = per_class
      per_class = pd.Series(history["val"]["scoreList"][-1]['0.5_all_per_class'])
      stats["iters_%d" % history["val"]["scoreList"][-1]["iters"]] = per_class
      print(stats)
      # print()
      print("---------------------------------------")

def visualize(main_dict, train_loader, val_loader):
  try:
    model = ms.load_model(main_dict, 
                            fname="train_best", 
                            reset=0)  
  except:
    model = ms.create_model(main_dict)
  fname = "train_%s_%s" % (main_dict["model_name"],
                           main_dict["dataset_name"])
  path = ("/mnt/home/issam/Summaries/experiments/%s/" % 
          fname)
  if os.path.exists(path):
    ms.remove_dir(path)

  for i, batch in enumerate(val_loader): 
    if i > 5: 
      break

    figure = model.visualize(batch, 
                              method="pred_gt", 
                              return_image=True)

    ms.imsave(path + "%s.png"%i, figure, size=(500,300))
    print("SAVED:", path + "%s.png"%i)

def fit_and_validate(main_dict, model,
                     train_loader, 
                     val_loader,  
                     multiprocess=True,
                     save_fname=None):
    model.eval()
    
    val_thread = None

    for e in range(1000):

      # Validation
      update_summary(main_dict, model)
      if e > -1:
        if multiprocess:
          verbose = 0
          val_thread = Thread(target=valEpoch, 
                              args=(main_dict, 
                                    copy.deepcopy(model), 
                                    val_loader,  
                                    model.history,
                                    save_fname, verbose))
          val_thread.start()
        else:

          valEpoch(main_dict, model, 
                   val_loader, 
                   model.history,
                   save_fname)
        


      # Training
      fitEpoch(main_dict, model, train_loader)
      
      update_summary(main_dict, model)
      ms.save_model(main_dict,  
                    fname="%s_latest" % save_fname,
                    model=model)

      

      # Wait for validation      
      if val_thread is not None:
        val_thread.join()
        # assert len(model.history["val"]["scoreList"]) == e + 1



def fitEpoch(main_dict, model, train_loader):

  n_batches = len(train_loader)
  history = model.history
  best_score_iters =  history["val"]["best_score_dict"]["iters"]
  best_score = history["val"]["best_score_dict"]["score"]
  metric_name = history["metric_name"]
  loss_sum = 0.


  for i, batch in enumerate(train_loader):
    
    loss = model.step(batch)

    GPUs = GPUtil.getGPUs()

    if len(GPUs) != 0:
      gpu = GPUs[0]
      gpu_load = gpu.load*100
      gpu_mem = gpu.memoryUtil*100
    
    print("%.2f - %d/%d - seen: %d - load: %d%% - mem:%d%%" \
          "  - loss: %.3f - Best %s (%d/%d): %.3f - %s" % 
          (history["iters"]/n_batches, history["iters"], n_batches,
            len(history["seen_image_ids"]), 
            gpu_load, gpu_mem, 
            loss, 
           metric_name, 
           best_score_iters, history["iters"], 
           best_score, ms.time_to_montreal()))
    loss_sum += loss
    history["iters"] += 1
    history["seen_image_ids"].add(batch["meta"]["image_id"][0])
    # history["seen_image_ids"].add(batch["meta"]["image_id"][0])

  history["train"] += [{"loss":loss_sum/n_batches, "iter":history["iters"]}]


@torch.no_grad()
def valEpoch(main_dict, model, val_loader,  
             history, save_fname=None, verbose=1):
  # Validate

  n_batches_val = len(val_loader)
  history_val = history["val"]
  iters = model.history["iters"]
  metric_name = main_dict["metric_name"]
  best_score = history_val["best_score_dict"]["score"]

  metric_class = main_dict["metric_dict"][main_dict["metric_name"]]
  metric_object = metric_class()

  for i, batch in enumerate(val_loader):
    metric_object.addBatch(model, batch)
    if verbose:
      print("%d/%d - validating - Best %s: %.3f" % (i+1, 
                  n_batches_val, metric_name, best_score))

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

  return score_dict


def get_loaders(main_dict, n_train, n_val,
                num_workers=1, val_indices=None):
  # load datasets

  dataset_name = main_dict["dataset_name"]

  train_set =  ms.load_dataset(main_dict, 
                               dataset_name=dataset_name, split="train")
  val_set =  ms.load_dataset(main_dict, 
                             dataset_name=dataset_name, split="val")

  
  if n_val == -1:
    n_val = len(val_set)

  if n_train == -1:
    n_train = len(val_set)*2


  if val_indices is None:
    val_indices = np.arange(n_val)

  sampler_val_tgt = FirstN(val_set, 
                           indices=val_indices)
  ###############
  # Samplers
  sampler_train = RandomN(train_set, n_samples=n_train)

  # loaders
  train_loader = torch.utils.data.DataLoader(
              train_set, 
              collate_fn=datasets.base_dataset.collate_fn_0_4,
               batch_size=1, sampler=sampler_train,
               shuffle=False, num_workers=num_workers, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
              val_set, 
              collate_fn=datasets.base_dataset.collate_fn_0_4,
               batch_size=1, sampler=sampler_val_tgt,
               shuffle=False, num_workers=num_workers, pin_memory=True) 

  print("\n%s (model)- %s " % (main_dict["model_name"], dataset_name))
  return {"train_loader":train_loader,  
          "val_loader":val_loader}

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
import ann_utils as au





def fit(main_dict, model, batch, n_iters=1):
 
  for i in range(n_iters):
    loss = model.step(batch)
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
