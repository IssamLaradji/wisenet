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
def test(model, test_set, metric_class, n_images=None, 
         loss_function=None):
    metric_name = metric_class.__name__
    n_batches = len(test_set)

    metric_object = metric_class()
    losses = 0.
    for i in range(n_batches):
        batch = ms.get_batch(test_set, [i])
        print("%d/%d - %s" % (i, n_batches, metric_name))

        if n_images is not None and i > n_images:
            break
        
        metric_object.addBatch(model, batch)
        import ipdb; ipdb.set_trace()  # breakpoint 989d6206 //

        if loss_function is not None:
            losses += loss_function(model, batch)

    score_dict = metric_object.compute_score_dict()

    if loss_function is not None:
        score_dict[loss_function.__name__] = losses

    return score_dict

def test_and_save(history, main_dict, model, test_set, metric_class,
                   verbose, epoch, loss_function=None):
    test_dict = test(model, test_set, metric_class, 
                     loss_function=loss_function)

    # val_dict[metric_name] = val_miou.item()
    test_dict["metric_name"] =  metric_name = metric_class.__name__
    test_dict["epoch"] = epoch
    # Update history
    history["val"] += [test_dict]


    # Higher is better
    if (history["best_model"] == {}
        or history["best_model"][metric_name] <= test_dict[metric_name]):

        history["best_model"] = test_dict
        ms.save_best_model(main_dict, model)

    ms.save_pkl(main_dict["path_history"], history)
    return history

@torch.no_grad()
def test_best_model(main_dict):
    val_set = ms.load_val(main_dict)
    model = ms.load_best_model(main_dict)

    results_dict = validate(model, val_set, main_dict)
    history = ms.load_history(main_dict)

    if history is None:
        history = {"test":[results_dict]}
    else:
        history["test"] = [results_dict]

    ms.save_pkl(main_dict["path_history"], history)



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
        


