import numpy as np
import misc as ms

from sklearn.metrics import confusion_matrix
import ann_utils as au
import copy
# from addons.pycocotools.coco import COCO
from addons.pycocotools import eval_funcs
# from addons.pycocotools.cocoeval import COCOeval
import torch.nn.functional as F
import torch

class MAE:
    def __init__(self):

        self.sum = 0.
        self.n_batches = 0.
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}


    def addBatch(self, model, batch, **options):
        pred_annList = model.predict(batch, method="annList")
        gt_annList = batch["annList"]

        self.sum += abs(len(gt_annList) - len(pred_annList))
        self.n_batches += 1.

        return pred_annList
        # return {"gt":, "pred":, ""}

    def compute_score_dict(self, history=None):
        mae = self.sum / self.n_batches
        curr_score = mae
        self.score_dict = {self.metric_name: mae}

        if history is not None:
            best_score = history["best_score"]
            history["validated_flag"] = 1

            history["best_iter_flag"] = 0
            if best_score >= curr_score or best_score == -1:
                print("New best model: "
                      "%.3f=>%.3f %s" % (best_score, curr_score,
                                         self.metric_name))

                history["best_iter_flag"] = 1
                history["best_iter"] = history["iters"]
                history["best_model"] = self.score_dict
                history["best_score"] = curr_score

        return self.score_dict, history

    def get_result(self):
        return "%s: %.3f" % (self.metric_name,
                             self.score_dict[self.metric_name])
