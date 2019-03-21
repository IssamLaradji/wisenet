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

class AP:
    def __init__(self, iouType, iouThr=None):
        self.iouType = iouType
        self.pred_annList = []
        self.gt_annList = []
        self.n_batches = 0.
        self.iouThr = iouThr
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}

    def addBatch(self, model, batch, **options):
        pred_annList = model.predict(batch, method="annList")
        self.pred_annList += pred_annList
        self.gt_annList += batch["annList"]

        self.n_batches += 1.

        return pred_annList
        # return {"gt":, "pred":, ""}

    def compute_score_dict(self):
        self.score_dict = self.compute_precision(self.gt_annList, 
                                        self.pred_annList,
                                  self.iouType)

        return self.score_dict


    def is_best_score_dict(self, best_score_dict):
        best_flag = False
        best_score = best_score_dict["score"]
        curr_score = self.score_dict["score"]
        if best_score <= curr_score or best_score == -1:
            print("New best model: "
                  "%.3f=>%.3f %s" % (best_score, curr_score, self.metric_name))
            best_flag = True

        self.score_dict["best_flag"] = best_flag

        return best_flag

    def compute_precision(self, gt_annList, pred_annList, iouType):
        gt_annList = copy.deepcopy(gt_annList)
        # pred_annList = copy.deepcopy(pred_annList)
        if len(gt_annList) == 0:
            return {self.metric_name: -1, "score":-1, "score_per_class":-1}
        if len(pred_annList) == 0:
            return {self.metric_name: -1, "score":-1, "score_per_class":-1}

        
        result_dict = eval_funcs.evaluate_annList(pred_annList=pred_annList,
                                                  gt_annList=gt_annList,
                                                  iouType=self.iouType,
                                                  iouThr_list=(self.iouThr,),
                                                  ap=1)

        key = '%s_all' % self.iouThr
        result_dict[self.metric_name] = result_dict[key]
        result_dict["score"] = result_dict[key]
        key = '%s_all_per_class' % self.iouThr
        result_dict["score_per_class"] = result_dict[key]

        return result_dict
        # ####
        # pred_annList_all = copy.deepcopy(pred_annList)
        # gt_annList_all = copy.deepcopy(gt_annList)
        # image_id_list = list(set([ann["image_id"] for ann in pred_annList_all]))
        # if 1:
        #     #####
        #     image_id = set(image_id_list[1:3])
        #     pred_annList = []
        #     gt_annList = []
        #     for ann in pred_annList_all:
        #         if ann["image_id"] in image_id:
        #             pred_annList += [ann]
        #     for ann in gt_annList_all:
        #         if ann["image_id"] in image_id:
        #             gt_annList += [ann]
        #     pred_annList = copy.deepcopy(gt_annList)
        #     print([{"id":category_id} for category_id in 
        #                       np.unique([a["category_id"] for a in gt_annList])])
 
        #     #######
        #     gt_annDict = annList2cocoDict(copy.deepcopy(gt_annList))
        #     cocoGt = COCO(gt_annDict)

        #     cocoDt = cocoGt.loadRes(copy.deepcopy(pred_annList))
        #     cocoEval = COCOeval(cocoGt, cocoDt, 
        #                         iouType=iouType)
        #     # cocoEval.params.iouThrs = np.array([.25, .5, .75])

        #     cocoEval.evaluate()
        #     cocoEval.accumulate()

        #     results = cocoEval.summarize_ap()
        #     result_dict = {}    
            
        #     for i in ["AP50"]:
        #         result_dict[i] = results[i]

        #     result_dict[self.metric_name] = result_dict["AP50"]
        #     # print(result_dict)
        #     return result_dict


class AP50_bbox(AP):
    def __init__(self):
        super().__init__("bbox", iouThr=0.5)
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}


class AP50_segm(AP):
    def __init__(self):
        super().__init__("segm", iouThr=0.5)
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}



class AP_segm(AP):
    def __init__(self):
        super().__init__("segm", iouThr=None)
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}


class AR100:
    def __init__(self, iouType):
        self.iouType = iouType
        self.pred_annList = []
        self.gt_annList = []
        self.n_batches = 0.
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}

    def addBatch(self, model, batch, **options):
        pred_annList = model.predict(batch, method="annList")
        self.pred_annList += pred_annList

        for ann in batch["annList"]:
            ann["category_id"] = 1

        self.gt_annList += batch["annList"]

        self.n_batches += 1.

        return pred_annList
        # return {"gt":, "pred":, ""}

    def compute_score_dict(self):
        results = self.compute_recall(self.gt_annList, 
                                  self.pred_annList,
                                  self.iouType)

        curr_score = results[self.metric_name]
        self.score_dict["score"] = curr_score

        return self.score_dict

    def is_best_score_dict(self, best_score_dict):
        best_flag = False
        best_score = best_score_dict["score"]
        curr_score = self.score_dict["score"]
        if best_score <= curr_score or best_score == -1:
            print("New best model: "
                  "%.3f=>%.3f %s" % (best_score, curr_score, self.metric_name))
            best_flag = True

        self.score_dict["best_flag"] = best_flag

        return best_flag

    def compute_recall(self, gt_annList, pred_annList, iouType):
        gt_annList = copy.deepcopy(gt_annList)
        # pred_annList = copy.deepcopy(pred_annList)
        if len(gt_annList) == 0:
            return {self.metric_name: 0}
        if len(pred_annList) == 0:
            return {self.metric_name: 0}


        score = eval_funcs.evaluate_annList(pred_annList=pred_annList,
                                                  gt_annList=gt_annList,
                                                  iouType=self.iouType,
                                                  ap=0)
        result_dict = {}
        result_dict[self.metric_name] = score
        return result_dict



class AR100_bbox(AR100):
    def __init__(self):
        super().__init__("bbox")
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}


class AR100_segm(AR100):
    def __init__(self):
        super().__init__("segm")
        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}



  






