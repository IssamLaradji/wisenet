import numpy as np
import datetime
from pycocotools import mask as maskUtils
import copy
from itertools import product


def evaluate_annList(pred_annList, gt_annList,
                     ap=1, 
                     iouType="bbox", 
                     iouThr_list=(None,), 
                     maxDets=100, 
                     aRngLabel_list = ("all",)):
    
    iouThrList = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
    recThrList = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)

    areaRngList = [[0 ** 2, 1e5 ** 2], 
                   [0 ** 2, 32 ** 2], 
                   [32 ** 2, 96 ** 2], 
                   [96 ** 2, 1e5 ** 2]]
    aRngLabelList = ['all', 'small', 'medium', 'large']

    maxDetList = [1, 10, 100]

    # Put them in dict
    gt_dict = {}  # gt for evaluation
    dt_dict = {}  # dt for evaluation
    imgList = set()
    catList = set()

    for i, gt in enumerate(copy.deepcopy(gt_annList)):
        key = (gt['image_id'], gt['category_id'])
        if key not in gt_dict:
            gt_dict[key] = []

        gt["id"] = i + 1
        gt_dict[key] += [gt]

        imgList.add(gt['image_id'])
        catList.add(gt['category_id'])

    for i, dt in enumerate(copy.deepcopy(pred_annList)):
        key = (dt['image_id'], dt['category_id'])
        if key not in dt_dict:
            dt_dict[key] = []

        if "bbox" not in dt:
            dt["bbox"] = maskUtils.toBbox(dt["segmentation"]).tolist()

        bb = dt["bbox"]
        dt['area'] = bb[2]*bb[3]

        dt["id"] = i + 1
        dt['iscrowd'] = 0

        dt_dict[key] += [dt]

        imgList.add(dt['image_id'])
        catList.add(dt['category_id'])

        # imgList.add(dt['image_id'])
        # catList.add(dt['category_id'])

    imgList = list(imgList)
    catList = list(catList)

    imgList.sort()
    catList.sort()

    # compute ious
    iou_dict = {}
    for imgId, catId in product(imgList, catList):
        key = (imgId, catId)
        
        if key in gt_dict:
            gt = gt_dict[key]
        else:
            gt = []

        if key in dt_dict:
            dt = dt_dict[key]
        else:
            dt = []

        iou_dict[key] = computeIoU(gt, dt,
                                    iouType=iouType,
                                    maxDets=maxDets)
    # evaluate detections
    evalImgs = []
    for catId, areaRng, imgId in product(catList,
                                        areaRngList,
                                        imgList):
        evalImgs += [
            evaluateImg(gt_dict, dt_dict, iou_dict, 
                        imgId, catId, areaRng,
                        maxDets, iouThrs=iouThrList)
        ]

    
    eval_dict = accumulate(evalImgs, imgList, 
                            catList, iouThrList, 
                            recThrList, 
                            areaRngList, maxDetList)

    result_dict = {}
    for iouThr, aRngLabel in product(iouThr_list, aRngLabel_list):
        key = "%s_%s" % (iouThr, aRngLabel)

        aind = [i for i, aRng in enumerate(aRngLabelList) if aRng == aRngLabel]
        mind = [i for i, mDet in enumerate(maxDetList) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval_dict['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == iouThrList)[0]
                s = s[t]
            s = s[:,:,:,aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval_dict['recall']
            if iouThr is not None:
                t = np.where(iouThr == iouThrList)[0]
                s = s[t]
            s = s[:,:,aind,mind]

        if len(s[s>-1])==0:
            mean_s = -1
            mean_s_per_class = -1
        else:
            # s_pos = s[s>-1]
            mean_s = np.mean(s[s>-1])
            mean_s_per_class = {}
            for k, c in enumerate(catList):
                sk = s[:,:,[k]] 

                if len(sk[sk>-1])==0:
                    mean_sk = -1
                else:
                    mean_sk =  np.mean(sk[sk>-1]) 
                mean_s_per_class["class_%d"%c] =mean_sk

        # tmp = s[:,:,:,aind, mind]; np.mean(tmp[tmp>-1])
        # for i in range(10):
        #     result = s[i, :, :, aind, mind]
        #     print(np.mean(result[result>-1]))
        # print(s.shape)
        result_dict[key] = mean_s
        result_dict[key + "_per_class"] = mean_s_per_class

    return result_dict


def computeIoU(gt, dt, iouType="bbox", maxDets=100):

    if len(gt) == 0 and len(dt) == 0:
        return []

    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    if len(dt) > maxDets:
        dt = dt[0:maxDets]

    if iouType == 'segm':
        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
    elif iouType == 'bbox':
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
    else:
        raise Exception('unknown iouType for iou computation')

    # compute iou between each dt and gt region
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = maskUtils.iou(d, g, iscrowd)

    # ####
    # print("id:%s - cat:%d" % (imgId, catId))
    # print("d:", d)
    # print("g:", g)
    # print("ious", ious)
    # ####
    return ious


def evaluateImg(gt_dict,
                dt_dict,
                ious,
                imgId,
                catId,
                aRng,
                maxDet,
                iouThrs):
    '''
    perform evaluation for single category and image
    :return: dict (single image results)
    '''
    # p = self.params
    key = (imgId, catId)
    if key in gt_dict:
        gt = gt_dict[key]
    else:
        gt = []

    if key in dt_dict:
        dt = dt_dict[key]
    else:
        dt = []

    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        if (g['area'] < aRng[0] or g['area'] > aRng[1]):            
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]
    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:maxDet]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = (ious[(imgId, catId)][:, gtind]
            if len(ious[(imgId, catId)]) > 0 else ious[(imgId, catId)])

    T = len(iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[(dind, gind)] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[(dind, gind)]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                  for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    # store results for given image and category
    return {
        'image_id': imgId,
        'category_id': catId,
        'aRng': aRng,
        'maxDet': maxDet,
        'dtIds': [d['id'] for d in dt],
        'gtIds': [g['id'] for g in gt],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dt],
        'gtIgnore': gtIg,
        'dtIgnore': dtIg,
    }


def accumulate(evalImgs, imgIds, catIds, iouThrs, recThrs, 
               areaRng,
               maxDets):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    '''
    # print('Accumulating evaluation results...')

    # allows input customized parameters

    T = len(iouThrs)
    R = len(recThrs)
    K = len(catIds)
    A = len(areaRng)
    M = len(maxDets)

    precision = -np.ones(
        (T, R, K, A, M))  # -1 for the precision of absent categories
    recall = -np.ones((T, K, A, M))
    scores = -np.ones((T, R, K, A, M))

    # create dictionary for future indexing
    catIds = catIds
    setK = set(catIds)
    setA = set(map(tuple, areaRng))
    setM = set(maxDets)
    setI = set(imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(catIds) if k in setK]
    m_list = [m for n, m in enumerate(maxDets) if m in setM]
    a_list = [
        n for n, a in enumerate(map(lambda x: tuple(x), areaRng)) if a in setA
    ]
    i_list = [n for n, i in enumerate(imgIds) if i in setI]
    I0 = len(imgIds)
    A0 = len(areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0 * A0 * I0
        for a, a0 in enumerate(a_list):
            Na = a0 * I0
            for m, maxDet in enumerate(m_list):
                E = [evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E],
                                     axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E],
                                      axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    q = np.zeros((R, ))
                    ss = np.zeros((R, ))

                    if nd:
                        recall[t, k, a, m] = rc[-1]
                    else:
                        recall[t, k, a, m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = np.searchsorted(rc, recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t, :, k, a, m] = np.array(q)
                    scores[t, :, k, a, m] = np.array(ss)
    eval_dict = {
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'recall': recall,
        'scores': scores,
    }

    return eval_dict