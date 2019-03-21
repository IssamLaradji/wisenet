import torch
import numpy as np 
import misc as ms
from . import base_model as bm
import ann_utils as au
import torch.nn.functional as F
from torch import optim 
import copy
from addons.pycocotools import mask as maskUtils
def pairwise_sum(fi, fj):
    diff = (fi - fj).pow(2).sum(1).clamp(min=0, max=50)
    return (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))

def log_pairwise_sum(fi, fj):
    return torch.log(pairwise_sum(fi, fj))

def diff_log_pairwise_sum(fi, fj):
    return torch.log(1. - pairwise_sum(fi, fj))



def get_batches(n_pixels, size=500000):
    batches = []
    for i in range(0, n_pixels, size):
        batches +=[(i, i+size)]
    return batches

class TwoHeads(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracter()
        self.blob_head = bm.Upsampler(self.feature_extracter.expansion_rate, 
                                    train_set.n_classes)
        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 
                                  64)

      
        

        self.n_bg_seeds = None
        self.similarity_function = log_pairwise_sum
        self.diff_function = diff_log_pairwise_sum
        kfname="/mnt/projects/counting/Saves/cvpr2019/kitti/k_parameter_search.pkl"

        base = "/mnt/projects/counting/Saves/cvpr2019/kitti/"
        self.load_state_dict(torch.load(base+"State_Dicts/best_model.pth"))
        self.history = ms.load_pkl(base + "history.pkl")

        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 
                                  64)
        self.opt = optim.Adam(self.parameters(), lr=1e-5,
                                  weight_decay=0.0005)


    def forward(self, x_input):

        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)

        return {"embedding_mask":embedding_mask, 
                "blob_mask":blob_mask}

    def step(self, batch):
        self.train()
        n,c,h,w = batch["images"].shape

        self.opt.zero_grad()
        with torch.enable_grad():
            O_dict = self(batch["images"].cuda())
            O = O_dict["embedding_mask"]

            loss = compute_metric_loss_sum(O, batch, 
                                           similarity_function=self.similarity_function,
                                           diff_function=self.diff_function)
        if loss != 0: 
            loss.backward()
            self.opt.step()

        return loss.item()



    @torch.no_grad()
    def visualize(self, batch, method="annList"):
        annList = self.predict(batch, method="annList")
        ms.images(batch["images"], annList=annList, denorm=1)

    @torch.no_grad()
    def predict(self, batch, method="blobs"):
        self.eval()
        n,c,h,w = batch["images"].shape
        
        O_dict = self(batch["images"].cuda())
        O = O_dict["blob_mask"]

        probs = F.softmax(O, dim=1)
        blob_dict = au.probs2blobs(probs)
        pointList = blob_dict["pointList"]

        if method == 'original':            
            return {"blobs":blob_dict['blobs'], 
                    "probs":blob_dict['probs'], 
                    "annList":blob_dict['annList'], 
                    "counts":blob_dict['counts']}

        elif method in ["best_objectness"]:
            
            counts = np.zeros(self.n_classes-1)
            if len(pointList) == 0:
                return {"blobs": np.zeros((h,w), int), "annList":[], "probs":probs,
                        "counts":counts}

            yList = []
            xList = []

            categories = np.zeros(len(pointList)+1)
            
            for i, p in enumerate(pointList):
                yList += [p["y"]]
                xList += [p["x"]]
                categories[i+1] = p["category_id"]
                counts[p["category_id"]-1] += 1
                
                blob_dict = au.pointList2BestObjectness(pointList,
                                                         batch)
                blobs = blob_dict["blobs"]
                annList = blob_dict["annList"]

            # ms.images(batch["images"], annList=blob_dict["annList"], denorm=1)
            return {"blobs":blobs, "probs":probs, "annList":annList, "counts":counts}

        elif method in ["annList", "blobs"]:
            
            propDict = au.pointList2propDict(pointList, batch, thresh=0.5)
                        
            # Segmenter
            O = O_dict["embedding_mask"]
            if self.n_bg_seeds is None:
                n_bg_seeds = len(propDict["propDict"])
            else:
                n_bg_seeds = self.n_bg_seeds

            seedList = propDict2seedList(propDict, E=O, n_bg_seeds=n_bg_seeds)
            fg_bg_seeds = CombineSeeds(seedList)
            blobs_categoryDict = get_embedding_blobs(self, O, fg_bg_seeds, 
                            similarity_function=self.similarity_function)
            blobs = blobs_categoryDict["blobs"]
            categoryDict = blobs_categoryDict["categoryDict"]
            
            blob_dict = au.blobs2BestDice(blobs, categoryDict, 
                        propDict, batch)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]

        return annList


def cosine_sum(fi, fj):
    
    scale = fi.norm() * fj.norm()
    sim = 0.5 * (1+ (fi*fj).sum(1)/scale)
    return sim


def sim_cosine_sum(fi, fj):
    return cosine_sum(fi, fj) - 1.

def diff_cosine_sum(fi, fj):
    return (1. - cosine_sum(fi, fj)).clamp(0)


class TwoHeads_sharpmask(TwoHeads):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.similarity_function = sim_cosine_sum
        self.diff_function = diff_cosine_sum

    @torch.no_grad()
    def predict(self, batch, method="blobs"):
        self.eval()

        from datasets import base_dataset
        sharp_proposals = base_dataset.SharpProposals(batch)

        annList = []
        for i in range(100):
            proposal = sharp_proposals[i]
            binmask = proposal["mask"]
            score = proposal["score"]
            seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
            seg["counts"] = seg["counts"].decode("utf-8")
            h, w = binmask.shape
            annList += [{"segmentation":seg,
                  "iscrowd":0,
                  "area":int(maskUtils.area(seg)),
                 "image_id":batch["meta"]["image_id"][0],
                 "category_id":1,
                 "height":h,
                 "width":w,
                 "score":score}]

        return annList

class TwoHeads_cosine(TwoHeads):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.similarity_function = sim_cosine_sum
        self.diff_function = diff_cosine_sum

class TwoHeads_kitti(TwoHeads):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        base = "/mnt/projects/counting/Saves/cvpr2019/kitti/"
        self.load_state_dict(torch.load(base+"State_Dicts/best_model.pth"))
        self.history = ms.load_pkl(base + "history.pkl")

class TwoHeads_iou(TwoHeads):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        import ipdb; ipdb.set_trace()  # breakpoint 9f0aef98 //
        
        base = "/mnt/projects/counting/Saves/cvpr2019/kitti/"
        self.load_state_dict(torch.load(base+"State_Dicts/best_model.pth"))
        self.history = ms.load_pkl(base + "history.pkl")

def get_embedding_blobs(self, O, fg_bg_seeds, similarity_function):
    n, c, h, w = O.shape
    # seeds = torch.cat([fg_seeds, bg_seeds], 2)[:,:,None]
    fA = O.view(1,c,-1)
    fS = O[:,:, fg_bg_seeds["yList"], fg_bg_seeds["xList"]]

    n_pixels = h*w
    blobs = torch.zeros(h*w)

    n_seeds =  fS.shape[-1]

    maximum = 5000000
    n_loops = int(np.ceil((n_pixels * n_seeds) / maximum))
    
    for (s,e) in get_batches(n_pixels, size=n_pixels//n_loops):
        # s,e = map(int, (s,e))
        diff = similarity_function(fS[:,:,None], fA[:,:,s:e,None]) 
        blobs[s:e] = diff.max(2)[1] + 1 
    
    bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()
    # assert len(fg_bg_seeds["yList"])//2 == bg_min_index
    blobs[blobs > int(bg_min_index)] = 0
    blobs = blobs.squeeze().reshape(h,w).long()

    categoryDict = {}
    for i, category_id in enumerate(fg_bg_seeds["categoryList"]):
        if category_id == 0:
             continue

        categoryDict[i+1] = category_id 

    return {"blobs":ms.t2n(blobs), "categoryDict":categoryDict}

### Prediction
def propDict2seedList(propDict, E=None, n_neighbors=100,
                      random_proposal=False, n_bg_seeds=5):
    seedList = []
    for prop in propDict["propDict"]:
        if len(prop["annList"]) == 0:
            seedList += [{"category_id":[prop["point"]["category_id"]],
                           "yList":[prop["point"]["y"]],   
                          "xList":[prop["point"]["x"]],   
                          "neigh":{"yList":[prop["point"]["y"]],
                                    "xList":[prop["point"]["x"]]}}]

        else:
            if random_proposal:
                i = np.random.randint(0, len(prop["annList"]))
                mask = prop["annList"][i]["mask"]
            else:
                mask = prop["annList"][0]["mask"]
                
            seedList += [{"category_id":[prop["point"]["category_id"]],
                           "yList":[prop["point"]["y"]],   
                          "xList":[prop["point"]["x"]],   
                          "neigh":get_random_indices(mask, n_indices=100)}]

    # Background
    background = propDict["background"]
    if background.sum() == 0:
        y_axis = np.random.randint(0, background.shape[1],100)
        x_axis = np.random.randint(0, background.shape[2],100)
        background[0,y_axis, x_axis] = 1

    bg_seeds = get_kmeans_indices(background, E=E, n_indices=n_bg_seeds)
    seedList += [{"category_id":[0]*len(bg_seeds["yList"]),
                  "yList":bg_seeds["yList"].tolist(), 
                  "xList":bg_seeds["xList"].tolist(), 
                  "neigh":get_random_indices(background, n_indices=100)}] 

    return seedList

def CombineSeeds(seedList, ind=None):
    yList = []
    xList = []
    categoryList = []

    if ind is None:
        ind = range(len(seedList))

    for i in ind:
        yList += seedList[i]["yList"]
        xList += seedList[i]["xList"]
        categoryList += seedList[i]["category_id"]

    assert len(categoryList) == len(yList) 
    return {"yList":yList, "xList":xList, "categoryList":categoryList}

def get_random_indices(mask, n_indices=10):
    mask_ind = np.where(mask.squeeze())
    n_pixels = mask_ind[0].shape[0]
    P_ind = np.random.randint(0, n_pixels, n_indices)
    yList = mask_ind[0][P_ind]
    xList = mask_ind[1][P_ind]

    return {"yList":yList, "xList":xList}

def get_kmeans_indices(mask, E, n_indices=10):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    _,_,h,w = E.shape

    clf = MiniBatchKMeans(n_clusters=n_indices,  max_iter=100)
    E_flat = E.view(1,64,-1).squeeze()

    mask_flat = mask.squeeze().ravel()
    indList = np.where(mask_flat)[0]

    E_new = E_flat[:, indList]
    X = ms.t2n(E_new.squeeze().transpose(0,1))
    clf.fit(X)
    closest, _ = pairwise_distances_argmin_min(clf.cluster_centers_, X)
    yx_list = indList[closest]

    yList, xList = np.unravel_index(yx_list, (h,w), order='C')

    # E_flat[:3, yx_list]
    # E[:,:3,yList,xList]
    
    # mask_ind = np.where(mask.squeeze())
    # n_pixels = mask_ind[0].shape[0]
    # P_ind = np.random.randint(0, n_pixels, n_indices)
    # yList = mask_ind[0][P_ind]
    # xList = mask_ind[1][P_ind]

    return {"yList":yList, "xList":xList}
### Loss

def compute_metric_loss_sum(O, batch, random_proposal=False, 
                            similarity_function=None,
                            diff_function=None):

    n,c,h,w = O.shape

    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None
    
    pointList = au.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:

        return loss

    
    if "single_point" in batch:
        single_point = True
    else:
        single_point = False

    # return O.sum()
    propDict = au.pointList2propDict(copy.deepcopy(pointList), 
                                            copy.deepcopy(batch), 
                                     single_point=single_point,
                                     thresh=0.5)


    background = propDict["background"]
    propDict = propDict["propDict"]


    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]

    fg_seeds = O[:, :, yList, xList]


    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    for i in range(n_seeds):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:

            if random_proposal:
                ann_i = np.random.randint(0, len(annList))
                mask = annList[ann_i]["mask"]
            else:
                mask = annList[0]["mask"]


       
        mask_ind = np.where(mask)
        prop_mask[mask!=0] = (i+1)

        
        f_A = fg_seeds[:,:,[i]]
        
        # Positive Embeddings
        n_pixels = mask_ind[0].shape[0]
        P_ind = np.random.randint(0, n_pixels, 100)
        yList = mask_ind[0][P_ind]
        xList = mask_ind[1][P_ind]
        fg_P = O[:,:,yList, xList]

        
        ap = - similarity_function(f_A, fg_P)
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - diff_function(f_A, f_N)
            loss += an.mean()

    

    # # Extract background seeds
    bg = np.where(background.squeeze())

    n_pixels = bg[0].shape[0]
    bg_ind = np.random.randint(0, n_pixels, n_seeds)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_A = O[:,:,yList, xList]


    bg_ind = np.random.randint(0, n_pixels, 100)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_P = O[:,:,yList, xList]


    # BG seeds towards BG pixels, BG seeds away from FG seeds
    ap = - similarity_function(f_A[:,:,None], f_P[:,:,:,None]) 
    an = - diff_function(f_A[:,:,None], fg_seeds[:,:,:,None])

    loss += ap.mean()
    loss += an.mean()

    n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():
            diff = similarity_function(O.view(1,c,-1)[:,:,:,None], 
                                       torch.cat([fg_seeds, f_A], 2)[:,:,None])    
            labels = diff.max(2)[1] + 1 
            labels = labels <= n_seeds
            labels = labels.squeeze().reshape(h,w)
            bg = labels.cpu().long()*torch.from_numpy(background)        
            # ms.images(labels.cpu().long()*torch.from_numpy(background))


        # Extract false positive pixels
        bg_ind = np.where(bg.squeeze())
        n_P = bg_ind[0].shape[0]
        if n_P != 0:
            A_ind = np.random.randint(0, n_P, n_seeds)
            f_P = O[:,:, bg_ind[0][A_ind], bg_ind[1][A_ind]]

            ap = - similarity_function(f_A[:,:,None],
                         f_P[:,:,:,None])
            an = - diff_function(f_P[:,:,None], 
                        fg_seeds[:,:,:,None])

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()


    return loss / max(n_seeds, 1)