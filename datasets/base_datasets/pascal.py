import os

from addons import transforms as myTransforms
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
#from pycocotools import mask as maskUtils
import misc as ms
from torchvision import transforms

def load_mask(mask_path):
    if ".mat" in mask_path:
        inst_mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
        inst_mask = Image.fromarray(inst_mask.astype(np.uint8))
    else:
        inst_mask = Image.open(mask_path)

    return inst_mask


class Pascal2012:
    def __init__(self, root, split, transform_function):
        super().__init__()
        self.split = split

        self.path = "/mnt/datasets/public/issam/VOCdevkit"

        self.categories = ms.load_json("/mnt/datasets/public/issam/"
                       "VOCdevkit/annotations/pascal_val2012.json")["categories"]

        # assert split in ['train', 'val', 'test']
        self.img_names = []
        self.mask_names = []
        self.cls_names = []

        base = "/mnt/projects/counting/Saves/main/"
        fname = base + "lcfcn_points/Pascal2012.pkl"
        self.pointDict = ms.load_pkl(fname)

        berkley_root =  os.path.join(self.path, 'benchmark_RELEASE')
        pascal_root = os.path.join(self.path)

        data_dict = get_augmented_filenames(pascal_root, 
                                            berkley_root, 
                                            mode=1)
        self.data_dict = data_dict
        # train
        assert len(data_dict["train_imgNames"]) == 10582
        assert len(data_dict["val_imgNames"]) == 1449

        berkley_path = berkley_root + '/dataset/'
        pascal_path = pascal_root + '/VOC2012/'

        corrupted=["2008_005262",
                    "2008_004172",
                    "2008_004562",
                    "2008_005145",
                    "2008_008051",
                    "2008_000763",
                    "2009_000573"]
        # sanity check
        path_base = "/mnt/projects/counting/Items/"
        self.path_gt_annList = path_base + "/Pascal2012/val_gt_annList.json"
        # self.gt_annList = 
        if split == 'train':
            for name in  data_dict["train_imgNames"]:
                name_img = os.path.join(berkley_path, 'img/' + name + '.jpg')
                if os.path.exists(name_img):
                    name_img = name_img
                    name_mask = os.path.join(berkley_path, 'cls/' + name + '.mat')
                else:
                    name_img = os.path.join(pascal_path, 'JPEGImages/' + name + '.jpg')
                    name_mask =  os.path.join(pascal_path, 'SegmentationObject/' +  name + '.jpg')


                self.img_names += [name_img]
                self.mask_names += [name_mask]

        elif split in ['val', "test"]:
            data_dict["val_imgNames"].sort() 
            for k, name in  enumerate(data_dict["val_imgNames"]):

                if name in corrupted:
                    continue
                name_img = os.path.join(pascal_path, 'JPEGImages/' + name + '.jpg')
                name_mask =  os.path.join(pascal_path, 'SegmentationObject/' + 
                                          name  + '.png')
                name_cls =  os.path.join(pascal_path, 'SegmentationClass/' + name + '.png')

                if not os.path.exists(name_img):
                    name_img = os.path.join(berkley_path, 'img/' + name + '.jpg')
                    name_mask =  os.path.join(berkley_path, 'inst/' + name + '.mat')
                    name_cls =  os.path.join(berkley_path, 'cls/' + name + '.mat')

                assert os.path.exists(name_img)
                assert os.path.exists(name_mask)
                assert os.path.exists(name_cls)

                self.img_names += [name_img]
                self.mask_names += [name_mask]
                self.cls_names += [name_cls]


        self.proposals_path = "/mnt/datasets/public/issam/VOCdevkit/VOC2012/ProposalsSharp/"
        if len(self.img_names) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.n_classes = 21
        self.transform_function = transform_function()
    
        self.ignore_index = 255
        self.pointsJSON = ms.jload(os.path.join( 
                                    '/mnt/datasets/public/issam/VOCdevkit/VOC2012',
                                    'whats_the_point/data', 
                                    "pascal2012_trainval_main.json"))

        self.lcfcn_pointListDict = ms.load_pkl("/mnt/projects/counting/Items/Pascal2012/lcfcn_pointList.pkl")
        if split == "val":
            annList_path = self.path + "/annotations/{}_gt_annList.json".format(split)
            self.annList_path = annList_path

    def load_mask(self, index):
        mask_path = self.cls_names[index]
        if ".mat" in mask_path:
            inst_mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            inst_mask = Image.fromarray(inst_mask.astype(np.uint8))
        else:
            inst_mask = Image.open(mask_path)

        return inst_mask

    def __getitem__(self, index):
        # Image       
        img_path = self.img_names[index]
        image = Image.open(img_path).convert('RGB')
        import ipdb; ipdb.set_trace()  # breakpoint 12d7b318 //

        mask_path = self.mask_names[index]
        maskObjects = np.array(load_mask(mask_path))
        maskClass = self.load_mask(index)

        annList = mask2annList(maskClass, maskObjects, image_id=img_path)
        # ms.vis_annList(np.array(image), annList)
        
        for ann in annList:
            print(ann)

        if self.transform_function is not None:
                image = self.transform_function([image])[0]

        if self.split == "train":
            
            
            return {"images":image,
                "annList":annList,
                "meta":{"index":index, "img_name":img_name, 
                        "split":self.split}}

        elif self.split in ["val", "test"]: 
            if self.transform_function is not None:
                image, points = self.transform_function([image, points])

            
            # maskVoid = torch.FloatTensor((np.array(maskClass) != 255).astype(float))

            # Mask
            
            # maskObjects[maskObjects==255] = 0


        return {"images":image,
            "annList":annList,
            "meta":{"index":index, "img_name":img_name, 
                    "split":self.split}}

    def __len__(self):
        return len(self.img_names)





#------ aux

def get_augmented_filenames(pascal_root, pascal_berkeley_root, mode=2):
    pascal_txts = get_pascal_segmentation_images_lists_txts(pascal_root=pascal_root)
    berkeley_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root=pascal_berkeley_root)

    pascal_name_lists = readlines_with_strip_array_version(pascal_txts)
    berkeley_name_lists = readlines_with_strip_array_version(berkeley_txts)

    pascal_train_name_set, pascal_val_name_set, _ = map(lambda x: set(x), pascal_name_lists)
    berkeley_train_name_set, berkeley_val_name_set = map(lambda x: set(x), berkeley_name_lists)

    
    all_berkeley = berkeley_train_name_set | berkeley_val_name_set
    all_pascal = pascal_train_name_set | pascal_val_name_set

    everything = all_berkeley | all_pascal

    # Extract the validation subset based on selected mode
    if mode == 1:

        # 1449 validation images, 10582 training images
        validation = pascal_val_name_set

    if mode == 2:

        # 904 validatioin images, 11127 training images
        validation = pascal_val_name_set - berkeley_train_name_set

    if mode == 3:

        # 346 validation images, 11685 training images
        validation = pascal_val_name_set - all_berkeley

    # The rest of the dataset is for training
    train = everything - validation

    # Get the part that can be extracted from berkeley
    train_from_berkeley = train & all_berkeley

    # The rest of the data will be loaded from pascal
    train_from_pascal = train - train_from_berkeley

    train_imgNames = list(train_from_pascal) + list(train_from_berkeley)
    val_imgNames = list(validation)

    

    ## Permutate
    # np.random.seed(3)

    
    train_imgNames = np.sort(train_imgNames)
    # train_imgNames = np.random.permutation(train_imgNames)
    assert train_imgNames.size == np.unique(train_imgNames).size
    train_imgNames = train_imgNames.tolist()

    return {"train_imgNames": train_imgNames, "val_imgNames": val_imgNames}



def readlines_with_strip_array_version(filenames_array):
    """The function that is similar to readlines_with_strip() but for array of filenames.
    Takes array of filenames as an input and applies readlines_with_strip() to each element.
    
    Parameters
    ----------
    array of filenams : array of strings
        Array of strings. Each specifies a path to a file.
    
    Returns
    -------
    clean_lines : array of (array of strings)
        Strings that were read from the file and cleaned up.
    """
    
    multiple_files_clean_lines = map(readlines_with_strip, filenames_array)
    
    return multiple_files_clean_lines

def readlines_with_strip(filename):
    """Reads lines from specified file with whitespaced removed on both sides.
    The function reads each line in the specified file and applies string.strip()
    function to each line which results in removing all whitespaces on both ends
    of each string. Also removes the newline symbol which is usually present
    after the lines wre read using readlines() function.
    
    Parameters
    ----------
    filename : string
        Full path to the root of PASCAL VOC dataset.
    
    Returns
    -------
    clean_lines : array of strings
        Strings that were read from the file and cleaned up.
    """
    
    # Get raw filnames from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Clean filenames from whitespaces and newline symbols
    clean_lines = map(lambda x: x.strip(), lines)
    
    return clean_lines

def get_pascal_segmentation_images_lists_txts(pascal_root):
    """Return full paths to files in PASCAL VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.
    
    Parameters
    ----------
    pascal_root : string
        Full path to the root of PASCAL VOC dataset.
    
    Returns
    -------
    full_filenames_txts : [string, string, string]
        Array that contains paths for train/val/trainval txts with images names.
    """
    
    segmentation_images_lists_relative_folder = 'VOC2012/ImageSets/Segmentation'
    
    segmentation_images_lists_folder = os.path.join(pascal_root,
                                                    segmentation_images_lists_relative_folder)
    
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    pascal_trainval_list_filname = os.path.join(segmentation_images_lists_folder,
                                                'trainval.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename,
            pascal_trainval_list_filname
           ]



def get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root):

    segmentation_images_lists_relative_folder = 'dataset'
    
    segmentation_images_lists_folder = os.path.join(pascal_berkeley_root,
                                                    segmentation_images_lists_relative_folder)
    
    
    # TODO: add function that will joing both train.txt and val.txt into trainval.txt
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename
           ]


def make_dataset(path, split):
    assert split in ['train', 'val', 'test']
    data_dict = {"img_names": [], "labels": []}

    if split == 'train':
        path = os.path.join(path, 'benchmark_RELEASE', 'dataset')
        img_path =  path +'/img'
        mask_path = path +'/cls'
        data_list = [l.strip('\n') for l in open(path + '/train.txt').readlines()]
        ext = ".mat"

    elif split == 'val':    
        path = os.path.join(path, 'VOC2012')
        img_path = path + '/JPEGImages'
        mask_path =  path + '/SegmentationClass'
        
        data_list = [l.strip('\n') for l in open(os.path.join(path,
            'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]

        ext = ".png"
        corrupted = ["2008_000763", "2008_004172",
                     "2008_004562", "2008_005145",
                     "2008_005262", "2008_008051"] 

        data_list = np.setdiff1d(data_list, corrupted)

                                                                      
    elif split == 'test':  
        path = os.path.join(path, 'VOC2012')
        img_path = path + '/JPEGImages'
        mask_path =  path + '/SegmentationClass'
        
        data_list = [l.strip('\n') for l in open(os.path.join(path,
            'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]

        ext = ".png"
        corrupted = ["2008_000763", "2008_004172",
                     "2008_004562", "2008_005145",
                     "2008_005262", "2008_008051"] 

        data_list = np.setdiff1d(data_list, corrupted)

    
    for it in data_list:
        data_dict["img_names"] += [os.path.join(img_path, it + '.jpg')]
        data_dict["labels"] += [os.path.join(mask_path, it + '%s' % ext)]

    return data_dict


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


# class PascalSmall(Pascal2012):
#     def __init__(self, root, split, transform_function):
#         if split == "train":
#             split = "train_small"
#         super().__init__(root, split, transform_function)
        
#         if split == "train_small":
#             self.img_names.sort()
#             self.cls_names.sort()
#             self.mask_names.sort()

#             self.img_names = np.array(self.img_names)
#             self.cls_names = np.array(self.cls_names)
#             self.mask_names = np.array(self.mask_names)

#             np.random.seed(3)

#             ind = np.random.choice(len(self.img_names), 883, replace=False)
#             self.img_names = self.img_names[ind]
#             self.cls_names = self.cls_names[ind]
#             self.mask_names = self.mask_names[ind]
#             self.split = "train"

#     def __getitem__(self, index):
#         # Image

#         img_path = self.img_names[index]
#         image = Image.open(img_path).convert('RGB')

#         # Points
#         name = ms.extract_fname(img_path).split(".")[0]
#         points, counts = ms.point2mask(self.pointsJSON[name], image, return_count=True, n_classes=self.n_classes-1)
#         points = transforms.functional.to_pil_image(points)

#         counts = torch.LongTensor(counts)
#         original = transforms.ToTensor()(image)
        

#         # Mask
#         mask_path = self.mask_names[index]
#         mask = load_mask(mask_path)

#         # Mask
#         cls_path = self.cls_names[index]
#         maskClass = load_mask(cls_path)
        
#         if self.transform_function is not None:
#             image, points, mask, maskClass = self.transform_function([image, points, 
#                 mask,maskClass])

#         maskVoid = maskClass != 255
#         maskClass[maskClass==255] = 0
#         mask[mask==255] = 0
#         lcfcn_pointList = self.get_lcfcn_pointList(name)
#         return {"images":image, 
#                 "original":original,
#                 "points":points, 
#                 "counts":counts,
#                 "index":index,
#                 "name":name,
#                 "image_id":int(name.replace("_","")),
#                 "maskObjects":mask,
#                 "maskClasses":maskClass,
#                 "maskVoid":maskVoid.long(),
#                 "dataset":"voc",
#                 "lcfcn_pointList":lcfcn_pointList,
#                 "proposals_path":self.proposals_path,
#                 "split":self.split,
#                 "path":self.path}


from datasets import base_dataset as bd
class PascalOriginal(Pascal2012):
    def __init__(self, root, split, transform_function, **dataset_options):
        super().__init__(root, split, transform_function)
        pascal_root = os.path.join(self.path)

        pascal_path = pascal_root + '/VOC2012/'
        if split == "train":
            self.img_names  = []
            self.mask_names = []
            self.cls_names = []

            for name in  self.data_dict["train_imgNames"]:
                name_img = os.path.join(pascal_path, 'JPEGImages/' + name + '.jpg')
                name_mask =  os.path.join(pascal_path, 'SegmentationObject/' +  name + '.png')
                name_cls =  os.path.join(pascal_path, 'SegmentationClass/' + name + '.png')

                if not os.path.exists(name_cls):
                    continue
               
               

                self.img_names += [name_img]
                self.mask_names += [name_mask]
                self.cls_names += [name_cls]

            self.img_names.sort()
            self.cls_names.sort()
            self.mask_names.sort()

            self.img_names = np.array(self.img_names)
            self.cls_names = np.array(self.cls_names)
            self.mask_names = np.array(self.mask_names)


        if "multiscale" in dataset_options:
            self.multiscale = dataset_options["multiscale"]
        else:
            self.multiscale = 0

        if "mold_image" in dataset_options:
            self.mold_image = dataset_options["mold_image"]
        else:
            self.mold_image = 0
        
        self.collate_fn = bd.collate_fn_0_4
            
    def __getitem__(self, index):
        # Image       

        img_path = self.img_names[index]
        image_pil = Image.open(img_path).convert('RGB')

        mask_path = self.mask_names[index]
        maskObjects = np.array(load_mask(mask_path))
        maskClass = self.load_mask(index)

        annList = mask2annList(maskClass, maskObjects, image_id=img_path)

        if self.transform_function is not None:
                image = self.transform_function(image_pil)



        # Mold Image
        if self.mold_image == 1: 
            resized_image =  bd.resize_image(np.array(image_pil))
        else:
            resized_image = {}


        return {"images":image,
            "annList":annList,
            "resized":resized_image,
            "meta":{"index":index, "image_id":img_path,
                    "split":self.split}}

import ann_utils as au
from addons.pycocotools import mask as maskUtils
def mask2annList(maskClass, maskObjects, image_id, classes=None):

    annList = []
    maskClass = np.array(maskClass)
    maskObjects = np.array(maskObjects)

    objects = np.setdiff1d(np.unique(maskObjects), [0, 255])
    for i, obj in enumerate(objects):
        binmask = maskObjects==obj
        category_id = np.unique(maskClass*binmask)[1] 
        
        if classes is not None and category_id not in classes:
            continue       

        ann = au.mask2ann(binmask, category_id=category_id, 
                          image_id=image_id, 
                          maskVoid=-1, 
                          score=1, 
                          point=-1)
        # "bbox" - [nx4] Bounding box(es) stored as [x y w h]
        ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
        # ann["id"] = i + 1
        annList += [ann]
        
    return annList



class PascalSmall_RCNN(Pascal2012):
    def __init__(self, root, split, transform_function):
        if split == "train":
            split = "train_small"
        super().__init__(root, split, transform_function)
        
        if split == "train_small":
            self.img_names.sort()
            self.cls_names.sort()
            self.mask_names.sort()

            self.img_names = np.array(self.img_names)
            self.cls_names = np.array(self.cls_names)
            self.mask_names = np.array(self.mask_names)
            
            self.split = "train"


class PascalSmall_WISENet(Pascal2012):
    def __init__(self, root, split, transform_function):
        if split == "train":
            split = "train_small"
        super().__init__(root, split, transform_function)
        
        if split == "train_small":
            self.img_names.sort()
            self.cls_names.sort()
            self.mask_names.sort()

            self.img_names = np.array(self.img_names)
            self.cls_names = np.array(self.cls_names)
            self.mask_names = np.array(self.mask_names)
            
            self.split = "train"
