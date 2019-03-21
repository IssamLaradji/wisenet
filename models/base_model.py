import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
import misc as ms
from skimage import morphology as morph
from torch.autograd import Function
import torchvision
# from core import proposals as prp
# from core import predict_methods as pm 
# from core import score_functions as sf 
import ann_utils as au


class Upsampler(nn.Module):
    def __init__(self, expansion_rate, n_output):
        super().__init__()
        
        self.score_32s = nn.Conv2d(512 *  4,
                                   n_output,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  4,
                                   n_output,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  4,
                                   n_output,
                                   kernel_size=1)
    
    def upsample(self, x_input, x_8s, x_16s, x_32s):
        input_spatial_dim = x_input.size()[2:]
        
        logits_8s = self.score_8s(x_8s)
        logits_16s = self.score_16s(x_16s)
        logits_32s = self.score_32s(x_32s)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
        
                
        logits_16s += nn.functional.interpolate(logits_32s,
                                        size=logits_16s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_8s += nn.functional.interpolate(logits_16s,
                                        size=logits_8s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_upsampled = nn.functional.interpolate(logits_8s,
                                       size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        return logits_upsampled


class BaseModel(nn.Module):
    def __init__(self, train_set, **model_options):
        super().__init__()
        self.options = model_options
        # self.predict_dict = ms.get_functions(pm)

        if hasattr(train_set, "n_classes"):
            self.n_classes = train_set.n_classes
        else:
            self.n_classes = train_set["n_classes"]  

        if hasattr(train_set, "ignore_index"):
            self.ignore_index = train_set.ignore_index
        else:
            self.ignore_index = -100

        self.blob_mode = None
        self.trained_batch_names = set()

    def sanity_checks(self, batch):
        if batch["split"][0] != "train":
            assert batch["name"][0] not in self.trained_batch_names 
        
    @torch.no_grad()
    def predict(self, batch, predict_method="probs"):
        self.sanity_checks(batch)
        self.eval()
        # ms.reload(pm)
        # self.predict_dict = ms.get_functions(pm)
        if predict_method == "counts":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)

            return {"counts":blob_dict["counts"]}

        elif predict_method == "probs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            return {"probs":probs}

        elif predict_method == "points":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)

            return {"points":blob_dict["points"], 
                    "pointList":blob_dict["pointList"],
                    "probs":probs}
            

        elif predict_method == "blobs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)
            
            return blob_dict

        else:
            print("Used predict method {}".format(predict_method))
            return self.predict_dict[predict_method](self, batch)




def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=(padding,padding))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)





class FeatureExtracter(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet50_32s.fc = nn.Sequential()
        self.resnet50_32s = resnet50_32s
        self.expansion_rate = resnet50_32s.layer1[0].expansion

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    
    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_8s, x_16s, x_32s