import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.device_count()

import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import test_batchify_fn 
from encoding.models.sseg import BaseNet
from lseg.additional_utils.models import LSeg_MultiEvalModule
from lseg.modules.lseg_module import LSegModule

import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from lseg.data import get_dataset
import torchvision.transforms as transforms

from dn_splatter.utils.lseg_utils import lseg_module_init, get_new_pallete, get_new_mask_pallete

model = lseg_module_init()

alpha = 0.5

# img_path = 'inputs/cat1.jpeg'
img_path = 'data/e9ac2fc517_original/DSC08479_original.png'

crop_size = 512 #480
padding = [0.0] * 3
image = Image.open(img_path)
image = np.array(image)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
image = transform(image).unsqueeze(0)
img = image[0].permute(1,2,0)
img = img * 0.5 + 0.5
plt.imshow(img)

# args.label_src = 'plant,grass,cat,stone,other'
label_src = 'monitor,cabinet,floor,ceiling,chair,table,wall,other'

labels = []
print('** Input label value: {} **'.format(label_src))
lines = label_src.split(',')
for line in lines:
    label = line
    labels.append(label)


with torch.no_grad():
    # [[1, 8, 512, 512]]
    # outputs = evaluator.parallel_forward(image, labels) #evaluator.forward(image, labels) #parallel_forward

    # outputs = [evaluator.forward(image.to("cuda"), labels)] # need to add this bracelet to make it a list

    # outputs = [model(image.to("cuda"), labels)[0].unsqueeze(0)] # poorer results than the above outputs

    # [1, 512, 256, 256]
    outputs = model.net.get_image_features(image.to("cuda"))
    # print(outputs)

    predicts = [
        torch.max(output, 1)[1].cpu().numpy() 
        for output in outputs
    ]
    
predict = predicts[0]

# show results
new_palette = get_new_pallete(len(labels))
mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=labels)
img = image[0].permute(1,2,0)
img = img * 0.5 + 0.5
img = Image.fromarray(np.uint8(255*img)).convert("RGBA")
seg = mask.convert("RGBA")
out = Image.blend(img, seg, alpha)
plt.axis('off')
plt.imshow(img)
plt.figure()
plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
plt.axis('off')
plt.imshow(seg)