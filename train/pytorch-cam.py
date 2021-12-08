import os, sys
from models import *
from albumentations.augmentations.functional import normalize
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import *
from lib import *
import random
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

def csv2list(csv_file):
    f = open(csv_file, 'r')
    csvf = csv.reader(f)
    lst = []
    for item in csvf:
        lst.append(item)
    return lst

input_tensor = None

imgs = csv2list("/home/eslab/wyh/spindle_test.csv")

data_path = Path("/home/eslab/wyh/data/img/resize/spindle_only_eeg_rgb/")
checkpoint_name = 'SPindle_test_withB_resnet18_rgb2.pth'
img_name = "001_3_/0001_0_.png"

net = resnet18_grayscale(num_classes=2)

net = net.to('cuda')
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+checkpoint_name)
#print(checkpoint)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
scheduler = checkpoint['scheduler']
optimizer = checkpoint['optimizer']
print("best acc: %lf" %(best_acc))

target_layers = [net.module.layer4[-1]]
#input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = EigenCAM(model=net, target_layers=target_layers, use_cuda=True)


transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])


for img_name in imgs:
    img_name = img_name[0]
    rgb_img = cv2.imread(str(data_path / img_name), 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    """
    with open(data_path / img_name, 'rb') as f:
        # Open image
        img = Image.open(f)

        input_tensor = transform(img).unsqueeze(1)
    """
    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    #print(grayscale_cam)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    final_img = Image.fromarray(visualization.astype('uint8'), 'RGB')

    final_img.save("./test/"+img_name.split("/")[0]+img_name.split("/")[-1])