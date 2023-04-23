import argparse
import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


class PathDataset(Dataset):
    def __init__(self, img_list, img_path, transform=None):
        """
        Args:
            txt_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(img_list, "r") as f:
            self.file_names = [os.path.join(img_path, x.strip() + '.jpg') for x in f.readlines()]

        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def _load_img(self, index):
        _img = np.array(Image.open(self.file_names[index]).convert('RGB')).astype(np.float32)
        return Image.fromarray(_img.astype('uint8'), 'RGB')

    def __getitem__(self, idx):
        _img = self._load_img(idx)

        if self.transform:
            _img = self.transform(_img)

        return _img


def dataset_embed(model, img_list, img_path, batch_size, workers, device, transform):
    cudnn.benchmark = True

    # Data loading code
    # TODO: Make this modular
    normalize = transforms.Normalize(mean=transform['mean'],
                                     std=transform['std'])

    loader = torch.utils.data.DataLoader(
        PathDataset(img_list, img_path, transforms.Compose([
            transforms.Resize(transform['Resize']),
            transforms.CenterCrop(transform['CenterCrop']),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return generateEmbeddings(loader, model, device, transform['downsample'])


def generateEmbeddings(val_loader, model, device, downsample):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        outputs = []
        for i, images in enumerate(val_loader):
            images = images.to(device)
            output = model(images)[-1][:, :, ::downsample, ::downsample].permute(0, 2, 3, 1)
            output = output.reshape(-1, output.size()[-1])
            outputs.append(output.cpu())
    outputs = torch.cat(outputs, dim=0)
    print("Done extracting features for cluster initialization")

    return outputs
