import argparse
import os

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from models.alexnet import *
from utils import *


def main(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating alexnet model ...")
    model = AlexNet(num_classes=args.num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # resume from a checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)

        # load model & optimizer
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit()

    cudnn.benchmark = True

    # create transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read image files
    image_path_list = []
    for (path, dir, files) in os.walk(args.data):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            ext_lower = ext.lower()
            if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg':
                image_path_list.append(os.path.join(path, filename))

    # switch to evaluate mode
    model.eval()

    top5_list = []
    with torch.no_grad():
        for image_path in image_path_list:
            # read image
            image = Image.open(image_path).convert('RGB')

            # transform image
            image = transform(image)

            # reshape image for model inference
            image = image.view(1, image.shape[0], image.shape[1], image.shape[2])

            # convert to cuda(if possible)
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)

            # infer model
            output = model(image)

            # calculate top k prediction
            _, pred = output.topk(args.topk, 1, True, True)
            pred = pred.cpu().numpy()[0]

            # print prediction
            print('filename: {}, pred: '.format(os.path.basename(image_path)), pred)

            # append to topk list
            top5_list.append((os.path.basename(image_path), pred))

    # write to csv
    with open(args.output, 'wt') as wf:
        for filename, pred in top5_list:
            wf.write('{},'.format(filename))
            for i, p in enumerate(pred):
                wf.write('{}'.format(p))
                # write comma except last
                if i < len(pred) - 1:
                    wf.write(',')
            wf.write('\n')

        print("=> write csv to {}".format(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='', metavar='DIR', help='path to dataset')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--num-classes', default=5, type=int, metavar='N', help='number of classes')
    parser.add_argument('--output', default='output.csv', help='output csv path')
    parser.add_argument('--topk', default=5, type=int, metavar='N', help='top k')
    args = parser.parse_args()

    main(args)