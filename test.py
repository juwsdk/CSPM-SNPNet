import os
import argparse
import time
import datetime
import sys
import shutil
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat
from model import CSPM_SNP

#############################################################################################
# Argument Parsing
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='CSPM_SNP')
parser.add_argument('--weight_name', '-w', type=str, default='CSPM_SNP_50')
parser.add_argument('--file_name', '-f', type=str, default='best.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--n_layer', '-nl', type=int, default=101)
parser.add_argument('--data_dir', '-dr', type=str, default='./mfnet_dataset/')
parser.add_argument('--model_dir', '-wd', type=str, default='./runs/CSPM_SNP')
args = parser.parse_args()


#############################################################################################

if __name__ == '__main__':

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # prepare save directory
    if os.path.exists("./runs"):
        os.makedirs("./runs/vis_results")
    else:
        sys.exit("no run folder!")

    model_dir = os.path.join(args.model_dir)
    if not os.path.exists(model_dir):
        sys.exit(f"the {model_dir} does not exist.")

    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file):
        print('use the loaded model file.')
    else:
        sys.exit('no model file found.')
    print(f'testing {args.model_name}-{args.n_layer} on GPU #{args.gpu} with pytorch')

    # 初始化混淆矩阵
    conf_total_rgb = np.zeros((args.n_class, args.n_class))
    conf_total_thermal = np.zeros((args.n_class, args.n_class))
    model = eval(args.model_name)(num_classes=args.n_class, num_layers=args.n_layer)
    if args.gpu >= 0: model.cuda(args.gpu)

    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1
    test_dataset = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    ave_time_cost = 0.0
    model.eval()

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_time = time.time()
            logits_rgb, logits_thermal = model(images)
            end_time = time.time()

            if it >= 5:
                ave_time_cost += (end_time - start_time)

            label = labels.cpu().numpy().squeeze().flatten()
            prediction_rgb = logits_rgb.argmax(1).cpu().numpy().squeeze().flatten()
            prediction_thermal = logits_thermal.argmax(1).cpu().numpy().squeeze().flatten()

            conf_rgb = confusion_matrix(y_true=label, y_pred=prediction_rgb, labels=range(args.n_class))  # RGB
            conf_total_rgb += conf_rgb
            conf_thermal = confusion_matrix(y_true=label, y_pred=prediction_thermal,
                                            labels=range(args.n_class))  # Thermal
            conf_total_thermal += conf_thermal

            visualize(image_name=names, predictions=logits_rgb.argmax(1), weight_name=args.weight_name, name='rgb')
            visualize(image_name=names, predictions=logits_thermal.argmax(1), weight_name=args.weight_name,
                      name='thermal')
            print(
                f"{args.model_name}, {args.weight_name}, frame {it + 1}/{len(test_loader)}, {names}, time cost: {(end_time - start_time) * 1000:.2f} ms, demo result saved.")

    precision_rgb, recall_rgb, iou_rgb= compute_results(conf_total_rgb)
    precision_thermal, recall_thermal, iou_thermal= compute_results(conf_total_thermal)

    conf_total_matfile = os.path.join("./runs", f'conf_{args.weight_name}.mat')
    savemat(conf_total_matfile, {'conf_rgb': conf_total_rgb, 'conf_thermal': conf_total_thermal})

    print('\n###########################################################################')
    print(
        f'\n{args.model_name}-{args.n_layer} test results (with batch size {batch_size}) on {datetime.date.today()} using {torch.cuda.get_device_name(args.gpu)}:')
    print(f'\n* the tested dataset name: {args.dataset_split}')
    print(f'* the tested image count: {len(test_loader)}')
    print(f'* the tested image size: {args.img_height}*{args.img_width}')
    print(f'* the weight name: {args.weight_name}')
    print(f'* the file name: {args.file_name}\n')

    print("################################ RGB ##################################")
    print("* IoU per class: " + ", ".join([f"{iou_rgb[i]:.6f}" for i in range(args.n_class)]))
    print(f"* mIoU: {np.nanmean(iou_rgb):.6f}")

    print("################################ Thermal ##################################")
    print("* IoU per class: " + ", ".join([f"{iou_thermal[i]:.6f}" for i in range(args.n_class)]))
    print(f"* mIoU: {np.nanmean(iou_thermal):.6f}")

    print('\n###########################################################################')
