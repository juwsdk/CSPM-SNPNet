import os, argparse, time, datetime, sys, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, MonoModalRandomCropOut
from util.util import compute_results, init_weight
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import CSPM_SNP
#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='CSPM_SNP')
parser.add_argument('--batch_size', '-b', type=int, default=6)
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.99)
parser.add_argument('--epoch_max', '-em', type=int, default=200)  # please stop training mannully
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--n_layer', '-nl', type=int, default=50)
parser.add_argument('--data_dir', '-dr', type=str, default='./mfnet_dataset')
parser.add_argument('--weight_file', '-wf', type=str, default=None)
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    MonoModalRandomCropOut(crop_rate=0.3, prob_rgb=0.5, prob_thermal=0.0)]


# train model
def train(epo, model, train_loader, optimizer, batch_size):
    model.train()

    for it, (images, labels, names) in enumerate(train_loader):

        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)

        start_t = time.time()
        optimizer.zero_grad()

        logits_rgb, logits_thermal = model(images)

        loss_rgb = F.cross_entropy(logits_rgb, labels)
        loss_thermal = F.cross_entropy(logits_thermal, labels)
        loss = loss_rgb + loss_thermal
        loss.backward()
        optimizer.step()
        lr_this_epo = 0

        for param_group in optimizer.param_groups: lr_this_epo = param_group['lr']

        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f (rgb=%.4f  thermal=%.4f), time %s' \
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss), float(loss_rgb), float(loss_thermal),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))

        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
            writer.add_scalar('Train/loss_rgb', loss_rgb, accIter['train'])
            writer.add_scalar('Train/loss_thermal', loss_thermal, accIter['train'])
        view_figure = True

        if accIter['train'] % 100 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1, 255 // args.n_class)
                groundtruth_tensor = labels.unsqueeze(1) * scale
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits_rgb.argmax(1).unsqueeze(1) * scale
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1


# validate model
def validation(epo, model, val_loader):
    model.eval()

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):

            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            start_t = time.time()

            logits_rgb, logits_thermal = model(images)

            loss_rgb = F.cross_entropy(logits_rgb, labels)
            loss_thermal = F.cross_entropy(logits_thermal, labels)
            loss = loss_rgb + loss_thermal

            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f (rgb=%.4f  thermal=%.4f), time %s' \
                  % (
                      args.model_name, epo, args.epoch_max, it + 1, len(val_loader),
                      len(names) / (time.time() - start_t),
                      float(loss), float(loss_rgb), float(loss_thermal),
                      datetime.datetime.now().replace(microsecond=0) - start_datetime))

            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
                writer.add_scalar('Validation/loss_rgb', loss_rgb, accIter['val'])
                writer.add_scalar('Validation/loss_thermal', loss_thermal, accIter['val'])
            view_figure = False

            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)
                    groundtruth_tensor = labels.unsqueeze(1) * scale
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits_rgb.argmax(1).unsqueeze(1) * scale
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1



def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            logits, _ = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()

            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            conf_total += conf

            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
            args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
            datetime.datetime.now().replace(microsecond=0) - start_datetime))

    precision, recall, IoU= compute_results(conf_total)

    # 将指标记录到 TensorBoard
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)

    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)

    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
            args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch:(iou) unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump,"
                "average(mIoU %)\n")

    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, ' % (100 * IoU[i]))
        miou = 100 * np.mean(np.nan_to_num(IoU))
        f.write('%0.4f, %0.4f\n' % (miou))

    print('saving testing results.')

    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

    miou = 100 * np.mean(np.nan_to_num(IoU))
    return miou


# main function to start
if __name__ == '__main__':
    bn_eps = 1e-5
    bn_momentum = 0.1
    cudnn.benchmark = True

    seed = 12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.autograd.set_detect_anomaly(True)

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # 初始化模型
    model = eval(args.model_name)(num_classes=args.n_class, num_layers=args.n_layer)
    if args.weight_file:
        if os.path.exists(args.weight_file):
            weight_file = os.path.join(args.weight_file)
            print('Use the weight file.')
        else:
            sys.exit('No weight file found.')
        pretrained_weight = torch.load(weight_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
        own_state = model.state_dict()
        for name, param in pretrained_weight.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
    else:
        init_weight(model.business_layer, nn.init.kaiming_normal_, nn.BatchNorm2d, bn_eps, bn_momentum, mode='fan_in',
                    nonlinearity='relu')

    if args.gpu >= 0:
        model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    if os.path.exists("./runs"):
        shutil.rmtree("./runs")
    weight_dir = os.path.join("./runs", args.model_name)
    os.makedirs(weight_dir)

    writer = SummaryWriter("./runs/tensorboard_log")

    print(f'training {args.model_name}-{args.n_layer} on GPU #{args.gpu}')
    print(f'from epoch {args.epoch_from} / {args.epoch_max}')
    print(f'weight will be saved in: {weight_dir}')

    # prepare for dataset
    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset = MF_dataset(data_dir=args.data_dir, split='val')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    testing_log = {"miou_idx": 0}

    # start
    for epo in range(args.epoch_from, args.epoch_max):
        print(f'\ntrain {args.model_name}-{args.n_layer}, epo #{epo} begin...')

        train(epo, model, train_loader, optimizer, args.batch_size)
        validation(epo, model, val_loader)

        if epo % 10 == 0 or epo >= args.epoch_max * 0.9:
            checkpoint_model_file = os.path.join(weight_dir, f'{epo}.pth')
            print(f'saving checkpoint {checkpoint_model_file}')
            torch.save(model.state_dict(), checkpoint_model_file)

        miou= testing(epo, model, test_loader)
        if testing_log['miou'] < miou:
            testing_log['miou'], testing_log['miou_idx'] = miou, epo
            checkpoint_model_file = os.path.join(weight_dir, 'best.pth')
            torch.save(model.state_dict(), checkpoint_model_file)

        # Record pointer
        with open("log.txt", 'a') as f:
            f.write(
                f"miou: {testing_log['miou']}  miou_idx: {testing_log['miou_idx']}\n")

        scheduler.step()
