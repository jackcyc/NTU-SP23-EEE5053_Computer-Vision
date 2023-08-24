import os
import random

import numpy as np
import torch
from dataset import build_dataloader
from opt import parse_args
from RITnet_v3 import DenseNet2D
from tqdm import tqdm
from utils import (CrossEntropyLoss2d, GeneralizedDiceLoss, Logger, get_nparams, get_predictions, mIoU, total_metric)

# reproducibility
seed = 42
# Set the random seed for CPU
random.seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
# Set the random seed for GPU (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# Set the random seed for PyTorch
torch.manual_seed(seed)


def lossandaccuracy(loader, model, factor):
    epoch_loss = []
    ious = []
    acc0s = []
    acc1s = []
    model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(tqdm(loader)):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device, non_blocking=True)

            target = labels.to(device, non_blocking=True).long()
            output, eye_pred = model(data)

            ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output, target)
            loss = CE_loss * (torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device) +
                              (spatialWeights).to(torch.float32).to(device))

            loss = torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output, target)
            # loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))

            ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (loss_dice) + loss

            epoch_loss.append(loss.item())
            predict = get_predictions(output)
            iou = mIoU(predict, labels)
            ious.append(iou)

            eye_gt = (labels.sum(dim=(1, 2)) != 0).numpy()
            mask_0 = (eye_gt == 0)
            mask_1 = (eye_gt == 1)
            eye_pred = eye_pred.softmax(1)[:, 1].cpu().numpy()

            if sum(mask_0) != 0:
                acc0s += (1 - eye_pred)[mask_0].tolist()
            if sum(mask_1) != 0:
                acc1s += eye_pred[mask_1].tolist()
    logger.write(f'eye_acc0: {np.average(acc0s)}, eye_acc1: {np.average(acc1s)}')
    return np.average(epoch_loss), np.average(ious)


if __name__ == '__main__':

    args = parse_args()
    kwargs = vars(args)

    device = torch.device("cuda")

    LOGDIR = '{}'.format(args.expname)
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(LOGDIR + '/models', exist_ok=True)
    logger = Logger(os.path.join(LOGDIR, 'logs.log'))

    model = DenseNet2D()
    ckpt = torch.load('./all.git_ok')
    msg = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(msg)
    model = model.to(device)

    model.train()
    nparams = get_nparams(model)

    trainloader, validloader = build_dataloader(args.data_path)

    optimizer = torch.optim.Adam([
        {
            'params': [p for n, p in model.named_parameters() if 'clf' not in n]
        },
        {
            'params': [p for n, p in model.named_parameters() if 'clf' in n]
        },
    ],
                                 lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[2e-4, 5e-3], total_steps=len(trainloader) * 10)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    # criterion_SL = SurfaceLoss()

    criterion_eye = torch.nn.CrossEntropyLoss(weight=torch.tensor([4, 1], dtype=torch.float32, device='cuda'))

    #    alpha = 1 - np.arange(1,args.epochs)/args.epoch
    ##The weighing function for the dice loss and surface loss
    # alpha=np.zeros(((args.epochs)))
    # alpha[0:np.min([125,args.epochs])]=1 - np.arange(1,np.min([125,args.epochs])+1)/np.min([125,args.epochs])
    # if args.epochs>125:
    #     alpha[125:]=1
    ious = []

    # val first
    # lossvalid , miou = lossandaccuracy(validloader,model,1)
    # totalperf = total_metric(nparams,miou)
    # f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'
    # logger.write(f.format(-1,lossvalid, miou,nparams,totalperf))

    for epoch in range(args.epochs - 10, args.epochs):
        model.train()
        for i, batchdata in enumerate(tqdm(trainloader)):
            #            print (len(batchdata))
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device, non_blocking=True)
            target = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            output, eye_cond = model(data)
            ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output, target)
            loss = CE_loss * (torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device) +
                              (spatialWeights).to(torch.float32).to(device))

            loss = torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output, target)
            # loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))

            ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            # loss = (1-alpha[epoch])*loss_sl+alpha[epoch]*(loss_dice)+loss
            loss = (loss_dice) + loss

            eye_gt = (labels.sum(dim=(1, 2)) != 0).long().cuda()
            loss = loss + criterion_eye(eye_cond, eye_gt)
            #
            predict = get_predictions(output)
            iou = mIoU(predict, labels)
            ious.append(iou)

            if i % 100 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), loss.item()))

            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.write('Epoch:{}, Train mIoU: {}'.format(epoch, np.average(ious)))
        lossvalid, miou = lossandaccuracy(validloader, model, 1)
        totalperf = total_metric(nparams, miou)
        f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'
        logger.write(f.format(epoch, lossvalid, miou, nparams, totalperf))

        # scheduler.step(lossvalid)

        ##save the model every epoch
        if epoch % 1 == 0:
            torch.save(model.state_dict(), '{}/models/dense_net{}.pkl'.format(LOGDIR, epoch))
