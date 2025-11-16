import argparse
import logging
import os
import pprint
import pdb
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import itertools
from dataset.semi_dg import SemiDataset
# from dataset.semi_mass import SemiDataset
# from dataset.semi_chn6 import SemiDataset
from model.semseg.patchnet import PatchNet
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter,criterion, dice_bce_loss
from util.dist_helper import setup_distributed
import torch.nn.functional as F
import random

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

IMG_SIZE = 512
def patch_mask(x, size=8):
    kernel = stride = IMG_SIZE // size
    return F.max_pool2d(x, kernel, stride)

def patch_pred(x, size=8):
    return F.adaptive_max_pool2d(x, (size, size))

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = PatchNet(cfg)
    
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'DICE_BCE':
        criterion_l = dice_bce_loss().cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_p = nn.BCELoss().cuda(local_rank)
    criterion_sfc = nn.MSELoss().cuda(local_rank)
    criterion_dice_s = dice_bce_loss().cuda(local_rank)
    criterion_scc = nn.L1Loss().cuda(local_rank)
    trainset_u = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    
    trainset_l = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)


    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_h = AverageMeter()
        total_loss_v = AverageMeter()
        total_loss_sfc = AverageMeter()
        total_loss_scc = AverageMeter()
        total_loss_u_h = AverageMeter()
        total_loss_u_v = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask)) in enumerate(loader):


            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()

            #start training
            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            outs, feas_h, feas_v, outs_h, outs_v = model([img_x, img_u_w, img_u_s1, img_u_s2])

            out_h, out_u_h = outs_h.split([num_lb, num_ulb])
            out_v, out_u_v = outs_v.split([num_lb, num_ulb])


            out, out_u = outs.split([num_lb, num_ulb])
            pred_u_w = out_u.detach()
            mask_u_w = pred_u_w.argmax(dim=1)

            loss_x = criterion_l(out, mask_x)
            loss_h = criterion_l(out_h, mask_x)
            loss_v = criterion_l(out_v, mask_x)

            loss_u_h = criterion_l(out_u_h, mask_u_w)
            loss_u_v = criterion_l(out_u_v, mask_u_w)

            # 根据配置组合损失
            loss = loss_x + loss_h + loss_v + loss_u_h + loss_u_v  # 基础损失

            # Initialize loss_sfc and loss_scc
            loss_sfc = torch.tensor(0.0).cuda(local_rank)
            loss_scc = torch.tensor(0.0).cuda(local_rank)

            
            loss_sfc = criterion_sfc(feas_h, feas_v)
            loss = loss + loss_sfc

            loss_scc = criterion_scc(outs_h, outs_v)
            loss = loss + loss_scc

            # Update all loss meters
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_h.update(loss_h.item())
            total_loss_v.update(loss_v.item())
            total_loss_sfc.update(loss_sfc.item())
            total_loss_scc.update(loss_scc.item())
            total_loss_u_h.update(loss_u_h.item())
            total_loss_u_v.update(loss_u_v.item())

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Synchronize loss across all processes
            torch.distributed.all_reduce(loss)
            loss = loss / world_size

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 2
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_h', loss_h.item(), iters)
                writer.add_scalar('train/loss_v', loss_v.item(), iters)
                writer.add_scalar('train/loss_sfc', loss_sfc.item(), iters)
                writer.add_scalar('train/loss_scc', total_loss_scc.avg, iters)
                writer.add_scalar('train/loss_u_h', loss_u_h.item(), iters)
                writer.add_scalar('train/loss_u_v', loss_u_v.item(), iters)

            if (i % (len(trainloader_l) // 8) == 0) and (rank == 0):
                sfc_info = f', Loss sfc: {total_loss_sfc.avg:.4f}' if use_sfc else ''
                scc_info = f', Loss scc: {total_loss_scc.avg:.4f}' if use_scc else ''
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss h: {:.3f}, Loss v: {:.3f}, '
                          'Loss u_h: {:.3f}, Loss u_v: {:.3f}{}{}'.format(
                    i, total_loss.avg, total_loss_x.avg, total_loss_h.avg, total_loss_v.avg,
                    total_loss_u_h.avg, total_loss_u_v.avg, sfc_info, scc_info))
        eval_mode = 'sliding_window'  if cfg['dataset'] == 'mass' else 'original'
        # Before evaluation
        torch.distributed.barrier()
        mIoU, iou_class, F1 = evaluate(model, valloader, eval_mode, cfg)

        # Add synchronization point
        torch.distributed.barrier()

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f} F1: {:.2f}\n'.format(eval_mode, mIoU, F1))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
