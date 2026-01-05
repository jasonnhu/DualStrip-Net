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
from segmentation_models_pytorch import Unet

from dataset.semi_dg import SemiDataset
from dataset.semi_mass import SemiDataset
from model.semseg.patchnet import PatchNet
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting DualStrip in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = Unet(encoder_name=cfg['backbone'], classes=2)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    output_device=local_rank, find_unused_parameters=True)

    testset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'test')

    testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=testsampler)

    checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
    model.load_state_dict(checkpoint['model'])

    if rank == 0:
        logger.info('************ Load best model from checkpoint.\n')
    eval_mode = 'original'
    mIoU, iou_class, F1 = evaluate(model, testloader, eval_mode, cfg, args.save_path)

    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Test ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Test {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
        logger.info('***** Test {} ***** >>>> F1: {:.2f}\n'.format(eval_mode, F1))


if __name__ == '__main__':
    main()
