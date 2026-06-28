#coding=utf-8
import argparse
import copy
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.optim
import sys
# from tensorboardX import SummaryWriter
from utils.random_seed import setup_seed
from IMFuse_no1skip import Model
from IMFuse import IMFuse
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
from predict import AverageMeter, test_softmax
from utils.subsetboost import (
    SubsetRiskTracker,
    lattice_ranking_loss,
    make_random_superset_mask,
    mask_to_subset_indices,
    per_sample_seg_loss,
    resample_with_weak_subsets,
    subset_cvar_loss,
)

DEBUG_ITER = 1

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=3e-5, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--mamba_skip', action='store_true', default=False)
parser.add_argument('--first_skip', action='store_true', default=False)
parser.add_argument('--wandb_mode', default='disabled', type=str, choices=['online', 'offline', 'disabled'])
parser.add_argument('--subsetboost', action='store_true', default=False)
parser.add_argument('--subsetboost_warmup_epoch', default=5, type=int)
parser.add_argument('--subsetboost_topk', default=3, type=int)
parser.add_argument('--subsetboost_ema', default=0.95, type=float)
parser.add_argument('--subsetboost_cvar_fraction', default=0.5, type=float)
parser.add_argument('--subsetboost_cvar_lambda', default=0.2, type=float)
parser.add_argument('--subsetboost_weak_lambda', default=0.2, type=float)
parser.add_argument('--subsetboost_weak_weight', default=1.0, type=float)
parser.add_argument('--subsetboost_rank_lambda', default=0.05, type=float)
parser.add_argument('--subsetboost_rank_margin', default=0.02, type=float)
parser.add_argument('--subsetboost_rank_interval', default=4, type=int)
parser.add_argument('--subset_adapter', action='store_true', default=False)
parser.add_argument('--residual_boost', action='store_true', default=False)
parser.add_argument('--booster_hidden', default=16, type=int)
parser.add_argument('--residual_alpha', default=0.1, type=float)
parser.add_argument('--subset_size_booster_gate', action='store_true', default=False)
parser.add_argument('--booster_min_gate', default=0.25, type=float)
parser.add_argument('--strong_weak_distill', action='store_true', default=False)
parser.add_argument('--distill_lambda', default=0.05, type=float)
parser.add_argument('--distill_conf', default=0.7, type=float)
parser.add_argument('--distill_interval', default=4, type=int)
parser.add_argument('--distill_warmup_epoch', default=20, type=int)
parser.add_argument('--ema_teacher', action='store_true', default=False)
parser.add_argument('--ema_decay', default=0.999, type=float)
parser.add_argument('--weak_subset_sampling', action='store_true', default=False)
parser.add_argument('--weak_sampling_start_epoch', default=50, type=int)
parser.add_argument('--weak_sampling_prob', default=0.5, type=float)
parser.add_argument('--val_interval', default=50, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
# writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

val_check = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 850, 900, 910, 920, 930, 940, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000] 
print(f"Validation checks: {val_check}")

def strong_to_weak_distill_loss(student_pred, teacher_pred, confidence=0.7, eps=1e-6):
    teacher_prob = teacher_pred.detach().clamp(min=eps, max=1.0)
    student_prob = student_pred.clamp(min=eps, max=1.0)
    confidence_map = teacher_prob.max(dim=1, keepdim=True).values
    valid = (confidence_map >= confidence).float()
    if torch.sum(valid) < 1:
        valid = torch.ones_like(valid)
    kl_map = torch.sum(teacher_prob * (torch.log(teacher_prob) - torch.log(student_prob)), dim=1, keepdim=True)
    return torch.sum(kl_map * valid) / torch.clamp(torch.sum(valid), min=1.0)


def get_fuse_prediction(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


@torch.no_grad()
def update_ema_model(ema_model, student_model, decay):
    ema_state = ema_model.state_dict()
    student_state = student_model.state_dict()
    for key, ema_value in ema_state.items():
        student_value = student_state[key]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(student_value)


def freeze_teacher(teacher):
    teacher.eval()
    teacher.module.is_training = False
    for param in teacher.parameters():
        param.requires_grad_(False)


def main():
    ##########setting seed
    setup_seed(args.seed)
    
    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID") 
    wandb_name_and_id = f'{args.dataname}_IMFuse{"no_1_skip" if not args.first_skip else ""}_{"Interleaved" if args.interleaved_tokenization else ""}{"Skip" if args.mamba_skip else ""}_jobid{slurm_job_id}'
    wandb_mode = args.wandb_mode
    wandb.init(
        project="SegmentationMM",
        name=wandb_name_and_id,
        # entity="NeuroTumor",
        id=wandb_name_and_id,
        mode=wandb_mode,
        resume="allow",
        config={
            "architecture": "IMFuse",
            "learning_rate": args.lr, 
            "batch_size": args.batch_size,
            "iter_per_epoch": args.iter_per_epoch,
            "num_epochs": args.num_epochs,
            "datapath": args.datapath,
            "region_fusion_start_epoch": args.region_fusion_start_epoch,
            "interleaved_tokenization": args.interleaved_tokenization,
            "mamba_skip": args.mamba_skip,
            "subsetboost": args.subsetboost,
            "subset_adapter": args.subset_adapter,
            "residual_boost": args.residual_boost,
            "strong_weak_distill": args.strong_weak_distill,
            "ema_teacher": args.ema_teacher,
            "weak_subset_sampling": args.weak_subset_sampling,
            "subset_size_booster_gate": args.subset_size_booster_gate,
        }
    )
    
    ##########setting models
    if args.dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    if args.first_skip:
        model = IMFuse(
                    num_cls=num_cls, 
                    interleaved_tokenization=args.interleaved_tokenization,
                    mamba_skip=args.mamba_skip,
                    subset_adapter=args.subset_adapter,
                    residual_boost=args.residual_boost,
                    booster_hidden=args.booster_hidden,
                    residual_alpha=args.residual_alpha,
                    subset_size_booster_gate=args.subset_size_booster_gate,
                    booster_min_gate=args.booster_min_gate,
            )
    else:
        model = Model(
                        num_cls=num_cls, 
                        interleaved_tokenization=args.interleaved_tokenization,
                        mamba_skip=args.mamba_skip,
                        subset_adapter=args.subset_adapter,
                        residual_boost=args.residual_boost,
                        booster_hidden=args.booster_hidden,
                        residual_alpha=args.residual_alpha,
                        subset_size_booster_gate=args.subset_size_booster_gate,
                        booster_min_gate=args.booster_min_gate,
                )
    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ########## Setting learning scheduler and optimizer    
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.RAdam(train_params)
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)

    ########## Setting data
    if args.dataname in ['BRATS2023', 'BRATS2020', 'BRATS2015']:
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits.csv'
        val_file = 'datalist/val15splits.csv'
        #test_file = 'datalist/test.txt'
        #val_file = 'datalist/val.txt'
    elif args.dataname == 'BRATS2018':
        #### BRATS2018 contains three splits (1,2,3)
        test_file = 'datalist/Brats18_test15splits.csv'
        val_file = 'datalist/Brats18_val15splits.csv'
        train_file = 'datalist/train3.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, 
                                    root=args.datapath, 
                                    num_cls=num_cls, 
                                    train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, 
                                    root=args.datapath, 
                                    test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, 
                                    root=args.datapath, 
                                    num_cls=num_cls, 
                                    val_file=val_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = min(len(train_loader), DEBUG_ITER) if args.debug else len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0
    subset_tracker = SubsetRiskTracker(momentum=args.subsetboost_ema, device=torch.device('cuda')) if args.subsetboost else None
    checkpoint = None

    ##########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume, weights_only=False)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        allow_new_subset_modules = args.subset_adapter or args.residual_boost
        load_result = model.load_state_dict(
            checkpoint['state_dict'],
            strict=not allow_new_subset_modules,
        )
        if allow_new_subset_modules:
            logging.info('Resume missing keys: {}'.format(load_result.missing_keys))
            logging.info('Resume unexpected keys: {}'.format(load_result.unexpected_keys))
        val_Dice_best = checkpoint['val_Dice_best']
        if 'optim_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optim_dict'])
            except ValueError as exc:
                logging.warning('Skip optimizer state because model parameters changed: {}'.format(exc))
        start_epoch = checkpoint['epoch'] + 1
        if subset_tracker is not None and 'subsetboost_tracker' in checkpoint:
            subset_tracker.load_state_dict(checkpoint['subsetboost_tracker'])

    ema_model = None
    if args.ema_teacher:
        ema_model = copy.deepcopy(model)
        if checkpoint is not None and 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        freeze_teacher(ema_model)
        logging.info('EMA self-teacher enabled with decay {}'.format(args.ema_decay))

    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        # writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        model.train()
        model.module.is_training = True
        if ema_model is not None:
            freeze_teacher(ema_model)

        prm_cross_loss_epoch = 0.0
        prm_dice_loss_epoch = 0.0
        fuse_cross_loss_epoch = 0.0
        fuse_dice_loss_epoch = 0.0
        sep_cross_loss_epoch = 0.0
        sep_dice_loss_epoch = 0.0
        loss_epoch = 0.0
        subsetboost_loss_epoch = 0.0
        subsetboost_cvar_epoch = 0.0
        subsetboost_weak_epoch = 0.0
        subsetboost_rank_epoch = 0.0
        subsetboost_distill_epoch = 0.0
        weak_sampling_count_epoch = 0

        ########## training epoch
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3] #x=(B, M=4, 128, 128, 128), target = (B, C, 128, 128, 128), mask = (B, 4)
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            weak_sample_count = 0

            if (
                args.weak_subset_sampling
                and subset_tracker is not None
                and (epoch >= args.weak_sampling_start_epoch or args.debug)
            ):
                mask, weak_sample_count = resample_with_weak_subsets(
                    mask,
                    subset_tracker,
                    topk=args.subsetboost_topk,
                    probability=args.weak_sampling_prob,
                )
                weak_sampling_count_epoch += weak_sample_count

            fuse_pred, sep_preds, prm_preds = model(x, mask)

            ###Loss compute
            ### fuse modality segmentation loss
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            fuse_cross_loss_epoch += fuse_cross_loss
            fuse_dice_loss_epoch += fuse_dice_loss

            ### separated modality segmentation loss
            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            sep_cross_loss_epoch += sep_cross_loss
            sep_dice_loss_epoch += sep_dice_loss

            ### pyramid segmentation loss
            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            prm_cross_loss_epoch += prm_cross_loss
            prm_dice_loss_epoch += prm_dice_loss

            subsetboost_loss = torch.zeros(1).cuda().float()
            subsetboost_cvar = torch.zeros(1).cuda().float()
            subsetboost_weak = torch.zeros(1).cuda().float()
            subsetboost_rank = torch.zeros(1).cuda().float()
            subsetboost_distill = torch.zeros(1).cuda().float()
            booster_scale = getattr(model.module.decoder_fuse, 'last_booster_scale', None)
            if args.subsetboost:
                subset_indices = mask_to_subset_indices(mask)
                fuse_sample_losses = per_sample_seg_loss(fuse_pred, target, num_cls, criterions)
                subset_tracker.update(subset_indices, fuse_sample_losses)

                if epoch >= args.subsetboost_warmup_epoch:
                    subsetboost_cvar = subset_cvar_loss(
                        fuse_sample_losses,
                        args.subsetboost_cvar_fraction,
                    )
                    subsetboost_weak = subset_tracker.weak_weighted_loss(
                        fuse_sample_losses,
                        subset_indices,
                        args.subsetboost_topk,
                        args.subsetboost_weak_weight,
                    )
                    subsetboost_loss = (
                        args.subsetboost_cvar_lambda * subsetboost_cvar
                        + args.subsetboost_weak_lambda * subsetboost_weak
                    )

                    if (
                        args.subsetboost_rank_lambda > 0
                        and args.subsetboost_rank_interval > 0
                        and step % args.subsetboost_rank_interval == 0
                    ):
                        superset_mask, valid_rank = make_random_superset_mask(mask)
                        if torch.any(valid_rank):
                            superset_fuse_pred = model(x, superset_mask)[0]
                            superset_losses = per_sample_seg_loss(
                                superset_fuse_pred,
                                target,
                                num_cls,
                                criterions,
                            )
                            subsetboost_rank = lattice_ranking_loss(
                                fuse_sample_losses,
                                superset_losses,
                                valid_rank,
                                args.subsetboost_rank_margin,
                            )
                            subsetboost_loss = (
                                subsetboost_loss
                                + args.subsetboost_rank_lambda * subsetboost_rank
                            )

            if (
                args.strong_weak_distill
                and args.distill_lambda > 0
                and args.distill_interval > 0
                and (epoch >= args.distill_warmup_epoch or args.debug)
                and step % args.distill_interval == 0
            ):
                full_mask = torch.ones_like(mask).bool()
                teacher_model = ema_model if ema_model is not None else model
                with torch.no_grad():
                    teacher_fuse_pred = get_fuse_prediction(teacher_model(x, full_mask))
                subsetboost_distill = strong_to_weak_distill_loss(
                    fuse_pred,
                    teacher_fuse_pred,
                    confidence=args.distill_conf,
                )
                subsetboost_loss = subsetboost_loss + args.distill_lambda * subsetboost_distill

            subsetboost_loss_epoch += subsetboost_loss
            subsetboost_cvar_epoch += subsetboost_cvar
            subsetboost_weak_epoch += subsetboost_weak
            subsetboost_rank_epoch += subsetboost_rank
            subsetboost_distill_epoch += subsetboost_distill

            ### total segmentation loss
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss
            loss = loss + subsetboost_loss

            loss_epoch += loss

            ### backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)
                freeze_teacher(ema_model)

            ###log
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            if args.subsetboost:
                msg += 'subsetboost:{:.4f}, cvar:{:.4f}, weak:{:.4f}, rank:{:.4f},'.format(
                    subsetboost_loss.item(),
                    subsetboost_cvar.item(),
                    subsetboost_weak.item(),
                    subsetboost_rank.item(),
                )
            if args.strong_weak_distill:
                msg += 'distill:{:.4f},'.format(subsetboost_distill.item())
            if args.weak_subset_sampling:
                msg += 'weak_resampled:{},'.format(weak_sample_count if 'weak_sample_count' in locals() else 0)
            if booster_scale is not None:
                msg += 'booster_scale:{:.4f},'.format(booster_scale.item())
            logging.info(msg)

            """
            if args.debug:
                break
            """

        logging.info('train time per epoch: {}'.format(time.time() - b))

        ########## log current epoch metrics and save current model 
        log_payload = {
            "train/epoch": epoch,
            "train/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/fusecross": fuse_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/fusedice": fuse_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/sepcross": sep_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/sepdice": sep_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/prmcross": prm_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/prmdice": prm_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
            "train/learning_rate": step_lr,
        }
        if args.subsetboost:
            log_payload.update({
                "subsetboost/loss": subsetboost_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "subsetboost/cvar": subsetboost_cvar_epoch.cpu().detach().item() / iter_per_epoch,
                "subsetboost/weak": subsetboost_weak_epoch.cpu().detach().item() / iter_per_epoch,
                "subsetboost/rank": subsetboost_rank_epoch.cpu().detach().item() / iter_per_epoch,
            })
            logging.info('SubsetBoost weak risk table: {}'.format(subset_tracker.summary(args.subsetboost_topk)))
        if args.strong_weak_distill:
            log_payload["subsetboost/distill"] = subsetboost_distill_epoch.cpu().detach().item() / iter_per_epoch
        if args.weak_subset_sampling:
            log_payload["subsetboost/weak_sampling_count"] = weak_sampling_count_epoch / iter_per_epoch
        if ema_model is not None:
            log_payload["subsetboost/ema_decay"] = args.ema_decay
        wandb.log(log_payload)

        file_name = os.path.join(ckpts, 'model_last.pth')
        checkpoint_payload = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            }
        if args.subsetboost:
            checkpoint_payload['subsetboost_tracker'] = subset_tracker.state_dict()
        if ema_model is not None:
            checkpoint_payload['ema_state_dict'] = ema_model.state_dict()
        torch.save(checkpoint_payload, file_name)
        
        ########## validation and test
        if epoch+1 in val_check or args.debug or (args.val_interval > 0 and (epoch + 1) % args.val_interval == 0):
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname)
        
            val_WT, val_TC, val_ET, val_ETpp = dice_score #validate(model, val_loader)
            logging.info('Validate epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}, loss = {:.2}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), seg_loss.cpu().item()))
            val_dice = (val_ET + val_WT + val_TC)/3
            wandb.log({
                "val/epoch":epoch,
                "val/val_ET_Dice": val_ET.item(),
                "val/val_ETpp_Dice": val_ETpp.item(),
                "val/val_WT_Dice": val_WT.item(),
                "val/val_TC_Dice": val_TC.item(),
                "val/val_Dice": val_dice.item(), 
                "val/seg_loss": seg_loss.cpu().item(),       
            })
            
            if val_dice > val_Dice_best:
                val_Dice_best = val_dice.item()
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                best_payload = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'val_Dice_best': val_Dice_best,
                    }
                if args.subsetboost:
                    best_payload['subsetboost_tracker'] = subset_tracker.state_dict()
                if ema_model is not None:
                    best_payload['ema_state_dict'] = ema_model.state_dict()
                torch.save(best_payload, file_name)
                
            print('testing ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    test_loader,
                    model,
                    dataname = args.dataname)
            test_WT, test_TC, test_ET, test_ETpp = dice_score   
            logging.info('Testing epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ET_postpro = {:.2}'.format(epoch, test_WT.item(), test_TC.item(), test_ET.item(), test_ETpp.item()))
            test_dice = (test_ET + test_WT + test_TC)/3
            wandb.log({
                "test/epoch":epoch,
                "test/test_WT_Dice": test_WT.item(),
                "test/test_TC_Dice": test_TC.item(),
                "test/test_ET_Dice": test_ET.item(),
                "test/test_ETpp": test_ETpp.item(),
                "test/test_Dice": test_dice.item(),  
                "test/seg_loss": seg_loss.cpu().item(),   
            })

            model.train()
            model.module.is_training=True


    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    """
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))
    """

if __name__ == '__main__':
    main()
