import torch
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader 
from IMFuse import IMFuse
from IMFuse_no1skip import Model
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--test_file', default='datalist/test15splits.csv', type=str)
parser.add_argument('--datapath', default="/work/grana_neuro/missing_modalities/BRATS2023_Training_npy", type=str)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--mamba_skip', action='store_true', default=False)
parser.add_argument('--first_skip', action='store_true', default=False)
parser.add_argument('--subset_adapter', action='store_true', default=False)
parser.add_argument('--residual_boost', action='store_true', default=False)
parser.add_argument('--booster_hidden', default=16, type=int)
parser.add_argument('--residual_alpha', default=0.1, type=float)
parser.add_argument('--subset_size_booster_gate', action='store_true', default=False)
parser.add_argument('--booster_min_gate', default=0.25, type=float)
parser.add_argument('--all_splits', action='store_true', default=False)
#parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

if __name__ == '__main__':
    args = parser.parse_args()
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
    
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = args.datapath
    test_file = args.test_file
    save_path = args.savepath
    num_cls = 4
    dataname = args.dataname
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    checkpoint = torch.load(args.resume, weights_only=False)
    state_dict = checkpoint['state_dict']
    checkpoint_has_adapter = any('subset_adapter' in key for key in state_dict)
    checkpoint_has_booster = any('residual_booster' in key for key in state_dict)
    subset_adapter = args.subset_adapter or checkpoint_has_adapter
    residual_boost = args.residual_boost or checkpoint_has_booster
    checkpoint_has_booster_gate = any('subset_size_gate_flag' in key for key in state_dict)
    subset_size_booster_gate = args.subset_size_booster_gate or checkpoint_has_booster_gate

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if args.first_skip:
        model = IMFuse(
                    num_cls=num_cls,
                    interleaved_tokenization=args.interleaved_tokenization,
                    mamba_skip=args.mamba_skip,
                    subset_adapter=subset_adapter,
                    residual_boost=residual_boost,
                    booster_hidden=args.booster_hidden,
                    residual_alpha=args.residual_alpha,
                    subset_size_booster_gate=subset_size_booster_gate,
                    booster_min_gate=args.booster_min_gate,
                )
    else:
        model = Model(
                    num_cls=num_cls,
                    interleaved_tokenization=args.interleaved_tokenization,
                    mamba_skip=args.mamba_skip,
                    subset_adapter=subset_adapter,
                    residual_boost=residual_boost,
                    booster_hidden=args.booster_hidden,
                    residual_alpha=args.residual_alpha,
                    subset_size_booster_gate=subset_size_booster_gate,
                    booster_min_gate=args.booster_min_gate,
                )
    model = torch.nn.DataParallel(model).cuda()
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print('missing keys:', load_result.missing_keys)
        print('unexpected keys:', load_result.unexpected_keys)
    best_epoch = checkpoint['epoch'] + 1
    out_path = args.savepath
    split_tag = 'all' if args.all_splits else str(index)
    output_path = f"{out_path}_{best_epoch}_{split_tag}.txt"
    eval_start = 0 if args.all_splits else index * 5
    eval_end = len(masks) if args.all_splits else (index + 1) * 5

    test_score = AverageMeter()
    subset_records = []
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for mask_global_idx, mask in enumerate(masks[eval_start:eval_end], start=eval_start):
            print('{}'.format(mask_name[mask_global_idx]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False,
                            save_masks=True,
                            save_dir=save_path,
                            index = split_tag)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            subset_records.append((mask_name[mask_global_idx], mask, torch.as_tensor(dice_score).float()))
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {} ({}), WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask_name[mask_global_idx], mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}\n'.format(test_score.avg))

        if subset_records:
            score_tensor = torch.stack([record[2] for record in subset_records], dim=0)
            primary_dice = score_tensor[:, :3].mean(dim=1)
            k = max(1, int(torch.ceil(torch.tensor(primary_dice.numel() * 0.3)).item()))
            cvar30 = torch.topk(primary_dice, k=k, largest=False).values.mean()
            weak_k = min(5, primary_dice.numel())
            weak_values, weak_indices = torch.topk(primary_dice, k=weak_k, largest=False)
            weak_names = ','.join(subset_records[int(idx.item())][0] for idx in weak_indices)
            mask_sizes = torch.tensor([sum(record[1]) for record in subset_records])
            single = primary_dice[mask_sizes == 1]
            full = primary_dice[mask_sizes == 4]
            worst_idx = int(torch.argmin(primary_dice).item())
            robust_lines = [
                'Robust metrics:',
                'Mean Dice = {:.4f}'.format(primary_dice.mean().item()),
                'Worst-subset Dice = {:.4f} ({})'.format(primary_dice[worst_idx].item(), subset_records[worst_idx][0]),
                'Top-5 weak subset Dice = {:.4f} ({})'.format(weak_values.mean().item(), weak_names),
                'CVaR-30% Dice = {:.4f}'.format(cvar30.item()),
                'Single-modality average Dice = {:.4f}'.format(single.mean().item() if single.numel() > 0 else float('nan')),
                'Full modality Dice = {:.4f}'.format(full.mean().item() if full.numel() > 0 else float('nan')),
            ]
            print('\n'.join(robust_lines))
            with open(output_path, 'a') as file:
                file.write('\n'.join(robust_lines) + '\n')
