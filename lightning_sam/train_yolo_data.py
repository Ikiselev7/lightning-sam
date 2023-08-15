import os
import time

import lightning as L
from lightning.fabric.strategies.fsdp import FSDPStrategy, fsdp_overlap_step_with_backward
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from tqdm import tqdm
from ultralytics.utils import IterableSimpleNamespace, yaml_load
import segment_anything

from config_yolo_set import cfg
from dataset import load_datasets, load_yolo_datasets
from lightning.fabric.fabric import _FabricOptimizer
# from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou

torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in tqdm(enumerate(val_dataloader)):
            images, bboxes, gt_masks = data['img'], data['bboxes'], data['masks']
            if not len(bboxes) or bboxes[0].nelement() == 0:
                continue
            if not len(gt_masks) or gt_masks[0].nelement() == 0:
                continue
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            # fabric.print(
            #     f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            # )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
        cfg: Box,
        fabric: L.Fabric,
        model: Model,
        optimizer: _FabricOptimizer,
        scheduler: _FabricOptimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        print(('\n' + '%15s' * 10) % ('Epoch', 'GPU_mem', "batch_time", "data_time", 'Focal', 'Dice', 'IoU', 'Total', 'Instances', 'Size'))
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), bar_format='{l_bar}{bar:7}{r_bar}' )

        for iter, data in pbar:
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data['img'], data['bboxes'], data['masks']
            if not len(bboxes) or bboxes[0].nelement() == 0:
                continue
            if not len(gt_masks) or gt_masks[0].nelement() == 0:
                continue
            boxes, masks = [], []
            for box, msk in zip(bboxes, gt_masks):
                random_indices = torch.randperm(box.size(0))[:cfg.model.chunk_size]
                # Use the random indices to slice both tensors
                boxes.append(box[random_indices])
                masks.append(msk[random_indices])
            bboxes = boxes
            gt_masks = masks
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, bboxes)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            # print(loss_total, loss_focal, loss_dice, loss_iou)
            with fsdp_overlap_step_with_backward(optimizer, model):
                loss_total.backward()

            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            # fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
            #              f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
            #              f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
            #              f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
            #              f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
            #              f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
            #              f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
            losses = [f'{focal_losses.val:.4f}/{focal_losses.avg:.4f}',
                      f'{dice_losses.val:.4f}/{dice_losses.avg:.4f}',
                      f'{iou_losses.val:.4f}/{iou_losses.avg:.4f}',
                      f'{total_losses.val:.4f}/{total_losses.avg:.4f}']

            pbar.set_description(
                ('%15s' * (4 + len(losses)) + '%15.4g' * 2) %
                (
                    f'{epoch}/{cfg.num_epochs}',
                    f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G',
                    f'{batch_time.val:.3f}/{batch_time.avg:.3f}',
                    f'{data_time.val:.3f}/{data_time.avg:.3f}',
                    *losses,
                    sum([c.shape[0] for c in data['cls']]),
                    data['img'].shape[-1]
                )
            )



def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizers = []
    for param in model.model.parameters():
        optimizer = torch.optim.Adam([param], lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay, foreach=False)
        optimizers.append(optimizer)

    # Assuming the step count will be consistent across all optimizers, use the first optimizer to create the scheduler.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers[0], lr_lambda)

    # Manually set other optimizer's learning rate using the first optimizer's learning rate.
    for optimizer in optimizers[1:]:
        for group, first_group in zip(optimizer.param_groups, optimizers[0].param_groups):
            group['lr'] = first_group['lr']

    return optimizers, scheduler


def main(cfg: Box) -> None:
    policy = {Model}
    strategy = FSDPStrategy(cpu_offload=True, auto_wrap_policy=policy)
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy=strategy,
                      # precision="16-mixed",
                      )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    yolo_cfg = IterableSimpleNamespace(**yaml_load(cfg["yolo_cfg"]))
    data_cfg = yaml_load(cfg["dataset"])
    yolo_cfg.imgsz = model.model.image_encoder.img_size

    train_data, val_data = load_yolo_datasets(yolo_cfg, data_cfg, cfg['batch_size'], cfg['num_workers'])
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, *optimizer = fabric.setup(model, *optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
