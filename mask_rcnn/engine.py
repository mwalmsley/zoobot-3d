import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, tb_writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    iterations = 0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iterations += 1
        if tb_writer is not None:
            step = (epoch * len(data_loader)) + iterations
            tb_writer.add_scalar('Training/Loss', losses_reduced, step)
            tb_writer.add_scalar('Training/Loss_box_reg', loss_dict["loss_box_reg"], step)
            tb_writer.add_scalar('Training/Loss_classifier', loss_dict["loss_classifier"], step)
            tb_writer.add_scalar('Training/Loss_objectness', loss_dict["loss_objectness"], step)
            tb_writer.add_scalar('Training/Loss_rpn_box_reg', loss_dict["loss_rpn_box_reg"], step)
            tb_writer.add_scalar('LR/Learning_rate', optimizer.param_groups[0]["lr"], step)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, epoch, tb_writer=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    if tb_writer is not None:
        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            tb_writer.add_scalar("AP/IoU/0.50-0.95/all/100", coco_eval.stats[0], epoch)
            tb_writer.add_scalar("AP/IoU/0.50/all/100", coco_eval.stats[1], epoch)
            tb_writer.add_scalar("AP/IoU/0.75/all/100", coco_eval.stats[2], epoch)
            tb_writer.add_scalar("AP/IoU/0.50-0.95/small/100", coco_eval.stats[3], epoch)
            tb_writer.add_scalar("AP/IoU/0.50-0.95/medium/100", coco_eval.stats[4], epoch)
            tb_writer.add_scalar("AP/IoU/0.50-0.95/large/100", coco_eval.stats[5], epoch)
            tb_writer.add_scalar("AR/IoU/0.50-0.95/all/1", coco_eval.stats[6], epoch)
            tb_writer.add_scalar("AR/IoU/0.50-0.95/all/10", coco_eval.stats[7], epoch)
            tb_writer.add_scalar("AR/IoU/0.50-0.95/all/100", coco_eval.stats[8], epoch)
            tb_writer.add_scalar("AR/IoU/0.50-0.95/small/100", coco_eval.stats[9], epoch)
            tb_writer.add_scalar("AR/IoU/0.50-0.95/medium/100", coco_eval.stats[10], epoch)
            tb_writer.add_scalar("AR/IoU/0.50-0.95/large/100", coco_eval.stats[11], epoch)

    return coco_evaluator


@torch.no_grad()
def evaluate_loss(model, data_loader, device, epoch, tb_writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation Loss:'
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        if tb_writer is not None:
            tb_writer.add_scalar('Validation/Loss', losses_reduced, epoch)
            tb_writer.add_scalar('Validation/Loss_box_reg', loss_dict["loss_box_reg"], epoch)
            tb_writer.add_scalar('Validation/Loss_classifier', loss_dict["loss_classifier"], epoch)
            tb_writer.add_scalar('Validation/Loss_objectness', loss_dict["loss_objectness"], epoch)
            tb_writer.add_scalar('Validation/Loss_rpn_box_reg', loss_dict["loss_rpn_box_reg"], epoch)

    return metric_logger