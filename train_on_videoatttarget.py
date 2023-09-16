
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from data_opt.load_data import VideoAttTarget_image
from config import *
from transforms import _get_transform
from eval import evaluate

from models.detr import build
import util.misc as utils
from pathlib import Path
import math
import argparse
import sys
import numpy as np
import  random
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default=None, help="initial weights")
parser.add_argument("--resume", action='store_true', help="resume training; requires init_weights")
parser.add_argument("--lr", type=float, default=2.5e-5, help="learning rate")
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument("--batch_size", type=int, default=3, help="batch size")
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=1, help="eval every ___ epochs")
parser.add_argument("--save_every", type=int, default=5, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="logs", help="directory to save log files")
parser.add_argument("--maxlabelbbox", type=int, default=maxlabeledbboxs, help="the max predict head box .this num is fixed.")
parser.add_argument("--heat_dim",type=list,default=[6,6,40],help="[w0,h0,gaussion]  heat dim  gaussion range.")
# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=maxlabeledbboxs, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=2, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=[1.0,2.5], type=list,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)

parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--inout_loss_coef', default=1, type=float)
parser.add_argument('--heatmap_loss_coef', default=2, type=float)

parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# class num  head_class inout_class
parser.add_argument('--head_class', default=2, type=float)
parser.add_argument('--inout_class', default=2, type=float)

parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--output_dir', default='checkpoint',
                        help='path where to save, empty for no saving')
args = parser.parse_args()


def train():
    transform = _get_transform()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prepare data
    print("Loading Data")

    train_dataset = VideoAttTarget_image(videoattentiontarget_train_label, transform, args,
                                         max_label_size=args.maxlabelbbox,)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collection_function)

    val_dataset = VideoAttTarget_image(videoattentiontarget_val_label, transform, args,
                                       max_label_size=args.maxlabelbbox)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                                collate_fn=val_dataset.collection_function)

    model, criterion=build(args)
    # print(model)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu", args.device)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    output_dir = Path(args.output_dir)

    max_norm=args.clip_max_norm
    print("Training in progress ...")
    with torch.cuda.amp.autocast(enabled=use_amp):
        for ep in range(args.start_epoch, args.epochs):
            model.train()
            criterion.train()
            print_freq = 10
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
            header = 'Epoch: [{}]'.format(ep)

            for img,targets,mask in metric_logger.log_every(train_loader,print_freq,header):


                images = img.to(device)
                mask=mask.to(device)
                outputs = model(utils.NestedTensor(images,mask))

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                              for k, v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

                metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)

            lr_scheduler.step()
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (ep + 1) % args.lr_drop == 0 or (ep + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{ep:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': ep,
                        'args': args,
                    }, checkpoint_path)

            test_stats= evaluate(
                model, criterion, val_loader, device, args.output_dir
            )

            log_stats = {**{k: meter.global_avg for k, meter in metric_logger.meters.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': ep}

            print(log_stats)
            

if __name__ == "__main__":
    train()
