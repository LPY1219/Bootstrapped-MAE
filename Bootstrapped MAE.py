# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import util.misc_bootstrapMAE as misc
from util.misc_bootstrapMAE import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain_bootstrapMAE import train_one_epoch
import torchvision
torchvision.datasets.CIFAR10("./dataset", train=True, download=True)
torchvision.datasets.CIFAR10("./dataset", train=False, download=True)
import models_reconstruct
import torch.nn as nn

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--nb_bootstrap',default=[2,3,4,5,6,7,8,10,20,40],type=list,help='number of bootstrap times')
    #为了探究bootstrapped的最佳次数，需要采用for循环对可能的情况进行遍历，故在此处添加一个参数指示所有需要测试的bootstrpped的次数
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--model', default='mae_deit_tiny_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--reconstruct_model', default='deit_tiny_reconstruct', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')
    # 由于需要采用CIFAR10数据集，此处需要修改输入大小
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')
    # Dataset parameters
    parser.add_argument('--data_path', default='./dataset', type=str,
                        help='dataset path')
    #改成相应的数据路径
    parser.add_argument('--output_dir', default='/data/lpy_data/MAE/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/data/lpy_data/MAE/output_dir',
                        help='path where to tensorboard log')
    #修改存储输出结果的路径
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #此处需要对最后的标准化操作进行修改，采用CIFAR10数据集的均值和方差
    dataset_train = torchvision.datasets.CIFAR10(
        root=r'./dataset',
        train=True,
        download=True,
        transform=transform_train
    )
    #下载CIFAR10
    #print(dataset_train)
    if False:  # args.distributed 由于采用单卡训练，所以可以直接跳过
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    for k in args.nb_bootstrap:#采用for循环遍历所有bootstrapped times
        print(f"Begin to train Bootstrap-{k} algorithm!")#打印提示信息
        origin_dir=args.output_dir#把初始的output_dir保存一下
        args.output_dir=args.output_dir+str(k)#pretrain的所有模型都放在了初始的output_dir路径下，因此需要遍历该文件夹，找出相应的模型。
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        bootstrap_epoch=int(args.epochs/k)#在bootstrapped-k 算法中每一次迭代需要消耗多少轮
        count=1#用来统计在一次bootstrap-k算法中进行到了第几轮bootstrap了
        if args.log_dir is not None:
            os.makedirs(args.log_dir+str(k), exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir+str(k))
        else:
            log_writer = None
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        # define the model
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        model.to(device)
        model_without_ddp = model
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        loss_scaler = NativeScaler()
        #misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        flag=True
        while count<=k:#迭代到第k次则bootstarpped-k算法停止，测试下一个k值
            '''
            下面进入核心阶段：
            MAE-1的重建目标仍然是像素，从MAE-2开始重建目标才是encoder的输出。所以首先采用条件判断语句进行区分。
            其次，需要对MAE-3进行特判，原因在于训练MAE-2时,所采用的重建目标模型target_model需要加载MAE-1的模型权重，而MAE-1的decoder的最后一层predict与其余的MAE-K不同，
            无法直接加载，因此需要先把target_model的网络结构调整到与MAE-1一致，等到MAE-2训练好以后再调整回来，之后就可以不用再改变了。
            '''
            #其实我这里的代码写得不是很好，如果你把target部分的decoder去掉就不用特判了。
            if count>1:
                args.resume=args.output_dir+'/'+ ('checkpoint-%s.pth' % (count-1))#调用第上一次迭代的训练模型
                misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler)
                checkpoint=torch.load(args.resume, map_location='cpu')
                print("Load pre-trained checkpoint from: %s" % args.resume)
                checkpoint_model = checkpoint['model']
                if flag==True:
                    target_model= models_reconstruct.__dict__[args.reconstruct_model](norm_pix_loss=args.norm_pix_loss)
                    target_model.to(device)#taget_model的作用是用来给出重建目标，也就是上一次迭代模型的encoder输出
                    model_without_ddp.decoder_pred = nn.Linear(model_without_ddp.decoder_embed_dim,
                                                               model_without_ddp.embed_dim, bias=True)
                    model_without_ddp.to(device)#model是本轮迭代需要训练的模型，由于修改了网络结构，需要重新送入CUDA，否则会报错。
                    flag=False
                    #只在count=2的时候才会进入
                if count==3:
                    target_model.decoder_pred=nn.Linear(target_model.decoder_embed_dim,
                                                               target_model.embed_dim, bias=True)
                    target_model.to(device)
                    #只在count=3的时候才会进入
                msg = target_model.load_state_dict(checkpoint_model, strict=False)
                print(msg)

            else:
                misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler)
                target_model = None
                print("Train MAE-1,no need to resume")

            print(f"Start training for MAE-{count} for the bootstrap-{k}!")
            start_time = time.time()
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model_without_ddp, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    log_writer=log_writer,
                    args=args,target_model=target_model
                )#相比于MAE的官方代码，此处多传入了一个参数 target_model
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'count': count, }

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
                if args.output_dir and ( (epoch + 1) % bootstrap_epoch ==0 or (epoch+1)%args.epochs==0):
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch,count=count)
                    count+=1
                    break

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {} for the {} bootstrap'.format(total_time_str,count))
        args.output_dir=origin_dir
        args.start_epoch=0
        args.resume=''
        #在进行下个k值测试时，需要将修改过的args参数还原。


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
