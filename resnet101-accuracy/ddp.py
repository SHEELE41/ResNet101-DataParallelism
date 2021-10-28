"""ResNet-101 Accuracy Benchmark"""
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from resnet import resnet101

def dataloaders(batch_size: int, num_workers: int = 32) -> Tuple[DataLoader, DataLoader]:
    num_workers = num_workers if batch_size <= 4096 else num_workers // 2

    post_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = torchvision.datasets.ImageNet(
        root='imagenet',
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            post_transforms,
        ])
    )
    test_dataset = torchvision.datasets.ImageNet(
        root='imagenet',
        split='val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            post_transforms,
        ])
    )

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        sampler=train_sampler,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        sampler=test_sampler,
    )

    return train_dataloader, test_dataloader


BASE_TIME: float = 0


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = click.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = click.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)


def parse_devices(ctx: Any, param: Any, value: Optional[str]) -> List[int]:
    if value is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(',')]


@click.command()
@click.pass_context
@click.option(
    '--epochs', '-e',
    type=int,
    default=10,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--skip-epochs', '-k',
    type=int,
    default=1,
    help='Number of epochs to skip in result (default: 1)',
)
@click.option(
    '--batch-size', '-b',
    type=int,
    default=128,
    help='Batch size to use (default: 128)',
)
@click.option(
    '--devices', '-d',
    metavar='0,1,2,3',
    callback=parse_devices,
    help='Device IDs to use (default: all CUDA devices)',
)
@click.option(
    '--world-size', '-np',
    type=int,
    default=1,
    help='Number of processes (default: 1)',
)
def cli(ctx: click.Context,
        epochs: int,
        skip_epochs: int,
        batch_size: int,
        devices: List[int],
        world_size: int,
        ) -> None:
    """ResNet-101 Accuracy Benchmark"""
    if skip_epochs > epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    mp.spawn(main, args=(world_size, batch_size, skip_epochs, epochs, ), nprocs=world_size, join=True)
    
def main(rank, world_size, batch_size, skip_epochs, epochs):
    # Initialize Process Group for Communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', world_size=world_size, rank=rank)

    relu_inplace = True
    model: nn.Module = resnet101(num_classes=1000, inplace=relu_inplace).to(rank)
    model = DDP(model, device_ids=[rank])

    # Prepare dataloaders.
    train_dataloader, valid_dataloader = dataloaders(batch_size)

    # Optimizer with LR scheduler
    steps = len(train_dataloader)
    lr_multiplier = max(1.0, batch_size / 256)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

    def gradual_warmup_linear_scaling(step: int) -> float:
        epoch = step / steps

        # Gradual warmup
        warmup_ratio = min(4.0, epoch) / 4.0
        multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

        if epoch < 30:
            return 1.0 * multiplier
        elif epoch < 60:
            return 0.1 * multiplier
        elif epoch < 80:
            return 0.01 * multiplier
        return 0.001 * multiplier

    scheduler = LambdaLR(optimizer, lr_lambda=gradual_warmup_linear_scaling)

    # HEADER ======================================================================================

    if rank == 0:
        title = '%d batch, %d-%d epochs'\
                '' % (batch_size, skip_epochs+1, epochs)
        click.echo(title)
        click.echo('python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
            platform.python_version(),
            torch.__version__,
            torch.backends.cudnn.version(),
            torch.version.cuda,
            torch.cuda.get_device_name(0)))

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def evaluate(dataloader: DataLoader) -> Tuple[float, float]:
        tick = time.time()
        steps = len(dataloader)
        data_tested = 0
        loss_sum = torch.zeros(1, device=rank)
        accuracy_sum = torch.zeros(1, device=rank)
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataloader):
                input, target = input.to(rank), target.to(rank)

                current_batch = input.size(0)
                data_tested += current_batch

                output = model(input)

                loss = F.cross_entropy(output, target)
                loss_sum += loss * current_batch

                _, predicted = torch.max(output, 1)
                correct = (predicted == target).sum()
                accuracy_sum += correct

                percent = i / steps * 100
                throughput = data_tested / (time.time() - tick)
                if rank == 0:
                    log('valid | %d%% | %.3f samples/sec (estimated)'
                        '' % (percent, throughput), clear=True, nl=False)

        loss = loss_sum / data_tested
        accuracy = accuracy_sum / data_tested

        return loss.item(), accuracy.item()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        torch.cuda.synchronize()
        tick = time.time()

        steps = len(train_dataloader)
        data_trained = 0
        loss_sum = torch.zeros(1, device=rank)
        model.train()
        for i, (input, target) in enumerate(train_dataloader):
            input, target = input.to(rank), target.to(rank)
            data_trained += batch_size
            input = input.to(rank, non_blocking=True)
            target = target.to(rank, non_blocking=True)

            output = model(input)
            loss = F.cross_entropy(output, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            loss_sum += loss.detach() * batch_size

            percent = i / steps * 100
            throughput = data_trained / (time.time()-tick)
            if rank == 0:
                log('train | %d/%d epoch (%d%%) | lr:%.5f | %.3f samples/sec (estimated)'
                    '' % (epoch+1, epochs, percent, scheduler.get_lr()[0], throughput),
                    clear=True, nl=False)

        torch.cuda.synchronize()
        tock = time.time()

        train_loss = loss_sum.item() / data_trained
        valid_loss, valid_accuracy = evaluate(valid_dataloader)
        torch.cuda.synchronize()

        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        if rank == 0:
            log('%d/%d epoch | lr:%.5f | train loss:%.3f %.3f samples/sec | '
                'valid loss:%.3f accuracy:%.3f'
                '' % (epoch+1, epochs, scheduler.get_lr()[0], train_loss, throughput,
                    valid_loss, valid_accuracy),
                clear=True)

        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    if rank == 0:
        hr()
    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(epoch)

        if epoch < skip_epochs:
            continue

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    _, valid_accuracy = evaluate(valid_dataloader)
    if rank == 0:
        hr()

    # RESULT ======================================================================================

    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    if rank == 0:
        click.echo('%s | valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
                '' % (title, valid_accuracy, throughput, elapsed_time))

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    cli()
    cleanup()
