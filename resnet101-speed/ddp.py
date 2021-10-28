"""ResNet-101 Speed Benchmark"""
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import os
import click
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.nn.parallel import DistributedDataParallel as DDP

from resnet import resnet101

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
    """ResNet-101 Speed Benchmark"""
    if skip_epochs >= epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    mp.spawn(main, args=(world_size, batch_size, skip_epochs, epochs, ), nprocs=world_size, join=True)

def main(rank, world_size, batch_size, skip_epochs, epochs):
    # Initialize Process Group for Communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', world_size=world_size, rank=rank)

    model: nn.Module = resnet101(num_classes=1000).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = SGD(model.parameters(), lr=0.1)

    # This experiment cares about only training speed, rather than accuracy.
    # To eliminate any overhead due to data loading, we use fake random 224x224
    # images over 1000 labels.
    dataset_size = 50000

    input = torch.rand(batch_size, 3, 224, 224)
    target = torch.randint(1000, (batch_size,))
    data = [(input, target)] * ((dataset_size//world_size)//batch_size)

    if dataset_size % (batch_size*world_size) != 0:
        last_input = input[:dataset_size % (batch_size*world_size)]
        last_target = target[:dataset_size % (batch_size*world_size)]
        input.append(last_input)
        target.append(last_target)
        data.append((last_input, last_target))

    # ds = TensorDataset(input, target)

    # train_sampler = DistributedSampler(ds)
    
    # train_loader = torch.utils.data.DataLoader(
    #     ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, sampler=train_sampler
    # )

    # HEADER ======================================================================================

    if rank == 0:
        title = f'np: {world_size}, {skip_epochs+1}-{epochs} epochs'
        click.echo(title)

        click.echo(f'batch size: {batch_size}, np: {world_size}, data_size: {len(data)}')

        click.echo('python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
            platform.python_version(),
            torch.__version__,
            torch.backends.cudnn.version(),
            torch.version.cuda,
            torch.cuda.get_device_name(0)))

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        torch.cuda.synchronize()
        tick = time.time()

        data_trained = 0
        for i, (input, target) in enumerate(data):
            input, target = input.to(rank), target.to(rank)

            data_trained += input.size(0)

            output = model(input)
            loss = F.cross_entropy(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
            percent = (i+1) / len(data) * 100
            throughput = data_trained / (time.time()-tick)
            if rank == 0:
                log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                    '' % (epoch+1, epochs, percent, throughput), clear=True, nl=False)

        torch.cuda.synchronize()
        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = dataset_size / elapsed_time
        if rank == 0:
            log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
                '' % (epoch+1, epochs, throughput, elapsed_time), clear=True)

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
    if rank == 0:    
        hr()

    # RESULT ======================================================================================

    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n
    elapsed_time = sum(elapsed_times) / n
    if rank == 0:
        click.echo('%s | %.3f samples/sec, %.3f sec/epoch (average)'
                '' % (title, throughput, elapsed_time))

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    cli()
    cleanup()
