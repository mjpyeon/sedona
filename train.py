import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import threading
import time
from queue import Queue

import torch.cuda.amp as amp
from contextlib import ExitStack

from module import LocalLossBlock
from dataset import DATASET_CONFIGS, get_dataloader
from config import config
from loss import SmoothCrossEntropyLoss
from utils import move_model_state_dict, AverageMeter, get_error

QUEUE_SIZE = config.queue_size

def train_step(x, y, i, model, optimizer, scheduler=None, device='cuda:0',
               mixed_precision=False, scaler=None):
    x, y = x.to(device), y.to(device)

    with ExitStack() as stack:
        if mixed_precision:
            stack.enter_context(amp.autocast())

        out = model(x, isolate_grad=True)
        losses = model.get_loss(y)
        loss = sum(losses)

    for j in range(len(optimizer)):
        optimizer[j].zero_grad()

    if mixed_precision:
        scaler.scale(loss).backward()
        for j in range(len(optimizer)):
            scaler.step(optimizer[j])
            if scheduler is not None:
                scheduler[j].step(i+1)
        scaler.update()
    else:
        loss.backward()
        for j in range(len(optimizer)):
            optimizer[j].step()
            if scheduler is not None:
                scheduler[j].step(i+1)

    return [loss.item() for loss in losses]

def block_train_step(model, optimizer, data=None, target=None,
                     queue_in=None, queue_out=None, scheduler=None, device=None,
                     mixed_precision=False, scaler=None):

    model.train()
    # get item from queue
    if data is None:
        data, target = queue_in.get()

    data, target = data.to(device), target.to(device)

    with ExitStack() as stack:
        if mixed_precision:
            stack.enter_context(amp.autocast())
        out = model(data)
        if queue_out is not None:
            queue_out.put((out.detach().clone(), target.detach().clone()))

        loss = model[-1].get_loss(target)

    del out  # necessary?
    del target  # necessary?

    optimizer.zero_grad()
    if mixed_precision:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()

def continuous_train_step(x, y, time_step, model, optimizer, alpha,
                          beta=None, categorical=False, scheduler=None, device='cuda:0'):
    model.train()
    x, y = x.to(device), y.to(device)
    out = model(x)
    alpha_softmax = F.softmax(alpha, dim=-1)
    beta_softmax = F.softmax(beta, dim=-1) if beta is not None else None
    dec_out_scale = 1/(alpha_softmax[:, 0].detach() + 1e-5)
    dec_out_scale = torch.cat((dec_out_scale, torch.ones((1,), device=device)))
    losses = model.get_loss(y, weights=beta_softmax, categorical=categorical, scale=dec_out_scale)
    loss = losses[-1]
    for i in range(len(losses)-2, -1, -1):
        loss = alpha_softmax[i, 1] * loss + alpha_softmax[i, 0] * losses[i]
    model.zero_grad()
    loss.backward()
    blocks = model.get_blocks()
    for i in range(model.num_blocks-1, -1, -1):
        block = blocks[i]
        for parameter in block.parameters():
            if parameter.grad is not None:
                parameter.grad /= alpha_softmax.data[:i, 1].prod()

    if isinstance(optimizer, list):
        for opt in optimizer:
            opt.step()
    else:
        optimizer.step()

    if scheduler is not None:
        if isinstance(scheduler, list):
            for sc in scheduler:
                sc.step(time_step+1)
        else:
            scheduler.step(time_step+1)

    return [loss.item() for loss in losses]

class BlockThread(threading.Thread):
    def __init__(self, idx, module, optimizer, scheduler, queue_in, queue_out,
                 device, train_iters=1e3, valid_freq=1000, do_validation=False,
                 mixed_precision=False, scaler=None):
        threading.Thread.__init__(self)
        self.idx = idx
        self.module = module.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.queue_in = queue_in
        self.queue_out = queue_out
        self.device = device
        self.train_iters = train_iters
        self.valid_freq = valid_freq
        self.do_validation = do_validation
        self.mixed_precision = mixed_precision
        self.scaler = scaler

        self.step = 0

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())

    def run(self):
        while self.step < self.train_iters:
            block_train_step(self.module, self.optimizer,
                             queue_in=self.queue_in,
                             queue_out=self.queue_out,
                             scheduler=self.scheduler,
                             device=self.device,
                             mixed_precision=self.mixed_precision,
                             scaler=self.scaler)

            if self.do_validation and ((self.step + 1) % self.valid_freq == 0 or self.step == 0):
                self.pause()

            self.step += 1

        while self.queue_out is not None and self.queue_out.qsize() > 0:
            time.sleep(1)

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    #should just resume the thread
    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()

    def is_paused(self):
        return self.paused



def train(model, optimizer, train_loader, train_iters,
          scheduler=None, device='cuda:0', valid_loader=None, valid_freq=100,
          alpha=None, beta=None,
          mixed_precision=False):
    model.train()

    pbar = tqdm(total=train_iters)
    step = 0
    time_per_iter = AverageMeter()

    scaler = amp.GradScaler() if mixed_precision else None
    if valid_loader:
        best_valid_err1 = 1.
        best_valid_err5 = 1.
        best_valid_loss = math.inf
        sd = None

    start_time = time.time()


    while True:
        if config.classes_per_batch > 0 and step > config.classes_per_batch_until_iter:
            train_loader = get_dataloader(train_loader.dataset, config.batch_size, shuffle=True)

        for x, y in train_loader:
            if config.eval_continuous:
                train_losses = continuous_train_step(x, y, step, model, optimizer, alpha, beta,
                                               False, scheduler, device)
            else:
                train_losses = train_step(x, y, step, model, optimizer, scheduler, device,
                                          mixed_precision=mixed_precision,
                                          scaler=scaler)
            if valid_loader and (step + 1) % valid_freq == 0:
                end_time = time.time()
                time_per_iter.update((end_time - start_time) / valid_freq, valid_freq)

                loss_avg, top1_err, top5_err = test(model, valid_loader, device, mixed_precision=mixed_precision)
                model.train()
                if top1_err < best_valid_err1:
                    sd = move_model_state_dict(model.state_dict(), 'cpu')
                    best_valid_err1 = top1_err
                    best_valid_err5 = top5_err
                    best_valid_loss = loss_avg

                pbar.set_description('[Step {}] tloss: {:.2f}, vloss: {:.2f}, err@1: {:.2f}%, err@5: {:.2f}%, latency: {:.3f}s'.\
                                     format(step, train_losses[-1], loss_avg, top1_err*100, top5_err*100, time_per_iter.avg),
                                     True
                )

                start_time = time.time()

            step += 1
            pbar.update(1)
            if step == train_iters:
                if valid_loader:
                    pbar.set_description('[Final] loss: {:.3f}, err@1: {:.2f}%, err@5: {:.2f}%, latency: {:.3f}s'.\
                                         format(best_valid_loss, best_valid_err1*100, best_valid_err5*100, time_per_iter.avg),
                                         True)
                    model.load_state_dict(move_model_state_dict(sd, device))

                pbar.close()
                return move_model_state_dict(model.state_dict(), 'cpu')


def train_parallel(model, optimizers, train_loader, train_iters,
                   schedulers, devices, valid_loader=None, valid_freq=100,
                   mixed_precision=False):
    '''Parallel training of local loss blocks'''
    model.train()
    modules = model.moduleList
    block_mapping = model.mapping_from_header_index
    scaler = [amp.GradScaler() for _ in range(model.num_blocks)] if mixed_precision else None
    time_per_iter = AverageMeter()

    if valid_loader:
        best_valid_err1 = 1.
        best_valid_err5 = 1.
        best_valid_loss = math.inf
        sd = None

    threads = []
    #queues = [Queue(1)] + [Queue(QUEUE_SIZE) for i in range(1, model.num_blocks)] + [None]  # last queue_out is None
    queues = [Queue(QUEUE_SIZE) for i in range(1, model.num_blocks)] + [None]  # last queue_out is None
    #for ix, block_module_indices in enumerate(block_mapping):
    initial_block = nn.Sequential(*[modules[module_index] for module_index in block_mapping[0]]).to(devices[0])
    for ix, block_module_indices in enumerate(block_mapping[1:], start=1):
        block = nn.Sequential(*[modules[module_index] for module_index in block_module_indices])
        #threads.append(BlockThread(ix, block, optimizers[ix], schedulers[ix], queues[ix], queues[ix+1], devices[ix],
        #                           train_iters, valid_freq, valid_loader is not None))
        threads.append(BlockThread(ix, block, optimizers[ix], schedulers[ix], queues[ix-1], queues[ix], devices[ix],
                                   train_iters, valid_freq, valid_loader is not None,
                                   mixed_precision, scaler[ix] if scaler is not None else None))

    start_time = time.time()
    for thread in threads:
        thread.start()

    step = 0
    pbar = tqdm(total=train_iters)
    #feed_queue = queues[0]
    while True:
        if config.classes_per_batch > 0 and step > config.classes_per_batch_until_iter:
            train_loader = get_dataloader(train_loader.dataset, config.batch_size, shuffle=True)

        for data, target in train_loader:

            #feed_queue.put((data, target))
            block_train_step(initial_block, optimizers[0], data, target,
                             queue_out=queues[0], device=devices[0],
                             scheduler=schedulers[0],
                             mixed_precision=mixed_precision, scaler=scaler[0] if scaler is not None else None)

            if valid_loader and (step + 1) % valid_freq == 0:
                # wait for all threads to pause
                while not all(thread.is_paused() for thread in threads):
                    time.sleep(0.01)

                end_time = time.time()
                time_per_iter.update((end_time - start_time) / valid_freq, valid_freq)

                # all threads paused, run validation
                loss_avg, top1_err, top5_err = test_multigpu(model, valid_loader, devices, mixed_precision=mixed_precision)
                model.train()

                if top1_err < best_valid_err1:
                    sd = move_model_state_dict(model.state_dict(), 'cpu')
                    best_valid_err1 = top1_err
                    best_valid_err5 = top5_err
                    best_valid_loss = loss_avg

                pbar.set_description('[Step {}] vloss: {:.2f}, err@1: {:.2f}%, err@5: {:.2f}%, latency: {:.3f}s'.\
                                     format(step, loss_avg, top1_err*100, top5_err*100, time_per_iter.avg),
                                     True)

                for thread in threads:
                    thread.resume()

                start_time = time.time()

            step += 1
            pbar.update(1)

            if step == train_iters:
                for thread in threads:
                    thread.join()

                model.to('cpu')
                if valid_loader:
                    pbar.set_description('[Final] loss: {:.3f}, err@1: {:.2f}%, err@5: {:.2f}%, latency: {:.3f}s'.\
                                         format(best_valid_loss, best_valid_err1*100, best_valid_err5*100, time_per_iter.avg),
                                         True)
                    model.load_state_dict(sd)

                pbar.close()

                return model.state_dict()


def test(model, test_loader, device='cuda:0', n_batches=0, mixed_precision=False):
    ''' Evaluate model on test set '''
    num_classes = DATASET_CONFIGS[config.dataset]['classes']
    model.eval()
    loss_fn = SmoothCrossEntropyLoss(config.smoothing)
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()

    # Loop test set
    for i, (data, target) in enumerate(test_loader):
        if n_batches > 0 and i == n_batches:
            break

        if config.cuda:
            data, target = data.to(device), target.to(device)

        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if mixed_precision:
                stack.enter_context(amp.autocast())
            output = model(data)
            test_loss = loss_fn(output, target)

        loss.update(test_loss.float().item(), target.size(0))
        top1_err, top5_err = get_error(output.float(), target, (1, 5))
        top1.update(top1_err.item(), target.size(0))
        top5.update(top5_err.item(), target.size(0))

    return loss.avg, top1.avg, top5.avg


def test_ensemble(model, test_loader, device='cuda:0', n_batches=0, num_ensemble=1, mixed_precision=False):
    num_classes = DATASET_CONFIGS[config.dataset]['classes']
    model.eval()
    loss_fn = SmoothCrossEntropyLoss(config.smoothing)
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()

    # Loop test set
    for i, (data, target) in enumerate(test_loader):
        if n_batches > 0 and i == n_batches:
            break

        if config.cuda:
            data, target = data.to(device), target.to(device)

        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if mixed_precision:
                stack.enter_context(amp.autocast())
            output = model(data)
            test_loss = loss_fn(output, target)
            outputs = []
            for module in reversed(model.moduleList):
                if isinstance(module, LocalLossBlock):
                    out = module.decoder(module.act)
                    outputs.append(F.log_softmax(out, dim=-1))

                if len(outputs) == num_ensemble:
                    break
            output = torch.stack(outputs, dim=0).sum(dim=0)

        top1_err, top5_err = get_error(output.float(), target, (1, 5))
        top1.update(top1_err.item(), target.size(0))
        top5.update(top5_err.item(), target.size(0))

    return loss.avg, top1.avg, top5.avg


def test_multigpu(model, test_loader, devices, n_batches=0, mixed_precision=False):
    ''' Evaluate model on test set on multiple GPUs'''
    num_classes = DATASET_CONFIGS[config.dataset]['classes']
    model.eval()
    loss_fn = SmoothCrossEntropyLoss(config.smoothing)
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss = AverageMeter()

    device = devices[0]

    # Loop test set
    for i, (data, target) in enumerate(test_loader):
        if n_batches > 0 and i == n_batches:
            break

        if config.cuda:
            data, target = data.to(devices[0]), target.to(devices[-1])

        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if mixed_precision:
                stack.enter_context(amp.autocast())
            # multi-gpu version
            x = data
            for ix, block_module_indices in enumerate(model.mapping_from_header_index):
                x = x.to(devices[ix])
                for module_index in block_module_indices:
                    if isinstance(model.moduleList[module_index], LocalLossBlock):
                        x = model.moduleList[module_index](x, save_act=False)
                    else:
                        x = model.moduleList[module_index](x)

            output = model.moduleList[-1].decoder(x)
            test_loss = loss_fn(output, target)

        loss.update(test_loss.float().item(), target.size(0))
        top1_err, top5_err = get_error(output.float(), target, (1, 5))
        top1.update(top1_err.item(), target.size(0))
        top5.update(top5_err.item(), target.size(0))

    return loss.avg, top1.avg, top5.avg


