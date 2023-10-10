import argparse
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from utils.loss import get_loss_func

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Tuples Transformer')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--work_dir', default='./work_dir/ntu/temp', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/ntu/ntu26_xsub_joint.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--run_mode', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--save_epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=2, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_ntu.Feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=0, help='the number of worker for data loader')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for model testing')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--cuda_visible_device', default='0,1', help='')
    parser.add_argument('--device', type=int, default=[0,1], nargs='+', help='the indexes of GPUs for training or testing')

    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5)
    parser.add_argument('--optimizer_betas', type=float, default=[0.9, 0.999])

    parser.add_argument('--loss', default='CrossEntropy', help='the loss will be used')
    parser.add_argument('--loss_args', default=dict(), help='the arguments of loss')

    parser.add_argument('--n_forward', type=int, default=5, help='minibatch forward times')

    return parser

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        value = value*1000
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Processor():

    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        self.load_model()
        self.load_data() 

        if arg.run_mode == 'train':
            self.load_optimizer()

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.run_mode == 'train':
            self.data_loader['train'] = DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.print_log('Data load finished')

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args)
        self.loss = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)['model']

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        self.print_log('Model load finished: ' + self.arg.model)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                betas=(self.arg.optimizer_betas[0], self.arg.optimizer_betas[1]))
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                betas=(self.arg.optimizer_betas[0], self.arg.optimizer_betas[1]))
        else:
            raise ValueError()
        self.print_log('Optimizer load finished: ' + self.arg.optimizer)

    def adjust_learning_rate(self, epoch):
        self.print_log('adjust learning rate, using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam'  or self.arg.optimizer == 'AdamW':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * ( self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self, epoch, save_model=False):
        losses = AverageMeter()

        self.model.train()
        self.adjust_learning_rate(epoch)
        
        for batch, (imgs, instr, joints, states_tensor, index) in enumerate(tqdm(self.data_loader['train'], desc="Training", ncols=100)):
            self.global_step += 1

            B, F, H, W, C = imgs.shape
            imgs = imgs.contiguous().view(B*F, H, W, C)
            _, _, V = joints.shape
            _, _, V2 = states_tensor.shape

            joints = joints.contiguous().view(-1, V)
            states_tensor = states_tensor.contiguous().view(-1, V2)

            imgs = imgs.permute(0, 3, 1, 2).unsqueeze(dim=2)
            instructions = [] 
            for i in instr:
                instructions += [i] * F
            
            # with torch.no_grad():
            imgs = imgs.float().cuda()
            joints = joints.float().cuda()
            states_tensor = states_tensor.float().cuda()

            # minibatch forward
            mini_batch = B*F//self.arg.n_forward
            for i in range(self.arg.n_forward):
                # forward
                # imgs: Torch tensor [mini_batch, C, T, H, W]
                # instruction: list of str len=mini_batch
                

                logits = self.model(imgs[mini_batch*i:mini_batch*(i+1)], instructions[mini_batch*i:mini_batch*(i+1)],states_tensor[mini_batch*i:mini_batch*(i+1)])
                # logits: Torch tensor [mini_batch, 1, ActionNum(22), ActionBin(256)]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                loss = self.loss(logits, joints[mini_batch*i:mini_batch*(i+1)])

                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                
                
                self.optimizer.step()

                losses.update(loss.item())


            self.lr = self.optimizer.param_groups[0]['lr']
        
        # print('ss')
        # print(value)
        
        self.print_log('training: epoch: {}, loss: {:.4f}, lr: {:.6f}'.format(
            epoch + 1, losses.avg, self.lr))

    def eval(self, epoch, loader_name=['test']):
        losses = AverageMeter()

        self.model.eval()
        for ln in loader_name:
            cnt=0
            for batch, (imgs, instr, joints, states_tensor, index) in enumerate(tqdm(self.data_loader[ln], desc="Evaluating", ncols=100)):
                cnt+=1
                if cnt>10:
                    break
                    
                B, F, H, W, C = imgs.shape
                imgs = imgs.contiguous().view(B*F, H, W, C)
                _, _, V = joints.shape
                _, _, V2 = states_tensor.shape
                joints = joints.contiguous().view(-1, V)
                states_tensor = states_tensor.contiguous().view(-1, V2)

                imgs = imgs.permute(0, 3, 1, 2).unsqueeze(dim=2)
                instructions = [] 
                for i in instr:
                    instructions += [i] * F
                
                with torch.no_grad():
                    imgs = imgs.float().cuda()
                    joints = joints.float().cuda()
                    states_tensor = states_tensor.float().cuda()

                # minibatch forward
                mini_batch = B*F//self.arg.n_forward
                with torch.no_grad():
                    for i in range(self.arg.n_forward):
                        # forward
                        # imgs: Torch tensor [mini_batch, C, T, H, W]
                        # instruction: list of str len=mini_batch
                        logits = self.model(imgs[mini_batch*i:mini_batch*(i+1)], instructions[mini_batch*i:mini_batch*(i+1)],states_tensor[mini_batch*i:mini_batch*(i+1)])
                        # logits: Torch tensor [mini_batch, 1, ActionNum(22), ActionBin(256)]
                        logits = logits.contiguous().view(-1, logits.shape[-1])
                        # print('rr')
                        # print(logits[0])
                        # print(joints[mini_batch*i])
                        # print(states_tensor[mini_batch*i][-10:])
                        loss = self.loss(logits, joints[mini_batch*i:mini_batch*(i+1)])
                        # loss = self.loss(logits, joints[mini_batch*i:mini_batch*(i+1)].contiguous().view(-1))

                        losses.update(loss.item())
        
            self.print_log('evaluating: loss: {:.4f}'.format(losses.avg))

    def start(self):

        if self.arg.run_mode == 'train':

            for argument, value in sorted(vars(self.arg).items()):
                self.print_log('{}: {}'.format(argument, value))

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            self.print_log('###***************start training***************###')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
        

                if ((epoch + 1) % self.arg.eval_interval == 0):
                    self.eval(epoch, loader_name=['test'])
                if (epoch+1)%self.arg.save_epoch==0:
                    state = {'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
                    print('save',self.arg.work_dir+ '/model_%d.pth' % (epoch+1))
                    torch.save(state,  self.arg.work_dir+ '/model_%d.pth' % (epoch+1))
            self.print_log('Done.\n')

        elif self.arg.run_mode == 'test':

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.eval(epoch=0, loader_name=['test'])
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.cuda_visible_device
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
