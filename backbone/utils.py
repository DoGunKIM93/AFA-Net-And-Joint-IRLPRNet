'''
utils.py
'''
version = "1.25.200427.1"

import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse

import types
import math
import numpy as np
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import apex.amp as amp
from apex.parallel import DistributedDataParallel as DDP

import param as p


#local function
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def denorm(x):
    out = x
    if p.valueRangeType == '-1~1':
        out = (x + 1) / 2
    
    return out.clamp(0, 1)

'''
def norm(x):
    out = (x - 0.5) * 2
    print(out)
    return out.clamp(-1,1)
'''

def calculateImagePSNR(a, b):

    pred = a.cpu().data[0].numpy().astype(np.float32)
    gt = b.cpu().data[0].numpy().astype(np.float32)

    np.nan_to_num(pred, copy=False)
    np.nan_to_num(gt, copy=False)

    if p.valueRangeType == '-1~1':
        pred = (pred + 1)/2
        gt = (gt + 1)/2

    if p.colorMode == 'grayscale':
        pred = np.round(pred * 219.)
        pred[pred < 0] = 0
        pred[pred > 219.] = 219.
        pred = pred[0,:,:] + 16
            
        gt = np.round(gt * 219.)
        gt[gt < 0] = 0
        gt[gt > 219.] = 219.
        gt = gt[0,:,:] + 16
    elif p.colorMode == 'color':
        pred = 16 + 65.481*pred[0:1,:,:] + 128.553*pred[1:2,:,:] + 24.966*pred[2:3,:,:]
        pred[pred < 16.] = 16.
        pred[pred > 235.] = 235.

        gt = 16 + 65.481*gt[0:1,:,:] + 128.553*gt[1:2,:,:] + 24.966*gt[2:3,:,:]
        gt[gt < 16.] = 16.
        gt[gt > 235.] = 235.

    

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    #print(20 * math.log10(255.0/ rmse), cv2.PSNR(gt, pred), cv2.PSNR(cv2.imread('sr.png'), cv2.imread('hr.png')))
    return 20 * math.log10(255.0/ rmse)
    
class NotOneCycleLR(_LRScheduler):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """
    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 last_epoch=-1):

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected non-negative integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected non-negative integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected non-negative integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(NotOneCycleLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        lrs = []
        step_num = self.last_epoch % self.total_steps

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                down_step_num = step_num - self.step_size_up
                computed_lr = self.anneal_func(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss



def logValues(writer, valueTuple, iter):
    writer.add_scalar(valueTuple[0], valueTuple[1], iter)

def logImages(writer, imageTuple, iter):
    saveImages = torch.clamp(imageTuple[1], 0, 1)
    for i in range(imageTuple[1].size(0)):
        writer.add_image(imageTuple[0], imageTuple[1][i,:,:,:], iter)


def loadModels(modelList, version, subversion, loadModelNum, isTest):
    startEpoch = 0
    lastLoss = torch.ones(1)*100
    bestPSNR = 0
    for mdlStr in modelList.getList():
        modelObj = getattr(modelList, mdlStr)
        optimizer = getattr(modelList, mdlStr + "_optimizer") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_optimizer")]) > 0 else None
        scheduler = getattr(modelList, mdlStr + "_scheduler") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_scheduler")]) > 0 else None
       

        modelObj.cuda()




        if (loadModelNum is not 'None' or len([attr for attr in vars(modelList) if attr == (mdlStr+"_pretrained")]) > 0 ): # 로드 할거야
            
            isPretrainedLoad = False
            if optimizer is None:
                isPretrainedLoad = True
            else:
                try:
                    if(loadModelNum == '-1'):
                        checkpoint = torch.load('./data/' + version + '/model/'+subversion+'/' + mdlStr + '.pth')
                    else:
                        checkpoint = torch.load('./data/' + version + '/model/'+subversion+'/' + mdlStr + '-' + loadModelNum+ '.pth')
                except:
                    print("utils.py :: Failed to load saved checkpoints.")
                    if modelList.getPretrainedPath(mdlStr) is not None:
                        isPretrainedLoad = True

            if isPretrainedLoad is True:
                print(f"utils.py :: load pretrained model... : {modelList.getPretrainedPath(mdlStr)}")
                loadPath = modelList.getPretrainedPath(mdlStr)
                checkpoint = torch.load(loadPath)
            
            
            # LOAD MODEL
            '''
            mk = list(modelObj.module.state_dict().keys())
            ck = list(checkpoint.keys())

            for i in range(len(mk)):
                if mk[i] != ck[i]:
                    print(mk[i], ck[i])
            
            '''
                

            try:
                modelObj.load_state_dict(checkpoint['model'],strict=True)
            except:
                try:
                    print("utils.py :: model load failed... load model in GLOBAL STRUCTURE mode..")
                    modelObj.load_state_dict(checkpoint ,strict=True)
                except:
                    try:
                        print("utils.py :: model load failed... load model in INNER MODEL GLOBAL STRUCTURE mode..")
                        modelObj.module.load_state_dict(checkpoint ,strict=True)
                    except:
                        try:
                            print("utils.py :: model load failed... load model in UNSTRICT mode.. (WARNING : load weights imperfectly)")
                            modelObj.load_state_dict(checkpoint['model'],strict=False)
                        except:
                            try:
                                print("utils.py :: model load failed... load model in GLOBAL STRUCTURE UNSTRICT mode.. (WARNING : load weights imperfectly)")
                                modelObj.load_state_dict(checkpoint ,strict=False)
                            except:
                                print("utils.py :: model load failed..... I'm sorry~")

            

            # LOAD OPTIMIZER
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optim'])
                    for param_group in optimizer.param_groups: param_group['lr'] = p.learningRate
                except:
                    optimDict = optimizer.state_dict()
                    preTrainedDict = {k: v for k, v in checkpoint.items() if k in optimDict}

                    optimDict.update(preTrainedDict)

            # LOAD VARs..
            try:
                startEpoch = checkpoint['epoch']
            except:
                pass#startEpoch = 0

            try:
                lastLoss = checkpoint['lastLoss']
            except:
                pass#lastLoss = torch.ones(1)*100
            
            try:
                bestPSNR = checkpoint['bestPSNR']
            except:
                pass#bestPSNR = 0
            
            
            if scheduler is not None:
                #scheduler.load_state_dict(checkpoint['scheduler'])
                scheduler.last_epoch = startEpoch
                scheduler.max_lr = p.learningRate
                scheduler.total_steps = p.schedulerPeriod

            try:
                if p.mixedPrecision is True:
                    amp.load_state_dict(checkpoint['amp'])
            except:
                pass

        #modelObj = nn.DataParallel(modelObj)  
        
        paramSize = 0
        for parameter in modelObj.parameters():
            paramSize = paramSize + np.prod(np.array(parameter.size()))
        print(mdlStr + ' : ' + str(paramSize))    

        if (isTest == True):
            modelObj.eval()
        else:
            modelObj.train()

    return startEpoch, lastLoss, bestPSNR
            
def saveModels(modelList, version, subversion, epoch, lastLoss, bestPSNR):

    for mdlStr in modelList.getList():
        modelObj = getattr(modelList, mdlStr)
        optimizer = getattr(modelList, mdlStr + "_optimizer") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_optimizer")]) > 0 else None
        scheduler = getattr(modelList, mdlStr + "_scheduler") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_scheduler")]) > 0 else None

        if optimizer is not None:
            saveData = {}
            saveData.update({'model': modelObj.state_dict()})
            saveData.update({'optim': optimizer.state_dict()})
            if scheduler is not None:
                saveData.update({'scheduler': scheduler.state_dict()})
            saveData.update({'epoch': epoch + 1})
            saveData.update({'lastLoss': lastLoss})
            saveData.update({'bestPSNR': bestPSNR})
            if p.mixedPrecision is True:
                saveData.update({'amp': amp.state_dict()})
            saveData.update({'epoch': epoch + 1})


            torch.save(saveData, './data/'+version+'/model/'+subversion+'/' + mdlStr + '.pth')
            if epoch % p.archiveStep == 0:
                torch.save(saveData, './data/'+version+'/model/'+subversion+'/'+ mdlStr +'-%d.pth' % (epoch + 1))



def backproagateAndWeightUpdate(modelList, loss, modelNames = None):

    modelObjs = []
    optimizers = []
    if modelNames is None:
        modelObjs = modelList.getModels()
        optimizers = modelList.getOptimizers()
    elif isinstance(modelNames, (tuple, list)): 
        for mdlStr in modelList.getList():
            if mdlStr in modelNames:
                modelObj = getattr(modelList, mdlStr)
                optimizer = getattr(modelList, mdlStr + '_optimizer')
                modelObjs.append(modelObj)
                optimizers.append(optimizer)
    else:
        modelObjs.append(getattr(modelList, mdlStr))
        optimizers.append(getattr(modelList, mdlStr + '_optimizer'))


    #init model grad
    for modelObj in modelObjs:
        modelObj.zero_grad()

    #backprop and calculate weight diff
    if p.mixedPrecision == False:
        loss.backward()
    else:
        with amp.scale_loss(loss, optimizers) as scaled_loss:
            scaled_loss.backward()

    #weight update
    for optimizer in optimizers:
        optimizer.step()


        
                




        

class ModelListBase():
    def __init__(self):
        super(ModelListBase, self).__init__()

    def initDataparallel(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]

        for mdlStr in mdlStrLst:
            setattr(self, mdlStr, nn.DataParallel(getattr(self, mdlStr)))

    def initApexAMP(self):
        if p.mixedPrecision is True:
            opt_level = 'O0' if p.mixedPrecision is False else 'O1'
            mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
            for mdlStr in mdlStrLst:
                mdlObj = getattr(self, mdlStr)
                mdlOpt = getattr(self, mdlStr + "_optimizer") if len([attr for attr in vars(self) if attr == (mdlStr+"_optimizer")]) > 0 else None

                if mdlOpt is None:
                    mdlObj = amp.initialize(mdlObj.to('cuda'), opt_level = opt_level)
                    setattr(self, mdlStr, mdlObj)
                else:
                    mdlObj, mdlOpt = amp.initialize(mdlObj.to('cuda'), mdlOpt, opt_level = opt_level)
                    setattr(self, mdlStr, mdlObj)
                    setattr(self, mdlStr + "_optimizer", mdlOpt)

    def getList(self):
        return [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]

    def getModels(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlObjLst = []
        for mdlStr in mdlStrLst:
            mdlObjLst.append(getattr(self, mdlStr))
        return mdlObjLst
    
    def getOptimizers(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlOptLst = []
        for mdlStr in mdlStrLst:
            mdlOptLst.append(getattr(self, mdlStr))
        return mdlOptLst

    def getSchedulers(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlSchLst = []
        for mdlStr in mdlStrLst:
            mdlSchLst.append(getattr(self, mdlStr))
        return mdlSchLst

    def getPretrainedPaths(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and attr.endswith("_pretrained")]
        mdlPpaLst = []
        for mdlStr in mdlStrLst:
            mdlPpaLst.append(getattr(self, mdlStr))
        return mdlPpaLst

    def getPretrainedPath(self, mdlStr):
        pP = p.pretrainedPath + getattr(self, mdlStr + "_pretrained")
        return pP



def printirene():
    print("	@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%//**,,,,*,**,,*(#&&&&&&@@@@@@@@@@@@@@@@@@@@&&&&%")
    print("	@@@@@@@@@@@@@@@@@@@@@@@@@&(**,,,,,,,,,,,,,,,,,,,,,,,,,*%@@@@@@@@@@@@@@@@(@@&&&&%")
    print("	@@@@@@@@@@@@@@@@@@@@@@@(,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*,(@@@@@@@@@@@@@@&&&&%%")
    print("	@@@@@@@@@@@@@@@@@@@@(*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*/@@@@@@@@@@@&&&&%%")
    print("	@@@@@@@@@@@@@@@@@#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*/#@@@@@@@@&&&&%#")
    print("	@@@@@@@@@@@@@&/,,,,,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,***#@@@@@@&&&&%#")
    print("	@@@@@@@@@@&/,,.,,,,,.,,,,,,,,,,,,,.,,,,,.,,,,,,,,,.,..,.,,**,*,,,,,,*&@@@@&&&%%#")
    print("	@@@@@@@&/,,,,,,,,.,,,,,.,,..,,.,,,.,,,,...        .. .....*(*,,,,,,,,*/@@@&&&%%#")
    print("	@@@@@&*,.,...,.,,.,..,,.,.....,,,...              ,. .....,/****,,,,*,**&@&&&%%#")
    print("	@@@@/,,....,,,........,..,,,.....               .,,***,*,,,,/. ,/**,,,**/&&&&%#(")
    print("	@@%,,..,.,...,.,,....,..,.......  .     ....  ../(/#%%%%%#(//*, .,,,,**,,/&&&%#(")
    print("	@#,,,.,,,,,,,...,,,..,,.,........      .....  .,/(/%&&&%%%%%%%#/. ,,*/****%&%%#/")
    print("	&*,,,,,,,,,......,.,............,.    . .. ....,(*/&&&&&&&&&&&%%%,.,///***%&%%#/")
    print("	*,.,,,,....,....... ,*.,.,...*.,..   .  .......*#(&&&&&&&&&&&&&&&&,,,****/%&%%(/")
    print("	,................ .,,,..,..,,.,..    . ...,....,*(%&&&&&&&&&&&&&&&%,..*,*/&&%#(*")
    print("	,,,............ ..,,,,.,*..*,/... . ...,,..,..**%%(*(/%%&&&&&&&&&@&, .,**/&&&#/,")
    print("	,,............ ,.,,,*,**.,,,..,... ......,.. .*##(%%%&%%%&&&&&&&%&&( .,,*&&&&@/,")
    print("	,,.....  ....,*,.*,,,*,*(,*,,,,..,....,,,.*. ,.. ,.,%&&&&&&&&%#(/**/ .,*#&&&&&&%")
    print("	..  .    ..*,/,./*,,*,,,,*.,,,,.,,..,.,.,,,,,/##(**(%&&&&&&&&&&%&&&%,,,/&&&&&&&&")
    print("	,,,.    ..(*/*,***,*.**,*,,,,**,*...*.*,,,.,*%&&&&&&&&&&&&&&%.  *./.,,(&&&&&&&&&")
    print("	,...   /.(/#*,,*/**,*.,*,,.,*,,,*,..,,,.,**,%&&&&&&&&&&&&&&&%#(/%..,,%&&&&&&&&&&")
    print("	@,.....*/#/*/../*/***/,,.,,**,/,,....,.*,,/(&&&&&&&&&&&%%&&&&&&&%%,(%%&&&&&&&&&&")
    print("	@@*...,#**/**/,/,**,*,,**,,,**/**..,.,,,,*(&&&&&&&&%%%&%&&&&&&&&&&%%%%%&&&&&&&&&")
    print("	@@@(%##*/(,,**,*#/,,,.,,,****/***,.*,,.,*/%&&&&&&&&#%#%###%%&&&&&&##%%%%&&&&&&&&")
    print("	@@@@@%/(,/*,.*,//*,,.   .,*******,,,,.,,/#%&&&&&&&&&&&&&&%&&&&&&&((##%%%&&&&&&&&")
    print("	@@@@@(/(( .***/*,.,        ,***,,.,.*.,*#&%%%&&&&&&&&&&&&&&&&&&&@###%%&%%#/,,.. ")
    print("	@@@&#/*(,,*.(,,  .          .,,,.....,((#%,/#%%&&.,(/,,//#&&&&@@@&&&(,*..,/**/,,")
    print("	@@&#(,**,,//..     .  . ,,(,,*,/,..,,**%%%&&&%&&%#(((//,,#&&&@@&&#  .,/*.....,/.")
    print("	@@##//,,/(.. .,,*,*,**,*//(/*(/,******,#%%%&&&&&&&%#(((#&&&&@&&&*  .  ....... ,.")
    print("	@#%/*(,//,.,*.****(/**#(/#**////#***,..,**(%&&&&&&&&&&&&&%&&&&&%..         .*(#%")
    print("	&&#**,(,/,,*(//(/,*//((,*,,... / ../,,,/((///#%%&&&&&&&%./#%&&#/*/    . .,#%%%%%")
    print("	&&#**(#(*,,,*/***.////,,.//....*...*,,.....,*,,/##%%%%.,.,//*,..*,  *,..,#%%%&%&")
    print("	&/*/(//(,,**,**.,*/,*#*..(.,,.(.,.(/(((###(/%(/,.. .,, .(//..... . ,,.,.*%%%%&&&")
    print("	#**//*.*,,*,./*,*,.,*..//..,#,**(/*,,**(#...     ,.  .**(,.**/////,,.,.*(%%&&&&&")
    print("	**//.,...,./**.,, ./..,,..,(,,*/////(,..*.   .**.  .,,,*.........,,.,*,.(%&%&&&%")
    print("	//*,..,,..**,*.,.  ..(,.,.(.,**,,*,   .*   .,,,  ./*,... .,,,..  ,*/(.,,#%%%%%%%")
    print("	/,,,..*..,**,.,.   .,,.,*.,,(,,,/.   . .  ,,*,... ,  ,*,,.....   ./,/*/. ,##%%%#")
    print("	/.,.,..,,,,,,,,,.. ......,,*,,....  .   .,*,..  ...,.,..        .*,........*(((*")
    print("	,,,,..* /.***,..   ,.....,,..  ..  .....  ... .,.  .,.             .     ......,")
    print("	..,...,,.,*,,.*     .........  ...,(  ..  .,,.*     ,..                      ...")
    print("	....* ,/,,**,   ,    .,.,,.,.. .**.   ,/.. ..  . ...,...                     ...")
    print("	.... ., .,,*...     .,,.,, ..*,  ..     .**,*,...  ..*,...           ,   .   ...")
    print("	. .. .....,...,.     .,*......   ,,     .....      ..........  ...... ,**,./*...")
    print("	.  .. ,/,.,... ...*...,.. (*. .,..    ...         ..   . ..,*..,* ..,,,,,..*(*,,")
    print("	.  .,..,..,...     ...# .....  . .     .        . .,,.,,..    .     #%,., ,..*#*")