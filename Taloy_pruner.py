'''
Examples of structured pruners with Taylor filter pruning
Author: shan.zhu@enflame-tech.com
'''

import argparse
import os
import json
import torch
import sys
import numpy as np
import torch.nn.parallel
import types
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
import time
from nni.compression.torch.utils.config_validation import CompressorSchema
from schema import And, Optional, SchemaError
from models.imagenet.resnet import resnet50
import torchvision
##write to tensorboard
from utils.loggers import *
##for ddp training
from utils.dist import *
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
##for pruning library
from nni.compression.torch import Pruner
from nni.compression.torch.pruning.one_shot import OneshotPruner
from nni.compression.torch.utils.counter import count_flops_params
from nni.compression.torch.pruning.weight_masker import WeightMasker
import logging
_logger = logging.getLogger('Taylor_Pruner_Debug')
_logger.setLevel(logging.DEBUG)
#define file handler and set formatter
file_handler=logging.FileHandler('taylor_prune_logfile.log')
formatter=logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add handler to _logger
_logger.addHandler(file_handler)
#reference code from torchvision classification examples
sys.path.append("/home/devdata/shan/vision/references/classification/")
from train import evaluate, train_one_epoch, load_data

class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__
def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def get_dummy_input_img(device):
    dummy_input=torch.randn([1,3,224,224]).to(device)
    return dummy_input
#overwrite the BN layer maskers
class BNTaylorPrunerMasker(WeightMasker):
    def __init__(self, model, pruner, statistics_batch_num,fre,tot_prune,save_path=None):
        super().__init__(model, pruner)
        self.pruner=pruner
        self.pruner.statistics_batch_num = statistics_batch_num
        self.pruner.set_wrappers_attribute("contribution", None)
        self.pruner.iterations = 0
        #self.pruner.patch_optimizer(self.calc_contributions)
        self.fre=fre
        self.stop_pruning=tot_prune
        self.global_threshold=0
        #to test all residual connections having the same wrapper contribution as well as wrapper weight_mask
        self.contribution_test=[None,None,None,None]
        self.relu_mask_test=[None,None,None,None]
        #for save the model and mask after pruning is done
        self.save_path=save_path
        #self.pruner.patch_optimizer(self.calc_contributions)

    def get_mask(self):
        """
        within pruning, update the contribution and perform pruning with specific frequency
        """
        self.pruner.optimizer.step()
        print('number of step iteration is {} and to iterations number is {}'.format(
            self.pruner.iterations,self.pruner.statistics_batch_num))
        if self.pruner.iterations<self.pruner.statistics_batch_num:
            self.pruner.iterations+=1
            return
        if get_world_size()>1:##for DDP barrier before calculate the new mask contribution
            barrier()
        weight_list = []
        weight_indx = []
        print('all modules to pruner is ',len(self.pruner.modules_wrapper))
        for wrapper in self.pruner.modules_wrapper:
            if wrapper.type=='ReLU' and 'layer1.0' in wrapper.name:
                self.relu_mask_test[0]=wrapper.contribution.data
                weight_list.extend(torch.unsqueeze(wrapper.contribution,0))
            elif wrapper.type=='ReLU' and 'layer2.0' in wrapper.name:
                self.relu_mask_test[1]=wrapper.contribution.data
                weight_list.extend(torch.unsqueeze(wrapper.contribution,0))
            elif wrapper.type=='ReLU' and 'layer3.0' in wrapper.name:
                self.relu_mask_test[2]=wrapper.contribution.data
                weight_list.extend(torch.unsqueeze(wrapper.contribution,0))
            elif wrapper.type=='ReLU' and 'layer4.0' in wrapper.name:
                self.relu_mask_test[3]=wrapper.contribution.data
                weight_list.extend(torch.unsqueeze(wrapper.contribution,0))
            elif wrapper.type=='ReLU':
                if 'layer1' in wrapper.name:
                    assert all(wrapper.contribution.data==self.relu_mask_test[0]), 'ERROR GATE contribution diff RELU'
                elif 'layer2' in wrapper.name:
                    assert all(wrapper.contribution.data==self.relu_mask_test[1]),'ERROR GATE contribution diff RELU'
                elif 'layer3' in wrapper.name:
                    assert all(wrapper.contribution.data==self.relu_mask_test[2]),'ERROR GATE contribution diff RELU'
                elif 'layer4' in wrapper.name:
                    assert all(wrapper.contribution.data==self.relu_mask_test[3]),'ERROR GATE contribution diff RELU'
            else:
                weight_list.extend(torch.unsqueeze(wrapper.contribution,0))
        all_bn_weights = torch.cat(weight_list)
        _logger.debug('A DEBUG message FOR ALL CONTRIBUTION ')
        _logger.debug(all_bn_weights)
        k = self.pruner.fre#current neutrons to prune
        print('DEBUG Fre======',k)
        print('DEBUG left to prune======',self.pruner.stop_neutrons)
        #print("DEBUG k num to prune is {} and length of all bn is {} and shape is {}".format(k,len(all_bn_weights),all_bn_weights.shape))
        if get_world_size()>1:##for DDP barrier before calculate the new mask contribution
            barrier()
        self.global_threshold = torch.topk(
            all_bn_weights, k, largest=False)[0].max()
        if get_world_size()>1:##for DDP barrier before calculate the new mask contribution
            barrier()
        _logger.debug('DEBUG message for the current thresholder')
        _logger.debug(self.global_threshold)
        _logger.debug('DEBUG message for current topK')
        _logger.debug(k)
        self.update_mask()
        self.pruner.iterations=0

    def update_mask(self):
        """
        after specific steps to update the mask and pruning more
        """
        count_prune=0
        for wrapper in self.pruner.modules_wrapper:
            #print('during update wrapper is {}'.format(wrapper))
            if wrapper.type=='BatchNorm2d':
                filters=wrapper.module.weight.size(0)
            elif wrapper.type=='ReLU':
                filters=wrapper.weight_mask.size(0)
            print('DEBUG====wrapper.type is {} and dimension is {}'.format(wrapper.type,filters))
            #weight=wrapper.module.weight
            #filters = weight.size(0)
            print('DEBUG current global_threshold is {}'.format(self.global_threshold))
            contrib_list=wrapper.contribution.squeeze().tolist()
            for idx,contrib in enumerate(contrib_list):
                if contrib<=self.global_threshold:
                    if wrapper.type=='BatchNorm2d':
                        wrapper.weight_mask.data[idx]*=0
                        wrapper.weight_mask_b.data[idx]*=0
                        count_prune+=1
                    elif wrapper.type=='ReLU':
                        wrapper.module_added.weight_mask.data[idx]*=0
                        wrapper.module_added.weight_mask_b.data[idx]*=0
                        if any([iden_name in wrapper.name for iden_name in ['layer1.0','layer2.0','layer3.0','layer4.0']]):
                            count_prune+=1
            if 'relu' in wrapper.name and any([iden_name in wrapper.name for iden_name in ['layer1.0','layer2.0','layer3.0','layer4.0']]):
                if '1.0' in wrapper.name:self.contribution_test[0]=wrapper.module_added.weight_mask.data
                elif '2.0' in wrapper.name:self.contribution_test[1]=wrapper.module_added.weight_mask.data
                elif '3.0' in wrapper.name:self.contribution_test[2]=wrapper.module_added.weight_mask.data
                elif '4.0' in wrapper.name:self.contribution_test[3]=wrapper.module_added.weight_mask.data
            elif 'relu' in wrapper.name:
                if 'layer1' in wrapper.name:
                    assert all(wrapper.weight_mask.data==self.contribution_test[0]), 'ERROR GATE RELU'
                elif 'layer2' in wrapper.name:
                    assert all(wrapper.weight_mask.data==self.contribution_test[1]),'ERROR GATE RELU'
                elif 'layer3' in wrapper.name:
                    assert all(wrapper.weight_mask.data==self.contribution_test[2]),'ERROR GATE RELU'
                elif 'layer4' in wrapper.name:
                    assert all(wrapper.weight_mask.data==self.contribution_test[3]),'ERROR GATE RELU'
            wrapper.contribution=None
        _logger.debug('DEBUG message for current counted pruned netrons are:')
        _logger.debug(count_prune)
        print('DEBUG===current fre {} and current count pruned {}'.format(self.pruner.fre,count_prune))
        #self.pruner.stop_neutrons-=count_prune
        if self.pruner.stop_neutrons<=self.pruner.fre:
            #pruning is done; reset the optimizer.step as well save the model
            self.pruner.reset_optimizer()
            if self.save_path is not None:
                if get_rank()==0:
                    self.pruner.export_model(os.path.join(self.save_path,'model_fineTuned.pth'),
                            os.path.join(self.save_path,'mask_done.pth'))
            else:
                _logger.debug('final pruning is done, but model and mask are not save')
        self.set_momentum_zero() # set momentum after pruning to be zero
        self.pruner.fre+=100

    def set_momentum_zero(self):
        """
        For the pruning part set gate layer momentum to be zero after every pruning step done
        """
        #self.pruner.optimizer
        for wrapper in self.pruner.modules_wrapper:
            if wrapper.type=='BatchNorm2d':
                filters=wrapper.module.weight.size(0)
            elif wrapper.type=='ReLU':
                filters=wrapper.weight_mask.size(0)
            for idx in range(filters):
                if wrapper.type=='BatchNorm2d':
                    if wrapper.weight_mask.data[idx]==0:continue
                    if 'momentum_buffer' in self.pruner.optimizer.state[wrapper.weight_mask].keys():
                        self.pruner.optimizer.state[wrapper.weight_mask]['momentum_buffer'][idx] *= 0.0
                    _logger.debug('momentum set to be zero!!!')
                    _logger.debug(idx)
                    _logger.debug(self.pruner.optimizer.state[wrapper.weight_mask]['momentum_buffer'][idx])
                elif wrapper.type=='ReLU':
                    if wrapper.module_added.weight_mask.data[idx]==0:
                        continue
                    if 'momentum_buffer' in self.pruner.optimizer.state[wrapper.module_added.weight_mask].keys():
                        self.pruner.optimizer.state[wrapper.module_added.weight_mask]['momentum_buffer'][idx] *= 0.0
                    _logger.debug('momentum set to be zero!!!')
                    _logger.debug(idx)
                    _logger.debug(self.pruner.optimizer.state[wrapper.module_added.weight_mask]['momentum_buffer'][idx])
            _logger.debug('momentum buffer set zero done for wrapper: ')
            _logger.debug(wrapper.name)

    def calc_contributions(self):
        """
        Calculate the estimated importance of filters as a sum of individual contribution
        based on the first order taylor expansion.
        """
        #print("current contribution calculation==")
        for wrapper in self.pruner.modules_wrapper:
            assert wrapper.type == 'BatchNorm2d' or wrapper.type=='ReLU', 'MyTaylorPruner only supports 2d batch normalization  and ReLU layer pruning'
            if wrapper.type=='BatchNorm2d':
                filters = wrapper.module.weight.size(0)
                contribution = (
                        wrapper.weight_mask*wrapper.weight_mask.grad).data.pow(2).view(filters, -1).sum(dim=1)
            elif wrapper.type=='ReLU':
                filters = wrapper.weight_mask.size(0)
                contribution = (
                        wrapper.module_added.weight_mask*wrapper.module_added.weight_mask.grad).data.pow(2).view(filters,-1).sum(dim=1)
            if wrapper.contribution is None:
                wrapper.contribution = contribution
            else:
                wrapper.contribution += contribution


    def get_channel_sum(self, wrapper, wrapper_idx):
        if self.pruner.iterations < self.pruner.statistics_batch_num:
            return None
        if wrapper.contribution is None:
            return None
        return wrapper.contribution
    

class TaylorReLUadd(torch.nn.Module):
    def __init__(self, input_features, output_features, size_mask):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.size_mask = size_mask
        self.register_buffer("weight_mask_b", torch.ones(output_features))
        self.register_parameter('weight_mask',torch.nn.Parameter(torch.ones(output_features)))
    def forward(self, input):
        return input*self.weight_mask.view(*self.size_mask)
class TaylorReLUwrapper(torch.nn.Module):
    def __init__(self,module,module_added,module_name,module_type,config,pruner):
        """
        wrapper relu with TaylorReLUadd layer
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.pruner = pruner
        self.module_added=module_added
        self.weight_mask=module_added.weight_mask.detach()
        self.bias_mask=None
    def forward(self,input):
        outputs=self.module(input)
        return self.module_added(outputs)

#overwrite BNTaylor wrapper
class TaylorBNWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, pruner):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module : pytorch module
            the module user wants to compress
        config : dict
            the configurations that users specify for compression
        module_name : str
            the name of the module to compress, wrapper module shares same name
        module_type : str
            the type of the module to compress
        pruner ： Pruner
            the pruner used to calculate mask
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.pruner = pruner
        self.filters=self.module.weight.shape[0]
        self.size_mask=[1,self.filters,1,1]
        # register buffer for mask
        self.register_buffer("weight_mask_b", torch.ones(self.module.weight.shape))
        self.register_parameter('weight_mask',torch.nn.Parameter(torch.ones(self.module.weight.shape)))
        self.bias_mask=None
        #update the bias mask
        #self.update_mask(prune_idx)
    
    def forward(self, *inputs):
        # apply mask to weight, bias
        outputs=self.module(*inputs)
        return outputs*self.weight_mask.view(*self.size_mask)

class TaylorPruner(Pruner):
    def __init__(self,device,model,config_list,config_step,dependency_aware=False,optimizer=None):
        self.device = device#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reluWrapper=[TaylorReLUadd(256,256,[1, -1, 1, 1]),TaylorReLUadd(512,512,[1, -1, 1, 1]),
                TaylorReLUadd(1024,1024,[1, -1, 1, 1]),TaylorReLUadd(2048,2048,[1, -1, 1, 1])]
        super().__init__(model, config_list, optimizer)
        self.fre=config_step['fre']
        iterations=0
        statistics_batch_num=config_step['bn_statics']
        self.stop_neutrons=config_step['tot_pru']
        save_path=config_step['save_path'] if 'save_path' in config_step else None
        self.masker = BNTaylorPrunerMasker(model, self, 
                statistics_batch_num=statistics_batch_num,fre=config_step['fre'],tot_prune=config_step['tot_pru'],
                save_path=save_path)
        self.org_step=None

    def keep_org_step(self):
        assert self.optimizer is not None,"error for empty optimizer"
        _logger.info('optimizer step RESET to orginal one without contribution calculation')
        self.org_step=self.optimizer.step
    
    def step(self):
        """
        update contribution after every step during pruning steps
        after pruning, normal optimizer step will be called
        """
        if self.stop_neutrons>=self.fre:
            self.masker.get_mask()
        else:
            self.optimizer.step()


    def compress(self):
        print(self.config_list)
        return self.bound_model


    def _wrap_modules(self, layer, config):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        print('DEBUG module detected to compress:',layer.name)
        if layer.type=='BatchNorm2d':
            wrapper = TaylorBNWrapper(layer.module, layer.name, layer.type, config, self)
            assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
            # move newly registered buffers to the same device of weight
        elif layer.type=='ReLU':
            print('DEBUG===relu wrapper is ',self.reluWrapper)
            if 'layer1' in layer.name:
                wrapper = TaylorReLUwrapper(layer.module,self.reluWrapper[0],layer.name,layer.type,config,self)
            elif 'layer2' in layer.name:
                wrapper= TaylorReLUwrapper(layer.module,self.reluWrapper[1],layer.name,layer.type,config,self)
            elif 'layer3' in layer.name:
                wrapper= TaylorReLUwrapper(layer.module,self.reluWrapper[2],layer.name,layer.type,config,self)
            elif 'layer4' in layer.name:
                wrapper= TaylorReLUwrapper(layer.module,self.reluWrapper[3],layer.name,layer.type,config,self)
            else:
                print('ERROR unsupported module wrapper appears')
                raise 
        #wrapper.to(layer.module.weight.device)
        wrapper.to(self.device)
        return wrapper
    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None, device=None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path : str
            path to save pruned model state_dict
        mask_path : str
            (optional) path to save mask dict
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        device : torch.device
            device of the model, used to place the dummy input tensor for exporting onnx file.
            the tensor is placed on cpu if ```device``` is None
        """
        assert model_path is not None, 'model_path must be specified'
        mask_dict = {}
        self._unwrap_model() # used for generating correct state_dict name without wrapper state

        for wrapper in self.get_modules_wrapper():
            weight_mask = wrapper.weight_mask
            bias_mask = wrapper.bias_mask
            if wrapper.type=='ReLU':
                weight_mask=wrapper.module_added.weight_mask
                bias_mask=wrapper.module_added.weight_mask
                mask_dict[wrapper.name] = {"weight": weight_mask, "bias": bias_mask}
                continue
            if weight_mask is not None:
                mask_sum = weight_mask.sum().item()
                mask_num = weight_mask.numel()
                _logger.debug('Layer: %s  Sparsity: %.4f', wrapper.name, 1 - mask_sum / mask_num)
                if wrapper.type!='ReLU':
                    wrapper.module.weight.data = wrapper.module.weight.data.mul(weight_mask)
            if bias_mask is not None and wrapper.type!='ReLU':
                wrapper.module.bias.data = wrapper.module.bias.data.mul(bias_mask)
            # save mask to dict
            _logger.debug('DEBUG message during model export')
            _logger.debug('wrapper is')
            _logger.debug(wrapper.name)
            _logger.debug('weight maks is')
            _logger.debug(weight_mask)
            mask_dict[wrapper.name] = {"weight": weight_mask, "bias": bias_mask}

        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)
        if mask_path is not None:
            torch.save(mask_dict, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)
        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            if device is None:
                device = torch.device('cpu')
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data.to(device), onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)

        self._wrap_model()
    
    def reset_optimizer(self):
        """
        reset optimizer to origin one
        """
        _logger.info('RESET OPTIMIZER STEP')
        self.optimizer.step = self.org_step
    
    def patch_optimizer(self, *tasks):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                # call origin optimizer step method
                output = old_step(*args, **kwargs)
                if get_world_size()>1:##for DDP barrier before calculate the new mask contribution
                    barrier()
                for task in tasks:
                    #print('for current task to call the methods',task)
                    task()
                return output
            return new_step
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)

    def select_config(self, layer):
        """
        overwite schema
        """
        ret = None
        for config in self.config_list:
            config = config.copy()
            # expand config if key `default` is in config['op_types']
            if 'op_types' in config and 'default' in config['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == 'default':
                        expanded_op_types.extend(default_layers.weighted_modules)
                    else:
                        expanded_op_types.append(op_type)
                config['op_types'] = expanded_op_types

            # check if condition is satisified
            if config['must_names'] not in layer.name:# or layer.name not in config['include_names']:
                continue
            if 'op_types' in config and layer.type not in config['op_types']:
                continue
            if 'include_names' in config and all([include_name not in layer.name for include_name in config['include_names']]):
                continue

            ret = config
        if ret is None or 'exclude' in ret:
            return None
        return ret
    def validate_config(self, model, config_list):
        schema = CompressorSchema([{
            'op_types': ['BatchNorm2d','ReLU'],
            Optional('op_names'): [str],
            Optional('must_names'):str,
            Optional('include_names'):[str]
        }], model, _logger)

        schema.validate(config_list)

        if len(config_list) > 1:
            _logger.warning('MyTaylor pruner only supports 1 configuration')

def get_data(dataset, data_dir, batch_size, test_batch_size):
    '''
    get data for imagenet
    '''
    nThread=1
    pin=True # for cuda device
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'validation')
    print('train_dir is ',traindir)
    dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False,get_world_size()>1)
    train_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=nThread, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
                        dataset_test, batch_size=test_batch_size,
                                        sampler=test_sampler, num_workers=nThread, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()

    return train_loader, val_loader, criterion

from nni.compression.torch.compressor import *
def train(pruner,args, model, device, train_loader, criterion, optimizer, epoch,logger, callback=None):
    model.train()
    paral=get_world_size()
    print(len(train_loader.dataset))
    print('device is ',device)
    print('current rank is ',get_rank())
    Nstep=len(train_loader.dataset)//paral
    loss_per_batch=AverageMeter()
    overall_time=AverageMeter()
    print('current device is {}'.format(device))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #print(data.shape)
        stime=time.time()
        output = model(data)
        #if batch_idx%args.log_interval==0:
        #    print('The performace of training is {} fps'.format(args.batch_size/(etime-stime)))
        loss = criterion(output, target)
        loss.backward()
        loss_per_batch.update(loss)
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        #optimizer.step()
        pruner.step()
        etime=time.time()
        overall_time.update(etime-stime)
        if batch_idx%args.log_interval==0:
            print('The performace of training is {} fps'.format(args.batch_size/(etime-stime)))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        tensorboard_log = []
        tensorboard_train_loss=[]
        tensorboard_lr=[]
        model.syn_buffer_params()
        wrap_mask=[(module[0],module[1].state_dict()['weight_mask'],module[1]) 
                for module in model.named_modules() if isinstance(module[1],TaylorBNWrapper)]
        relu_mask=[(module[0],module[1].module_added.state_dict()['weight_mask'],module[1]) 
                for module in model.named_modules() if isinstance(module[1],TaylorReLUwrapper) and '0.relu' in module[0]]
        wrap_mask.extend(relu_mask)
        masks=[(mask[0],mask[1].cpu().numpy()) for mask in wrap_mask]
        #print('DEBUG all masks are ',masks)
        def ratio(array):
            N=len(array)
            remain=sum([np.all(array[i]==1) for i in range(N)])
            return (remain,N)
        mask_remain=[(mask[0],ratio(mask[1])) for mask in masks]
        for i, (name,ratios) in enumerate(mask_remain):
            tensorboard_log += [(f"{name}_num_filters", ratios[1])]
            tensorboard_log += [(f"{name}_num_filters_remain", ratios[0])]
        tensorboard_train_loss += [("loss", loss.item())]
        tensorboard_lr += [("lr", optimizer.param_groups[0]['lr'])]
        logger.list_of_scalars_summary('train', tensorboard_log, 
                args.batch_size*batch_idx+(epoch)*Nstep)
        logger.list_of_scalars_summary('train_loss', tensorboard_train_loss,
                args.batch_size*batch_idx+(epoch)*Nstep)
        logger.list_of_scalars_summary('learning_rate', tensorboard_lr,
                args.batch_size*batch_idx+(epoch)*Nstep)
        #bn_weights = gather_bn_weights(model.module_list, prune_idx)
        #logger.writer.add_histogram('bn_weights/hist', bn_weights.numpy(), epoch, bins='doane')
    overall_time.reduce('mean')
    print('over_all card average time is',overall_time.avg)



def test(model, device, criterion, val_loader,step,logger):
    paral=get_world_size()
    model.eval()
    test_loss = 0
    correct_curr = 0
    correct=AverageMeter()
    print('current device is {}'.format(device))
    with torch.no_grad():
        for idx,(data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            stime=time.time()
            output = model(data)
            etime=time.time()
            if idx%args.log_interval==0:
                print('Performance for inference is {} second'.format(etime-stime))
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct_curr += pred.eq(target.view_as(pred)).sum().item()
            correct.update(pred.eq(target.view_as(pred)).sum().item())
            if idx % args.log_interval == 0:
                print('Evaluation: [{}/{} ({:.0f}%)]\tcorrect: {:.6f}'.format(
                    idx * len(data), len(val_loader.dataset),
                    100. * idx / len(val_loader), correct_curr))
            #logger.list_of_scalars_summary('valid', test_loss, idx)

    print('Done for the validation dataset')
    test_loss /= (len(val_loader.dataset)/paral)
    correct.reduce('sum')
    accuracy = correct.sum/ len(val_loader.dataset)
    print('corrent all is {} and accuracy is {}'.format(correct.avg,accuracy))
    curr_rank=get_rank()
    logger.list_of_scalars_summary('valid_loss',[('loss',test_loss)],step)
    logger.list_of_scalars_summary('valid_accuracy',[('accuracy',accuracy)],step)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct.avg, len(val_loader.dataset), 100. * accuracy))

    return accuracy



def get_dummy_input(args, device):
    if args.dataset=='imagenet':
        dummy_input=torch.randn([args.test_batch_size,3,224,224]).to(device)
    return dummy_input


def get_input_size(dataset):
    if dataset == 'mnist':
        input_size = (1, 1, 28, 28)
    elif dataset == 'cifar10':
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        input_size = (1, 3, 256, 256)
    return input_size


def update_model(model,pruner):
    # add by shan, update model at every epoch
    pruner.bound_model=model
    pruner.update_mask
    return pruner.bound_model

def main(args):
    # prepare dataset
    torch.manual_seed(0)
    model=resnet50()
    model.load_state_dict(torch.load(args.pretrained_model_dir))
    inited=init_distributed(True) #use nccl fro communication
    print('all cudas numbers are ',get_world_size())
    distributed=(get_world_size()>1) and inited
    paral=get_world_size()
    #device = torch.device('cuda',args.local_rank) if distributed else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #args.rank = get_rank()
    device = set_device(args.cuda, args.local_rank)
    #write to tensorboard
    logger = Logger("logs/"+str(args.local_rank))
    print(distributed)
    print('local rank is {}'.format(args.local_rank))
    train_loader, val_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    print('to distribute ',distributed)
    model=model.to(args.local_rank)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])#, output_device=args.local_rank)
    #elif args.mgpu:
    #    model=torch.nn.DataParallel(model).cuda()
    else:
        model=model.cuda()
    #model = torch.nn.DataParallel(model).cuda()
    criterion=criterion.to(args.local_rank)
    #for module in model.named_modules():
    #    print('%'*20)
    #    print(module)
    # module types to prune, only "BatchNorm2d" supported for channel pruning with Taylor 
    config_step={'bn_statics': args.bn_statistics,
        'fre':args.freq,
        'tot_pru':args.tot_pru,
        'save_path':args.experiment_data_dir}
    config_list = [{
        'op_types': ['BatchNorm2d','ReLU'],
        'must_names':'layer',
        'include_names':['bn1','bn2','relu1']
    }]
    dummy_input = get_dummy_input(args, args.local_rank)
    if args.pruner == 'TaylorPruner':
        pruner=TaylorPruner(device,model,config_list,config_step,dependency_aware=False,optimizer=None)
    else:
        raise ValueError(
            "Pruner not supported.")

    # Pruner.compress() returns the masked model
    model = pruner.compress()
    if args.pruner == 'TaylorPruner':
        params_update=[param[1] for param in model.named_parameters() if 'mask' not in param[0]]
        not_updates=[param[1] for param in model.named_parameters() if 'mask' in param[0]]
        #print(params_update)
        params_all=[{'params':params_update},{'params':not_updates,'lr':0.0}]
    if args.fine_tune:
        if args.dataset in ['imagenet'] and args.model == 'resnet50':
            optimizer = torch.optim.SGD(params_all, lr=0.01, momentum=0.5, weight_decay=1e-8)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.fine_tune_epochs*0.3), int(args.fine_tune_epochs*0.6),int(args.fine_tune_epochs*0.8)], gamma=0.1)
            pruner.optimizer=optimizer
            pruner.patch_optimizer(pruner.masker.calc_contributions)
            pruner.keep_org_step()
        else:
            raise ValueError

    def short_term_fine_tuner(model, epochs=1):
        for epoch in range(epochs):
            train(pruner,args, model, device, train_loader, criterion, optimizer, epoch,logger)

    def trainer(pruner,model, optimizer, criterion, epoch, callback):
        return train(pruner,args, model, device, train_loader, criterion, optimizer, epoch=epoch, logger=logger, callback=callback)

    def evaluator(model,step):
        return test(model, device, criterion, val_loader,step,logger)

    # used to save the performance of the original & pruned & finetuned models
    result = {'flops': {}, 'params': {}, 'performance':{}}

    #print(model)
    for module in model.named_modules():
        print('DEBUG===name of module is ',module[0])
        print('buffers are: ',[buff for buff in module[1].buffers()])
        print('para are: ',[para for para in module[1].named_parameters()])
    flops, params = count_flops_params(model, get_input_size(args.dataset))
    result['flops']['original'] = flops
    result['params']['original'] = params

    evaluation_result = evaluator(model,0)
    print('Evaluation result (original model): %s' % evaluation_result)
    result['performance']['original'] = evaluation_result


    if args.local_rank==0 and args.save_model:
        pruner.export_model(
            os.path.join(args.experiment_data_dir, 'model_masked.pth'), os.path.join(args.experiment_data_dir, 'mask.pth'))
        print('Masked model saved to %s', args.experiment_data_dir)

    def wrapped(module):
        return isinstance(module,BNTaylorPrunerMasker)
    wrap_mask=[module for module in model.named_modules() if wrapped(module[1])]
    for idx,mm in enumerate(wrap_mask):
        print('====****'*10,idx)
        print(mm[0])
        print(mm[1].state_dict().keys())
        print('weight mask is ',mm[1].state_dict()['weight_mask'])
        if 'bias_mask' in mm[1].state_dict():
            print('bias mask is ',mm[1].state_dict()['bias_mask'])

    if args.mgpu:model=torch.nn.DataParallel(model).cuda()
    print('local rank is',args.local_rank)
    if args.fine_tune:
        best_acc = 0
        for epoch in range(args.fine_tune_epochs):
            print('start fine tune for epoch {}/{}'.format(epoch,args.fine_tune_epochs))
            stime=time.time()
            train(pruner,args, model, args.local_rank, train_loader, criterion, optimizer, epoch,logger)
            scheduler.step()
            acc = evaluator(model,epoch)
            print('end fine tune for epoch {}/{} for {} seconds'.format(epoch,
                args.fine_tune_epochs,time.time()-stime))
            if acc > best_acc and args.local_rank==0:
                best_acc = acc
                torch.save(model,os.path.join(args.experiment_data_dir,args.model,'finetune_model.pt'))
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))

    print('Evaluation result (fine tuned): %s' % best_acc)
    print('Fined tuned model saved to %s', args.experiment_data_dir)
    result['performance']['finetuned'] = best_acc

    if args.local_rank==0:
        with open(os.path.join(args.experiment_data_dir, 'result.json'), 'w+') as f:
            json.dump(result, f)


if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='PyTorch Example for SimulatedAnnealingPruner')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to use resnet18 or resnet50')
    parser.add_argument('--cuda',type=str2bool,default=True,
                        help='whether use cuda')
    parser.add_argument('--load-pretrained-model', type=str2bool, default=False,
                        help='whether to load pretrained model')
    parser.add_argument('--pretrained-model-dir', type=str, default='./',
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='number of epochs to pretrain the model')
    parser.add_argument("--mgpu",type=str2bool,default=False,
            help='Local rank. Necessary for distributed train with dataparallel')
    parser.add_argument("--local_rank",type=int,help='Local rank. Necessary for distributed train')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=10,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data/resnet_bn',
                        help='For saving experiment data')

    # pruner
    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner',
                        help='pruner to use')
    parser.add_argument('--bn_statistics', type=int, default=30,
                        help='contributions collection step num')
    parser.add_argument('--freq', type=int, default=100,
                        help='number of netrons being pruned every bn_statisctis steps')
    parser.add_argument('--tot_pru', type=int, default=2200,
                        help='maximum number of netrons being pruned')

    # others
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)
