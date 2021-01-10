"""
speed up the resnet model with structured mask added to model
Author: shan.zhu@enflame-tech.com
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.imagenet.resnet import resnet50
#from torchvision import datasets, transforms,models
from models.cifar10.vgg import VGG
from nni.compression.torch import apply_compression_results, ModelSpeedup
import collections
import sys
from nni.compression.torch.utils.counter import count_flops_params 
sys.path.append("/home/devdata/shan/vision/references/classification/")
from train import evaluate, train_one_epoch, load_data
import logging
_logger = logging.getLogger('Taylor_Pruner_Debug')
_logger.setLevel(logging.DEBUG)
#define file handler and set formatter
file_handler=logging.FileHandler('speedup_model_architecture_1700.log')
formatter=logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
# add handler to _logger
_logger.addHandler(file_handler)
torch.manual_seed(0)
use_mask = True
use_speedup = True
compare_results = True

data_path = "/home/devdata/datasets/imagenet_raw/"
batch_size = 32

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')
dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False, False)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=4, pin_memory=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=4, pin_memory=True)


def model_inference(config):
    model_trained = './experiment_data/resnet_bn/model_fine_tuned_first.pth'
    rn50=resnet50()
    m_paras=torch.load(model_trained)
    ##delete mask in pth
    m_new=collections.OrderedDict()
    mask=dict()
    for key in m_paras:
        if 'weight_mask_b' in key: continue
        if 'weight_mask' in key:
            if 'module_added' not in key:
                mask[key.replace('.weight_mask','')]=dict()
                mask[key.replace('.weight_mask','')]['weight']=m_paras[key]
                mask[key.replace('.weight_mask','')]['bias']=m_paras[key]
            else:
                mask[key.replace('.relu1.module_added.weight_mask','.bn3')]={}
                mask[key.replace('.relu1.module_added.weight_mask','.bn3')]['weight']=m_paras[key]
                mask[key.replace('.relu1.module_added.weight_mask','.bn3')]['bias']=m_paras[key]
                if '0.relu1' in key:
                    mask[key.replace('relu1.module_added.weight_mask','downsample.1')]={}
                    mask[key.replace('relu1.module_added.weight_mask','downsample.1')]['weight']=m_paras[key]
                    mask[key.replace('relu1.module_added.weight_mask','downsample.1')]['bias']=m_paras[key]
            continue
        if 'module_added' in key:
            continue
        elif 'module' in key:
            m_new[key.replace('module.','')]=m_paras[key]
        else:
            m_new[key]=m_paras[key]
    for key in mask:
        #modify the weight and bias of model with pruning
        m_new[key+'.weight']=m_new[key+'.weight'].data.mul(mask[key]['weight'])
        m_new[key+'.bias']=m_new[key+'.bias'].data.mul(mask[key]['bias'])
    rn50.load_state_dict(m_new)
    rn50.cuda()
    rn50.eval()
    torch.save(mask,'taylor_mask.pth')
    mask_file='./taylor_mask.pth'
    dummy_input = torch.randn(64,3,224,224).cuda()
    use_mask_out = use_speedup_out = None
    rn=rn50
    rn_mask_out = rn(dummy_input)
    model=rn50
    if use_mask:
        torch.onnx.export(model,dummy_input,'resnet_masked_taylor_1700.onnx',export_params=True,opset_version=12,do_constant_folding=True,
                     input_names=['inputs'],output_names=['proba'],
                     dynamic_axes={'inputs':[0],'mask':[0]},keep_initializers_as_inputs=True)

        start = time.time()
        for _ in range(32):
            use_mask_out = model(dummy_input)
        elapsed_t=time.time()-start
        print('elapsed time when use mask: ',elapsed_t)
        _logger.info('for batch size 64 and with 32 runs, the elapsed time is {}'.format(elapsed_t))
    print('before speed up===================')
    flops,paras=count_flops_params(model,(1,3,224,224))
    _logger.info('flops and parameters before speedup is {} FLOPS and {} params'.format(flops,paras))
    if use_speedup:
        dummy_input.cuda()
        m_speedup = ModelSpeedup(model, dummy_input, mask_file,'cuda')
        m_speedup.speedup_model()
        print('=='*20)
        print('Start inference')
        torch.onnx.export(model,dummy_input,'resnet_taylor_1700.onnx',export_params=True,opset_version=12,do_constant_folding=True,
                     input_names=['inputs'],output_names=['proba'],
                     dynamic_axes={'inputs':[0],'mask':[0]},keep_initializers_as_inputs=True)
        start=time.time()
        for _ in range(32):
            use_speedup_out = model(dummy_input)
        elasped_t1=time.time()-start
        print('elapsed time when use speedup: ', elasped_t1)
        _logger.info('elasped time with batch_size 64 and in 32 runs is {}'.format(elasped_t1))
    #print('After speedup model is ',model)
    _logger.info('model structure after speedup is ====')
    _logger.info(model)
    print('=================')
    print('After speedup')
    flops,paras=count_flops_params(model,(1,3,224,224))
    _logger.info('After speedup flops are {} and number of parameters are {}'.format(flops,paras))
    if compare_results:
        print(rn_mask_out)
        print('another is',use_speedup_out)
        if torch.allclose(rn_mask_out, use_speedup_out, atol=1e-6):#-07):
            print('the outputs from use_mask and use_speedup are the same')
        else:
            raise RuntimeError('the outputs from use_mask and use_speedup are different')
    # start the accuracy check
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        start=time.time()
        evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)
        print('elapsed time is ',time.time()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("speedup")
    parser.add_argument("--masks_file", type=str, default='./taylor_mask.pth', help="the path of the masks file")
    args = parser.parse_args()
    config={}
    if args.masks_file is not None:
        config['masks_file'] = args.masks_file
    if not os.path.exists(config['masks_file']):
        msg = '{} does not exist! You should specify masks_file correctly, ' \
                'or use default one which is generated by model_prune_torch.py'
        raise RuntimeError(msg.format(config[args.example_name]['masks_file']))
    model_inference(config)
