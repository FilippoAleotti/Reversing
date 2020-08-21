import os
import numpy as np
import torch

def to_regular_format(tensor):
    '''
    Pytorch uses the format:
        [Batch] x Channels x Height x Width 
    Outside Pytorch the format is instead:
        [Batch] x Height x Width x Channels
    This function rearrange the tensor in the latter way
    Note that Batch axis is optional
    Params:
        tensor: torch.Tensor, numpy ndarray or list with shape Batch x Channels x Height x Width
    Returns:
        tensor with shape Batch x Height x Width x Channels of the same type of the input
    '''
    dim = len(tensor.shape)
    if dim == 4:
        new_axis = (0,2,3,1)
    else:
        new_axis = (1,2,0)
    if isinstance(tensor, torch.Tensor):
        return tensor.permute(new_axis)
    
    is_list= False
    if isinstance(tensor,(list,)):
        tensor = np.asarray(tensor)
        is_list = True
    rearranged_tensor = np.transpose(tensor, new_axis)
    if is_list:
        return rearranged_tensor.tolist()
    return rearranged_tensor

def to_tensor(tensor): 
    '''
    This function rearrange a tensor or a numpy ndarray in order to have
        [Batch] x Channels x Height x Width
    Note that Batch dimension is optional
    Params:
        tensor: torch.Tensor, numpy ndarray or list with shape [Batch] x Height x Width x Channels
    Returns:
        torch tensor with shape [Batch] x Height x Width x Channels 
    '''    
    cast = False

    if isinstance(tensor, np.ndarray):
        cast = True

    if isinstance(tensor, (list,)):
        inp = np.asarray(inp)
        cast = True

    if cast:
        tensor = torch.from_numpy(tensor)

    dim = len(tensor.shape)
    if dim == 4:
        tensor = tensor.permute((0,3,1,2))
    else:
        tensor = tensor.permute((2,0,1))
    return tensor

def get_size(tensor):
    '''
        Extract [Height,Width] of a tensor
    '''
    _,_,h,w = list(tensor.size())
    return [h,w]

def to_cpu(tensor):
    '''
        Move tensor to cpu'
    '''
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor
