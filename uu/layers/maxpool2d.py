import torch
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
import numpy as np
from torch.autograd.variable import Variable
import math
import pdb
import maxpool_2d_bkw_cpp, maxpool_2d_bkw_cuda
from uu.utils import correctness_check 

class cMaxPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        #print("\n^^^^^cMaxPool2dFunction fwd")
        input = inputs[0]

        kernel_size = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        ctx.info = inputs[4]
        uniq_id = inputs[5]
        is_ccheckpoint = inputs[6]

        if not is_ccheckpoint:
            out = F.max_pool2d(input, kernel_size, stride, padding, return_indices=True)
            out_value = out[0]
            out_index = out[1]

            # save status for bkward
            ctx.stride = stride
            ctx.kernel_size = kernel_size
            ctx.padding = padding

            ctx.input = input
            ctx.output = out_value
            ctx.arg_max = out_index

            ctx.uniq_id = uniq_id
        else:
            out = F.max_pool2d(input, kernel_size, stride, padding, return_indices=True)

            out_value = out[0]
            out_index = out[1]

        return out_value
    
    @staticmethod
    def backward(ctx, grad_output):
        print("\n^^^^^cMaxPool2dFunction bwd")
        # print(ctx.input.size())
        # print(grad_output.size())
        # print(ctx.arg_max.size())

        # #case1
        # if ctx.input.is_cuda:
        #     grad_in = maxpool_2d_bkw_cuda.backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        # else:
        #     grad_in = maxpool_2d_bkw_cpp.backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        

        f_info = ctx.info[0][ctx.uniq_id]
        b_info = ctx.info[1][ctx.uniq_id]

        # not sure if these logic is correct
        crop = []
        for i in range(len(f_info)):
            crop.append(b_info[i] - f_info[i])
        #print("crop", crop)

        #NCHW
        left = crop[0]
        right = ctx.arg_max.shape[3]-crop[1]
        top = crop[2]
        bottom = ctx.arg_max.shape[2]-crop[3]
        new_arg_max = ctx.arg_max[:, :, top:bottom, left:right]


        #case2
        grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        
        
        # print("##############grad_in in maxp", grad_in.size()) 
        # print("grad in", grad_in)
        return grad_in, None, None, None, None

class cMaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t = None,
                 padding: _size_2_t = (0,0), dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False,
                 is_ccheckpoint = False, #mdepth = 1, num_maxp = 1
                 ):
        super(cMaxPool2d, self).__init__(kernel_size, stride,
                 padding, dilation, return_indices, ceil_mode)
        self.is_ccheckpoint = is_ccheckpoint

        # self.mdepth = mdepth # depth of a maxpool in the checkpoint segment
        # self.num_maxp = num_maxp

        

# do I need to create auto-fucntion for MaxPool??
    def forward(self, *inputs):
        if type (inputs[0]) == tuple:
            # to remove additional packing in tuple
            inputs = list(inputs[0])

        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        elif len(inputs) == 3:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        else:
            print("missing info in cMaxPool2d")
            assert False
        cmaxplool = cMaxPool2dFunction.apply
       
        uniq_id = id(self)
        next_input = cmaxplool(input, self.kernel_size, self.stride,
                            self.padding, info, uniq_id, is_ccheckpoint)
        # need to handle padded for next if needed. 
        return next_input, info, self.is_ccheckpoint
    
        
 
