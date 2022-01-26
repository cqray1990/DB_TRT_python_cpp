#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt
import cv2

import pycuda.driver as cuda
import pycuda.autoinit

# from image_batcher import ImageBatcher
# from visualize import visualize_detections
import math
from typing import Tuple, List


def resize_image(img, algorithm, side_len=1536, add_padding=True):

    stride = 32
    height, width, _ = img.shape
    flag = None
    if height > width:
        flag = True
        new_height = side_len
        new_width = int(math.ceil(new_height / height * width / stride) * stride)
    else:
        flag = False
        new_width = side_len
        new_height = int(math.ceil(new_width / width * height / stride) * stride)
    resized_img = cv2.resize(img, (new_width, new_height))
    if add_padding is True:
        if flag:
            padded_image = cv2.copyMakeBorder(resized_img, 0, 0,
                                              0, side_len - new_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            padded_image = cv2.copyMakeBorder(resized_img, 0, side_len - new_height,
                                              0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        return resized_img
    return padded_image

class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        self.context.active_optimization_profile = 0
        self.input_binding_idxs, self.output_binding_idxs = self.get_binding_idxs(self.engine,self.context.active_optimization_profile)

        assert len(self.input_binding_idxs) >=1
        assert len(self.output_binding_idxs)>=1


    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def inferengine(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        ## Run inference.
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])
        return outputs

    def predict(self,args):
        img1 = self.get_img(args)
        h, w = img1.shape[2:4]
        #img_numpy = np.array([img1,img1])
        img_numpy = np.array([img1])
        s = time.time()
        outputs = self.inferengine(img_numpy)
        out = outputs[0].reshape(int(args.batch_size), h, w)
        cv2.imwrite(args.output+os.sep+ args.algorithm + '_trt_img1.jpg', out[0] * 255)

    def get_img(self,args):
        img = cv2.imread(args.img_path)
        img = resize_image(img, args.algorithm, side_len=args.max_size, add_padding=args.add_padding)

        cv2.imwrite('./onnx/' + args.algorithm + '_ori_img.jpg', img)
        image = img / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image

    def load_engine(self,filename: str):
        # Load serialized engine file into memory
        with open(filename, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_binding_idxs(self,engine: trt.ICudaEngine, profile_index: int):
        # Calculate start/end binding indices for current context's profile
        num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
        start_binding = profile_index * num_bindings_per_profile
        end_binding = start_binding + num_bindings_per_profile
        print("Engine/Binding Metadata")
        print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
        print("\tNumber of bindings per profile: {}".format(num_bindings_per_profile))
        print("\tFirst binding for profile {}: {}".format(profile_index, start_binding))
        print("\tLast binding for profile {}: {}".format(profile_index, end_binding - 1))

        # Separate input and output binding indices for convenience
        input_binding_idxs = []
        output_binding_idxs = []
        for binding_index in range(start_binding, end_binding):
            if engine.binding_is_input(binding_index):
                input_binding_idxs.append(binding_index)
            else:
                output_binding_idxs.append(binding_index)

        return input_binding_idxs, output_binding_idxs



    def get_input_host(self,args):
        host_inputs = []
        for bind_indx in self.input_binding_idxs:
            input_shape = self.context.get_binding_shape(bind_indx)
            input_name = self.engine.get_binding_name(bind_indx)
            input_dtype = self.engine.get_binding_dtype(bind_indx)

            # if self.is_dynamic(input_shape):
            #     # profile_index = self.context.active_optimization_profile
            #     # profile_shapes = self.engine.get_profile_shape(profile_index, bind_indx)
            #     # print("\tProfile Shapes for [{}]: [kMIN {} | kOPT {} | kMAX {}]".format(input_name, *profile_shapes))
            #     # # 0=min, 1=opt, 2=max, or choose any shape, (min <= shape <= max)
            #     # dims = profile_shapes[1]
            #     # print("\tInput [{}] shape was dynamic, setting inference shape to {}".format(input_name, dims))
            img1 = self.get_img(args)
            h, w = img1.shape[2:4]
            # img_numpy = np.array([img1,img1])
            #img_numpy = np.array([img1])
            host_inputs.append(np.ascontiguousarray(img1))

            return host_inputs


    def is_dynamic(self,shape):
        return any(dim is None or dim < 0 for dim in shape)

    def is_fixed(self,shape):
        return not self.is_dynamic(shape)

    def setup_binding_shapes(self,
            engine: trt.ICudaEngine,
            context: trt.IExecutionContext,
            host_inputs: List[np.ndarray],
            input_binding_idxs: List[int],
            output_binding_idxs: List[int],
    ):
        # Explicitly set the dynamic input shapes, so the dynamic output
        # shapes can be computed internally
        for host_input, binding_index in zip(host_inputs, input_binding_idxs):
            context.set_binding_shape(binding_index, host_input.shape)

        assert context.all_binding_shapes_specified

        host_outputs = []
        device_outputs = []
        for binding_index in output_binding_idxs:
            output_shape = context.get_binding_shape(binding_index)
            # Allocate buffers to hold output results after copying back to host
            buffer = np.empty(output_shape, dtype=np.float32)
            host_outputs.append(buffer)
            # Allocate output buffers on device
            device_outputs.append(cuda.mem_alloc(buffer.nbytes))
        return host_outputs, device_outputs

    def infer(self,args):
        img1 = self.get_img(args)
        h, w = img1.shape[2:4]
        # img_numpy = np.array([img1,img1])
        img_numpy = np.array([img1])
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
           # self.context.active_optimization_profile = 0
            shape = self.engine.get_binding_shape(i)

            # 增加部分
            if is_input and (shape[-1] == -1):
                shape[0],shape[-2], shape[-1] = (img_numpy.shape[0],h,w)
                self.context.set_binding_shape(i, (shape))
            else:
                shape[0], shape[-2], shape[-1] = (img_numpy.shape[0], h, w)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        outputs = self.inferengine(img_numpy)
        out = outputs[0].reshape(int(args.batch_size), h, w)

        cv2.imwrite(args.output + os.sep + args.algorithm + '_trt_img1.jpg', out[0] * 255)



    def infer_v2(self,args):
        host_inputs = self.get_input_host(args)
        # allocate devidce mem for inputs
        device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]
        #copy host  inputs to device
        for h_input, d_input in zip(host_inputs, device_inputs):
            cuda.memcpy_htod(d_input, h_input)
        #if inputs shape change this needs to bu called,if inputs shape always thse same call this once ,reuse this allcocation
        host_outputs, device_outputs = self.setup_binding_shapes(
            self.engine, self.context, host_inputs, self.input_binding_idxs, self.output_binding_idxs,
        )
        output_names = [self.engine.get_binding_name(binding_idx) for binding_idx in self.output_binding_idxs]
        # Bindings are a list of device pointers for inputs and outputs
        bindingsallocation = device_inputs + device_outputs

        self.context.execute_v2(bindingsallocation)

        # Copy outputs back to host to view results
        for h_output, d_output in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh(h_output, d_output)
        out = host_outputs[0].reshape(int(args.batch_size), host_inputs[0].shape[2], host_inputs[0].shape[3])
        cv2.imwrite(args.output + os.sep + args.algorithm + '_trt_img1.jpg', out[0] * 255)

    #reuse alloc mechnim
    def alloc_buffers(self,is_explicit_batch=False, input_shape=None):
        inputs = []
        outputs = []
        bindings = []

        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()

        for binding in self.engine:
            dims = self.engine.get_binding_shape(binding)
            print(dims)
            if dims[-1] == -1:
                assert (input_shape is not None)
                dims[-2], dims[-1] = input_shape
            size = trt.volume(dims) * self.engine.max_batch_size  # The maximum batch size which can be used for inference.
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):  # Determine whether a binding is an input binding.
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

def main(args):
    output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(args.trt_engine_path)
    trt_infer = TensorRTInfer(args.trt_engine_path)
    #trt_infer.predict(args)
    for i in range(100):
        s = time.time()
        trt_infer.infer_v2(args)
        print(time.time() - s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--trt_engine_path', nargs='?', type=str, default="./db_sim_author_me.engine")
    parser.add_argument('--img_path', nargs='?', type=str,
                        default="/home/lgx/.sensorsto/mmnlpNet/script/onnx/DB_ori_img.jpg")
    parser.add_argument('--output',nargs='?',type=str,default="./results")
    parser.add_argument('--batch_size', nargs='?', type=str, default=1)
    parser.add_argument('--max_size', nargs='?', type=int, default=960)
    parser.add_argument('--algorithm', nargs='?', type=str, default="DB")
    parser.add_argument('--add_padding', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
