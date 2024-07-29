import torch
import math
from torchvision import models
from TransferLearningInterpretation import config as cfg
import os
import numpy as np


class Receptive_Field():
    def __init__(self, model=None, dataset_rf=None):
        if model != None:
            self.model = model
            self.kernels = []
            self.strides = []
            self.paddings = []
            self.rfs_info = []
            conv_layer_counter = 0
            layer_counter = 0

            self.layer_map = dict()

            # Go over layers, get changes to output size
            if model.name == "vgg19" or model.name == "alexnet":
                for i, (name, module) in enumerate(self.model.features._modules.items()):
                    if type(module) == torch.nn.Conv2d:
                        # print("conv2d", name)
                        self.kernels.append(module.kernel_size[0])
                        self.strides.append(module.stride[0])
                        self.paddings.append(module.padding[0])
                        conv_layer_counter += 1
                        layer_counter += 1
                    if type(module) == torch.nn.modules.pooling.MaxPool2d:
                        # print("maxpool", name)
                        self.kernels.append(module.kernel_size)
                        self.strides.append(module.stride)
                        self.paddings.append(module.padding)
                        layer_counter += 1
                    
                    self.layer_map[i] = layer_counter - 1
            elif model.name == "densenet":
                i = 0
                for module in self.model.features.children():
                    if type(module) == torch.nn.Conv2d:
                        # print("conv2d", name)
                        self.kernels.append(module.kernel_size[0])
                        self.strides.append(module.stride[0])
                        self.paddings.append(module.padding[0])
                        conv_layer_counter += 1
                        layer_counter += 1
                        self.layer_map[i] = layer_counter - 1
                        i += 1
                    elif type(module) == torch.nn.modules.pooling.MaxPool2d:
                        # print("maxpool", name)
                        self.kernels.append(module.kernel_size)
                        self.strides.append(module.stride)
                        self.paddings.append(module.padding)
                        layer_counter += 1
                        self.layer_map[i] = layer_counter - 1
                        i += 1
                    elif type(module) == models.densenet._DenseBlock:
                        for j in module.children():
                            for child in j.children():
                                if type(child) == torch.nn.Conv2d:
                                    # print("conv2d", name)
                                    self.kernels.append(child.kernel_size[0])
                                    self.strides.append(child.stride[0])
                                    self.paddings.append(child.padding[0])
                                    conv_layer_counter += 1
                                    layer_counter += 1
                                elif type(child) == torch.nn.modules.pooling.MaxPool2d:
                                    # print("maxpool", name)
                                    self.kernels.append(child.kernel_size)
                                    self.strides.append(child.stride)
                                    self.paddings.append(child.padding)
                                    layer_counter += 1
                                self.layer_map[i] = layer_counter - 1
                                i += 1
                    elif type(module) == models.densenet._Transition:
                        for child in module.children():
                            if type(child) == torch.nn.Conv2d:
                                # print("conv2d", name)
                                self.kernels.append(child.kernel_size[0])
                                self.strides.append(child.stride[0])
                                self.paddings.append(child.padding[0])
                                conv_layer_counter += 1
                                layer_counter += 1
                            elif type(child) == torch.nn.modules.pooling.MaxPool2d:
                                # print("maxpool", name)
                                self.kernels.append(child.kernel_size)
                                self.strides.append(child.stride)
                                self.paddings.append(child.padding)
                                layer_counter += 1
                            self.layer_map[i] = layer_counter - 1
                            i += 1
                    else:
                        self.layer_map[i] = layer_counter - 1
                        i += 1
            elif model.name == "resnet50":
                i = 0
                for module in self.model.children():
                    if type(module) == torch.nn.Conv2d:
                        # print("conv2d", name)
                        self.kernels.append(module.kernel_size[0])
                        self.strides.append(module.stride[0])
                        self.paddings.append(module.padding[0])
                        conv_layer_counter += 1
                        layer_counter += 1
                        self.layer_map[i] = layer_counter - 1
                        i += 1
                    elif type(module) == torch.nn.modules.pooling.MaxPool2d:
                        # print("maxpool", name)
                        self.kernels.append(module.kernel_size)
                        self.strides.append(module.stride)
                        self.paddings.append(module.padding)
                        layer_counter += 1
                        self.layer_map[i] = layer_counter - 1
                        i += 1
                    elif type(module) == torch.nn.Sequential:
                        for j in module.children():
                            for child in j.children():
                                if type(child) == torch.nn.Conv2d:
                                    # print("conv2d", name)
                                    self.kernels.append(child.kernel_size[0])
                                    self.strides.append(child.stride[0])
                                    self.paddings.append(child.padding[0])
                                    conv_layer_counter += 1
                                    layer_counter += 1
                                elif type(child) == torch.nn.modules.pooling.MaxPool2d:
                                    # print("maxpool", name)
                                    self.kernels.append(child.kernel_size)
                                    self.strides.append(child.stride)
                                    self.paddings.append(child.padding)
                                    layer_counter += 1
                                self.layer_map[i] = layer_counter - 1
                                i += 1
                    else:
                        self.layer_map[i] = layer_counter - 1
                        i += 1

            # print("kernels ", self.kernels)
            # print("strides ", self.strides)
            # print("paddings ", self.paddings)
            # print("mappings ", self.layer_map)

            # Compute all the layer sizes for rf
            self.computes_all_layers_rf_info(cfg.INPUT_SIZE[0])
        else:
            self.dataset_rf = dataset_rf

    def computes_all_layers_rf_info(self, img_size):
        # First layer
        rf_info = [img_size, 1, 1, 0.5]
        self.rfs_info.append(rf_info)

        # Compute for other layers
        for i in range(len(self.kernels)):
            filter_size = self.kernels[i]
            stride_size = self.strides[i]
            padding_size = self.paddings[i]
            layer_rf_info = self.compute_layer_rf_info(filter_size, stride_size, padding_size, self.rfs_info[-1])
            self.rfs_info.append(layer_rf_info)

        # for rf in self.rfs_info:
        #     print(rf)

    def compute_layer_rf_info(self, layer_filter_size, layer_stride, layer_padding,
                              previous_layer_rf_info):
        n_in = previous_layer_rf_info[0]  # input size
        j_in = previous_layer_rf_info[1]  # receptive field jump of input layer
        r_in = previous_layer_rf_info[2]  # receptive field size of input layer
        start_in = previous_layer_rf_info[3]  # center of receptive field of input layer

        if layer_padding == 'SAME':
            n_out = math.ceil(float(n_in) / float(layer_stride))
            if (n_in % layer_stride == 0):
                pad = max(layer_filter_size - layer_stride, 0)
            else:
                pad = max(layer_filter_size - (n_in % layer_stride), 0)
            assert (n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1)  # sanity check
            assert (pad == (n_out - 1) * layer_stride - n_in + layer_filter_size)  # sanity check
        elif layer_padding == 'VALID':
            n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
            pad = 0
            assert (n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1)  # sanity check
            assert (pad == (n_out - 1) * layer_stride - n_in + layer_filter_size)  # sanity check
        else:
            # layer_padding is an int that is the amount of padding on one side
            pad = layer_padding * 2
            n_out = math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1

        pL = math.floor(pad / 2)

        j_out = j_in * layer_stride
        r_out = r_in + (layer_filter_size - 1) * j_in
        start_out = start_in + ((layer_filter_size - 1) / 2 - pL) * j_in
        return [n_out, j_out, r_out, start_out]

    def compute_rf_at_spatial_location(self, img_size, height_index, width_index, layer_index):
        target_layer_index = self.layer_map[layer_index]
        layer_rf_info = self.rfs_info[target_layer_index]
        # print('layer index ', layer_index, 'height_index ',height_index, ' width_index ',width_index)
        # print('layer_rf_info ',layer_rf_info)
        n = layer_rf_info[0]
        j = layer_rf_info[1]
        r = layer_rf_info[2]
        start = layer_rf_info[3]
        assert (height_index < n)
        assert (width_index < n)

        center_h = start + (height_index * j)
        center_w = start + (width_index * j)

        rf_start_height_index = max(int(center_h - (r / 2)), 0)
        rf_end_height_index = min(int(center_h + (r / 2)), img_size)

        rf_start_width_index = max(int(center_w - (r / 2)), 0)
        rf_end_width_index = min(int(center_w + (r / 2)), img_size)

        return [rf_start_height_index, rf_end_height_index,
                rf_start_width_index, rf_end_width_index]

    def rf_processing(self, data_dir, rf_dir):
        agg_rf = {}
        for class_index in self.dataset_rf.keys():
            for layer_index in self.dataset_rf[class_index].keys():
                for channel_index in self.dataset_rf[class_index][layer_index].keys():
                    for image_name in self.dataset_rf[class_index][layer_index][channel_index].keys():
                        if 'checkpoint' not in image_name:
                            img_info = self.dataset_rf[class_index][layer_index][channel_index][image_name]
                            img_rf = img_info[0]
                            class_name = img_info[1]
                            if 'checkpoint' not in class_name:
                                max_value = img_info[2]
                                print(image_name)
                                agg_rf = self.make_spot(agg_rf, class_name, layer_index, channel_index)

                                if 'dog' in image_name or 'cat' in image_name:
                                    image_name = image_name[0:3] + '.' + image_name[3:]

                                if not os.path.exists(rf_dir + class_name):
                                    os.mkdir(rf_dir + class_name + '/')
                                if not os.path.exists(rf_dir + class_name + '/' + str(layer_index) + '/'):
                                    os.mkdir(rf_dir + class_name + '/' + str(layer_index) + '/')
                                if not os.path.exists(
                                        rf_dir + class_name + '/' + str(layer_index) + '/' + str(channel_index)):
                                    os.mkdir(rf_dir + class_name + '/' + str(layer_index) + '/' + str(channel_index) + '/')
                                if not os.path.exists(rf_dir + class_name + '/' + str(layer_index) + '/'
                                                      + str(channel_index) + '/' + 'original_img'):
                                    os.mkdir(rf_dir + class_name + '/' + str(layer_index) + '/'
                                             + str(channel_index) + '/' + 'original_img')
                                if not os.path.exists(rf_dir + class_name + '/' + str(layer_index) + '/'
                                                      + str(channel_index) + '/' + 'patch'):
                                    os.mkdir(rf_dir + class_name + '/' + str(layer_index) + '/'
                                             + str(channel_index) + '/' + 'patch')

                                print(data_dir + 'test/' + class_name + '/' + image_name + '.jpg.jpg')

                                input_image, c_n, i_n = HF.read_images(
                                    data_dir + 'test/' + class_name + '/' + image_name + '.jpg.jpg')

                                cropped_img = input_image[img_rf[0]:img_rf[1], img_rf[2]:img_rf[3], :]
                                cropped_img_2save = np.copy(np.uint8(cropped_img * 255))
                                cropped_img_2save = cropped_img_2save[:, :, ::-1]
                                cv2.imwrite(rf_dir + class_name + '/' + str(layer_index) + '/' + str(channel_index)
                                            + '/' + 'patch/' + image_name + '.jpg',
                                            cropped_img_2save)

                                input_image_rectangle = self.draw_rectangle(np.copy(input_image), img_rf, (255, 0, 0), 2)
                                input_image_rectangle = np.uint8(input_image_rectangle * 255)
                                input_image_rectangle = input_image_rectangle[:, :, ::-1]
                                cv2.imwrite(rf_dir + class_name + '/' + str(layer_index) + '/' + str(channel_index)
                                            + '/' + 'original_img/' + image_name + '_rectangle' + '.jpg',
                                            input_image_rectangle)
                                '''****
                                if agg_rf[class_name][layer_index][channel_index] is None:
                                    cropped_img = cv2.resize(cropped_img, (128, 128))
                                    cropped_img = HF.normalize_numpy(cropped_img)
                                    agg_rf[class_name][layer_index][channel_index] = cropped_img
                                else:
                                    agg_img = agg_rf[class_name][layer_index][channel_index]
                                    cropped_img = cv2.resize(cropped_img, (128, 128))
                                    cropped_img = HF.normalize_numpy(cropped_img)
                                    agg_img = agg_img + cropped_img
                                    agg_rf[class_name][layer_index][channel_index] = agg_img
                                '''

                                if agg_rf[class_name][layer_index][channel_index] is None:
                                    agg_rf[class_name][layer_index][channel_index] = []
                                    cropped_img = cv2.resize(cropped_img, (128, 128))
                                    cropped_img = HF.normalize_numpy(cropped_img)
                                    agg_rf[class_name][layer_index][channel_index].append((cropped_img,max_value))
                                else:
                                    cropped_img = cv2.resize(cropped_img, (128, 128))
                                    cropped_img = HF.normalize_numpy(cropped_img)
                                    agg_rf[class_name][layer_index][channel_index].append((cropped_img, max_value))

        '''
        for class_name in agg_rf.keys():
            for layer_index in agg_rf[class_name].keys():
                for channel_index in agg_rf[class_name][layer_index].keys():
                    agg_img = agg_rf[class_name][layer_index][channel_index]
                    agg_img = (agg_img - np.min(agg_img)) / (np.max(agg_img) - np.min(agg_img))
                    agg_img = np.uint8(agg_img * 255)
                    agg_img = agg_img[:, :, ::-1]
                    cv2.imwrite(rf_dir + class_name + '/' + str(layer_index) + '/' + str(channel_index) + '.jpg',
                                agg_img)
        '''
        for class_name in agg_rf.keys():
            for layer_index in agg_rf[class_name].keys():
                for channel_index in agg_rf[class_name][layer_index].keys():
                    cropped_img_list = []
                    acts_list = []
                    for crpimg_actv in agg_rf[class_name][layer_index][channel_index]:
                        cropped_img_list.append(crpimg_actv[0])
                        acts_list.append(crpimg_actv[1])

                    if len(acts_list) > 60:
                        acts_arr = np.asarray(acts_list)
                        acts_index = np.argsort(acts_arr)[-60:]
                        acts_arr = acts_arr[acts_index]
                        acts_arr = (acts_arr - np.min(acts_arr)) / (np.max(acts_arr) - np.min(acts_arr))
                        cropped_img_list_selected = []
                        for i in acts_index:
                            for j in range(len(cropped_img_list)):
                                if j == i:
                                    cropped_img_list_selected.append(cropped_img_list[i])
                                    break
                        cropped_img_list = cropped_img_list_selected
                    else: 

                        acts_arr = np.asarray(acts_list)
                        acts_arr = (acts_arr - np.min(acts_arr)) / (np.max(acts_arr) - np.min(acts_arr))

                    agg_img = None
                    for i in range(len(cropped_img_list)):
                        if agg_img is None:
                            agg_img = acts_arr[i] * cropped_img_list[i]
                        else:
                            agg_img += (acts_arr[i] * cropped_img_list[i])

                    agg_img = (agg_img - np.min(agg_img)) / (np.max(agg_img) - np.min(agg_img))
                    agg_img = np.uint8(agg_img * 255)
                    agg_img = agg_img[:, :, ::-1]
                    cv2.imwrite(rf_dir + class_name + '/' + str(layer_index) + '/' + str(channel_index) + '.jpg',
                                agg_img)


    def draw_rectangle(self, input_image, img_rf, color, thickness):
        input_image[img_rf[0]:img_rf[1] - 1, img_rf[2], :] = [1, 0, 0]  # left-side
        input_image[img_rf[0]:img_rf[1] - 1, img_rf[3] - 1, :] = [1, 0, 0]  # right-side
        input_image[img_rf[0], img_rf[2]:img_rf[3] - 1, :] = [1, 0, 0]  # upper-side
        input_image[img_rf[1] - 1, img_rf[2]:img_rf[3] - 1, :] = [1, 0, 0]  # down-side

        return input_image

    def make_spot(self, agg_rf, class_name, layer_index, channel_index):
        if class_name not in agg_rf.keys():
            agg_rf[class_name] = {}
        if layer_index not in agg_rf[class_name].keys():
            agg_rf[class_name][layer_index] = {}
        if channel_index not in agg_rf[class_name][layer_index].keys():
            agg_rf[class_name][layer_index][channel_index] = None
        return agg_rf

