import cv2

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
from torchvision import models, transforms, utils
import numpy as np
import copy
import logging
import sys
import time
import hashlib

class FeatureExtractor():
    def __init__(self, config, model, padding=True):
        self.config = config
        self.model = copy.deepcopy(model).to(config.device)
        self.model = self.model.eval()
        self.feature_maps = []

        self.index = []
        self.f = []
        self.stride = []
        for i, module in enumerate(self.model.children()):
            if isinstance(module, nn.Conv2d):
                self.index.append(i)
                self.f.append(module.kernel_size[0])
                self.stride.append(module.stride[0])
            if isinstance(module, nn.MaxPool2d):
                if padding:
                    module.padding = 1
                self.index.append(i)
                self.f.append(module.kernel_size)
                self.stride.append(module.stride)

        self.rf = np.array(self.calc_rf(self.f, self.stride))

        self.cache = {}

    def save_tmp_feature_map(self, module, input, output):
        self.tmp_feature_map = output.detach()

    def calc_rf(self, f, stride):
        rf = []
        for i in range(len(f)):
            if i == 0:
                rf.append(3)
            else:
                rf.append(rf[i-1] + (f[i]-1)*self.product(stride[:i]))
        return rf

    def product(self, lis):
        if len(lis) == 0:
            return 0
        else:
            res = 1
            for x in lis:
                res *= x
            return res

    def calc_l_star(self, template, k):
        l = np.sum(self.rf <= min(list(template.size()[-2:]))) - 1
        l_star = max(l - k, 1)
        return l_star
    
    def calc_NCC(self, F, M):
        c, h_f, w_f = F.shape[-3:]
        norm_f = F.norm()

        self.config.logger.debug("image feature map: {}".format(M.shape))
        self.config.logger.debug("template feature map: {}".format(F.shape))

        stime = time.time()
        h_diff = M.shape[-2] - h_f + 1
        w_diff = M.shape[-1] - w_f + 1
        ncc = torch.empty((h_diff, w_diff)).to(self.config.device)

        for i in range(h_diff):
            for j in range(w_diff):
                M_tilde = M[:, :, i:i+h_f, j:j+w_f]
                ncc[i, j] = torch.sum(F * M_tilde / (M_tilde.norm() * norm_f))

        self.config.logger.debug('time of calc NCC: {}'.format(time.time() - stime))
        return ncc

    def remove_cache(self, image):
        hash = self.__hash(image)
        if hash in self.cache:
            del self.cache[hash]

    def __get_feature_map(self, img, l_star):
        hash = self.__hash(img)
        if hash not in self.cache:
            self.cache[hash] = {}

        feature_map = self.cache[hash].get(l_star, None)
        if feature_map is None:
            handle = self.model[self.index[l_star]].register_forward_hook(self.save_tmp_feature_map)
            self.model(img)
            handle.remove()
            self.cache[hash][l_star] = self.tmp_feature_map

        return self.cache[hash][l_star] 

    def __call__(self, template, image):
        template = template.to(self.config.device)
        image = image.to(self.config.device)

        self.l_star = self.calc_l_star(template, self.config.klayer)

        self.config.logger.debug("save features...")

        self.template_feature_map = self.__get_feature_map(template, self.l_star)
        self.image_feature_map = self.__get_feature_map(image, self.l_star)

        template_feature_maps = self.__create_scaled_template_feature_maps(self.template_feature_map, self.image_feature_map)

        self.config.logger.debug("calc NCC...")

        boxes = []
        scores = []
        for template_map in template_feature_maps:
            ncc = self.calc_NCC(template_map, self.image_feature_map).cpu().numpy()
            result = self.__calc_scores(ncc, image, template)
            boxes += result[0]
            scores += result[1]

        return np.array(boxes), np.array(scores)

    def __hash(self, img):
        return hashlib.md5(img.numpy()).hexdigest()

    def __create_scaled_template_feature_maps(self, template_feature_map, image_feature_map):
        h_t, w_t = template_feature_map.shape[-2:]
        h_i, w_i = image_feature_map.shape[-2:]

        scaled_template_feature_maps = []
        # scales = [1/3, 2/3, 1.0]
        # for scale in scales:
        #     scaled_size = [int(h_t * scale), int(w_t * scale)]
        #     if h_i <= scaled_size[0] or w_i <= scaled_size[1]:
        #         continue
        #
        #     scaled_template_feature_maps.append(func.interpolate(template_feature_map, size=scaled_size, mode='bilinear', align_corners=True))

        if len(scaled_template_feature_maps) == 0:
            size = [h_t, w_t]
            if h_i <= h_t:
                size[0] = h_i - 1
            if w_i <= w_t:
                size[1] = w_i - 1
            scaled_template_feature_maps.append(func.interpolate(template_feature_map, size=size, mode='bilinear', align_corners=True))

        return scaled_template_feature_maps

    def __calc_scores(self, ncc, image, template):
        # 最もスコアの高いものを一つだけ返す
        # 一つのsearch画像内に同じtemplate画像が複数出てくることは今回の用途ではないため
        max_indices = np.array([np.unravel_index(np.argmax(ncc), ncc.shape)])
        self.config.logger.debug("NCC shape: {}, max indices: {}".format(ncc.shape, max_indices))
        self.config.logger.debug("detected boxes: {}".format(len(max_indices)))

        size_template_feature = self.template_feature_map.size()

        boxes = []
        centers = []
        scores = []
        x_ranges = [ [], [0], [0, 1], [-1, 0, 1], [-2, -1, 0, 1], ]
        y_ranges = [ [], [0], [0, 1], [-1, 0, 1], ]
        for max_index in max_indices:
            i_star, j_star = max_index
            i_min = max(0, i_star-1)
            j_min = max(0, j_star-2)
            NCC_part = ncc[i_min:i_star+2, j_min:j_star+2]

            x_center = (j_star + size_template_feature
                        [-1]/2) * image.size()[-1] // self.image_feature_map.size()[-1]
            y_center = (i_star + size_template_feature
                        [-2]/2) * image.size()[-2] // self.image_feature_map.size()[-2]

            x1_0 = x_center - template.size()[-1]/2
            x2_0 = x_center + template.size()[-1]/2
            y1_0 = y_center - template.size()[-2]/2
            y2_0 = y_center + template.size()[-2]/2

            stride_product = self.product(self.stride[:self.l_star])

            shape = NCC_part.shape
            x1 = np.sum(
                NCC_part * (x1_0 + np.array(x_ranges[shape[1]]) * stride_product)[None, :]) / np.sum(NCC_part)
            x2 = np.sum(
                NCC_part * (x2_0 + np.array(x_ranges[shape[1]]) * stride_product)[None, :]) / np.sum(NCC_part)
            y1 = np.sum(
                NCC_part * (y1_0 + np.array(y_ranges[shape[0]]) * stride_product)[:, None]) / np.sum(NCC_part)
            y2 = np.sum(
                NCC_part * (y2_0 + np.array(y_ranges[shape[0]]) * stride_product)[:, None]) / np.sum(NCC_part)

            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))
            x_center = int(round(x_center))
            y_center = int(round(y_center))

            boxes.append([(x1, y1), (x2, y2)])
            scores.append(np.average(NCC_part))

        return boxes, scores

