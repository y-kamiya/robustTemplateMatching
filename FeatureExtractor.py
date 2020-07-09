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

    def save_template_feature_map(self, module, input, output):
        self.template_feature_map = output.detach()

    def save_image_feature_map(self, module, input, output):
        self.image_feature_map = output.detach()

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

    def calc_l_star(self, template, k=3):
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

    def remove_cache(self, image_path):
        del self.cache[image_path]

    def __call__(self, template_path, template, image_path, image, threshold=None, use_cython=True):
        template = template.to(self.config.device)
        image = image.to(self.config.device)

        self.l_star = self.calc_l_star(template)

        if image_path not in self.cache:
            self.cache[image_path] = {}

        if template_path not in self.cache:
            self.cache[template_path] = {}

        self.config.logger.debug("save features...")

        if self.l_star not in self.cache[template_path]:
            # save template feature map (named F in paper)
            template_handle = self.model[self.index[self.l_star]].register_forward_hook(
                self.save_template_feature_map)
            self.model(template)
            template_handle.remove()
            self.cache[template_path][self.l_star] = self.template_feature_map
        else:
            self.template_feature_map = self.cache[template_path][self.l_star]

        if self.l_star not in self.cache[image_path]:
            # save image feature map (named M in papar)
            image_handle = self.model[self.index[self.l_star]].register_forward_hook(
                self.save_image_feature_map)
            self.model(image)
            image_handle.remove()
            self.cache[image_path][self.l_star] = self.image_feature_map
        else:
            self.image_feature_map = self.cache[image_path][self.l_star]

        h_t, w_t = self.template_feature_map.shape[-2:]
        h_i, w_i = self.image_feature_map.shape[-2:]
        size = [h_t, w_t]
        if h_i < h_t:
            size[0] = h_i
        if w_i < w_t:
            size[1] = w_i
        self.template_feature_map = func.interpolate(self.template_feature_map, size=size, mode='bilinear', align_corners=True)

        self.config.logger.debug("calc NCC...")

        if use_cython:
            F = self.template_feature_map.cpu().numpy()[0].astype(np.float32)
            M = self.image_feature_map.cpu().numpy()[0].astype(np.float32)
            import cython_files.cython_calc_NCC as cython_calc_NCC
            self.NCC = np.zeros(
                (M.shape[1] - F.shape[1]) * (M.shape[2] - F.shape[2])).astype(np.float32)
            cython_calc_NCC.c_calc_NCC(M.flatten().astype(np.float32), np.array(M.shape).astype(
                np.int32), F.flatten().astype(np.float32), np.array(F.shape).astype(np.int32), self.NCC)
            self.NCC = self.NCC.reshape(
                [M.shape[1] - F.shape[1], M.shape[2] - F.shape[2]])
        else:
            self.NCC = self.calc_NCC(self.template_feature_map, self.image_feature_map).cpu().numpy()

        # if threshold is None:
        #     threshold = 0.95 * np.max(self.NCC)
        # max_indices = np.array(np.where(self.NCC > threshold)).T

        # 最もスコアの高いものを一つだけ返す
        # 一つのsearch画像内に同じtemplate画像が複数出てくることは今回の用途ではないため
        max_indices = np.array([np.unravel_index(np.argmax(self.NCC), self.NCC.shape)])
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
            NCC_part = self.NCC[i_min:i_star+2, j_min:j_star+2]

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
            # centers.append((x_center, y_center))
            scores.append(np.average(NCC_part))

        return np.array(boxes), np.array(scores)
        # return boxes, centers, scores
