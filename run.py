import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import models, transforms, utils
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from FeatureExtractor import FeatureExtractor
import sys

class Evaluator:
    def __init__(self, config):
        self.config = config

    def nms(self, dets, scores, thresh):
        x1 = dets[:, 0, 0]
        y1 = dets[:, 0, 1]
        x2 = dets[:, 1, 0]
        y2 = dets[:, 1, 1]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
        
    def is_img(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.png', '.jpg']

    def extract_labels(self, path, is_template=False):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        if is_template:
            return [name]

        return name.split('@')[:-1]

    def get_best_match_templates(self, score_map):
        for template_path, entry in score_map.items():
            labels = self.extract_labels(template_path, True)
            return {'label': labels[0], 'boxes': entry[0], 'scores': entry[2]}

    def output_result(self, result):
        n_all = 0
        n_correct = 0

        for image_path, template_label in result.items():
            n_all += 1
            labels = self.extract_labels(image_path)
            if template_label in labels:
                n_correct += 1

        print('accuracy: {}'.format(n_correct / n_all * 100))

        with open(os.path.join(self.config.output_dir, 'result.txt'), 'w') as f:
            for image_path, template_label in result.items():
                f.write('{}\t{}\n'.format(str(image_path), template_label))

    def execute(self):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        vgg_feature = models.vgg13(pretrained=True).features
        FE = FeatureExtractor(vgg_feature, use_cuda=self.config.use_cuda, padding=True)

        image_paths = []
        for path in os.listdir(self.config.image_dir):
            if self.is_img(path):
                image_paths.append(os.path.join(self.config.image_dir, path))

        template_paths = []
        for path in os.listdir(self.config.template_dir):
            if self.is_img(path):
                template_paths.append(os.path.join(self.config.template_dir, path))

        print(image_paths)
        print(template_paths)

        result = {}
        for image_path in image_paths:
            score_map = {}
            raw_image = cv2.imread(image_path)[..., ::-1]
            image = image_transform(raw_image.copy()).unsqueeze(0)

            for template_path in template_paths:
                raw_template = cv2.imread(template_path)[..., ::-1]
                template = image_transform(raw_template.copy()).unsqueeze(0)

                boxes, centers, scores = FE(
                    template, image, threshold=0.1, use_cython=self.config.use_cython)
                score_map[template_path] = [boxes, centers, scores]

            best_matches = self.get_best_match_templates(score_map)
            print('{} matches {}'.format(image_path, best_matches))

            boxes = best_matches['boxes']
            scores = best_matches['scores']
            label = best_matches['label']
            if len(boxes) == 0:
                continue

            d_img = raw_image.astype(np.uint8).copy()
            nms_res = self.nms(np.array(boxes), np.array(scores), thresh=0.5)
            print("detected objects: {}".format(len(nms_res)))
            for i in nms_res:
                d_img = cv2.rectangle(d_img, boxes[i][0], boxes[i][1], (255, 0, 0), 3)
                # d_img = cv2.circle(d_img, centers[i], int(
                #     (boxes[i][1][0] - boxes[i][0][0])*0.2), (0, 0, 255), 2)
                
            cv2.imwrite(os.path.join(self.config.output_dir, os.path.basename(image_path)), d_img[..., ::-1])
            
            result[image_path] = label

        self.output_result(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='robust template matching using CNN')
    parser.add_argument('image_dir')
    parser.add_argument('template_dir')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_cython', action='store_true')
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.execute()
