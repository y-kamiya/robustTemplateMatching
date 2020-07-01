import os
import time
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
import logging

class Evaluator:
    SCORE_THRESHOLD = 0.3

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

    def get_matched_templates(self, score_map, n_top):
        entry_all = []
        for template_path, entry in score_map.items():
            for i in range(len(entry[0])):
                entry_all.append((template_path, entry[0][i], entry[1][i]))

        sorted_entries = sorted(entry_all, key=lambda x:-x[2])[:n_top]
        if len(sorted_entries) <= 0:
            sorted_entries.append(('none.png', [[0,0],[0,0]], [0.0]))

        return sorted_entries
        # labels = self.extract_labels(template_path, True)
        # return {'label': labels[0], 'boxes': entry[0], 'scores': entry[2]}

    def output_result(self, result, is_log=True):
        n_all = accuracy = precision = recall = 0

        for image_path, template_paths in result.items():
            n_all += 1
            image_labels = set(self.extract_labels(image_path))
            template_labels = set([self.extract_labels(path, True)[0] for path in template_paths])
            inter = image_labels & template_labels
            union = image_labels | template_labels
            accuracy += len(inter) / len(union)
            precision += len(inter) / len(template_labels)
            recall += len(inter) / len(image_labels)

        accuracy = accuracy / n_all * 100
        precision = precision / n_all * 100
        recall = recall / n_all * 100

        if is_log:
            self.config.logger.info('accuracy: {}, precision: {}, recall: {}'.format(accuracy, precision, recall))

            with open(os.path.join(self.config.output_dir, 'result.txt'), 'w') as f:
                for image_path, template_label in result.items():
                    f.write('{}\t{}\n'.format(str(image_path), template_label))

        return accuracy, precision, recall

    def execute(self):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        vgg_feature = models.vgg13(pretrained=True).features
        FE = FeatureExtractor(self.config, vgg_feature, padding=True)

        image_paths = []
        for path in os.listdir(self.config.image_dir):
            if self.is_img(path):
                image_paths.append(os.path.join(self.config.image_dir, path))

        template_paths = []
        for path in os.listdir(self.config.template_dir):
            if self.is_img(path):
                template_paths.append(os.path.join(self.config.template_dir, path))

        self.config.logger.debug(image_paths)
        self.config.logger.debug(template_paths)

        result = {}
        for image_path in image_paths:
            score_map = {}
            raw_image = cv2.imread(image_path)[..., ::-1]
            image = image_transform(raw_image.copy()).unsqueeze(0)

            for template_path in template_paths:
                start_time = time.time()
                raw_template = cv2.imread(template_path)[..., ::-1]
                template = image_transform(raw_template.copy()).unsqueeze(0)

                boxes, scores = FE(template_path, template, image_path, image, use_cython=self.config.use_cython)

                # 複数返す場合は重複削除処理
                # indexes = self.nms(boxes, scores, thresh=0.5)
                # self.config.logger.debug("detected objects: {}".format(len(indexes)))
                # score_map[template_path] = [boxes[indexes], scores[indexes]]

                if self.SCORE_THRESHOLD <= scores[0]:
                    score_map[template_path] = [boxes, scores]

                self.config.logger.info('{:.2f}\t{:.4}\t{}\t{}'.format(time.time() - start_time, scores[0], template_path, image_path))

            FE.remove_cache(image_path)

            matched_entries = self.get_matched_templates(score_map, 2)
            self.config.logger.debug('{} matches {}'.format(image_path, matched_entries))

            if len(matched_entries) == 0:
                continue

            result[image_path] = []
            d_img = raw_image.astype(np.uint8).copy()
            for entry in matched_entries:
                box = entry[1]
                d_img = cv2.rectangle(d_img, (box[0][0],box[0][1]), (box[1][0],box[1][1]), (255, 0, 0), 3)

                result[image_path].append(entry[0])
                
            cv2.imwrite(os.path.join(self.config.output_dir, os.path.basename(image_path)), d_img[..., ::-1])
            self.config.logger.info('result: {}\t{}'.format(image_path, result[image_path]))
            

        self.output_result(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='robust template matching using CNN')
    parser.add_argument('image_dir')
    parser.add_argument('template_dir')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--use_cython', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--loglevel', default='INFO')
    parser.add_argument('--logfile', default=None)
    args = parser.parse_args()

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda"
    args.device = torch.device(args.device_name)

    logging.basicConfig(level=getattr(logging, args.loglevel))

    logger = logging.getLogger('default')
    args.logger = logger
    if args.logfile is not None:
        handler = logging.FileHandler(args.logfile)
        handler.setLevel(getattr(logging, args.loglevel))
        logger.addHandler(handler)

    evaluator = Evaluator(args)
    evaluator.execute()
