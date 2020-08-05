# coding: UTF-8
import os
import time
import numpy as np
import cv2
from torchvision import models, transforms, utils
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from FeatureExtractor import FeatureExtractor
import sys
import logging
import ast
import json

class ImageDataset(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg']

    def __init__(self, image_dir, config):
        self.config = config
        self.logger = config.logger
        self.paths = []
        self.cache = {}

        if os.path.isdir(image_dir):
            for file in os.listdir(image_dir):
                if self.is_img(file):
                    self.paths.append(os.path.join(image_dir, file))

        elif os.path.isfile(image_dir):
            with open(image_dir) as f:
                self.paths = [row.strip() for row in f.readlines()]

        else:
            assert False, 'need directory path containing images or file path containing image path list'

        self.paths.sort()

        self.__validate()

    def is_img(self, fname):
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    def __validate(self):
        wrong_paths = []
        for path in self.paths:
            if not os.path.isfile(path):
                wrong_paths.append(path)

        if wrong_paths:
            self.logger.warning('!!!!!!!!!!!!! images do not exist !!!!!!!!!!!!!')
            self.logger.warning(wrong_paths)

    def __transform(self, w, h):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(h), int(w))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def __image_name(self, path):
        return os.path.basename(path)

    def __getitem__(self, index):
        path = self.paths[index]
        name = self.__image_name(path)

        if index not in self.cache:
            img = cv2.imread(path)[..., ::-1]
            h, w, _ = img.shape

            ratio = self.config.resize
            transform = self.__transform(w * ratio, h * ratio)
            image = transform(img)

            self.cache[index] = image

        return {'image': self.cache[index], 'path': path}

    def __len__(self):
        return len(self.paths)

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
        
    def extract_labels(self, path, is_template=False):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        if is_template:
            return [name]

        splited = name.split('@')
        if 1 < len(splited):
            return splited[:-1]

        return [name]

    def get_matched_templates(self, score_map, n_top, image_path):
        entry_all = []
        for name, entry in score_map.items():
            for i in range(len(entry[0])):
                entry_all.append((name, entry[0][i], entry[1][i]))

        entries = sorted(entry_all, key=lambda x:-x[2])
        self.config.logger.info('sorted list: {}\t{}'.format(os.path.basename(image_path), entries[:5]))

        sorted_entries = entries[:n_top]

        if len(sorted_entries) <= 0:
            sorted_entries.append(('none.png', [[0,0],[0,0]], [0.0]))

        return sorted_entries
        # labels = self.extract_labels(name, True)
        # return {'label': labels[0], 'boxes': entry[0], 'scores': entry[2]}

    def summary_result(self, path):
        with open(path, 'r') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            result = {}
            for line in lines:
                # parse until empty line
                if len(line) <= 1:
                    break
                print(line)
                key = line[0]
                values = ast.literal_eval(line[1])
                result[key] = values

        if self.config.predict:
            self.output_predict_result(result)
        else:
            self.output_eval_result(result)

    def output_eval_result(self, result, is_log=True):
        n_all = accuracy = precision = recall = 0

        wrongs = []
        for image_path, template_paths in result.items():
            n_all += 1
            image_labels = set(self.extract_labels(image_path))
            template_labels = set([self.extract_labels(path, True)[0] for path in template_paths])
            inter = image_labels & template_labels
            union = image_labels | template_labels
            accuracy += len(inter) / len(union)
            precision += len(inter) / len(template_labels)
            recall += len(inter) / len(image_labels)
            if len(image_labels) != len(template_labels) or len(image_labels) != len(inter):
                wrongs.append((image_path, template_paths))

        accuracy = accuracy / n_all * 100
        precision = precision / n_all * 100
        recall = recall / n_all * 100

        if is_log:
            text_result = 'accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}'.format(accuracy, precision, recall)
            self.config.logger.info(text_result)

            with open(os.path.join(self.config.output_dir, 'result.txt'), 'w') as f:
                for image_path, template_label in result.items():
                    f.write('{}\t{}\n'.format(str(image_path), template_label))
                f.write('\n=========================================\n')
                f.write('wrongs\n')
                for image_path, template_paths in wrongs:
                    f.write('{}\t{}\n'.format(str(image_path), str(template_paths)))

                f.write('\n=========================================\n')
                f.write(text_result)

        return accuracy, precision, recall

    def output_predict_result(self, result):
        result_by_template = {}
        for image_path, template_paths in result.items():
            for template_path in template_paths:
                if template_path not in result_by_template:
                    result_by_template[template_path] = []
                result_by_template[template_path].append(image_path)

        with open(os.path.join(self.config.output_dir, 'result.txt'), 'w') as f:
            f.write(json.dumps(result_by_template))

    def execute(self):
        vgg_feature = models.vgg13(pretrained=True).features
        FE = FeatureExtractor(self.config, vgg_feature, padding=True)

        dataset_search = ImageDataset(self.config.search_dir, self.config)
        dataset_template = ImageDataset(self.config.template_dir, self.config)
        dataloader_search = DataLoader(dataset_search, batch_size=1)
        dataloader_template = DataLoader(dataset_template, batch_size=1)

        logger = self.config.logger
        result = {}
        for data_search in dataloader_search:
            image = data_search['image']
            image_path = data_search['path'][0]
            score_map = {}

            for data_template in dataloader_template:
                template = data_template['image']
                template_path = data_template['path'][0]
                start_time = time.time()
                boxes, scores = FE(template_path, template, image_path, image)

                # 複数返す場合は重複削除処理
                # indexes = self.nms(boxes, scores, thresh=0.5)
                # logger.debug("detected objects: {}".format(len(indexes)))
                # score_map[template_path] = [boxes[indexes], scores[indexes]]

                if self.config.score_threshold <= scores[0]:
                    score_map[template_path] = [boxes, scores]

                logger.info('{:.2f}\t{:.4}\t{}\t{}'.format(time.time() - start_time, scores[0], template_path, image_path))

            FE.remove_cache(image_path)

            matched_entries = self.get_matched_templates(score_map, self.config.ntop, image_path)
            logger.debug('{} matches {}'.format(image_path, matched_entries))

            if len(matched_entries) == 0:
                continue

            result[image_path] = []
            raw_image = cv2.imread(image_path)[..., ::-1]
            d_img = raw_image.astype(np.uint8).copy()
            for entry in matched_entries:
                box = entry[1]
                d_img = cv2.rectangle(d_img, (box[0][0],box[0][1]), (box[1][0],box[1][1]), (255, 0, 0), 3)

                result[image_path].append(entry[0])
                
            cv2.imwrite(os.path.join(self.config.output_dir, os.path.basename(image_path)), d_img[..., ::-1])
            logger.info('result: {}\t{}'.format(image_path, result[image_path]))
            

        if self.config.predict:
            self.output_predict_result(result)
        else:
            self.output_eval_result(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='robust template matching using CNN')
    parser.add_argument('search_dir')
    parser.add_argument('template_dir')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--loglevel', default='INFO')
    parser.add_argument('--logfile', default=None)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--summary_result', default=None)
    parser.add_argument('--ntop', type=int, default=2)
    parser.add_argument('--klayer', type=int, default=3)
    parser.add_argument('--resize', type=float, default=1.0)
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

    logger.info(args)

    evaluator = Evaluator(args)

    if args.summary_result is not None:
        evaluator.summary_result(args.summary_result)
        sys.exit()

    evaluator.execute()
