#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a DeepIM network on an image database."""

import cv2
from termcolor import colored
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import random
import scipy.io
from PIL import Image
from tqdm import tqdm

import _init_paths
from fcn.test_common import _vis_minibatch_segmentation
from fcn.test_dataset import AverageMeter, clustering_features, crop_rois, filter_labels_depth, match_label_crop
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from utils.blob import add_noise, chromatic_transform
from utils.evaluation import multilabel_metrics

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Unseen Clustering Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default='/home/user/scx/Code/UnseenObjectClustering/data/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth', type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default='/home/user/scx/Code/UnseenObjectClustering/data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint.pth', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='/home/user/scx/Code/UnseenObjectClustering/experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml', type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='ocid_object_test', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='seg_resnet34_8s_embedding', type=str)
    parser.add_argument(
        "--result_save_root", dest='result_save_root',
        type=str,
        default="/media/user/data1/rcao/result/uois/Housecat6D/UCN_rgb_mask",
        help="path to save inference result"
    )

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def process_label(foreground_labels):
    """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

        @param foreground_labels: a [H x W] numpy array of labels

        @return: foreground_labels
    """
    # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
    unique_nonnegative_indices = np.unique(foreground_labels)
    mapped_labels = foreground_labels.copy()
    for k in range(unique_nonnegative_indices.shape[0]):
        mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
    foreground_labels = mapped_labels
    return foreground_labels

def test_segnet(network, output_dir, network_crop):

    batch_time = AverageMeter()

    # switch to test mode
    network.eval()
    if network_crop is not None:
        network_crop.eval()

    metrics_all = []
    metrics_all_refined = []

    # load dataset
    image_paths = []
    depth_paths = []
    anno_paths = []

    data_root = '/media/user/data1/dataset/Housecat6D'

    data_list_path = os.path.join(data_root, "data_list.txt")
    # 读取 data_list.txt 文件
    with open(data_list_path, 'r') as file:
        data_list = file.read().splitlines()

    for item in data_list:
        image_path = os.path.join(data_root, item)
        depth_path = image_path.replace('/rgb/', '/depth/')
        anno_path = image_path.replace('/rgb/', '/instance/')

        image_paths.append(image_path)
        depth_paths.append(depth_path)
        anno_paths.append(anno_path)

    assert len(image_paths) == len(depth_paths)
    assert len(image_paths) == len(anno_paths)
    print(colored("Evaluation on PhoCAL dataset: {} rgbs, {} depths, {} visible_masks".format(
                        len(image_paths), len(depth_paths), len(anno_paths)), "green"))
    epoch_size = len(image_paths)        
    
    for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(image_paths), depth_paths, anno_paths)):     
        image_dir, image_name = rgb_path.split('/')[-3], os.path.basename(rgb_path).split('.')[0]

        im = cv2.imread(rgb_path)
        im = cv2.resize(im, (640, 480))
        # if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
        #     im = chromatic_transform(im)
        # if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
        #     im = add_noise(im)
        im_tensor = torch.from_numpy(im) / 255.0

        im_tensor_bgr = im_tensor.clone()
        im_tensor_bgr = im_tensor_bgr.permute(2, 0, 1)

        pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()

        im_tensor -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)

        # Label
        foreground_labels = np.array(Image.open(anno_path))
        # mask table as background
        # foreground_labels[foreground_labels == 1] = 0
        foreground_labels = process_label(foreground_labels)
        foreground_labels = foreground_labels.astype(np.int32)  # 或者 np.int64
        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)

        # index = filename.find('OCID')
        sample = {'image_color': image_blob.unsqueeze(0),
                'image_color_bgr': im_tensor_bgr.unsqueeze(0),
                'label': label_blob.unsqueeze(0),
                'filename': image_name}

    # for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        image = sample['image_color'].cuda()
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth = sample['depth'].cuda()
        else:
            depth = None
        label = sample['label'].cuda()

        # run network
        features = network(image, label, depth).detach()
        out_label, selected_pixels = clustering_features(features, num_seeds=100)

        if depth is not None:
            # filter labels on zero depth
            out_label = filter_labels_depth(out_label, depth, 0.5)

        # if 'osd' in test_loader.dataset.name and depth is not None:
        #     # filter labels on zero depth
        #     out_label = filter_labels_depth(out_label, depth, 0.8)

        # evaluation
        # gt = sample['label'].squeeze().numpy()
        # prediction = out_label.squeeze().detach().cpu().numpy()
        # metrics = multilabel_metrics(prediction, gt)
        # metrics_all.append(metrics)
        # print(metrics)

        prediction = out_label.squeeze().detach().cpu().numpy()

        # result save
        prediction = (prediction / np.max(prediction)) * 255
        result = Image.fromarray(prediction.astype(np.uint8))
        mask_save_path = os.path.join(args.result_save_root, image_dir)
        os.makedirs(mask_save_path, exist_ok=True)
        result.save(os.path.join(mask_save_path, '{}.png'.format(image_name)))

        # zoom in refinement
        out_label_refined = None
        if network_crop is not None:
            rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
            if rgb_crop.shape[0] > 0:
                features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
                labels_crop, selected_pixels_crop = clustering_features(features_crop)
                out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

        # evaluation
        if out_label_refined is not None:
            prediction_refined = out_label_refined.squeeze().detach().cpu().numpy()
        else:
            prediction_refined = prediction.copy()
        # metrics_refined = multilabel_metrics(prediction_refined, gt)
        # metrics_all_refined.append(metrics_refined)
        # print(metrics_refined)

        prediction_refined = (prediction_refined / np.max(prediction_refined)) * 255
        result_zoomin = Image.fromarray(prediction_refined.astype(np.uint8))
        mask_save_path = os.path.join(args.result_save_root.replace('_mask', '_zoomin_mask'), image_dir)
        os.makedirs(mask_save_path, exist_ok=True)
        result_zoomin.save(os.path.join(mask_save_path, '{}.png'.format(image_name)))

        # if cfg.TEST.VISUALIZE:
        #     _vis_minibatch_segmentation(image, depth, label, out_label, out_label_refined, features, 
        #         selected_pixels=selected_pixels, bbox=None)
        # else:
        #     # save results
        #     result = {'labels': prediction, 'labels_refined': prediction_refined, 'filename': sample['filename']}
        #     filename = os.path.join(output_dir, '%06d.mat' % i)
        #     print(filename)
        #     scipy.io.savemat(filename, result, do_compression=True)

        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time.val))

    # # sum the values with same keys
    # print('========================================================')
    # result = {}
    # num = len(metrics_all)
    # print('%d images' % num)
    # print('========================================================')
    # for metrics in metrics_all:
    #     for k in metrics.keys():
    #         result[k] = result.get(k, 0) + metrics[k]

    # for k in sorted(result.keys()):
    #     result[k] /= num
    #     print('%s: %f' % (k, result[k]))

    # print('%.6f' % (result['Objects Precision']))
    # print('%.6f' % (result['Objects Recall']))
    # print('%.6f' % (result['Objects F-measure']))
    # print('%.6f' % (result['Boundary Precision']))
    # print('%.6f' % (result['Boundary Recall']))
    # print('%.6f' % (result['Boundary F-measure']))
    # print('%.6f' % (result['obj_detected_075_percentage']))

    # print('========================================================')
    # print(result)
    # print('====================Refined=============================')

    # result_refined = {}
    # for metrics in metrics_all_refined:
    #     for k in metrics.keys():
    #         result_refined[k] = result_refined.get(k, 0) + metrics[k]

    # for k in sorted(result_refined.keys()):
    #     result_refined[k] /= num
    #     print('%s: %f' % (k, result_refined[k]))
    # print(result_refined)
    # print('========================================================')


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = args.gpu_id
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(args.gpu_id))

    # prepare dataset
    if cfg.TEST.VISUALIZE:
        shuffle = True
        np.random.seed()
    else:
        shuffle = False
    cfg.MODE = 'TEST'
    # dataset = get_dataset(args.dataset_name)
    # worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    num_workers = 1
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=shuffle,
        # num_workers=num_workers)
    # print('Use dataset `{:s}` for training'.format(dataset.name))

    # overwrite intrinsics
    # if len(cfg.INTRINSICS) > 0:
    #     K = np.array(cfg.INTRINSICS).reshape(3, 3)
    #     dataset._intrinsic_matrix = K
    #     print(dataset._intrinsic_matrix)

    # output_dir = get_output_dir(dataset, None)
    output_dir = '/home/user/scx/Code/UnseenObjectClustering/output/tabletop_object/phocal'
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](1, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=args.gpu_id)
    network = torch.nn.DataParallel(network, device_ids=[args.gpu_id])
    cudnn.benchmark = True

    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](1, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=args.gpu_id)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[args.gpu_id])
    else:
        network_crop = None

    # test network
    test_segnet(network, output_dir, network_crop)

    

