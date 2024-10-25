import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
import cv2

import argparse
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(utils_dir)
from utils.evaluation import multilabel_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def read_file(file_path):
    f = open(file_path,"r")
    lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip('\n'))
    return data_list


def eval_scene(image_path, cfgs):
    result = np.zeros(7)
    dataset = cfgs.dataset
    dataset_root = os.path.join(cfgs.dataset_root, dataset)
    segment_root = os.path.join(cfgs.segment_result, dataset)
    segment_method = cfgs.segment_method
    image_name = os.path.basename(image_path).split('.')[0]
    if dataset == 'OCID':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'label'))
        pred_mask_path = os.path.join(segment_root, '{}_mask'.format(segment_method), image_dir, '{}.png'.format(image_name))
        gt_mask = np.array(Image.open(gt_mask_path))
        gt_mask[gt_mask == 1] = 0
        if 'table' in gt_mask_path:
            gt_mask[gt_mask == 2] = 0
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        
    elif dataset == 'OSD':
        gt_mask_path = os.path.join(dataset_root, image_path.replace('image_color', 'annotation'))
        pred_mask_path = os.path.join(segment_root, '{}_mask'.format(segment_method), '{}.png'.format(image_name))
        gt_mask = np.array(Image.open(gt_mask_path))
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        
    elif dataset == 'PhoCAL':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'mask'))
        pred_mask_path = os.path.join(segment_root, '{}_mask'.format(segment_method), image_dir, '{}.png'.format(image_name))
        gt_mask = np.array(Image.open(gt_mask_path))
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        
    elif dataset == 'Housecat6D':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'instance'))
        pred_mask_path = os.path.join(segment_root, '{}_mask'.format(segment_method), image_dir, '{}.png'.format(image_name))
        gt_mask = np.array(Image.open(gt_mask_path))
        gt_mask[gt_mask == 255] = 0
        width, height = gt_mask.shape[1], gt_mask.shape[0]

    gt_mask = process_label(gt_mask)
    pred_mask = np.array(Image.open(pred_mask_path))
    pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    eval_metrics = multilabel_metrics(pred_mask, gt_mask, 0.75)
    
    result[0] = eval_metrics['Objects F-measure']
    result[1] = eval_metrics['Objects Precision']
    result[2] = eval_metrics['Objects Recall']
    result[3] = eval_metrics['Boundary F-measure']
    result[4] = eval_metrics['Boundary Precision']
    result[5] = eval_metrics['Boundary Recall']
    result[6] = eval_metrics['obj_detected_075_percentage']
    return result


def parallel_eval(image_list, cfgs, proc=2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes=proc)
    result_list = []
    for image_path in image_list:
        scene_result = p.apply_async(eval_scene, (image_path, cfgs))
        result_list.append(scene_result)
    p.close()
    p.join()
    return result_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='OSD', choices=['OSD', 'OCID', 'PhoCAL', 'Housecat6D'])
    parser.add_argument('--dataset_root', default='/media/user/data1/dataset')
    parser.add_argument('--segment_result', default='/media/user/data1/rcao/result/uois')
    parser.add_argument('--segment_method', default='UCN_rgb', help='Segmentation method')
    cfgs = parser.parse_args()
    print(cfgs)
    
    image_list = read_file(os.path.join(cfgs.dataset_root, cfgs.dataset, 'data_list.txt'))
    result_list = parallel_eval(image_list, cfgs=cfgs, proc=20)
    results = [result.get() for result in result_list]
    results = np.stack(results, axis=0)
    
    print('Overlap Prec:{}, Rec:{}, F_score:{}, Boundary Prec:{}, Rec:{}, F_score:{}, %75:{}'. \
        format(np.mean(results[:, 1]), np.mean(results[:, 2]), np.mean(results[:, 0]),
               np.mean(results[:, 4]), np.mean(results[:, 5]), np.mean(results[:, 3]), np.mean(results[:, 6])))
    np.save('{}_{}_results.npy'.format(cfgs.dataset, cfgs.segment_method), results)