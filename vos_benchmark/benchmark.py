import os
from os import path
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
import tqdm

from .evaluator import Evaluator


class VideoEvaluator:
    """
    A processing function object.
    This returns metrics for a single video.
    """

    def __init__(self, gt_root, mask_root, skip_first_and_last=True, full_mask=True):
        self.gt_root = gt_root
        self.mask_root = mask_root
        self.skip_first_and_last = skip_first_and_last
        self.full_mask = full_mask

    def __call__(self, vid_name):
        vid_gt_path = path.join(self.gt_root, vid_name)
        vid_mask_path = path.join(self.mask_root, vid_name)
        if self.full_mask:
            assert sorted(os.listdir(vid_gt_path)) == sorted(os.listdir(vid_mask_path)), "gt and mask not match"
            frames = sorted(os.listdir(vid_gt_path))
        else:
            assert set(sorted(os.listdir(vid_gt_path))).issubset(set(sorted(os.listdir(vid_mask_path)))), \
                f"additional mask: {set(sorted(os.listdir(vid_gt_path))) - set(sorted(os.listdir(vid_mask_path)))}"
            frames = sorted(os.listdir(vid_mask_path))

        if self.skip_first_and_last:
            # the first and the last frames are skipped in DAVIS semi-supervised evaluation
            frames = frames[1:-1]
        evaluator = Evaluator(name=vid_name)
        for f in frames:
            try:
                gt_array = np.array(Image.open(path.join(vid_gt_path, f)))
                mask_array = np.array(Image.open(path.join(vid_mask_path, f)))
                assert gt_array.shape[-2:] == mask_array.shape[-2:], \
                        f'Dimensions mismatch: GT: {gt_array.shape}, predicted: {mask_array.shape}. '\
                        f'GT path: {path.join(vid_gt_path, f)}; ' \
                        f'predicted path: {path.join(vid_mask_path, f)}'
            except FileNotFoundError:
                print(f'{f} not found in {vid_mask_path}.')
                exit(1)

            evaluator.feed_frame(mask_array, gt_array)
        iou, boundary_f = evaluator.conclude()
        return vid_name, iou, boundary_f


def benchmark(gt_roots,
              mask_roots,
              video_names,
              strict=True,
              full_mask=True,
              overwrite=False,
              num_processes=None,
              *,
              verbose=True,
              skip_first_and_last=True):
    """
    gt_roots: a list of paths to datasets, i.e., [path_to_DatasetA, path_to_DatasetB, ...] 
                with the below directory structure
        DatasetA - 
            Video 1 - 
                xxxx.png
                ...
            Video 2 - 
                xxxx.png
                ...
            ...
        DatasetB - 
            ...
    mask_roots: same as above, but the .png are masks predicted by the model
    video_names: a list of paths to text files, i.e., [path_to_DatasetA, path_to_DatasetB, ...],
                each containing a list of video names to be evaluated.
                If not provided, evaluated videos will depend on strict.
    strict: only used when video_names is not provided.
            when True, all videos in the dataset must have corresponding predictions.
            Setting it to False is useful in cases where the ground-truth contains both train/val
                sets, but the model only predicts the val subset.
            Either way, if a video is predicted (i.e., the corresponding folder exists), 
                then it must at least contain all the masks in the ground truth annotations.
                Masks that are in the prediction but not in the ground-truth
                (i.e., sparse annotations) are ignored.
    overwrite: whether overwrite existing results.csv
    skip_first_and_last: whether we should skip the first and the last frame in evaluation.
                            This is used by DAVIS 2017 in their semi-supervised evaluation.
                            It should be disabled for unsupervised evaluation.
    """

    assert len(gt_roots) == len(mask_roots) and (len(video_names) == 0 or len(video_names) == len(gt_roots))
    single_dataset = (len(gt_roots) == 1)

    # check if results.csv already exists, decide to skip or overwrite
    skip_indices = []
    for i in range(len(mask_roots)):
        mask_root = mask_roots[i]
        if path.exists(path.join(mask_root, 'results.csv')):
            if not overwrite:
                print(f'{mask_root}/results.csv already exists. Skipping.')
                skip_indices.append(i)
            else:
                print(f'{mask_root}/results.csv already exists. But we will remove and overwrite it.')
                os.remove(path.join(mask_root, 'results.csv'))
    mask_roots = [mask_roots[i] for i in range(len(mask_roots)) if i not in skip_indices]
    gt_roots = [gt_roots[i] for i in range(len(gt_roots)) if i not in skip_indices]
    video_names = [video_names[i] for i in range(len(video_names)) if i not in skip_indices] if len(video_names) > 0 else []

    if verbose:
        if skip_first_and_last:
            print(
                'We are *SKIPPING* the evaluation of the first and the last frame (standard for semi-supervised video object segmentation).'
            )
        else:
            print(
                'We are *NOT SKIPPING* the evaluation of the first and the last frame (*NOT STANDARD* for semi-supervised video object segmentation).'
            )

    pool = Pool(num_processes)
    start = time.time()
    to_wait = []
    for i, (gt_root, mask_root) in enumerate(zip(gt_roots, mask_roots)):
        #Validate folders
        validated = True
        gt_videos = os.listdir(gt_root)
        mask_videos = os.listdir(mask_root)

        # if the user passed the root directory instead of Annotations
        if len(gt_videos) != len(mask_videos):
            if 'Annotations' in gt_videos:
                if '.png' not in os.listdir(path.join(gt_root, 'Annotations'))[0]:
                    gt_root = path.join(gt_root, 'Annotations')
                    gt_videos = os.listdir(gt_root)

        # remove non-folder items
        gt_videos = list(filter(lambda x: path.isdir(path.join(gt_root, x)), gt_videos))
        mask_videos = list(filter(lambda x: path.isdir(path.join(mask_root, x)), mask_videos))

        # read evaluated videos from text file or according to strict
        if len(video_names) > 0:
            with open(video_names[i], mode='r') as f:
                video_name = set(f.read().splitlines())

            if video_name != set(mask_videos):
                print(f'Videos in {video_names[i]} do not match videos in {mask_root}.')
                validated = False
            if not video_name.issubset(set(gt_videos)):
                print(f'Videos in {video_names[i]} are not in {gt_root}.')
                validated = False

            videos = sorted(list(video_name))
        elif not strict:
            videos = sorted(list(set(gt_videos) & set(mask_videos)))
        else:
            gt_extras = set(gt_videos) - set(mask_videos)
            mask_extras = set(mask_videos) - set(gt_videos)

            if len(gt_extras) > 0:
                print(f'Videos that are in {gt_root} but not in {mask_root}: {gt_extras}')
                validated = False
            if len(mask_extras) > 0:
                print(f'Videos that are in {mask_root} but not in {gt_root}: {mask_extras}')
                validated = False

            videos = sorted(gt_videos)
        if not validated:
            print('Validation failed. Exiting.')
            exit(1)

        if verbose:
            print(f'In dataset {gt_root}, we are evaluating on {len(videos)} videos: {videos}')

        if single_dataset:
            if verbose:
                results = tqdm.tqdm(pool.imap(
                    VideoEvaluator(gt_root, mask_root, skip_first_and_last=skip_first_and_last, full_mask=full_mask),
                    videos),
                                    total=len(videos))
            else:
                results = pool.map(
                    VideoEvaluator(gt_root, mask_root, skip_first_and_last=skip_first_and_last, full_mask=full_mask),
                    videos)
        else:
            to_wait.append(
                pool.map_async(
                    VideoEvaluator(gt_root, mask_root, skip_first_and_last=skip_first_and_last, full_mask=full_mask),
                    videos))

    pool.close()

    all_global_jf, all_global_j, all_global_f = [], [], []
    all_object_metrics = []
    for i, mask_root in enumerate(mask_roots):
        if not single_dataset:
            results = to_wait[i].get()

        all_iou = []
        all_boundary_f = []
        object_metrics = {}
        for name, iou, boundary_f in results:
            all_iou.extend(list(iou.values()))
            all_boundary_f.extend(list(boundary_f.values()))
            object_metrics[name] = (iou, boundary_f)

        global_j = np.array(all_iou).mean()
        global_f = np.array(all_boundary_f).mean()
        global_jf = (global_j + global_f) / 2

        time_taken = (time.time() - start)
        """
        Build string for reporting results
        """
        # find max length for padding
        ml = max(*[len(n) for n in object_metrics.keys()], len('Global score'))
        # build header
        out_string = f'{"sequence":<{ml}},{"obj":>3}, {"J&F":>4}, {"J":>4}, {"F":>4}\n'
        out_string += f'{"Global score":<{ml}},{"":>3}, {global_jf:.1f}, {global_j:.1f}, {global_f:.1f}\n'
        # append one line for each object
        for name, (iou, boundary_f) in object_metrics.items():
            for object_idx in iou.keys():
                j, f = iou[object_idx], boundary_f[object_idx]
                jf = (j + f) / 2
                out_string += f'{name:<{ml}},{object_idx:03}, {jf:>4.1f}, {j:>4.1f}, {f:>4.1f}\n'

        # print to console
        if verbose:
            print(out_string.replace(',', ' '), end='')
            print('\nSummary:')
            print(f'Global score: J&F: {global_jf:.1f} J: {global_j:.1f} F: {global_f:.1f}')
            print(f'Time taken: {time_taken:.2f}s')

        # print to file
        with open(path.join(mask_root, 'results.csv'), 'w') as f:
            f.write(out_string)

        all_global_jf.append(global_jf)
        all_global_j.append(global_j)
        all_global_f.append(global_f)
        all_object_metrics.append(object_metrics)

    return all_global_jf, all_global_j, all_global_f, all_object_metrics
