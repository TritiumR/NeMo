import sys
import os
from PIL import Image
sys.path.append('./')

import logging

import numpy as np
from tqdm import tqdm

from nemo.utils import pose_error


def inference_3d_pose_estimation(
    cfg,
    cate,
    model,
    dataloader,
    cached_pred=None
):
    save_pred = {}
    pose_errors = []
    running = []
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            preds = model.evaluate(sample)
            
            for pred, name_ in zip(preds, sample['this_name']):
                save_pred[str(name_)] = pred
        else:
            for name_ in sample['this_name']:
                save_pred[str(name_)] = cached_pred[str(name_)]
        for idx, pred in enumerate(preds):
            # _err = pose_error(sample, pred["final"][0])
            _err = pred['pose_error']
            print('pi6_acc: ', np.mean(np.array(_err) < np.pi / 6), 'pi18_acc: ', np.mean(np.array(_err) < np.pi / 18), 'med_err: ', np.median(np.array(_err)) / np.pi * 180.0)
            if np.mean(np.array(_err) < np.pi / 6) < 0.5:
                image_ori = sample['img_ori'][idx].cpu().numpy().transpose(1, 2, 0)
                image_ori = (image_ori * 255).astype(np.uint8)
                image_ori = Image.fromarray(image_ori)
                if not os.path.exists('./visual/failure'):
                    os.makedirs('./visual/failure')
                image_ori.save(f'./visual/failure/err_{np.mean(np.array(_err))}.png')
            pose_errors.append(_err)
            running.append((cate, _err))
    pose_errors = np.array(pose_errors)

    results = {}
    results["running"] = running
    results["pi6_acc"] = np.mean(pose_errors < np.pi / 6)
    results["pi18_acc"] = np.mean(pose_errors < np.pi / 18)
    results["med_err"] = np.median(pose_errors) / np.pi * 180.0
    results["save_pred"] = save_pred

    return results


def inference_correlation(
        cfg,
        cate,
        model,
        dataloader,
        cached_pred=None
):
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            _ = model.evaluate_corr(sample)

    return None


def inference_part_locate(
        cfg,
        cate,
        model,
        dataloader,
        cached_pred=None
):
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            _ = model.evaluate_locate(sample)

    return None


def inference_part_locate_and_rotate(
        cfg,
        cate,
        model,
        dataloader,
        cached_pred=None
):
    save_pred = {}
    pose_errors = []
    running = []
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            preds, part_preds = model.evaluate_part(sample)

            for pred, name_ in zip(preds, sample['this_name']):
                save_pred[str(name_)] = pred
        else:
            for name_ in sample['this_name']:
                save_pred[str(name_)] = cached_pred[str(name_)]
        for idx, pred in enumerate(preds):
            # _err = pose_error(sample, pred["final"][0])
            _err = pred['pose_error']
            # print('pi6_acc: ', np.mean(np.array(_err) < np.pi / 6), 'pi18_acc: ', np.mean(np.array(_err) < np.pi / 18),
            #       'med_err: ', np.median(np.array(_err)) / np.pi * 180.0)
            if np.mean(np.array(_err) < np.pi / 6) < 0.5:
                image_ori = sample['img_ori'][idx].cpu().numpy().transpose(1, 2, 0)
                image_ori = (image_ori * 255).astype(np.uint8)
                image_ori = Image.fromarray(image_ori)
                if not os.path.exists('./visual/failure'):
                    os.makedirs('./visual/failure')
                image_ori.save(f'./visual/failure/err_{np.mean(np.array(_err))}.png')
            pose_errors.append(_err)
            running.append((cate, _err))
    pose_errors = np.array(pose_errors)

    results = {}
    results["running"] = running
    results["pi6_acc"] = np.mean(pose_errors < np.pi / 6)
    results["pi18_acc"] = np.mean(pose_errors < np.pi / 18)
    results["med_err"] = np.median(pose_errors) / np.pi * 180.0
    results["save_pred"] = save_pred

    return results


def inference_image_part(
        cfg,
        cate,
        model,
        dataloader,
        cached_pred=None
):
    if cate == 'car':
        anno_parts = ['body', 'wheel', 'mirror']
        body_ious = []
        wheel_ious = []
        mirror_ious = []
    elif cate == 'aeroplane':
        anno_parts = ['head', 'body', 'engine', 'wing', 'tail']
        head_ious = []
        body_ious = []
        engine_ious = []
        wing_ious = []
        tail_ious = []

    mious = []

    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            results = model.evaluate_imagepart(sample)
            if results is None:
                continue
            print('results: ', results)
            if cate == 'car':
                body_ious.append(results['body'])
                wheel_ious.append(results['wheel'])
                mirror_ious.append(results['mirror'])
            elif cate == 'aeroplane':
                head_ious.append(results['head'])
                body_ious.append(results['body'])
                engine_ious.append(results['engine'])
                wing_ious.append(results['wing'])
                tail_ious.append(results['tail'])
            mious.append(results['mIoU'])

    if cate == 'car':
        wheel_ious = np.array(wheel_ious)
        mirror_ious = np.array(mirror_ious)
        body_ious = np.array(body_ious)
        results = {}
        results["wheel_iou"] = np.mean(wheel_ious)
        results["mirror_iou"] = np.mean(mirror_ious)
        results["body_iou"] = np.mean(body_ious)
        results["mIoU"] = np.mean(mious)
        print('final results: ')
        print('wheel_iou: ', results["wheel_iou"], '  mirror_iou: ', results["mirror_iou"], '  body_iou: ',
              results["body_iou"], '  mIoU: ', results["mIoU"])
    elif cate == 'aeroplane':
        head_ious = np.array(head_ious)
        body_ious = np.array(body_ious)
        engine_ious = np.array(engine_ious)
        wing_ious = np.array(wing_ious)
        tail_ious = np.array(tail_ious)
        results = {}
        results["head_iou"] = np.mean(head_ious)
        results["body_iou"] = np.mean(body_ious)
        results["engine_iou"] = np.mean(engine_ious)
        results["wing_iou"] = np.mean(wing_ious)
        results["tail_iou"] = np.mean(tail_ious)
        results["mIoU"] = np.mean(mious)
        print('final results: ')
        print('head_iou: ', results["head_iou"], '  body_iou: ', results["body_iou"], '  engine_iou: ',
              results["engine_iou"], '  wing_iou: ', results["wing_iou"], '  tail_iou: ', results["tail_iou"],
              '  mIoU: ', results["mIoU"])
    return None


def print_3d_pose_estimation(
    cfg,
    all_categories,
    running_results
):
    logging.info(f"\n3D Pose Estimation Results:")
    logging.info(f"Dataset:     {cfg.dataset.name} (root={cfg.dataset.root_path})")
    logging.info(f"Category:    {all_categories}")
    logging.info(f"# samples:   {len(running_results)}")
    logging.info(f"Model:       {cfg.model.name} (ckpt={cfg.args.checkpoint})")

    cate_line = f'            '
    pi_6_acc = f'pi/6 acc:   '
    pi_18_acc = f'pi/18 acc:  '
    med_err = f'Median err: '
    for cate in all_categories:
        pose_errors_cate = np.array([x[1] for x in running_results if x[0] == cate])
        cate_line += f'{cate[:6]:>8s}'
        pi_6_acc += f'  {np.mean(pose_errors_cate < np.pi / 6)*100:5.1f}%'
        pi_18_acc += f'  {np.mean(pose_errors_cate < np.pi / 18)*100:5.1f}%'
        med_err += f'  {np.median(pose_errors_cate)/np.pi*180.0:6.2f}'
    cate_line += f'    mean'
    pose_errors_cate = np.array([x[1] for x in running_results])
    pi_6_acc += f'  {np.mean(pose_errors_cate < np.pi / 6)*100:5.1f}%'
    pi_18_acc += f'  {np.mean(pose_errors_cate < np.pi / 18)*100:5.1f}%'
    med_err += f'  {np.median(pose_errors_cate)/np.pi*180.0:6.2f}'

    logging.info('\n'+cate_line+'\n'+pi_6_acc+'\n'+pi_18_acc+'\n'+med_err)


helper_func_by_task = {"3d_pose_estimation": inference_3d_pose_estimation, "3d_pose_estimation_print": print_3d_pose_estimation,
                       "correlation_marking": inference_correlation, "part_locate": inference_part_locate,
                       "part_locate_and_rotate": inference_part_locate_and_rotate, "image_part": inference_image_part}
