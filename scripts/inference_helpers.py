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
        body_intersections = 0
        wheel_intersections = 0
        mirror_intersections = 0
        body_unions = 0
        wheel_unions = 0
        mirror_unions = 0
    elif cate == 'aeroplane':
        anno_parts = ['head', 'body', 'engine', 'wing', 'tail']
        head_intersections = 0
        body_intersections = 0
        engine_intersections = 0
        wing_intersections = 0
        tail_intersections = 0
        head_unions = 0
        body_unions = 0
        engine_unions = 0
        wing_unions = 0
        tail_unions = 0
    elif cate == 'bicycle':
        anno_parts = ['frame', 'handle', 'saddle', 'wheel']
        frame_intersections = 0
        handle_intersections = 0
        saddle_intersections = 0
        wheel_intersections = 0
        frame_unions = 0
        handle_unions = 0
        saddle_unions = 0
        wheel_unions = 0
    elif cate == 'boat':
        anno_parts = ['body', 'sail']
        body_intersections = 0
        sail_intersections = 0
        body_unions = 0
        sail_unions = 0
    bg_intersections = 0
    bg_unions = 0
    mious = []

    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            results = model.evaluate_imagepart(sample)
            if results is None:
                continue
            print('results: ', results)
            if cate == 'car':
                body_intersections += results['intersections'][0]
                wheel_intersections += results['intersections'][1]
                mirror_intersections += results['intersections'][2]
                bg_intersections += results['intersections'][3]
                body_unions += results['unions'][0]
                wheel_unions += results['unions'][1]
                mirror_unions += results['unions'][2]
                bg_unions += results['unions'][3]
            elif cate == 'aeroplane':
                head_intersections += results['intersections'][0]
                body_intersections += results['intersections'][1]
                engine_intersections += results['intersections'][2]
                wing_intersections += results['intersections'][3]
                tail_intersections += results['intersections'][4]
                bg_intersections += results['intersections'][5]
                head_unions += results['unions'][0]
                body_unions += results['unions'][1]
                engine_unions += results['unions'][2]
                wing_unions += results['unions'][3]
                tail_unions += results['unions'][4]
                bg_unions += results['unions'][5]
            elif cate == 'bicycle':
                frame_intersections += results['intersections'][0]
                handle_intersections += results['intersections'][1]
                saddle_intersections += results['intersections'][2]
                wheel_intersections += results['intersections'][3]
                bg_intersections += results['intersections'][4]
                frame_unions += results['unions'][0]
                handle_unions += results['unions'][1]
                saddle_unions += results['unions'][2]
                wheel_unions += results['unions'][3]
                bg_unions += results['unions'][4]
            elif cate == 'boat':
                body_intersections += results['intersections'][0]
                sail_intersections += results['intersections'][1]
                bg_intersections += results['intersections'][2]
                body_unions += results['unions'][0]
                sail_unions += results['unions'][1]
                bg_unions += results['unions'][2]
            mious.append(results['mIoU'])

    print('exp name: ', cfg.args.save_dir.split('/')[-1])
    if cate == 'car':
        wheel_ious = wheel_intersections / wheel_unions
        mirror_ious = mirror_intersections / mirror_unions
        body_ious = body_intersections / body_unions
        bg_ious = bg_intersections / bg_unions
        results = {}
        results["wheel_iou"] = wheel_ious
        results["mirror_iou"] = mirror_ious
        results["body_iou"] = body_ious
        results["mIoU"] = np.nanmean(mious)
        results["bg_iou"] = bg_ious
        print('final results: ')
        print('wheel_iou: ', results["wheel_iou"], '  mirror_iou: ', results["mirror_iou"], '  body_iou: ',
              results["body_iou"], '  bg_iou: ', results["bg_iou"], '  mIoU: ', results["mIoU"])
    elif cate == 'aeroplane':
        head_ious = head_intersections / head_unions
        body_ious = body_intersections / body_unions
        engine_ious = engine_intersections / engine_unions
        wing_ious = wing_intersections / wing_unions
        tail_ious = tail_intersections / tail_unions
        bg_ious = bg_intersections / bg_unions
        results = {}
        results["head_iou"] = head_ious
        results["body_iou"] = body_ious
        results["engine_iou"] = engine_ious
        results["wing_iou"] = wing_ious
        results["tail_iou"] = tail_ious
        results["bg_iou"] = bg_ious
        results["mIoU"] = np.nanmean(mious)
        print('final results: ')
        print('head_iou: ', results["head_iou"], '  body_iou: ', results["body_iou"], '  engine_iou: ',
              results["engine_iou"], '  wing_iou: ', results["wing_iou"], '  tail_iou: ', results["tail_iou"],
              '  bg_iou: ', results["bg_iou"], '  mIoU: ', results["mIoU"])
    elif cate == 'bicycle':
        frame_ious = frame_intersections / frame_unions
        handle_ious = handle_intersections / handle_unions
        saddle_ious = saddle_intersections / saddle_unions
        wheel_ious = wheel_intersections / wheel_unions
        bg_ious = bg_intersections / bg_unions
        results = {}
        results["frame_iou"] = frame_ious
        results["handle_iou"] = handle_ious
        results["saddle_iou"] = saddle_ious
        results["wheel_iou"] = wheel_ious
        results["bg_iou"] = bg_ious
        results["mIoU"] = np.nanmean(mious)
        print('final results: ')
        print('frame_iou: ', results["frame_iou"], '  handle_iou: ', results["handle_iou"], '  saddle_iou: ',
              results["saddle_iou"], '  wheel_iou: ', results["wheel_iou"], '  bg_iou: ', results["bg_iou"],
              '  mIoU: ', results["mIoU"])
    elif cate == 'boat':
        body_ious = body_intersections / body_unions
        sail_ious = sail_intersections / sail_unions
        bg_ious = bg_intersections / bg_unions
        results = {}
        results["body_iou"] = body_ious
        results["sail_iou"] = sail_ious
        results["bg_iou"] = bg_ious
        results["mIoU"] = np.nanmean(mious)
        print('final results: ')
        print('body_iou: ', results["body_iou"], '  sail_iou: ', results["sail_iou"], '  bg_iou: ', results["bg_iou"],
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
