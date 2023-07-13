import numpy as np
import os
from PIL import Image, ImageDraw


imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple/train/car'
render_imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_render/train/car'
save_path = '../visual/DiffusionQualityDirty'

if not os.path.exists(save_path):
    os.makedirs(save_path)

instance_ids = os.listdir(imgs_path)
print('instance number: ', len(instance_ids))

for instance_id in instance_ids:
    instance_path = os.path.join(save_path, f'{instance_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    img_path = os.path.join(imgs_path, instance_id)
    if not os.path.exists(img_path):
        continue
    render_img_path = os.path.join(render_imgs_path, instance_id)

    img_fns = os.listdir(img_path)
    print('image number: ', len(img_fns))
    for idx, img_fn in enumerate(img_fns):
        img = np.array(Image.open(os.path.join(img_path, img_fn)).resize((480, 480)))
        # print('shape: ', img.shape)

        render_img_fn = img_fn[:-7] + '.png'
        render_img = np.array(Image.open(os.path.join(render_img_path, render_img_fn)))
        render_img = render_img[:, :, :3]
        # print('render shape: ', render_img.shape)

        mixed_image = (img * 0.4 + render_img * 0.6).astype(np.uint8)
        Image.fromarray(mixed_image).save(os.path.join(instance_path, f'{idx}.jpg'))
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(instance_path, f'img_{idx}.jpg'))
        Image.fromarray(render_img.astype(np.uint8)).save(os.path.join(instance_path, f'render_{idx}.jpg'))
