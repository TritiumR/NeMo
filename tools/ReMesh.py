# need bpy python=3.7 with "pip install bpy==2.91a0 && bpy_post_install"
import bpy
import os
import bmesh
from scipy.spatial import KDTree
import numpy as np

imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/train/car'
meshs_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02958343'
remesh_path = '/mnt/sde/angtian/data/ShapeNet/ReMesh/02958343'

if not os.path.exists(remesh_path):
    os.makedirs(remesh_path)

instance_ids = os.listdir(imgs_path)
print('instance number: ', len(instance_ids))

idx = 0
for instance_id in instance_ids:
    if '.' in instance_id:
        continue
    instance_path = os.path.join(remesh_path, f'{instance_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    mesh_fn = os.path.join(meshs_path, instance_id, 'models', 'model_normalized.obj')

    bpy.ops.import_scene.obj(filepath=mesh_fn, filter_glob="*.obj")
    print('import done')

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
    print('using GPU done')

    # Grab original mesh.
    orig_mesh = bpy.context.selected_objects[0]
    print('orig', orig_mesh)
    print('get mesh done')

    bpy.context.view_layer.objects.active = orig_mesh
    print('active done')

    orig_mesh.select_set(True)
    print('select done')

    # Apply remesh modifier.
    bpy.ops.object.modifier_add(type='REMESH')
    print('add remesh done')

    # adjust the remesh settings as desired
    remesh = orig_mesh.modifiers["Remesh"]

    remesh.mode = 'SMOOTH'  # can also be 'SMOOTH' or 'SHARP'
    remesh.octree_depth = 9  # resolution of the mesh, increase for more detail
    remesh.sharpness = 1  # how much to preserve sharp corners, 1 is max
    remesh.use_remove_disconnected = False

    # apply the remesh modifier
    bpy.ops.object.modifier_apply(modifier="Remesh")

    # add decimate modifier
    bpy.ops.object.modifier_add(type='DECIMATE')

    # adjust decimate settings as desired
    decimate = orig_mesh.modifiers["Decimate"]

    decimate.ratio = 0.01  # reduce the vertex count to 10%

    # apply decimate modifier
    bpy.ops.object.modifier_apply(modifier="Decimate")

    new_mesh = bpy.context.selected_objects[0]
    print('new', new_mesh)

    bpy.ops.export_scene.obj(filepath=os.path.join(instance_path, 'model_remeshed.obj'), use_selection=True)
    print('export remesh done')

    idx += 1
    if idx > 5:
        break
    # bpy.ops.object.delete()