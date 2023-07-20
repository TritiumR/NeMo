# need bpy python=3.7 with "pip install bpy==2.91a0 && bpy_post_install"
import bpy
import bmesh
from scipy.spatial import KDTree
import numpy as np


def get_mesh(obj):
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.editmode_toggle()

    return bmesh.from_edit_mesh(obj.data)


def toggle_edit_mode():
    bpy.ops.object.editmode_toggle()


obj = bpy.context.active_object

# Grab original mesh.
orig_mesh = get_mesh(obj)
orig_mesh.faces.ensure_lookup_table()
print('orig', orig_mesh)

# Face map.
mats = []
vecs = np.zeros((len(orig_mesh.faces), 3))

for i in range(len(orig_mesh.faces)):
    v = orig_mesh.faces[i].calc_center_median()
    m = orig_mesh.faces[i].material_index

    vecs[i][0] = v.x
    vecs[i][1] = v.y
    vecs[i][2] = v.z

    mats.append(m)

toggle_edit_mode()

# Create KDTree.
tree = KDTree(vecs)

# Apply remesh modifier.
bpy.ops.object.modifier_add(type='REMESH')
obj.modifiers['Remesh'].voxel_size = 0.1
bpy.ops.object.modifier_apply(modifier='Remesh')

new_mesh = get_mesh(obj)
new_mesh.faces.ensure_lookup_table()
print('new', new_mesh)

for i in range(len(new_mesh.faces)):
    v = new_mesh.faces[i].calc_center_median()

    mi = tree.query((v.x, v.y, v.z))[1]
    m = mats[mi]

    new_mesh.faces[i].material_index = m

bmesh.update_edit_mesh(obj.data)

toggle_edit_mode()