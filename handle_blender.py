# %%
import os
import bpy  # blender python api

path = 'c:/Users/Niko/Documents/GitHub/PhoenixHand/segmentations'
name = 'Segmentation'


name_list = [
    'DAU DP',
    'DAU MCP',
    'ZF DP',
    'ZF MCP',
]


def load_finger_elements():
    obj_list = []
    for loc_name in name_list:
        mesh_name = f'{name}_{loc_name}'
        tr_name = f'{name}_TRACKER {loc_name}'.replace(' DP', '')

        # import mesh
        bpy.ops.import_mesh.stl(
            filepath=f'{path}/{mesh_name}.stl')
        loc_obj = bpy.data.objects[f'{mesh_name}']
        obj_list.append(loc_obj)

        # import tracker
        bpy.ops.import_mesh.stl(
            filepath=f'{path}/{tr_name}.stl')
        loc_obj = bpy.data.objects[f'{tr_name}']
        obj_list.append(loc_obj)

    return obj_list


load_finger_elements()
