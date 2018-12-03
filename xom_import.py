import bpy
import bmesh
import os
import struct
import math
import mathutils

from mathutils import *
from math import *
from bpy.props import *
from string import *
from struct import *
from math import *
from bpy_extras.io_utils import unpack_list, unpack_face_list
from bpy_extras.image_utils import load_image

bl_info = {
    "name": "Import Xom 3D Model / Animation",
    "author": "Psycrow",
    "version": (1, 0, 2),
    "blender": (2, 60, 0),
    "location": "File > Import > Import Xom 3D Model / Animation (.xom3d, .xac)",
    "description": "Import Xom 3D Model / Animation from XomView format (.xom3d, .xac).",
    "warning": "",
    "wiki_url": "",
    "category": "Import-Export",
}


def trans_matrix(v):
    return mathutils.Matrix.Translation(v)


def scale_matrix(v):
    mat = mathutils.Matrix.Identity(4)
    mat[0][0], mat[1][1], mat[2][2] = v[0], v[1], v[2]
    return mat


def rotate_matrix(v):
    rot = mathutils.Euler(v, 'XYZ').to_quaternion()
    return rot.to_matrix().to_4x4()


def find_bezier(x, a1, b1, c1, d1, a2, b2, c2, d2):
    def bezier_f(t, a, b, c, d):
        t1 = 1 - t
        return (t1 ** 3 * a) + (3 * t * t1 ** 2 * b) + (3 * t ** 2 * t1 * c) + (t ** 3 * d)

    def find(t1, t2, t3):
        xdiv = bezier_f(t2, a1, b1, c1, d1)
        if abs(xdiv - x) < 0.001:
            return t2

        if xdiv > x:
            return find(t1, (t1 + t2) / 2, t2)

        return find(t2, (t2 + t3) / 2, t3)

    return bezier_f(find(0, 0.5, 1.0), a2, b2, c2, d2)


def calculate_frame_value(k, frames_data):
    left_frame, right_frame, cur_frame = None, None, None

    for f in frames_data:
        if f[0] < k:
            left_frame = f
        elif f[0] == k:
            cur_frame = f
            break
        else:
            right_frame = f
            break

    if cur_frame:
        return cur_frame[1]

    if left_frame and right_frame:
        a = left_frame[0] + (right_frame[0] - left_frame[0]) * math.cos(left_frame[5]) * left_frame[4] / 3
        b = left_frame[1] + (right_frame[1] - left_frame[1]) * math.sin(left_frame[5]) * left_frame[4] / 3
        c = right_frame[0] - (right_frame[0] - left_frame[0]) * math.cos(right_frame[3]) * right_frame[2] / 3
        d = right_frame[1] - (right_frame[1] - left_frame[1]) * math.sin(right_frame[3]) * right_frame[2] / 3

        return find_bezier(k, left_frame[0], a, c, right_frame[0], left_frame[1], b, d, right_frame[1])

    if left_frame:
        return left_frame[1]

    return right_frame[1]


def read_byte(fd):
    return struct.unpack('<b', fd.read(1))[0]


def read_float(fd):
    return struct.unpack('<f', fd.read(4))[0]


def read_short(fd):
    return struct.unpack('<h', fd.read(2))[0]


def read_bool(fd):
    return struct.unpack('<?', fd.read(1))[0]


def read_string(fd):
    res = ''
    byte = fd.read(1)
    while byte != b'\x00':
        res += (byte.decode('ascii'))
        byte = fd.read(1)
    return res


def read_vector(fd):
    return struct.unpack('3f', fd.read(12))


def read_short_vector(fd):
    return struct.unpack('<hhh', fd.read(6))


def read_color(fd):
    return fd.read(1)[0] / 255.0, fd.read(1)[0] / 255.0, fd.read(1)[0] / 255.0


def read_xnode(fd, parent_node=None, armature=None):
    global path, nodes_data

    x_type = read_string(fd)

    # XNodeInterion
    if x_type == "XN":
        return None

    x_name = read_string(fd)

    matrix_type = read_byte(fd)
    trmatix = None
    pos = None
    rot = None
    scale = None
    jointorient = None
    rotorient = None
    ndata = None

    if matrix_type == 1:
        trmatix = mathutils.Matrix(((read_float(fd), read_float(fd), read_float(fd), 0),
                                    (read_float(fd), read_float(fd), read_float(fd), 0),
                                    (read_float(fd), read_float(fd), read_float(fd), 0),
                                    (read_float(fd), read_float(fd), read_float(fd), 1.0))).transposed()
        pos = trmatix.to_translation()
        rot = trmatix.to_euler()
        scale = mathutils.Vector((1, 1, 1))
    elif matrix_type == 2:
        pos = mathutils.Vector(read_vector(fd))
        rot = mathutils.Euler(read_vector(fd))
        scale = read_vector(fd)
        trmatix = trans_matrix(pos) * rotate_matrix(rot) * scale_matrix(scale)
    elif matrix_type == 3:
        pos = mathutils.Vector(read_vector(fd))
        rot = mathutils.Euler(read_vector(fd))
        jointorient = read_vector(fd)
        rotorient = read_vector(fd)
        scale = read_vector(fd)

        # todo: testing
        trmatix = trans_matrix(pos) * rotate_matrix(rot) * scale_matrix(scale)
    else:
        trmatix = trans_matrix((0, 0, 0))
        pos = mathutils.Vector((0, 0, 0))
        rot = mathutils.Euler((0, 0, 0))
        scale = mathutils.Vector((1, 1, 1))

    children_num = 0
    object = None

    # XInteriorNode
    if x_type == "IN":
        children_num = read_short(fd)

        armature = bpy.data.armatures.new(x_name)
        object = bpy.data.objects.new(x_name, armature)
        object.show_x_ray = True

        scene = bpy.context.scene
        scene.objects.link(object)
        scene.objects.active = object
        object.select = True

        bpy.ops.object.mode_set(mode='EDIT')

    # XGroup
    elif x_type == "GO" or x_type == "GR":
        children_num = read_short(fd)

        object = armature.edit_bones.new(x_name)
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)

        if type(parent_node["object"]) == bpy.types.EditBone:
            object.parent = parent_node["object"]
            object.matrix = parent_node["object"].matrix * trmatix
        else:
            object.matrix = trmatix

    # XGroupShape or XSkin
    elif x_type == "GS" or x_type == "SK":
        children_num = read_short(fd)
        ndata = parent_node

    # XBoneGroup
    elif x_type == "BG":
        children_num = read_short(fd)

    # XBone
    elif x_type == "BO":
        children_num = read_short(fd)

        object = armature.edit_bones.new(parent_node["name"])
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)

        object.matrix = trmatix.inverted_safe()

        if type(parent_node["parent"]["object"]) == bpy.types.EditBone:
            object.parent = parent_node["parent"]["object"]

        # todo: testing
        for node in nodes_data:
            if node == parent_node:
                node["object"] = object

    # todo:
    # XBinModifier or XChildSelector
    elif x_type == "BM" or x_type == "CS":
        children_num = read_short(fd)

        object = armature.edit_bones.new(x_name)
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)

        if type(parent_node["object"]) == bpy.types.EditBone:
            object.parent = parent_node["object"]
            object.matrix = parent_node["object"].matrix

    # XShape or XSkinShape
    elif x_type == "SH" or x_type == "SS":
        md = bpy.data.meshes.new(x_name)

        faces_num = read_short(fd)
        faces = [read_short_vector(fd) for _ in range(faces_num)]

        vertices_num = read_short(fd)
        vertices = [read_vector(fd) for _ in range(vertices_num)]

        md.from_pydata(vertices, [], faces)

        # if it's a normal vector
        if read_bool(fd):
            for i in range(vertices_num):
                md.vertices[i].normal = read_vector(fd)

        for p in md.polygons:
            p.use_smooth = True

        # todo
        # if it's a color
        if read_bool(fd):
            colors = [read_color(fd) for _ in range(vertices_num)]

            vcolor = md.vertex_colors.new()
            for p in md.polygons:
                for i in p.loop_indices:
                    vcolor.data[i].color = colors[md.loops[i].vertex_index]

            md.vertex_colors.active = vcolor

        # if it's an UV
        if read_bool(fd):
            bm = bmesh.new()
            bm.from_mesh(md)
            bm.faces.ensure_lookup_table()

            uv_layer = bm.loops.layers.uv.new()
            uv = [struct.unpack('<ff', fd.read(8)) for _ in range(vertices_num)]

            for i, f in enumerate(faces):
                face = bm.faces[i]
                face.loops[0][uv_layer].uv = uv[f[0]]
                face.loops[1][uv_layer].uv = uv[f[1]]
                face.loops[2][uv_layer].uv = uv[f[2]]

            bm.to_mesh(md)
            bm.free()

        mat = bpy.data.materials.new(name='Material')

        # if it's a material
        if read_bool(fd):
            mat.name = read_string(fd)
            mat.mirror_color = read_color(fd)
            mat.diffuse_color = read_color(fd)
            mat.specular_color = read_color(fd)
            mat.specular_intensity = 1.0
            selfIllum_color = read_color(fd)
        else:
            mat.diffuse_color = (1, 1, 1)

        # if it's a texture
        if read_bool(fd):
            texture_name = read_string(fd)
            tex = bpy.data.textures.new(texture_name, type='IMAGE')
            slot = mat.texture_slots.add()
            image = load_image(texture_name, path)

            if image:
                tex.image = image
                depth = image.depth

                for uv_text in md.uv_textures.active.data:
                    uv_text.image = image

                slot.texture = tex
                slot.texture_coords = 'UV'
                slot.use_map_color_diffuse = True

                if depth in {32, 128}:
                    slot.use_map_alpha = True
                    tex.use_mipmap = True
                    tex.use_interpolation = True
                    image.use_alpha = True
                    mat.use_transparency = True

        md.materials.append(mat)
        md.update()

        # XSkinShape
        if x_type == "SS":
            groups_num = read_short(fd)
            groups_names = [read_string(fd) for _ in range(groups_num)]
            groups_weights = [[read_float(fd) for _ in range(groups_num)] for _ in range(vertices_num)]
        else:
            groups_names = [parent_node["name"]]
            groups_weights = [[1.0] for _ in range(vertices_num)]

        object = { 'data': md, 'groups_names': groups_names, 'groups_weights': groups_weights }

    if not ndata:
        ndata = {
            "name": x_name,
            "type": x_type,
            "matrix_type": matrix_type,
            "matrix": trmatix,
            "position": pos,
            "rotation": rot,
            "scale": scale,
            "jointorient": jointorient,
            "rotorient": rotorient,
            "parent": parent_node,
            "object": object
        }
        nodes_data.append(ndata)

    for _ in range(children_num):
        read_xnode(fd, parent_node=ndata, armature=armature)

    return ndata


def import_xom3d_mesh(infile):
    global path, nodes_data
    path = os.path.dirname(infile)

    file = open(infile, 'rb')

    file_type = read_string(file)
    assert file_type == "X3D", "Incorrect Xom3D Model format!"

    nodes_data = []
    node_in = read_xnode(file)

    file.close()

    if not node_in:
        return

    armature = node_in["object"]
    scene = bpy.context.scene

    bpy.ops.object.mode_set(mode='OBJECT')
    for node in nodes_data:
        if node["type"] != "SH" and node["type"] != "SS":
            continue

        mesh = bpy.data.objects.new(node["name"], node["object"]["data"])

        # todo: for BM
        parent_bone = armature.pose.bones.get(node["parent"]["name"])

        if parent_bone:
            mesh.matrix_local = parent_bone.matrix
        mesh.parent = armature
        scene.objects.link(mesh)

        modifier = mesh.modifiers.new(type='ARMATURE', name='Armature')
        modifier.object = armature

        groups_names = node["object"]["groups_names"]
        groups_weights = node["object"]["groups_weights"]
        for i, g in enumerate(groups_names):
            vg = mesh.vertex_groups.new(g)
            for v, w in enumerate(groups_weights):
                if w[i] > 0:
                    vg.add([v], w[i], 'REPLACE')

        if node["parent"]["type"] == 'CS':
            parent_bone.xom_child_selector.add().child_name = node["name"]

    bpy.ops.object.mode_set(mode='POSE')

    for b in armature.pose.bones:
        b.rotation_mode = 'XYZ'

        for node in nodes_data:
            if node['name'] == b.name:
                b.xom_type = node['type']
                b.xom_location = node['position']
                b.xom_rotation = node['rotation']
                b.xom_scale = node['scale']

                b.xom_base_location = mathutils.Vector((0, 0, 0))
                b.xom_base_rotation = mathutils.Vector((0, 0, 0))
                b.xom_base_scale = mathutils.Vector((0, 0, 0))

                if b.xom_type == 'BG':
                    b.rotation_euler = node["jointorient"]
                break

    bpy.ops.object.mode_set(mode='OBJECT')
    armature.rotation_euler = [1.5708, 0.0, 3.14159]
    bpy.ops.object.transforms_to_deltas(mode='ROT')

    scene.update()


def import_xom3d_animation(infile, base_not_animation):
    scene = bpy.context.scene

    armature_object = scene.objects.active

    if not armature_object:
        return

    armature = armature_object.data
    if type(armature) != bpy.types.Armature:
        return

    file = open(infile, 'rb')

    bpy.ops.object.mode_set(mode='POSE')

    anim_name = read_string(file)
    maxkey = read_float(file)
    num = read_short(file)
    fps = scene.render.fps
    last_frame = fps * maxkey

    loc_data, rot_data, scale_data, tex_data = {}, {}, {}, {}

    for _ in range(num):
        obj_name = read_string(file)
        prs_type = read_short(file)
        xyz_type = read_short(file)
        keys = read_short(file)

        bone = armature_object.pose.bones.get(obj_name)

        if xyz_type == 0:
            axis = 0
        elif xyz_type == 256:
            axis = 1
        else:
            axis = 2

        animation_data = []

        for _ in range(keys):
            c1 = read_float(file)
            c2 = read_float(file)
            c3 = read_float(file)
            c4 = read_float(file)

            frame = read_float(file) * fps
            value = read_float(file)

            # fix the scale of the worm animation
            if bone and bone.xom_has_base and prs_type == 2308:
                value *= 0.5

            animation_data.append([frame, value, c1, c2, c3, c4])

        # check the correct bone
        if not bone or not bone.xom_type:
            continue

        # todo: for others prs_type
        if prs_type == 258:
            if not loc_data.get(obj_name):
                loc_data[obj_name] = [None, None, None]
            loc_data[obj_name][axis] = animation_data
        elif prs_type == 259:
            if not rot_data.get(obj_name):
                rot_data[obj_name] = [None, None, None]
            rot_data[obj_name][axis] = animation_data
        elif prs_type == 2308:
            if not scale_data.get(obj_name):
                scale_data[obj_name] = [None, None, None]
            scale_data[obj_name][axis] = animation_data
        elif prs_type == 4352:
            tex_data[obj_name] = animation_data

    file.close()

    scene.frame_start = 0
    scene.frame_end = int(last_frame)

    action = bpy.data.actions.new(anim_name)

    # scale
    for bone_name, data in scale_data.items():
        bone = armature_object.pose.bones[bone_name]

        if bone.xom_type == 'BG':
            scale_origin_data = mathutils.Vector((1, 1, 1))
        else:
            scale_origin_data = mathutils.Vector(bone.xom_scale)

        bone_string = "pose.bones[\"{}\"].".format(bone_name)

        if bone_name in action.groups.keys():
            group = action.groups[bone_name]
        else:
            group = action.groups.new(name=bone_name)

        for axis in (0, 1, 2):
            frames_data = data[axis]

            if not frames_data:
                continue

            curve = action.fcurves.new(data_path=bone_string + "scale", index=axis)
            curve.group = group

            frames_num = math.ceil((max(frames_data, key=lambda _f: _f[0])[0])) + 1
            for k in range(frames_num):
                val = 1.0 + bone.xom_base_scale[axis] + calculate_frame_value(k, frames_data) - scale_origin_data[axis]
                curve.keyframe_points.add(1)
                curve.keyframe_points[k].co = k, val
                curve.keyframe_points[k].interpolation = 'LINEAR'

    # rotation
    for bone_name, data in rot_data.items():
        bone = armature_object.pose.bones[bone_name]

        if bone.xom_type == 'BG':
            rot_origin_data = mathutils.Euler((0, 0, 0))
        else:
            rot_origin_data = mathutils.Vector(bone.xom_rotation)

        bone_string = "pose.bones[\"{}\"].".format(bone_name)

        if bone_name in action.groups.keys():
            group = action.groups[bone_name]
        else:
            group = action.groups.new(name=bone_name)

        for axis in (0, 1, 2):
            frames_data = data[axis]

            if not frames_data:
                continue

            curve = action.fcurves.new(data_path=bone_string + "rotation_euler", index=axis)
            curve.group = group

            frames_num = math.ceil((max(frames_data, key=lambda _f: _f[0])[0])) + 1
            for k in range(frames_num):
                val = bone.xom_base_rotation[axis] + calculate_frame_value(k, frames_data) - rot_origin_data[axis]
                curve.keyframe_points.add(1)
                curve.keyframe_points[k].co = k, val
                curve.keyframe_points[k].interpolation = 'LINEAR'

    # location
    original_frame = scene.frame_current

    for bone_name, data in loc_data.items():
        bone = armature_object.pose.bones[bone_name]

        loc_origin_data = mathutils.Vector(bone.xom_location)
        is_bone = bone.xom_type == 'BG'

        bone_string = "pose.bones[\"{}\"].".format(bone_name)

        if bone_name in action.groups.keys():
            group = action.groups[bone_name]
        else:
            group = action.groups.new(name=bone_name)

        frames_num = 0
        for frames in data:
            if frames:
                max_num = math.ceil(max(frames, key=lambda _f: _f[0])[0])
                if frames_num < max_num:
                    frames_num = max_num
        frames_num += 1

        loc_apply_data = [loc_origin_data.copy() for _ in range(frames_num)]

        curves = [action.fcurves.new(data_path=bone_string + "location", index=axis) for axis in (0, 1, 2)]

        for axis in (0, 1, 2):
            frames_data = data[axis]
            curves[axis].group = group

            if not frames_data:
                continue

            for k in range(frames_num):
                loc_apply_data[k][axis] = bone.xom_base_location[axis] + calculate_frame_value(k, frames_data)

        for k in range(frames_num):
            scene.frame_set(k)

            vector1 = loc_origin_data.copy()
            vector2 = loc_apply_data[k].copy()

            if bone.parent:
                x_ax = bone.parent.x_axis
                y_ax = bone.parent.y_axis
                z_ax = bone.parent.z_axis

                vec1 = vector1.copy()
                vec2 = vector2.copy()

                for i in (0, 1, 2):
                    vector1[i] = vec1[0] * x_ax[i] + vec1[1] * y_ax[i] + vec1[2] * z_ax[i]
                    vector2[i] = vec2[0] * x_ax[i] + vec2[1] * y_ax[i] + vec2[2] * z_ax[i]

            dif_vec = vector2 - vector1

            x_ax = bone.x_axis
            y_ax = bone.y_axis
            z_ax = bone.z_axis

            # todo: fix that
            if x_ax.length == 0.0:
                x_ax = mathutils.Vector((1, 0, 0))
            if y_ax.length == 0.0:
                y_ax = mathutils.Vector((0, 1, 0))
            if z_ax.length == 0.0:
                z_ax = mathutils.Vector((0, 0, 1))

            if is_bone:
                vec = dif_vec.copy()
                dif_vec[0] = vec[0] * x_ax[0] + vec[1] * x_ax[1] + vec[2] * x_ax[2]
                dif_vec[1] = vec[0] * y_ax[0] + vec[1] * y_ax[1] + vec[2] * y_ax[2]
                dif_vec[2] = vec[0] * z_ax[0] + vec[1] * z_ax[1] + vec[2] * z_ax[2]
            else:
                d = x_ax[0] * y_ax[1] * z_ax[2] + x_ax[1] * y_ax[2] * z_ax[0] + x_ax[2] * y_ax[0] * z_ax[1] - x_ax[2] * \
                    y_ax[1] * z_ax[0] - x_ax[0] * y_ax[2] * z_ax[1] - x_ax[1] * y_ax[0] * z_ax[2]
                d1 = dif_vec[0] * y_ax[1] * z_ax[2] + x_ax[1] * y_ax[2] * dif_vec[2] + x_ax[2] * dif_vec[1] * z_ax[1] - \
                     x_ax[2] * y_ax[1] * dif_vec[2] - dif_vec[0] * y_ax[2] * z_ax[1] - x_ax[1] * dif_vec[1] * z_ax[2]
                d2 = x_ax[0] * dif_vec[1] * z_ax[2] + dif_vec[0] * y_ax[2] * z_ax[0] + x_ax[2] * y_ax[0] * dif_vec[2] - \
                     x_ax[2] * dif_vec[1] * z_ax[0] - x_ax[0] * y_ax[2] * dif_vec[2] - dif_vec[0] * y_ax[0] * z_ax[2]
                d3 = x_ax[0] * y_ax[1] * dif_vec[2] + x_ax[1] * dif_vec[1] * z_ax[0] + dif_vec[0] * y_ax[0] * z_ax[1] - \
                     dif_vec[0] * y_ax[1] * z_ax[0] - x_ax[0] * dif_vec[1] * z_ax[1] - x_ax[1] * y_ax[0] * dif_vec[2]

                dif_vec[0] = d1 / d
                dif_vec[1] = d2 / d
                dif_vec[2] = d3 / d

            for axis in (0, 1, 2):
                curves[axis].keyframe_points.add(1)
                curves[axis].keyframe_points[k].co = k, dif_vec[axis]
                curves[axis].keyframe_points[k].interpolation = 'LINEAR'

    scene.frame_set(original_frame)

    # texture animation (CS)
    for bone_name, data in tex_data.items():
        bone = armature_object.pose.bones[bone_name]

        for i, c in enumerate(bone.xom_child_selector):
            mesh = bpy.context.scene.objects.get(c.child_name)

            if not c:
                continue

            mesh_action = bpy.data.actions.new(mesh.name)
            group = mesh_action.groups.new(name=mesh.name)

            curve1 = mesh_action.fcurves.new(data_path='hide')
            curve1.group = group

            curve2 = mesh_action.fcurves.new(data_path='hide_render')
            curve2.group = group

            for d in data:
                val = math.ceil(bone.xom_base_tex + d[1]) - 1

                curve1.keyframe_points.add(1)
                curve2.keyframe_points.add(1)

                curve1.keyframe_points[-1].co = d[0], val != i
                curve2.keyframe_points[-1].co = d[0], val != i

                curve1.keyframe_points[-1].interpolation = 'CONSTANT'
                curve2.keyframe_points[-1].interpolation = 'CONSTANT'

            if anim_name == 'Base' and base_not_animation:
                bpy.data.actions.remove(mesh_action)
            else:
                mesh.animation_data_clear()
                mesh.animation_data_create().action = mesh_action

    # save base data
    if anim_name == 'Base':
        for bone_name, data in scale_data.items():
            bone = armature_object.pose.bones[bone_name]
            bone.xom_has_base = True

            vec = mathutils.Vector((0, 0, 0))
            for axis in (0, 1, 2):
                if data[axis]:
                    vec[axis] = data[axis][0][1] * 0.5
            bone.xom_base_scale = vec

        for bone_name, data in rot_data.items():
            bone = armature_object.pose.bones[bone_name]
            bone.xom_has_base = True

            vec = mathutils.Vector((0, 0, 0))
            for axis in (0, 1, 2):
                if data[axis]:
                    vec[axis] = data[axis][0][1]
            bone.xom_base_rotation = vec

        for bone_name, data in loc_data.items():
            bone = armature_object.pose.bones[bone_name]
            bone.xom_has_base = True

            vec = mathutils.Vector((0, 0, 0))
            for axis in (0, 1, 2):
                if data[axis]:
                    vec[axis] = data[axis][0][1]
            bone.xom_base_location = vec

        for bone_name, data in tex_data.items():
            bone = armature_object.pose.bones[bone_name]
            bone.xom_has_base = True
            bone.xom_base_tex = data[0][1]

    if anim_name == 'Base' and base_not_animation:
        bpy.data.actions.remove(action)
    else:
        armature_object.animation_data_clear()
        armature_object.animation_data_create().action = action

    bpy.ops.object.mode_set(mode='OBJECT')


def import_xom3d_file(self, filepath, base_not_animation):
    filepath_lc = filepath.lower()
    if filepath_lc.endswith('.xom3d'):
        import_xom3d_mesh(filepath)
    elif filepath_lc.endswith('.xac'):
        import_xom3d_animation(filepath, base_not_animation)


# -----------------------------------------------------------------------------
# Operator

class IMPORT_OT_xom3d(bpy.types.Operator):
    """Import Xom 3D Model from XomView format"""
    bl_idname = "import_scene.xom3d"
    bl_label = "Import Xom 3D Model / Animation"
    bl_region_type = "WINDOW"
    bl_options = {'UNDO'}

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    filepath = StringProperty(
        name="File Path",
        subtype='FILE_PATH',
        description="File path used for importing the XMD3D/XAC file",
        maxlen=1024,
        default="",
        options={'HIDDEN'}
    )
    filter_glob = StringProperty(
        default="*.xom3d;*.xac",
        options={'HIDDEN'},
    )

    base_not_animation = BoolProperty(
        name="Only data for Base",
        description="Do not create Base animation",
        default=True
    )

    def execute(self, context):
        import_xom3d_file(self, self.filepath, self.base_not_animation)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# -----------------------------------------------------------------------------
# Register

class XomChildItem(bpy.types.PropertyGroup):
    child_name = bpy.props.StringProperty(name="XOM Child Name", options={'HIDDEN'})


def import_xom3d_button(self, context):
    self.layout.operator(IMPORT_OT_xom3d.bl_idname,
                         text="Xom 3DModel (.xom3d, .xac)")


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(import_xom3d_button)
    bpy.types.PoseBone.xom_type = bpy.props.StringProperty(name="XOM Type", options={'HIDDEN'})
    bpy.types.PoseBone.xom_location = bpy.props.FloatVectorProperty(name="XOM Location", options={'HIDDEN'})
    bpy.types.PoseBone.xom_rotation = bpy.props.FloatVectorProperty(name="XOM Rotation", options={'HIDDEN'})
    bpy.types.PoseBone.xom_scale = bpy.props.FloatVectorProperty(name="XOM Scale", options={'HIDDEN'})

    bpy.types.PoseBone.xom_child_selector = bpy.props.CollectionProperty(type=XomChildItem, options={'HIDDEN'})

    bpy.types.PoseBone.xom_has_base = bpy.props.BoolProperty(name="XOM Has Base", options={'HIDDEN'})
    bpy.types.PoseBone.xom_base_location = bpy.props.FloatVectorProperty(name="XOM Base Location", options={'HIDDEN'})
    bpy.types.PoseBone.xom_base_rotation = bpy.props.FloatVectorProperty(name="XOM Base Rotation", options={'HIDDEN'})
    bpy.types.PoseBone.xom_base_scale = bpy.props.FloatVectorProperty(name="XOM Base Scale", options={'HIDDEN'})
    bpy.types.PoseBone.xom_base_tex = bpy.props.FloatProperty(name="XOM Base Texture", options={'HIDDEN'})


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(import_xom3d_button)


if __name__ == "__main__":
    register()
