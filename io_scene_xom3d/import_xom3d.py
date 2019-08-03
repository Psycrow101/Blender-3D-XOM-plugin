import bpy
import bmesh
import os
import struct
import math
import mathutils
from bpy_extras.image_utils import load_image
from bpy_extras import node_shader_utils


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
    return fd.read(1)[0] / 255.0, fd.read(1)[0] / 255.0, fd.read(1)[0] / 255.0, 1.0


def read_xnode(fd, context, parent_node=None, armature=None):
    global path, nodes_data

    x_type = read_string(fd)

    # XNodeInterion
    # nothing to do here
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
        trmatix = trans_matrix(pos) @ rotate_matrix(rot) @ scale_matrix(scale)
    elif matrix_type == 3:
        pos = mathutils.Vector(read_vector(fd))
        rot = mathutils.Euler(read_vector(fd))
        jointorient = mathutils.Euler(read_vector(fd))
        rotorient = read_vector(fd)
        scale = read_vector(fd)

        # todo: testing
        trmatix = trans_matrix(pos) @ rotate_matrix(rot) @ scale_matrix(scale)
    else:
        trmatix = trans_matrix((0, 0, 0))
        pos = mathutils.Vector((0, 0, 0))
        rot = mathutils.Euler((0, 0, 0))
        scale = mathutils.Vector((1, 1, 1))

    children_num = 0
    obj = None

    # XInteriorNode
    # create armature
    if x_type == "IN":
        children_num = read_short(fd)

        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection

        armature = bpy.data.armatures.new(x_name)
        obj = bpy.data.objects.new(x_name, armature)
        obj.show_in_front = True

        collection.objects.link(obj)
        view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')

    # XGroup or XGroupObject
    # create bone
    elif x_type in ("GR", "GO"):
        children_num = read_short(fd)

        obj = armature.edit_bones.new(x_name)
        obj.head = (0, 0, 0)
        obj.tail = (0, 0, 1)

        if type(parent_node["object"]) == bpy.types.EditBone:
            obj.parent = parent_node["object"]
            obj.matrix = parent_node["object"].matrix @ trmatix
        else:
            obj.matrix = trmatix

    # XGroupShape or XSkin or XBinModifier
    # skip node
    elif x_type in ("GS", "SK", "BM"):
        children_num = read_short(fd)
        ndata = parent_node

    # XBoneGroup
    # create nothing
    elif x_type == "BG":
        children_num = read_short(fd)

    # XBone
    # create bone from parent data
    elif x_type == "BO":
        children_num = read_short(fd)

        obj = armature.edit_bones.new(parent_node["name"])
        obj.head = (0, 0, 0)
        obj.tail = (0, 0, 1)

        obj.matrix = trmatix.inverted_safe()

        if type(parent_node["parent"]["object"]) == bpy.types.EditBone:
            obj.parent = parent_node["parent"]["object"]

        # todo: testing
        for node in nodes_data:
            if node == parent_node:
                node["object"] = obj

    # XChildSelector
    # create bone (used only as a container of children)
    elif x_type == "CS":
        children_num = read_short(fd)

        obj = armature.edit_bones.new(x_name)
        obj.head = (0, 0, 0)
        obj.tail = (0, 0, 1)

        if type(parent_node["object"]) == bpy.types.EditBone:
            obj.parent = parent_node["object"]
            obj.matrix = parent_node["object"].matrix.copy()

    # XShape or XSkinShape
    # create mesh
    elif x_type in ("SH", "SS"):
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

        # todo: testing
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

        mat = bpy.data.materials.new(name=x_name)
        mat_wrap = node_shader_utils.PrincipledBSDFWrapper(mat, is_readonly=False)

        # if it's a material
        # todo:
        if read_bool(fd):
            mat.name = read_string(fd)
            _mat_mirror_color = read_color(fd)
            _mat_diffuse_color = read_color(fd)
            _mat_specular_color = read_color(fd)
            # mat.specular_intensity = 1.0
            _mat_selfIllum_color = read_color(fd)
        else:
            _mat_diffuse_color = (1, 1, 1)

        # if it's a texture
        # todo:
        if read_bool(fd):
            texture_name = read_string(fd)
            tex = bpy.data.textures.get(texture_name)
            if not tex:
                tex = bpy.data.textures.new(texture_name, type='IMAGE')
                image = load_image(texture_name, path)
            else:
                image = bpy.data.images.get(texture_name)

            if image:
                tex.image = image
                depth = image.depth

                nodetex = mat_wrap.base_color_texture
                nodetex.image = image
                nodetex.texcoords = 'UV'

                node_tree = mat.node_tree
                nodes = node_tree.nodes

                texture_node = nodes.get('Image Texture')
                uv_map_node = nodes.new('ShaderNodeUVMap')
                mapping_node = nodes.new('ShaderNodeMapping')

                mapping_node.vector_type = 'TEXTURE'

                node_tree.links.new(uv_map_node.outputs[0], mapping_node.inputs[0])
                node_tree.links.new(mapping_node.outputs[0], texture_node.inputs[0])

                if depth in {32, 128}:
                    mat.blend_method = 'BLEND'

                    output_node = nodes.get('Material Output')
                    bsdf_node = nodes.get('Principled BSDF')
                    mix_shader_node = nodes.new('ShaderNodeMixShader')
                    transparent_node = nodes.new('ShaderNodeBsdfTransparent')

                    node_tree.links.new(texture_node.outputs[1], mix_shader_node.inputs[0])
                    node_tree.links.new(transparent_node.outputs[0], mix_shader_node.inputs[1])
                    node_tree.links.new(bsdf_node.outputs[0], mix_shader_node.inputs[2])

                    node_tree.links.new(mix_shader_node.outputs[0], output_node.inputs[0])

                if parent_node['type'] == 'CS':
                    mat.blend_method = 'BLEND'

        md.materials.append(mat)
        md.validate()
        md.update()

        # XSkinShape
        # set up bones weights
        if x_type == "SS":
            groups_num = read_short(fd)
            groups_names = [read_string(fd) for _ in range(groups_num)]
            groups_weights = [[read_float(fd) for _ in range(groups_num)] for _ in range(vertices_num)]
        else:
            groups_names = [parent_node["name"]]
            groups_weights = [[1.0] for _ in range(vertices_num)]

        obj = {'data': md, 'groups_names': groups_names, 'groups_weights': groups_weights}

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
            "object": obj
        }
        nodes_data.append(ndata)

    # next children
    for _ in range(children_num):
        read_xnode(fd, context, parent_node=ndata, armature=armature)

    # return root node
    return ndata


def load_xom3d_mesh(filepath, context, global_matrix):
    global path, nodes_data
    path = os.path.dirname(filepath)

    with open(filepath, 'rb') as fd:
        file_type = read_string(fd)
        assert file_type == "X3D", "Incorrect Xom3D Model format!"

        nodes_data = []
        node_in = read_xnode(fd, context)

    if not node_in:
        return

    view_layer = context.view_layer
    armature_object = node_in["object"]

    # set up mesh objects
    bpy.ops.object.mode_set(mode='OBJECT')
    for node in nodes_data:
        if node["type"] != "SH" and node["type"] != "SS":
            continue

        mesh_object = bpy.data.objects.new(node["name"], node["object"]["data"])

        parent_bone = armature_object.pose.bones.get(node["parent"]["name"])
        if parent_bone:
            mesh_object.matrix_local = parent_bone.matrix.copy()

        mesh_object.scale = node["parent"]["scale"]

        mesh_object.parent = armature_object
        view_layer.active_layer_collection.collection.objects.link(mesh_object)

        modifier = mesh_object.modifiers.new(type='ARMATURE', name='Armature')
        modifier.object = armature_object

        groups_names = node["object"]["groups_names"]
        groups_weights = node["object"]["groups_weights"]
        for i, g in enumerate(groups_names):
            vg = mesh_object.vertex_groups.new(name=g)
            for v, w in enumerate(groups_weights):
                if w[i] > 0:
                    vg.add([v], w[i], 'REPLACE')

        # remove doubles
        view_layer.objects.active = mesh_object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')

        # append mesh name to XChildSelector
        if node["parent"]["type"] == 'CS':
            parent_bone.xom_child_selector.add().child_name = mesh_object.name
            if len(parent_bone.xom_child_selector) > 1:
                mesh_object.hide_viewport = True
                mesh_object.hide_render = True

    # set up pose bones
    view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode='POSE')
    for b in armature_object.pose.bones:
        b.rotation_mode = 'XYZ'

        for node in nodes_data:
            if node['name'] == b.name:
                b.xom_type = node['type']
                b.xom_location = node['position']
                b.xom_rotation = node['rotation']
                b.xom_scale = node['scale']

                if b.xom_type == 'BG':
                    b.rotation_euler = node["jointorient"]
                    b.xom_jointorient = b.rotation_euler.copy()

                break

    bpy.ops.object.mode_set(mode='OBJECT')
    armature_object.matrix_world = global_matrix

    for o in view_layer.objects:
        o.select_set(o == armature_object)

    view_layer.update()


def load_xom3d_animation(filepath, context, use_def_pose):
    view_layer = context.view_layer

    armature_object = view_layer.objects.active

    if not armature_object:
        return

    armature = armature_object.data
    if type(armature) != bpy.types.Armature:
        return

    bpy.ops.object.mode_set(mode='POSE')

    # reset pose
    if use_def_pose:
        armature_object.animation_data_clear()
        for b in armature_object.pose.bones:
            b.rotation_euler = mathutils.Euler(b.xom_jointorient)
            b.location.zero()
            b.scale = mathutils.Vector((1, 1, 1))

            # reset Child Selector
            if b.xom_child_selector:
                for i, c in enumerate(b.xom_child_selector):
                    mesh_object = view_layer.objects.get(c.child_name)
                    if mesh_object:
                        mesh_object.animation_data_clear()
                        mesh_object.hide_viewport = (i != 0)
                        mesh_object.hide_render = (i != 0)

    # reset texture offsets
    for obj in view_layer.objects:
        if obj.parent == armature_object:
            node_tree = obj.active_material.node_tree
            node_tree.animation_data_clear()
            mapping_node = node_tree.nodes.get('Mapping')
            if mapping_node:
                mapping_node.translation = mathutils.Vector((0.0, 0.0, 0.0))

    file = open(filepath, 'rb')

    anim_name = read_string(file)
    maxkey = read_float(file)
    num = read_short(file)
    fps = context.scene.render.fps
    last_frame = fps * maxkey

    loc_data, rot_data, scale_data, tex_data, child_data = {}, {}, {}, {}, {}

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

        if prs_type == 1025:
            if view_layer.objects.get(obj_name):
                if not tex_data.get(obj_name):
                    tex_data[obj_name] = [None, None]
                tex_data[obj_name][axis] = animation_data
        # check the correct bone
        elif bone and bone.xom_type:
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
                child_data[obj_name] = animation_data

    file.close()

    context.scene.frame_start = 0
    context.scene.frame_end = int(last_frame)

    action = bpy.data.actions.new(anim_name)

    # scale
    for bone_name, data in scale_data.items():
        bone = armature_object.pose.bones[bone_name]

        if bone.xom_type == 'BG':
            scale_origin_data = mathutils.Vector((1, 1, 1))
        else:
            scale_origin_data = bone.xom_scale

        bone_string = 'pose.bones["{}"].'.format(bone_name)

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
            rot_origin_data = bone.xom_rotation

        bone_string = 'pose.bones["{}"].'.format(bone_name)

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

    # fix prevent zero vectors
    temp_scale = {}
    for b in armature_object.pose.bones:
        temp_scale[b], b.scale = b.scale.copy(), mathutils.Vector((1, 1, 1))
    view_layer.update()

    # location
    for bone_name, data in loc_data.items():
        bone = armature_object.pose.bones[bone_name]

        loc_origin_data = mathutils.Vector(bone.xom_location)
        is_bone = bone.xom_type == 'BG'

        bone_string = 'pose.bones["{}"].'.format(bone_name)

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

        if bone.parent:
            p_x_ax = bone.parent.x_axis
            p_y_ax = bone.parent.y_axis
            p_z_ax = bone.parent.z_axis
        else:
            p_x_ax = (1, 0, 0)
            p_y_ax = (0, 1, 0)
            p_z_ax = (0, 0, 1)

        x_ax = bone.x_axis
        y_ax = bone.y_axis
        z_ax = bone.z_axis

        for k in range(frames_num):
            vector1 = loc_origin_data.copy()
            vector2 = loc_apply_data[k].copy()

            if bone.parent:
                vec1 = vector1.copy()
                vec2 = vector2.copy()

                for i in (0, 1, 2):
                    vector1[i] = vec1[0] * p_x_ax[i] + vec1[1] * p_y_ax[i] + vec1[2] * p_z_ax[i]
                    vector2[i] = vec2[0] * p_x_ax[i] + vec2[1] * p_y_ax[i] + vec2[2] * p_z_ax[i]

            dif_vec = vector2 - vector1

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

    for b in armature_object.pose.bones:
        b.scale = temp_scale[b]

    # texture animation
    for obj_name, data in tex_data.items():
        obj = view_layer.objects[obj_name]
        node_tree = obj.active_material.node_tree

        obj_action = bpy.data.actions.new(node_tree.name)
        group = action.groups.new(name=node_tree.name)

        for axis in (0, 1):
            frames_data = data[axis]

            if not frames_data:
                continue

            curve = obj_action.fcurves.new(data_path='nodes["Mapping"].translation', index=axis)
            curve.group = group

            frames_num = math.ceil((max(frames_data, key=lambda _f: _f[0])[0])) + 1
            for k in range(frames_num):
                val = obj.xom_base_tex[axis] + calculate_frame_value(k, frames_data)
                curve.keyframe_points.add(1)
                curve.keyframe_points[k].co = k, -val
                curve.keyframe_points[k].interpolation = 'LINEAR'

        node_tree.animation_data_create().action = obj_action

    # child selector animation (CS)
    for bone_name, data in child_data.items():
        bone = armature_object.pose.bones[bone_name]

        for i, c in enumerate(bone.xom_child_selector):
            mesh_object = view_layer.objects.get(c.child_name)
            if not mesh_object:
                continue

            mesh_action = bpy.data.actions.new(mesh_object.name)
            group = mesh_action.groups.new(name=mesh_object.name)

            curve1 = mesh_action.fcurves.new(data_path='hide_viewport')
            curve1.group = group

            curve2 = mesh_action.fcurves.new(data_path='hide_render')
            curve2.group = group

            for d in data:
                val = math.ceil(bone.xom_base_cs + d[1]) - 1

                curve1.keyframe_points.add(1)
                curve2.keyframe_points.add(1)

                curve1.keyframe_points[-1].co = d[0], val != i
                curve2.keyframe_points[-1].co = d[0], val != i

                curve1.keyframe_points[-1].interpolation = 'CONSTANT'
                curve2.keyframe_points[-1].interpolation = 'CONSTANT'

            mesh_object.animation_data_create().action = mesh_action

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

        for obj_name, data in tex_data.items():
            obj = view_layer.objects[obj_name]
            vec = mathutils.Vector((0, 0, 0))
            for axis in (0, 1):
                if data[axis]:
                    vec[axis] = data[axis][0][1]
            obj.xom_base_tex = vec

        for bone_name, data in child_data.items():
            bone = armature_object.pose.bones[bone_name]
            bone.xom_has_base = True
            bone.xom_base_cs = data[0][1]

    armature_object.animation_data_create().action = action

    bpy.ops.object.mode_set(mode='OBJECT')


def load(context, filepath, *, use_def_pose, global_matrix=None):
    filepath_lc = filepath.lower()
    if filepath_lc.endswith('.xom3d'):
        load_xom3d_mesh(filepath, context, global_matrix)
    elif filepath_lc.endswith('.xac'):
        load_xom3d_animation(filepath, context, use_def_pose)

    return {'FINISHED'}
