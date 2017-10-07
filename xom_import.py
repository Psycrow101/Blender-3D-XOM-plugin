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
    "name": "Import Xom 3DModel",
    "author": "Psycrow, AlexBond",
    "version": (1, 0, 1),
    "blender": (2, 60, 0),
    "location": "File > Import > Import Xom 3DModel (.xom3d)",
    "description": "Import Xom 3DModel from XomView format (.xom3d).",
    "warning": "",
    "wiki_url": "",
    "category": "Import-Export",
}


def trans_matrix(v):
    return mathutils.Matrix.Translation(v)

def scale_matrix(v):
    mat = mathutils.Matrix.Identity(4) 
    mat[0][0], mat[1][1], mat[2][2] = v[0],v[1],v[2]
    return mat

def rotate_matrix(v):
    rot = mathutils.Euler(v,'XYZ').to_quaternion()
    return rot.to_matrix().to_4x4()

def findBezier(x,a1,b1,c1,d1,a2,b2,c2,d2):

    def BezierF(t,a,b,c,d):
        t1 = 1 - t
        return t1*t1*t1*a + 3*t*t1*t1*b + 3*t*t*t1*c + t*t*t*d

    def find(t1,t2,t3):
        xdiv = BezierF(t2,a1,b1,c1,d1)
        if abs(xdiv-x) < 0.001:
            return t2
        
        if xdiv > x:
            return find(t1,(t1+t2)/2,t2)
        
        return find(t2,(t2+t3)/2,t3)
   
    return BezierF(find(0,0.5,1.0),a2,b2,c2,d2)

def read_byte(file):
    return struct.unpack('<b', file.read(1))[0]

def read_float(file):
   return struct.unpack('<f', file.read(4))[0]
   
def read_short(file):
   return struct.unpack('<h', file.read(2))[0]

def read_bool(file):
    return struct.unpack('<?', file.read(1))[0]

def read_string(file):
    str = ''
    byte = file.read(1)
    while byte != b'\x00':
        str += (byte.decode('ascii'))
        byte = file.read(1)
    return str

def read_vector(file):
    return struct.unpack('3f', file.read(12))

def read_short_vector(file):
    return struct.unpack('<hhh', file.read(6))

def read_color(file):
    return (file.read(1)[0] / 255.0, file.read(1)[0] / 255.0, file.read(1)[0] / 255.0)

def read_xnode(file, parentNode = None, parentArmature = None):
    global path, nodesData

    XType = read_string(file)
        
    if XType == "XN": return None
    
    XName = read_string(file)
           
    matrixtype = read_byte(file)
    trmatix = None
    pos = None
    rot = None
    scale = None
    jointorient = None
    rotorient = None
    
    if matrixtype == 1:
        trmatix = mathutils.Matrix(( (read_float(file), read_float(file), read_float(file), 0),
		  (read_float(file), read_float(file), read_float(file), 0),
		  (read_float(file), read_float(file), read_float(file), 0),
		  (read_float(file), read_float(file), read_float(file), 1.0))).transposed()
        pos = trmatix.to_translation()
        rot = trmatix.to_euler()
    elif matrixtype == 2:
        pos = mathutils.Vector(read_vector(file))
        rot = mathutils.Euler(read_vector(file))
        scale = read_vector(file)
        trmatix = trans_matrix(pos) * rotate_matrix(rot) * scale_matrix(scale)
    elif matrixtype == 3:
        pos = mathutils.Vector(read_vector(file))
        rot = mathutils.Euler(read_vector(file))
        jointorient = read_vector(file)
        rotorient = read_vector(file)
        scale = read_vector(file)
        trmatix = trans_matrix(pos) * rotate_matrix(rot) * scale_matrix(scale) # !!!
    else:
        trmatix = trans_matrix([0, 0, 0])
        pos = mathutils.Vector([0, 0, 0])
        rot = mathutils.Euler([0, 0, 0])
        scale = mathutils.Vector([1, 1, 1])
        
    childsNum = 0
    object = None
    armature = None

    if XType == "IN":
        childsNum = read_short(file)
        
        armature = bpy.data.armatures.new(XName)
        object = bpy.data.objects.new(XName, armature)
        object.show_x_ray = True
                
        bpy.context.scene.objects.link(object)
        bpy.context.scene.objects.active = object
        object.select = True
             
        bpy.ops.object.mode_set(mode='EDIT')
        
    elif XType == "GO" or XType == "GR":
        childsNum = read_short(file)
        
        armature = parentArmature
        
        object = armature.edit_bones.new(XName)
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)
                        
        if type(parentNode["object"]) == bpy.types.EditBone:
            object.parent = parentNode["object"]
            object.matrix = parentNode["object"].matrix * trmatix
        else:
            object.matrix = trmatix
    
    elif XType == "GS" or XType == "SK":
        childsNum = read_short(file)
                    
        for x in range(childsNum):
            read_xnode(file, parentNode = parentNode, parentArmature = parentArmature)
    
        return parentNode
    
    elif XType == "BG":
        childsNum = read_short(file)            
        armature = parentArmature
    
    elif XType == "BO":
        childsNum = read_short(file)
            
        armature = parentArmature
                
        XName = XName[:len(XName) - 5]
        
        object = armature.edit_bones.new(XName)
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)
                             
        object.matrix = trmatix.inverted_safe()
        
        bg_object = object
        
        if type(parentNode["parent"]["object"]) == bpy.types.EditBone:
            object.parent = parentNode["parent"]["object"]
        
        for node in nodesData:
            if node == parentNode:
                node["object"] = object # !!!
    
    elif XType == "CS": #доделать
        childsNum = read_short(file)
        
        armature = parentArmature
        
        object = armature.edit_bones.new(XName)
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)
                
        if type(parentNode["object"]) == bpy.types.EditBone:
            object.parent = parentNode["object"]
            object.matrix = parentNode["object"].matrix
    
    elif XType == "BM": #доделать
        childsNum = read_short(file)
        
        armature = parentArmature
        
        object = armature.edit_bones.new(XName)
        object.head = (0, 0, 0)
        object.tail = (0, 0, 1)
                
        if type(parentNode["object"]) == bpy.types.EditBone:
            object.parent = parentNode["object"]
            object.matrix = parentNode["object"].matrix

    elif XType == "SH" or XType == "SS":
        mesh = bpy.data.meshes.new(XName)
        
        faces = []
        facesNum = read_short(file)
        for i in range(facesNum):
            faces.append(read_short_vector(file))
            
        verts = []
        vertsNum = read_short(file)
        for i in range(vertsNum):
            verts.append(read_vector(file))
            
        mesh.from_pydata(verts, [], faces)
            
        isNormal = read_bool(file)
        if isNormal:
            for i in range(vertsNum):
                mesh.vertices[i].normal = read_vector(file)
                
        for p in mesh.polygons:
            p.use_smooth = True
        
        # Протестировать с такими мешами
        isColor = read_bool(file)
        if isColor:
            colors = []
            for i in range(vertsNum):
                colors.append(read_color(file))
                
            vcolor = mesh.vertex_colors.new()
            for p in mesh.polygons:
                for i in p.loop_indices:
                    vcolor.data[i].color = colors[mesh.loops[i].vertex_index]   
      
            m.vertex_colors.active = vcolor  
            #bpy.ops.object.mode_set(mode='VERTEX_PAINT')
                
        isUV = read_bool(file)
        if isUV:
            uv = []
            for i in range(vertsNum):
                uv.append(struct.unpack('<ff', file.read(8)))
                
            mesh.uv_textures.new(XName)
            bm = bmesh.new()
            bm.from_mesh(mesh)
        
            uv_layer = bm.loops.layers.uv[0]
            bm.faces.ensure_lookup_table()
            for f in range(facesNum):
                bm.faces[f].loops[0][uv_layer].uv = uv[faces[f][0]]
                bm.faces[f].loops[1][uv_layer].uv = uv[faces[f][1]]
                bm.faces[f].loops[2][uv_layer].uv = uv[faces[f][2]]
        
            bm.to_mesh(mesh)
        
        # Доделать материалы
        mat = bpy.data.materials.new(name="Material")
        
        isMat = read_bool(file)
        if isMat:
            mat.name = read_string(file)
            mat.mirror_color = read_color(file)
            mat.diffuse_color = read_color(file)
            mat.specular_color = read_color(file)
            mat.specular_intensity = 1.0
            selfIllum_color = read_color(file)
        else:
            mat.diffuse_color = (1, 1, 1)  
        
        isTexture = read_bool(file)
        if isTexture:
            textName = read_string(file)       
            tex = bpy.data.textures.new(textName, type='IMAGE')
            slot = mat.texture_slots.add()
            image = load_image(textName, path)
            
            if image is not None:
                tex.image = image
                depth = image.depth
                for uv_text in mesh.uv_textures[0].data:  
                    uv_text.image = image
                    
                #for area in bpy.context.screen.areas:
                #    if area.type == 'VIEW_3D':
                #        for space in area.spaces:
                #            if space.type == 'VIEW_3D':
                #                space.viewport_shade = 'TEXTURED'
                    
            slot.texture = tex
            slot.texture_coords = 'UV'
            slot.use_map_color_diffuse = True
            
            if depth in {32, 128}:
                slot.use_map_alpha = True
                tex.use_mipmap = True
                tex.use_interpolation = True
                image.use_alpha = True
                mat.use_transparency = True
            
        mesh.materials.append(mat)
        mesh.update()
        
        object = [mesh]
                
        if XType == "SS":
            bonesNum = read_short(file)
            object.append(bonesNum)
            
            groupsName = []
            for b in range(bonesNum):
                groupsName.append(read_string(file))
            object.append(groupsName)
                                
            weights = []
            for vert_index in range(vertsNum):
                weightsGroup = []
                for b in range(bonesNum):
                    weightsGroup.append(read_float(file))
                weights.append(weightsGroup)
            object.append(weights)
                         
    ndata = dict()
    ndata["name"] = XName
    ndata["type"] = XType
    ndata["matrix_type"] = matrixtype
    ndata["matrix"] = trmatix
    ndata["position"] = pos
    ndata["rotation"] = rot
    ndata["scale"] = scale
    ndata["jointorient"] = jointorient
    ndata["rotorient"] = rotorient
    ndata["parent"] = parentNode
    ndata["object"] = object
    nodesData.append(ndata)
    
    for x in range(childsNum):
        read_xnode(file, parentNode = ndata, parentArmature = armature)
    
    return ndata
	
def xom3dimport(infile, xacpath):
	global path, nodesData
	
	file = open(infile, 'rb')
	path = os.path.dirname(infile)
	
	filetype = read_string(file)

	nodesData = []

	if filetype == "X3D":
		nodeIN = read_xnode(file)
												
		bpy.ops.object.mode_set(mode='OBJECT')
		for node in nodesData:
			if node["type"] == "SH":
				object = bpy.data.objects.new(node["name"], node["object"][0])
				bpy.context.scene.objects.link(object)
				object.matrix_local = nodeIN["object"].pose.bones.get(node["parent"]["name"]).matrix

				object.vertex_groups.new(node["parent"]["name"])
				for vert_index in range(len(object.data.vertices)): 
					object.vertex_groups[0].add([vert_index], 1.0, 'REPLACE')
					
				bpy.ops.object.select_all(action='DESELECT')
				object.select = True
				nodeIN["object"].select = True
				bpy.context.scene.objects.active = nodeIN["object"]
				bpy.ops.object.parent_set(type='ARMATURE', keep_transform=False)
				
			if node["type"] == "SS":
				object = bpy.data.objects.new(node["name"], node["object"][0])
				bpy.context.scene.objects.link(object)
				
				for b in range(node["object"][1]):
					object.vertex_groups.new(node["object"][2][b]) # !!
					
				for vert_index in range(len(object.data.vertices)):
					for b in range(node["object"][1]):                    
						if node["object"][3][vert_index][b] > 0:
							object.vertex_groups[b].add([vert_index], node["object"][3][vert_index][b], 'REPLACE')
				
				bpy.ops.object.select_all(action='DESELECT')
				object.select = True
				nodeIN["object"].select = True
				bpy.context.scene.objects.active = nodeIN["object"]
				bpy.ops.object.parent_set(type='ARMATURE', keep_transform=False)
				
		for b in nodeIN["object"].pose.bones:
			b.rotation_mode = 'XYZ'
			
		bpy.ops.object.mode_set(mode='POSE')
		for node in nodesData:
			if node["type"] == "BG":            
				obj = nodeIN["object"].pose.bones.get(node["name"]) # !!!
				obj.rotation_euler = node["jointorient"]
				obj.keyframe_insert(data_path="rotation_euler", frame=0, index=-1)
			  
		bpy.ops.object.mode_set(mode='OBJECT')
		nodeIN["object"].rotation_euler = [1.5708, 0.0, 3.14159]
		bpy.ops.object.transforms_to_deltas(mode='ROT')
		bpy.context.scene.update()
		
		file = open(path + '/' + xacpath, 'rb') # check
		
		bpy.context.scene.objects.active = nodeIN["object"]
		bpy.ops.object.mode_set(mode='POSE')
		
		animName = read_string(file) #!
		maxkey = read_float(file)
		num = read_short(file)

		scene = bpy.context.scene
		fps = scene.render.fps
		scene.frame_start = 0
		scene.frame_end = fps * maxkey
			
		loc_data = dict()
		rot_data = dict()
		scale_data = dict()
				
		for i in range(num):
			objname = read_string(file)
			PRSType = read_short(file)
			XYZType = read_short(file)
			keys = read_short(file)
						
			if XYZType == 0:
				index = 0
			elif XYZType == 256:
				index = 1
			else:
				index = 2
				
			# И для других PRSType
			if PRSType == 258:
				if loc_data.get(objname) is None:
					loc_data[objname] = [None, None, None]
			elif PRSType == 259:
				if rot_data.get(objname) is None:
					rot_data[objname] = [None, None, None]
			elif PRSType == 2308:
				if scale_data.get(objname) is None:
					scale_data[objname] = [None, None, None]
					
			anim_data = [None for x in range(scene.frame_end + 1)]
			
			for node in nodesData:
				if node["type"] == "BG":
					continue
				
				if node["name"] == objname:
								 
					for j in range(keys):        
						c1 = read_float(file)
						c2 = read_float(file)
						c3 = read_float(file)
						c4 = read_float(file)
			
						frame = read_float(file) * fps
						value = read_float(file)
			
						anim_data[int(frame)] = [c1, c2, c3, c4, value]
				
					if PRSType == 258:
						loc_data[objname][index] = anim_data
					elif PRSType == 259:
						rot_data[objname][index] = anim_data
					elif PRSType == 2308:
						scale_data[objname][index] = anim_data
							
					break
				
		
		for i in rot_data:
			obj = nodeIN["object"].pose.bones.get(i) # !!!
			
			rot_origin_data = mathutils.Euler((0, 0, 0))
			for node in nodesData:
				if node["type"] == "BG" or node["type"] == "BO":
					continue         
				if node["name"] == i:
					rot_origin_data = node["rotation"]
					break
					
			for x in range(3):
				if rot_data[i][x] is not None:
					for f in range(scene.frame_end + 1):
						if rot_data[i][x][f] is None:
							key1 = None
							key2 = None
											
							for k in range(0, f):
								if rot_data[i][x][k] is not None:
									key1 = rot_data[i][x][k]
									frame1 = k
								
							for k in range(scene.frame_end, f, -1):
								if rot_data[i][x][k] is not None:
									key2 = rot_data[i][x][k]
									frame2 = k
								
							if key1 is not None and key2 is not None:
								a = frame1 + (frame2 - frame1) * math.cos(key1[3]) * key1[2] / 3
								b = key1[4] + (key2[4] - key1[4]) * math.sin(key1[3]) * key1[2] / 3
								c = frame2 - (frame2 - frame1) * math.cos(key2[1]) * key2[0] / 3
								d = key2[4] - (key2[4] - key1[4]) * math.sin(key2[1]) * key2[0] / 3
								
								obj.rotation_euler[x] = findBezier(f, frame1, a, c, frame2, key1[4], b, d, key2[4]) - rot_origin_data[x]
								obj.keyframe_insert(data_path="rotation_euler", frame=f, index=x)
						else:
							obj.rotation_euler[x] = rot_data[i][x][f][4] - rot_origin_data[x]
							obj.keyframe_insert(data_path="rotation_euler", frame=f, index=x)
							
		for i in scale_data:
			obj = nodeIN["object"].pose.bones.get(i) # !!!
			
			scale_origin_data = mathutils.Vector((1, 1, 1))
			for node in nodesData:
				if node["type"] == "BG" or node["type"] == "BO":
					continue         
				if node["name"] == i:
					scale_origin_data = node["scale"]
					break
			
			
			for x in range(3):
				if scale_data[i][x] is not None:
					for f in range(scene.frame_end + 1):
						if scale_data[i][x][f] is None:
							key1 = None
							key2 = None
											
							for k in range(0, f):
								if scale_data[i][x][k] is not None:
									key1 = scale_data[i][x][k]
									frame1 = k
								
							for k in range(scene.frame_end, f, -1):
								if scale_data[i][x][k] is not None:
									key2 = scale_data[i][x][k]
									frame2 = k
								
							if key1 is not None and key2 is not None:
								a = frame1 + (frame2 - frame1) * math.cos(key1[3]) * key1[2] / 3
								b = key1[4] + (key2[4] - key1[4]) * math.sin(key1[3]) * key1[2] / 3
								c = frame2 - (frame2 - frame1) * math.cos(key2[1]) * key2[0] / 3
								d = key2[4] - (key2[4] - key1[4]) * math.sin(key2[1]) * key2[0] / 3
							
								obj.scale[x] = 1.0 + findBezier(f, frame1, a, c, frame2, key1[4], b, d, key2[4]) - scale_origin_data[x]
								obj.keyframe_insert(data_path="scale", frame=f, index=x)
						else:
							obj.scale[x] = scale_data[i][x][f][4]
							obj.keyframe_insert(data_path="scale", frame=f, index=x)
							
		for i in loc_data:
			obj = nodeIN["object"].pose.bones.get(i) # !!!    
			is_bone = False
			for node in nodesData:
				if node["type"] == "BO":
					continue
				
				if node["name"] == i:
					loc_origin_data = node["position"]
					if node["type"] == "BG":
						is_bone = True
					break
				
			loc_apply_data = [loc_origin_data.copy() for f in range(scene.frame_end + 1)] # !!!  
			for x in range(3):
				if loc_data[i][x] is not None:
					for f in range(scene.frame_end + 1):
						if loc_data[i][x][f] is None:
							
							key1 = None
							key2 = None
									 
							for k in range(0, f):
								if loc_data[i][x][k] is not None:
									key1 = loc_data[i][x][k]
									frame1 = k
								
							for k in range(scene.frame_end, f, -1):
								if loc_data[i][x][k] is not None:
									key2 = loc_data[i][x][k]
									frame2 = k
								  
							if key1 is not None and key2 is not None:
								a = frame1 + (frame2 - frame1) * math.cos(key1[3]) * key1[2] / 3
								b = key1[4] + (key2[4] - key1[4]) * math.sin(key1[3]) * key1[2] / 3
								c = frame2 - (frame2 - frame1) * math.cos(key2[1]) * key2[0] / 3
								d = key2[4] - (key2[4] - key1[4]) * math.sin(key2[1]) * key2[0] / 3
							
								loc_apply_data[f][x] = findBezier(f, frame1, a, c, frame2, key1[4], b, d, key2[4])
							elif key1 is not None:
								loc_apply_data[f][x] = key1[4]
							else:
								loc_apply_data[f][x] = key2[4]
							
						else:
							loc_apply_data[f][x] = loc_data[i][x][f][4]
						
			for f in range(scene.frame_end + 1):
				scene.frame_set(f)
				
				vector1 = loc_origin_data.copy()
				vector2 = loc_apply_data[f].copy()
																																
				if obj.parent is not None:
					xAx = obj.parent.x_axis
					yAx = obj.parent.y_axis
					zAx = obj.parent.z_axis
					
					vec1 = vector1.copy()
					vec2 = vector2.copy()
					
					vector1[0] = vec1[0] * xAx[0] + vec1[1] * yAx[0] + vec1[2] * zAx[0]
					vector1[1] = vec1[0] * xAx[1] + vec1[1] * yAx[1] + vec1[2] * zAx[1]
					vector1[2] = vec1[0] * xAx[2] + vec1[1] * yAx[2] + vec1[2] * zAx[2]
					
					vector2[0] = vec2[0] * xAx[0] + vec2[1] * yAx[0] + vec2[2] * zAx[0]
					vector2[1] = vec2[0] * xAx[1] + vec2[1] * yAx[1] + vec2[2] * zAx[1]
					vector2[2] = vec2[0] * xAx[2] + vec2[1] * yAx[2] + vec2[2] * zAx[2]
													  
				dif_vec = vector2 - vector1
				
				if is_bone:
					dif_vec = dif_vec * obj.bone.matrix.to_quaternion().to_matrix()
				else:            
					xAx = obj.x_axis
					yAx = obj.y_axis
					zAx = obj.z_axis
				
					d = xAx[0] * yAx[1] * zAx[2] + xAx[1] * yAx[2] * zAx[0] + xAx[2] * yAx[0] * zAx[1] - xAx[2] * yAx[1] * zAx[0] - xAx[0] * yAx[2] * zAx[1] - xAx[1] * yAx[0] * zAx[2]
					d1 = dif_vec[0] * yAx[1] * zAx[2] + xAx[1] * yAx[2] * dif_vec[2] + xAx[2] * dif_vec[1] * zAx[1] - xAx[2] * yAx[1] * dif_vec[2] - dif_vec[0] * yAx[2] * zAx[1] - xAx[1] * dif_vec[1] * zAx[2]
					d2 = xAx[0] * dif_vec[1] * zAx[2] + dif_vec[0] * yAx[2] * zAx[0] + xAx[2] * yAx[0] * dif_vec[2] - xAx[2] * dif_vec[1] * zAx[0] - xAx[0] * yAx[2] * dif_vec[2] - dif_vec[0] * yAx[0] * zAx[2]
					d3 = xAx[0] * yAx[1] * dif_vec[2] + xAx[1] * dif_vec[1] * zAx[0] + dif_vec[0] * yAx[0] * zAx[1] - dif_vec[0] * yAx[1] * zAx[0] - xAx[0] * dif_vec[1] * zAx[1] - xAx[1] * yAx[0] * dif_vec[2]   
				 
					dif_vec[0] = d1 / d
					dif_vec[1] = d2 / d
					dif_vec[2] = d3 / d
														
				obj.location = dif_vec
				obj.keyframe_insert(data_path="location", frame=f, index=-1)
		
	else:
		print("Incorrect Xom3DModel format!")
	
def getInputFilenameXom3d(self, filename, xacpath):
    print ("------------",filename)
    xom3dimport(filename,xacpath)

# -----------------------------------------------------------------------------
# Operator

class IMPORT_OT_xom3d(bpy.types.Operator):
    """Import Xom 3DModel from XomView format"""
    bl_idname = "import_scene.xom3d"
    bl_label = "Import Xom 3DModel"
    bl_region_type = "WINDOW"
    bl_options = {'UNDO'}

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    filepath = StringProperty(
            subtype='FILE_PATH',
            )
    filter_glob = StringProperty(
            default="*.xom3d",
            options={'HIDDEN'},
            )

    xacpath = StringProperty(
            name="Xac name (.xac)",
            description="Xac animation name",
            )     			
            
    def execute(self, context):
        global scene
        scene = context.scene    
        getInputFilenameXom3d(self, self.filepath, self.xacpath)
        return {'FINISHED'}   
    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

# -----------------------------------------------------------------------------
# Register
def import_xom3d_button(self, context):
    self.layout.operator(IMPORT_OT_xom3d.bl_idname,
                         text="Xom 3DModel (.xom3d)")


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(import_xom3d_button)


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(import_xom3d_button)
print('ok')

if __name__ == "__main__":
    register()
