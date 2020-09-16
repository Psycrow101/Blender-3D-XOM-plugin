import bpy
from bpy.props import (
    BoolProperty,
    FloatProperty,
    StringProperty,
    FloatVectorProperty,
    CollectionProperty,
)
from bpy.types import (
    PoseBone,
)
from bpy_extras.io_utils import (
    ImportHelper,
    orientation_helper,
    axis_conversion,
)

bl_info = {
    "name": "Import Xom 3D Model / Animation",
    "author": "Psycrow",
    "version": (1, 2, 2),
    "blender": (2, 81, 0),
    "location": "File > Import-Export",
    "description": "Import Xom 3D Model / Animation from XomView format (.xom3d, .xac)",
    "warning": "",
    "wiki_url": "",
    "support": 'COMMUNITY',
    "category": "Import-Export"
}

if "bpy" in locals():
    import importlib

    if "import_xom3d" in locals():
        importlib.reload(import_xom3d)


@orientation_helper(axis_forward='-Z', axis_up='Y')
class ImportXom3D(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.xom3d"
    bl_label = "Import Xom 3D Model / Animation"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".xom3d"
    filter_glob: StringProperty(default="*.xom3d;*.xac", options={'HIDDEN'})

    use_def_pose: BoolProperty(
        name="Reset pose",
        description="Reset pose to default before loading animation",
        default=True,
    )

    remove_doubles: BoolProperty(
        name="Remove doubles vertices",
        description="Remove doubles vertices",
        default=True,
    )

    def execute(self, context):
        from . import import_xom3d

        keywords = self.as_keywords(ignore=("axis_forward",
                                            "axis_up",
                                            "filter_glob",
                                            ))
        keywords["global_matrix"] = axis_conversion(from_forward=self.axis_forward,
                                                    from_up=self.axis_up,
                                                    ).to_4x4()

        return import_xom3d.load(context, **keywords)


def menu_func_import(self, context):
    self.layout.operator(ImportXom3D.bl_idname,
                         text="Xom 3DModel (.xom3d, .xac)")


class XomChildItem(bpy.types.PropertyGroup):
    child_name: StringProperty(name="XOM Child Name", options={'HIDDEN'})


classes = (
    ImportXom3D,
    XomChildItem,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

    PoseBone.xom_type = StringProperty(name="XOM Type", options={'HIDDEN'})
    PoseBone.xom_location = FloatVectorProperty(name="XOM Location", options={'HIDDEN'})
    PoseBone.xom_rotation = FloatVectorProperty(name="XOM Rotation", options={'HIDDEN'})
    PoseBone.xom_scale = FloatVectorProperty(name="XOM Scale", options={'HIDDEN'})
    PoseBone.xom_jointorient = FloatVectorProperty(name="XOM Joint Orientation", options={'HIDDEN'})

    PoseBone.xom_child_selector = CollectionProperty(type=XomChildItem, options={'HIDDEN'})

    PoseBone.xom_has_base = BoolProperty(name="XOM Has Base", options={'HIDDEN'})
    PoseBone.xom_base_location = FloatVectorProperty(name="XOM Base Location", options={'HIDDEN'})
    PoseBone.xom_base_rotation = FloatVectorProperty(name="XOM Base Rotation", options={'HIDDEN'})
    PoseBone.xom_base_scale = FloatVectorProperty(name="XOM Base Scale", options={'HIDDEN'})
    PoseBone.xom_base_cs = bpy.props.FloatProperty(name="XOM Base Child Selector", options={'HIDDEN'})

    bpy.types.Object.xom_base_tex = bpy.props.FloatVectorProperty(name="XOM Base Texture Offset", options={'HIDDEN'})


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
