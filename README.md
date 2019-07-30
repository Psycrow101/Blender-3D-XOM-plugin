# xom-import

**xom-import** is a plugin for Blender 3D that allows you to import 3D models
from Worms games (Worms 3D, Worms Ultimate Mayhem, Worms Forts: Under Siege, etc) into the editor.
It works with two file formats: models (`.xom3d`) and animations (`.xac`).
The plugin cannot export loaded models. So you can't use it to make changes to the Worms games.

![](https://next21.ru/wp-content/uploads/2018/12/ezgif-2-6272f3db265e.gif)

## Requirements

* Blender 3D (2.79 or 2.80)
* [XomView v3.0](https://www.dropbox.com/s/vawww6wf8xdhq66/xom%20view.zip?dl=0) by AlexBond

XomView is used to extract resources from the Worms game bundles, including the required `.xom3d`
and `.xac` files. The plugin itself doesn't use XomView.

## Importing models

First you need to get the necessary models, textures and animations using XomView.
After exporting models, check that the textures are located in the same directory
with the model (if it uses them of course). Files with animations can be located anywhere.

Then simply import the model into Blender 3D.

![](https://next21.ru/wp-content/uploads/2018/12/ezgif-2-bf3dcae8a8e7.gif)

To load `.xac` file, make armature active and import the animation you want.

Some animations require a Base pose. To properly import such animations, you must first load
the data from the file named `[Base].xac`. To do this, simply import `[Base].xac` as an animation.
After that, you can import other animations.

![](https://next21.ru/wp-content/uploads/2018/12/ezgif-2-df30e3146c68.gif)

Optionally, you can uncheck "Reset pose" to keep the previous position before loading the animation.

![](https://next21.ru/wp-content/uploads/2018/12/ezgif-2-e37a18c7ec37.gif)
