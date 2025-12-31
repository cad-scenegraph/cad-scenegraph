import os
import carb
import numpy as np
import omni.replicator.core as rep
from pxr import UsdLux
from PIL import Image 

import MeshFunctions


# Enable scripts (important for saving picture as file)
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

def create_light(p_stage, intensity=500):
    """ Creates an ambient light to the world """
    light_path = "/World/AmbientLight"
    light_prim = p_stage.DefinePrim(light_path, "DomeLight")
    light = UsdLux.DomeLight(light_prim)
    light.CreateIntensityAttr(intensity)

class Camera:
    # predefined camera angles capturing an overlooking view of the corresponding environment of the related work
    pre_cam_positions = [((345.6,290.1,28.3),(0,-40,90)),
                         ((355.9,277.1,26.6),(0,-20,0)),
                         ((344.6,267.8,29.9),(0,-40,-90)),
                         ((340.4,283.2,24),(0,-15,160)),
                         ((337,292,28),(0,-20,115))]
    # predefined camera angles capturing a top-down view of the corresponding environment of the related work
    pre_cam_topview = [((345.26697,277.1153,84),(0,-90,-90))]

    def __init__(self, out_dir, cam, rp, ldr, simulation_app):
        os.makedirs(out_dir, exist_ok=True)
        if os.path.isdir(out_dir):
            self.out_dir = out_dir
        else:
            raise Exception("Invalid path for output") 
        self.cam = cam
        self.rp = rp
        self.ldr = ldr
        self.simulation_app = simulation_app

    def write_rgb(self, data, path):
        """Saves the given image data (format RGBA) as a png file at given path."""
        rgb_img = Image.fromarray(data, mode="RGBA")
        print(str(rgb_img) + " at " + path + ".png")
        rgb_img.save(path + ".png")

    def shoot_prim_img(self, target_prim, file_name, setting=0):
        """Shoots and saves an image of a given prim from one of the selectable camera view settings."""
        x,y,z = MeshFunctions.compute_bbox_midpoint(target_prim)
        lx,ly,lz = MeshFunctions.compute_bbox_dimensions(target_prim)

        add_dist = 1 # additional distance of the camera
        # Predefined views on a mesh based of its bounding box parameters
        match setting:
            case 0:
                set_position = (x+lx+add_dist,y+ly+add_dist,z+lz+add_dist)
            case 1:
                set_position = (x+lx+add_dist,y-ly-add_dist,z+lz)
            case 2:
                set_position = (x-lx-add_dist,y-ly-add_dist,z+lz+add_dist)
            case 3:
                set_position = (x-lx-add_dist,y-ly-add_dist,z)
            case 4:
                set_position = (x+lx+add_dist,y+ly+add_dist,z)
            case 5:
                set_position = (x-lx-add_dist,y,z+lz+add_dist)
            case 6:
                set_position = (x,y-ly-add_dist,z+lz+add_dist)
            case _:
                set_position = (x+lx+add_dist,y+ly+add_dist,z+lz+add_dist) # default is case 0

        # Move camera
        try:
            with self.cam:
                rep.modify.pose(
                    position=set_position,
                    look_at=(x,y,z))
        except:
            print("Error during camera modification")
        for _ in range(5):
            self.simulation_app.update()

        # Take picture and return image file path
        return self.shoot_img(file_name + str(setting))
    
    def shoot_pos_img(self, position, rotation, file_name):
        """Shoots and saves an image from a given angle"""
        self.set_pos(position,rotation)
        print("Position changed")
        return self.shoot_img(file_name)
    
    def shoot_img(self, file_name):
        """Shoots and saves an image with the camera in its current state.
        Depending on the used annotater data needs to be handled accordingly."""
        for _ in range(5):
            self.simulation_app.update()
        # Attach render product
        self.ldr.attach(self.rp)

        # Evaluate the camera render
        rep.orchestrator.preview()
        rep.orchestrator.step()
        rgb_data = self.ldr.get_data()

        # Handle image data depending on the annotator
        annot_name = self.ldr.name
        if annot_name == "LdrColor":
            # LdrColor - normal RGBA image - nothing to do
            rgb_data = rgb_data
        elif annot_name == "semantic_segmentation":
            # Semantic information of visible meshes in different colors
            # Option 1: Colorization is set true - 
            rgb_data = rgb_data["data"] # if colorization True - handle as normal rgba image
            # Option 2: Colorization is set False - manual colorization necessary 
            #rgb_data = self.rgb_colorize(rgb_data) # if colorization False
        else:
            # Exception case
            if type(rgb_data) == dict and "data" in rgb_data.keys():
                rgb_data = rgb_data["data"]

        # Save image
        path = os.path.join(self.out_dir, file_name)
        self.write_rgb(rgb_data, path)

        # Release resources by detaching
        self.ldr.detach()

        return path
    
    def rgb_colorize(self, id_data):
        """Colorizes image data taken with the camera annotator semantic_segmentation and colorization set to false.
        Colors are set manually for each semantic class."""
        info = id_data["info"]["idToLabels"]
        data = id_data["data"]
        out = np.zeros((1080, 1980, 4)) # recolored image
        out = out.astype(np.uint8) # essential correct typing

        # Recolorize each pixel with the according semantic color
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = info[str(data[i, j])]
                if "group" in value.keys():
                    if value["group"] == "pipe assembly":
                        out[i, j] = np.array([127, 0, 255, 255])
                    elif value["group"] == "valve assembly":
                        out[i, j] = np.array([0, 128, 255, 255])
                    elif value["group"] == "connection assembly":
                        out[i, j] = np.array([0, 100, 100, 255])
                    elif value["group"] == "tank assembly":
                        out[i, j] = np.array([255, 0, 255, 255])
                    elif value["group"] == "pump unit":
                        out[i, j] = np.array([153, 0, 76, 255])
                    elif value["group"] == "gauge unit":
                        out[i, j] = np.array([0, 204, 0, 255])
                    elif value["group"] == "structure":
                        out[i, j] = np.array([128, 128, 128, 255])
                    elif value["group"] in ["scene graph edge", "scene graph node"]:
                        out[i, j] = np.array([255, 0, 0, 255])
                    else:
                        out[i, j] = np.array([0, 0, 0, 255])
                else:
                    out[i, j] = (0, 0, 0, 0)
        return out

    def set_pos(self, position, rotation):
        """Sets the camera's position and rotation."""
        try:
            with self.cam:
                rep.modify.attribute("focalLength",16)
                rep.modify.pose(
                    position=position,
                    rotation=rotation)
        except:
            print("Error during camera modification")
        for _ in range(5):
            self.simulation_app.update()

    def set_look_at(self, position, look_at):
        """Sets the camera's position and makes it face the look_at position"""
        try:
            with self.cam:
                rep.modify.pose(
                    position=position,
                    look_at=look_at)
        except:
            print("Error during camera modification")
        for _ in range(5):
            self.simulation_app.update()

    def set_normal(self):
        """Sets the camera's type to a normal voew"""
        with self.cam:
            rep.modify.attribute("focalLength",16)
            rep.modify.attribute("projection_type","pinhole")
        for _ in range(5):
            self.simulation_app.update()

    def set_top_view(self):
        """Takes an image from a top-down view. Note that camera settings change with executing this function."""
        with self.cam:
            rep.modify.attribute("focalLength",50)
            rep.modify.pose(
                    position=self.pre_cam_topview[0][0],
                    rotation=self.pre_cam_topview[0][1])
        for _ in range(5):
            self.simulation_app.update()
