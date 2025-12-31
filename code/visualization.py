import networkx as nx
import numpy as np
import colorsys
import omni.isaac.core.utils.semantics as sem_utils
from pxr import UsdGeom, Gf, UsdShade, Sdf

import MeshFunctions

material_path = "/SceneGraph/Materials"
nodes_path =  "/SceneGraph/Nodes"
edges_path = "/SceneGraph/Edges"

def show_scene_graph_isaac(stage, scene_graph, prim_path = "/World/Meshes", show_parent_node = False):
    """Visualize the scene graph inside isaac sim by inserting colored nodes and edges in the environment"""
    meshes = MeshFunctions.collect_xforms(stage.GetPrimAtPath(prim_path))
    clusters  = list(nx.connected_components(scene_graph))

    # Associate each cluster with a color
    colors = generate_colors(len(clusters))
    def get_cluster_index(prim_name):
        for i in range(len(clusters)):
            if prim_name in clusters[i]:
                return i

    # Create sphere balls at median point of each mesh and colorize it accordingly
    mesh_median_register = nx.get_node_attributes(scene_graph, "median_point")
    for idx, mesh in enumerate(meshes):
        mesh_name = str(mesh.GetPath()).replace('/','')
        place_at = mesh_median_register[mesh_name]
        create_node_marker("L1", place_at, str(idx), stage, colors[get_cluster_index(mesh_name)], size=0.05)

    if show_parent_node:
        # Create parent node for each cluster
        for idx, my_cluster in enumerate(clusters):
            my_cluster = list(my_cluster)
            cluster_color = colors[get_cluster_index(my_cluster[0])]
            print(f"Cluster {idx} consists of {len(my_cluster)} nodes and has color {str(cluster_color)}.")

            # Calculate rough center of the cluster and create a parent node there above the meshes
            cluster_points = [mesh_median_register[n] for n in my_cluster]
            cluster_center = sum(cluster_points) / len(cluster_points)
            cluster_max_height = max([mesh_median_register[n][2] for n in my_cluster])
            cluster_center = Gf.Vec3d(cluster_center[0],cluster_center[1],cluster_max_height + 1.5)
            create_node_marker("L2", cluster_center, idx, stage, color=cluster_color, size=0.1)

            # Connect each mesh of the cluster with edges to the parent node
            for idx2, n_pos in enumerate(cluster_points):
                create_edge_marker("E1_2", n_pos, cluster_center, f"L2Node{idx}_{idx2}", stage, color=cluster_color, size=0.005)

    # Create cylinders as a representation of edges in the colors of the nodes
    for idx, (mesh1, mesh2, _) in enumerate(scene_graph.edges(data=True)):
        pos1 = mesh_median_register[mesh1]
        pos2 = mesh_median_register[mesh2]
        create_edge_marker("E1", pos1, pos2, str(idx), stage, colors[get_cluster_index(mesh1)], size=0.005)

def generate_colors(num_colors=10):
    """Generates various RGB colors"""
    colors = []
    for i in range(num_colors):
        # Hue varies from 0 to 360 degrees
        hue = i / num_colors  # Normalize the hue
        # Full saturation (1) and lightness (0.5) for bright, vivid colors
        rgb = colorsys.hls_to_rgb(hue, 0.5, 1)  # Convert HSL to RGB
        colors.append(rgb)
    return colors

def assign_color(prim, color, stage, mtr_path = material_path):
    """Assigns a color to a prim and creates the color material object if necessary"""
    def bind_material(mtl):
        # Setup a MaterialBindingAPI on the mesh prim
        bindingAPI = UsdShade.MaterialBindingAPI.Apply(prim)
        # Use the constructed binding API to bind the material
        bindingAPI.Bind(mtl)

    suffix_str = "/Diff"+ "_".join([str(v).replace('.', '')[:3] for v in color])
    t_material_path = Sdf.Path(mtr_path + suffix_str)

    # If material doesn't exist yet, create it
    if not stage.GetPrimAtPath(t_material_path):
        material = UsdShade.Material.Define(stage, t_material_path)
        shader = UsdShade.Shader.Define(stage, t_material_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
        surface_output = material.CreateSurfaceOutput()
        surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")
    
    bind_material(UsdShade.Material(stage.GetPrimAtPath(t_material_path)))

def create_node_marker(marker_name, position, index, stage, color=(0,0,0), size=0.1):
    """Creates a sphere object"""
    x,y,z = position
    position = Gf.Vec3d((x,y,z))  # Ensure position is a Gf.Vec3d
    marker_path = nodes_path + f"/{marker_name}/Node_{index}"

    stage.RemovePrim(Sdf.Path(marker_path))  # Remove if it already exists
    sphere_prim = stage.DefinePrim(marker_path, "Sphere")
    xform = UsdGeom.Xformable(sphere_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(position)
    assign_color(sphere_prim,color,stage)
    UsdGeom.Sphere(sphere_prim).GetRadiusAttr().Set(size)

def create_edge_marker(marker_name, position, position2, index, stage, color=(0,0,0), size=0.01):
    """Create a cylinder object spanning between the two given positions"""
    x,y,z = position
    position = Gf.Vec3d((x,y,z))
    x,y,z = position2
    position2 = Gf.Vec3d((x,y,z))
    direction = position2 - position
    length = np.linalg.norm(direction)
    marker_path = edges_path + f"/{marker_name}/Edge_{index}"

    stage.RemovePrim(Sdf.Path(marker_path))
    cylinder_prim = stage.DefinePrim(marker_path, "Cylinder")
    UsdGeom.Cylinder(cylinder_prim).GetRadiusAttr().Set(size)
    UsdGeom.Cylinder(cylinder_prim).GetHeightAttr().Set(length)
    xform = UsdGeom.Xformable(cylinder_prim)
    xform.ClearXformOpOrder()

    midpoint = Gf.Vec3f((position[0] + position2[0]) / 2,
                    (position[1] + position2[1]) / 2,
                    (position[2] + position2[2]) / 2) 
    xform.AddTranslateOp().Set(midpoint)

    # Compute orientation: rotate default up (0,0,1) to direction_norm
    direction_norm = direction / length
    default_up = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = np.dot(default_up, direction_norm)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)  # in radians

    # Axis for rotation
    axis = np.cross(default_up, direction_norm)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        # direction is almost parallel or anti-parallel to default_up
        # pick any axis orthogonal to default_up
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / axis_norm

    angle_degrees = np.degrees(angle)

    # Create a quaternion from axis-angle
    # Gf.Rotation takes axis (Gf.Vec3d) and angle in degrees
    rot = Gf.Rotation(Gf.Vec3d(axis[0], axis[1], axis[2]), angle_degrees).GetQuat()

    # Use orient op
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op.Set(rot)

    assign_color(cylinder_prim,color,stage)

def insert_semantics_isaac(stage, mesh_semantics, attribute_label = None, prim_path = "/World/Meshes"):
    """Adds a semantic attribute inside isaac sim"""
    meshes = MeshFunctions.collect_xforms(stage.GetPrimAtPath(prim_path))
    for mesh in meshes:
        prim_name = str(mesh.GetPath()).replace('/','')
        if prim_name in mesh_semantics.keys():
            if attribute_label:
                sem_utils.add_update_semantics(mesh, semantic_label=mesh_semantics[prim_name][attribute_label], type_label=attribute_label)
            else:
                sem_utils.add_update_semantics(mesh, semantic_label=mesh_semantics[prim_name], type_label="semantic")

def insert_cluster_semantic_isaac(stage, scene_graph, prim_path = "/World/Meshes"):
    """Adds a semantic attribute for scene graph clusters inside isaac sim"""
    meshes = MeshFunctions.collect_xforms(stage.GetPrimAtPath(prim_path))
    clusters  = list(nx.connected_components(scene_graph))

    def get_cluster_index(prim_name):
        for i in range(len(clusters)):
            if prim_name in clusters[i]:
                return i

    for mesh in meshes:
        mesh_name = str(mesh.GetPath()).replace('/','')
        if mesh_name in scene_graph.nodes():
            sem_utils.add_update_semantics(mesh, semantic_label=str(get_cluster_index(mesh_name)), type_label="cluster_id")


def compute_parent_position(scene_graph, parent_node_label="functional unit", hierarchy_edge_label="functional unit", attribute_name="type"):
    """Computes the position of parent nodes based of their children. Parent nodes are identified by attribute_name as key and the value parent_node_label. Edges to the children also with attribute_name as key and hierarchy_edges_label"""
    mesh_median_register = nx.get_node_attributes(scene_graph, "median_point")
    parent_nodes = [n for n,d in scene_graph.nodes(data=True) if attribute_name in d.keys() and d[attribute_name] == parent_node_label]

    for parent in parent_nodes:
        child_edges = [(n,m) for n,m,d in scene_graph.edges(data=True) if attribute_name in d.keys() and d[attribute_name] == hierarchy_edge_label]
        children = [n for n,p in child_edges if n not in parent_nodes and p == parent] + [n for p,n in child_edges if n not in parent_nodes and p == parent]

        cluster_points = [np.array(mesh_median_register[n]) for n in children]
        cluster_center = sum(cluster_points) / len(cluster_points)
        scene_graph.nodes()[parent]["median_point"] = cluster_center.tolist()

def show_connections(stage, scene_graph, nodes):
    """Visualize a selected part of the scene graph inside isaac sim by inserting colored nodes and edges in the environment"""
    mesh_median_register = nx.get_node_attributes(scene_graph, "median_point")
    for idx, node in enumerate(nodes):
        place_at = mesh_median_register[node]
        create_node_marker("Node_" + str(node).replace(' ', ''), place_at, str(idx), stage, color=(1, 0, 0), size=0.05)

    edges = [(n,m) for n,m in scene_graph.edges() if n in nodes and m in nodes]
    for idx, (node1, node2) in enumerate(edges):
        pos1 = mesh_median_register[node1]
        pos2 = mesh_median_register[node2]
        create_edge_marker("Edge_" + str(node1).replace(' ', '') + "_" + str(node2).replace(' ', ''), pos1, pos2, idx, stage, (1,0,0), size=0.015)