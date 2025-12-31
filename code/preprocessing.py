# Libraries
import omni
from pxr import UsdShade
from omni.isaac.core.prims import XFormPrim
from scipy.spatial import distance

# Personal python files
import MeshFunctions


def mesh_grouping(stage, prim_path = "/World", volume_threshold = 0.001):
    #Collect all valid meshes in a list
    meshes = MeshFunctions.collect_attached_meshes(stage.GetPrimAtPath(prim_path))
    meshes_path = [m.GetPath() for m in meshes]
    print(f"Collected {len(meshes_path)} meshes and prims")

    # Grouping of meshes with small volume to bigger meshes
    big_meshes = []
    meshes_info = {}
    small_meshes = []
    material_list = []
    material_count = {}

    path0 = "/World/Materials"
    prim0 = XFormPrim(path0, name="Materials")

    print(f"Starting grouping of {len(meshes)} meshes and its materials")
    for idx, mesh in enumerate(meshes):
        prim_name = str(mesh.GetPath()).replace('/','')
        prim_obj = stage.GetPrimAtPath(mesh.GetPath())
        print(str(idx) +  "  " +prim_name)

        # Get access to the mesh's material property and move the material object to a common diretory
        material_path = ""
        material_api = UsdShade.MaterialBindingAPI(mesh)
        if material_api:
            material_rel = material_api.GetDirectBindingRel()
            if material_rel:
                material_paths = material_rel.GetTargets()
                if material_paths:
                    material_path = material_paths[0] # Assuming only one material is bound - return the first one
                    material_name = stage.GetPrimAtPath(material_path).GetName()
                    # Check if material already moved when dealing a previous mesh
                    if material_name not in material_list:
                        material_list.append(material_name)
                        material_count[material_name] = 1
                    else:
                        material_count[material_name] = material_count[material_name] + 1

                    # Define old and new path for material
                    from_path = material_path
                    to_path = path0 + "/" + material_name

                    # Move material object
                    omni.kit.commands.execute("MovePrim", path_from=from_path, path_to=to_path)


        # Classification between big and small meshes for grouping
        sx,sy,sz = MeshFunctions.compute_bbox_dimensions(prim_obj)
        if sx * sy * sz > volume_threshold: 
            big_meshes.append(mesh)
        else:
            small_meshes.append(mesh) 
        meshes_info[mesh] = material_path
    print(f"A total of {material_count} material objects were identified and moved")
    print(f"Continue with mesh grouping")

    path1 = "/World/Meshes"
    prim1 = XFormPrim(path1, name="Meshes")
    path3 = "/World/Ignored"
    prim3 = XFormPrim(path3, name="Ignored")

    created_paths = []
    moving_instructions = {}

    print("Preparing new mesh directory...")
    # Move each big mesh under it's own Xform 
    for idx, mesh in enumerate(big_meshes):
        from_path = str(mesh.GetPath())
        prim_name = str(from_path).replace('/','')

        # Create new Xform
        prim_x = XFormPrim(path1 + "/" + f"Xform_{idx}", "Xform")
        # Move big mesh under Xform
        to_path = path1 + "/" + f"Xform_{idx}/Mesh"
        moving_instructions[prim_name] = to_path
        created_paths.append(to_path)


    # Retrieve geometric points of all meshes
    mesh_points_register = {}
    for mesh in big_meshes:
        prim_name = str(mesh.GetPath()).replace('/','')
        mesh_points_register[prim_name] = MeshFunctions.get_mesh_points(mesh)
    for mesh in small_meshes:
        prim_name = str(mesh.GetPath()).replace('/','')
        mesh_points_register[prim_name] = MeshFunctions.get_mesh_points(mesh)

    print("Finding nearest neighbour meshes...")
    # Allocating all small meshes to a big mesh 
    failed_counter = 0
    for small_mesh in small_meshes:
        min_dist = -1
        closest_mesh = None
        mesh_name_x = str(small_mesh.GetPath()).replace('/','')
        points_x = mesh_points_register[mesh_name_x]

        # Get close candidate meshes
        near_big_meshes = MeshFunctions.get_close_meshes(big_meshes, small_mesh)

        # Identify nearest mesh from candidates
        for b_mesh in near_big_meshes:
            mesh_name_y = str(b_mesh.GetPath()).replace('/','')
            points_y = mesh_points_register[mesh_name_y]
            new_dist = distance.cdist(points_x, points_y, 'euclidean').min()
            if new_dist < min_dist or min_dist < 0:
                min_dist = new_dist
                closest_mesh = str(b_mesh.GetPath()).replace('/','')

        # Check if there is a closest mesh
        if closest_mesh:
            # Prepare new location and name of the small mesh
            to_path = (moving_instructions[closest_mesh])[:-5] + "/sMesh_"
            counter = 0
            while to_path + str(counter) in created_paths:
                    counter = counter + 1
            to_path = to_path + str(counter)

            mesh_name = str(small_mesh.GetPath()).replace('/','')
            moving_instructions[mesh_name] = to_path
            created_paths.append(to_path)
        else:
            # Collect failed meshes
            failed_counter = failed_counter + 1
            mesh_name = str(small_mesh.GetPath()).replace('/','')
            moving_instructions[mesh_name] = path3
    print(f"Failed finding fitting meshes for {failed_counter} small meshes.")


    print("Moving meshes...")
    # Move big meshes to new location
    for idx, mesh in enumerate(big_meshes):
        from_path = str(mesh.GetPath())
        mesh_name = from_path.replace('/','')
        to_path = moving_instructions[mesh_name]

        omni.kit.commands.execute("MovePrim", path_from=from_path, path_to=to_path)
        print(f"Big Mesh-{idx}/{len(big_meshes)}")

    # Move small meshes to new location
    for idx, mesh in enumerate(small_meshes):
        from_path = str(mesh.GetPath())
        mesh_name = from_path.replace('/','')
        to_path = moving_instructions[mesh_name]
        
        omni.kit.commands.execute("MovePrim", path_from=from_path, path_to=to_path)
        print(f"Small Mesh-{idx}/{len(small_meshes)}")