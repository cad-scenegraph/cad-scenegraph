from pxr import UsdGeom, Usd
import numpy as np


def collect_attached_meshes(prim: Usd.Prim) -> list[UsdGeom.Mesh]:
    """Recursively collect visible meshes under the given prim."""
    meshes = []
    if prim.IsValid() and prim.GetAttribute("visibility").Get() != "invisible":
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
        for child in prim.GetChildren():
            meshes.extend(collect_attached_meshes(child))
    return meshes

def collect_xforms(prim: Usd.Prim) -> list[Usd.Prim]:
    """Collects all visible direct children prims of the given prim."""
    meshes = []
    for child in prim.GetChildren():
        if child.IsValid() and child.GetAttribute("visibility").Get() != "invisible":
            meshes.append(child)
    return meshes

def get_mesh_points(mesh, dist_threshold=4, sparse_factor = 5, voxelize = True):
    """Returns all geometric points of a mesh. Also adds computed ones along the triangle edges in order to cover the structure better."""
    mesh = UsdGeom.Mesh(mesh)
    local_points = mesh.GetPointsAttr().Get()
    matrix = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    np_matrix = np.matrix(matrix)
    np_local_points = np.array(local_points)
    points = np_local_points.dot(np_matrix[:3, :3]) + np_matrix[3, :3]
    faces = mesh.GetFaceVertexIndicesAttr().Get()
    all_points = add_points_to_mesh_faces(points, faces)

    if voxelize:
        all_points = get_voxelized_points(all_points)
    return all_points

def get_prim_points(prim: Usd.Prim, sparseness=1):
    """Returns all geometric points of a mesh or the prim's meshes."""
    if prim.IsA(UsdGeom.Mesh):
        # Case prim is a mesh - extract its points
        # Apply sparseness to reduce amount of points
        return get_mesh_points(prim)[::sparseness]
    else:
        # Case prim consists of meshes - collect points from each mesh
        prim_meshes = collect_attached_meshes(prim)
        np_points = []
        for prim_i in prim_meshes:
            if len(np_points) > 0:
                np_points = np.concatenate((np_points,get_prim_points(prim_i)),axis=0)
            else: 
                np_points = get_prim_points(prim_i)
        return np_points
    
def get_voxelized_points(points, digits=2):
    """Maps all given points into a grid with x-digits of the distance unit."""
    voxel_points = []
    for i in range(points.shape[0]):
        new_point = [round(points[i,0],digits),round(points[i,1],digits),round(points[i,2],digits)]
        if new_point not in voxel_points:
            voxel_points.append(new_point)
    return np.array(voxel_points)
    
def add_points_to_mesh_faces(mesh_points, mesh_faces, step_size=0.01):
    """Takes the geometric points of a mesh and adds points in steps at the edges of each mesh face in order to fill out faces roughly with points."""
    faces = [[mesh_faces[i],mesh_faces[i+1],mesh_faces[i+2]] for i in range(0,len(mesh_faces),3)]
    
    added_points = np.array([[0,0,0]])
    for (a,b,c) in faces:
        for p1,p2 in [[a,b],[a,c],[b,c]]:
            if np.linalg.norm(mesh_points[b] - mesh_points[a]) > 0.05:
                v1 = mesh_points[p2] - mesh_points[p1]
                i = step_size
                while i < np.linalg.norm(v1):
                    new_point = mesh_points[p1] + (i/np.linalg.norm(v1)) * v1 
                    added_points = np.concatenate((added_points,new_point),axis=0)
                    i = i + step_size

    added_points = added_points[1:,:]
    #Return all points (original and added ones) in a common list
    return (np.concatenate((mesh_points,added_points),axis=0))

def compute_bbox_midpoint(prim: Usd.Prim):
    """Returns the bounding box middle point for a given prim object."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    midpoint = bbox.ComputeCentroid()
    return midpoint[0],midpoint[1],midpoint[2]

def compute_bbox_dimensions(prim: Usd.Prim):
    """Returns the bounding box dimensions for a given prim object."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    x, y, z  = bbox.ComputeAlignedRange().GetSize()
    return x, y, z

def get_bbox_points(prim: Usd.Prim):
    """Returns the defining points (maximal point and minimal point) of a prims' bounding box."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    (x,y,z) = bbox.GetMin()
    (x2,y2,z2) = bbox.GetMax()
    return ((x,y,z),(x2,y2,z2))

def get_mesh_median(points, sparseness=5):
    """Return the median point of all mesh points"""
    def weiszfeld_algorithm(vectors, tol=1e-6, max_iter=20):
        # Initial guess: centroid of the points
        p = np.mean(vectors, axis=0)
        for _ in range(max_iter):
            # Compute the new point as a weighted average
            numerator = np.zeros(3)
            denominator = 0.0
            
            for v in vectors:
                dist = np.linalg.norm(p - v)
                if dist > 0:  # Avoid division by zero
                    weight = 1.0 / dist
                    numerator += v * weight
                    denominator += weight
            
            # New point estimate
            p_new = numerator / denominator
            
            # Convergence check
            if np.linalg.norm(p_new - p) < tol:
                return p_new
            
            p = p_new
        return p      
    return list(weiszfeld_algorithm(points[::sparseness]))

def get_corners_bbox(bbox_min, bbox_max):
    """Returns all edge points of a bbox."""
    x_min, y_min, z_min = bbox_min
    x_max, y_max, z_max = bbox_max
    corners = [
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ]
    return corners

def get_close_meshes(meshes_list, mesh, radius=1):
    """Given a list of meshes and a selected mesh, this function returns all meshes that are in distance of 1 distance unit of the selected mesh in any dimension."""
    (min,max) = get_bbox_points(mesh)
    close_meshes = []
    for mesh2 in meshes_list:
        t_min,t_max = get_bbox_points(mesh2)
        points = get_corners_bbox(t_min,t_max)
        close = False
        for point in points:
            for i in range(3):
                if point[i] > min[i] - radius and point[i] < max[i] + radius:
                    close = True
                    break
            if close:
                break
        if close:
            close_meshes.append(mesh2)
    return close_meshes
