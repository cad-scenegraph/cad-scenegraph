import numpy as np
import json
import os
import networkx as nx
from scipy.spatial import distance
from sklearn.cluster import dbscan

import MeshFunctions


def mesh_clustering(stage, prim_path = "/World/Meshes", full_distance_graph_path=None, save_distance_graph_path=None, dbscan_epsilon=0.01, dbscan_min_samples=1):
    #Collect all valid meshes in a list
    meshes = MeshFunctions.collect_xforms(stage.GetPrimAtPath(prim_path))
    meshes_path = [m.GetPath() for m in meshes]
    meshes_name = [str(mp).replace('/','') for mp in meshes_path]

    # Determine for each mesh the relevant points that are going to be used during distance calculation
    mesh_points_register = {}
    for mesh in meshes:
        mesh_name = str(mesh.GetPath()).replace('/','')
        mesh_points_register[mesh_name] = MeshFunctions.get_prim_points(mesh)

    if full_distance_graph_path:
        # Load existing graph 
        if os.path.isfile(full_distance_graph_path):
            print("Loading perviously generated full distance graph")
            G = nx.node_link_graph(json.load(open(full_distance_graph_path)))
        else:
            raise Exception("Error - No previously generated information found")
    else:
        # Initialize new graph
        G = nx.Graph()

        # Add for each a mesh a node in the graph
        for mesh in meshes:
            mesh_name = str(mesh.GetPath()).replace('/','')
            G.add_node(mesh_name)

        # Compute a full graph with distance value between each two meshes 
        for mesh_x in meshes:
            mesh_name_x = str(mesh_x.GetPath()).replace('/','')
            points_x = mesh_points_register[mesh_name_x]
            for mesh_y in meshes:
                mesh_name_y = str(mesh_y.GetPath()).replace('/','')
                points_y = mesh_points_register[mesh_name_y]
                if (mesh_name_x == mesh_name_y):
                    continue
                if(not G.has_edge(mesh_name_x,mesh_name_y)):
                    min_dist = distance.cdist(points_x, points_y, 'euclidean').min()
                    G.add_edge(mesh_name_x,mesh_name_y,weight=min_dist)

        # Save graph
        if save_distance_graph_path:
            data = nx.node_link_data(G)
            with open(save_distance_graph_path, 'w') as convert_file: 
                convert_file.write(json.dumps(data))

    ### Graph Preprocessing ###
    # Deleting non-existent nodes (manually deleted nodes)
    print("Amount of nodes: " + str(G.number_of_nodes()))
    print("Amount of edges: " + str(G.number_of_edges()))
    print("### Deleting nodes not in the environment ###")
    G.remove_nodes_from([a for a in G.nodes() if a not in meshes_name])
    print("Amount of nodes: " + str(G.number_of_nodes()))
    print("Amount of edges: " + str(G.number_of_edges()))

    #Copy of full graph
    # Gextra = G.copy()

    # Distance matrix of graph G
    G1 = nx.adjacency_matrix(G)

    # DBSCAN execution 
    # G = Gextra.copy()
    print(f"Executing DBSCAN with epsilon {dbscan_epsilon} and min_samples {dbscan_min_samples}")
    clustering1, clustering2 = dbscan(G1, metric='precomputed', eps=dbscan_epsilon, min_samples=dbscan_min_samples)

    print("Amount of clusters - " + str(len(set(clustering2))))
    print("Amount of outliers - " + str(np.count_nonzero(clustering2==-1)))

    print("### Removing not neighbour edges ###")
    G.remove_edges_from([(a,b) for a, b, attrs in G.edges(data=True) if attrs["weight"] > 0.01])
    print("### Removing cluster crossing edges ###")
    G.remove_edges_from([(a,b) for a, b in G.edges() if clustering2[list(G.nodes()).index(a)] != clustering2[list(G.nodes()).index(b)]])
    print("Amount of nodes: " + str(G.number_of_nodes()))
    print("Amount of edges: " + str(G.number_of_edges()))

    # Adding geometrical features to each node as attribute
    mesh_median_register = {}
    for mesh in meshes:
        mesh_name = str(mesh.GetPath()).replace('/','')
        mesh_median_register[mesh_name] = MeshFunctions.get_mesh_median(mesh_points_register[mesh_name])
    nx.set_node_attributes(G, mesh_median_register, "median_point")

    return G
