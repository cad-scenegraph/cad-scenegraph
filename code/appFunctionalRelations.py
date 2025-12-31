def extract_functional_relations(graph, functional_labels, connecting_labels, semantic_att="group"):
    """Identifies functional unit clusters based on functional labels and finds connections going over nodes with connecting labels"""
    # Retrieve functional units
    functional_units, list_completed = identify_functional_units(graph, functional_labels, semantic_att)

    # Only consider nodes that have that attribute
    subgraph = graph.subgraph([n for n,d in graph.nodes(data=True) if semantic_att in d.keys()])

    # Expand each functional unit onto nodes with an attribute value contained in connecting_labels
    did_change = True
    while did_change:
        did_change = False
        # Expand from each functional unit an attribute value contained in connecting_labels
        for k in functional_units.keys():
            current_expansion = functional_units[k]
            next_expansion = []
            # Get all neighbours not taken neighbours and add them to the current functional unit
            for node in current_expansion:
                next_expansion = next_expansion + [n for n in subgraph.neighbors(node) if n not in list_completed and subgraph.nodes()[n][semantic_att] in connecting_labels]
            functional_units[k] = current_expansion + next_expansion
            list_completed = list_completed + next_expansion
            
            if len(next_expansion) > 0:
                did_change = True

    # Make markings for all expanded nodes
    graph_marking = {}
    for k in functional_units.keys():
        for n in functional_units[k]:
            graph_marking[n] = k

    # Check for edges with both nodes differently marked -> relation between two distinct functional units
    new_edges = []
    for a,b in graph.edges():
        if a in graph_marking.keys() and b in graph_marking.keys():
            idx_a = graph_marking[a]
            idx_b = graph_marking[b]
            if idx_a != idx_b and (idx_a,idx_b) not in new_edges:
                new_edges.append((idx_a,idx_b))

    # Add functional relations to the graph
    graph.add_edges_from(new_edges, type="functional relation")

def identify_functional_units(graph, functional_labels, semantic_att="group"):
    """Returns sets of nodes that are interconnected and have the same attritue value"""
    # Only consider nodes that have that attribute
    subgraph = graph.subgraph([n for n,d in graph.nodes(data=True) if semantic_att in d.keys()])

    node_completed = []
    functional_units = {}
    for node,data in subgraph.copy().nodes(data=True):
        if data[semantic_att] in functional_labels and node not in node_completed:
            # Expand onto all nodes of identical functional group
            expanded_nodes = expand_node_group(subgraph, node, semantic_att)
            new_node = warp_up_node(graph, expanded_nodes, data[semantic_att])

            node_completed = node_completed + expanded_nodes
            functional_units[new_node] = expanded_nodes

    return functional_units, node_completed

def expand_node_group(graph, start_node, semantic_att="group"):
    """Returns all nodes that are connected to the given start node by nodes with the same attribute value as the start node"""
    # Initializations
    list1 = [start_node]
    list2 = []
    collected_nodes = [start_node]
    group = graph.nodes()[start_node][semantic_att]

    while len(list1):
        for node in list1:
            new_nodes = [n for n in graph.neighbors(node) if n not in collected_nodes and graph.nodes()[n][semantic_att] == group]
            list2 = list2 + new_nodes
            collected_nodes = collected_nodes + new_nodes
        list1 = list2
        list2 = []

    return collected_nodes

def warp_up_node(graph, nodes, node_name):
    """Adds a parent node for the given nodes in the graph"""
    index = 0
    while ((str(node_name) + str(index)) in graph.nodes()):
        index = index + 1
    new_node_name = node_name + str(index)
    graph.add_node(new_node_name, abst=True, type="functional unit")

    for node in nodes:
        graph.add_edge(node,new_node_name, type="functional unit")

    return new_node_name
