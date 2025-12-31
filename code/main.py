import argparse
import sys

from isaacsim import SimulationApp

# Configuration settings for opening Isaac Sim window
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

# Set up in-code arguments

# Set up directories (only linux supported)
import pathlib
base_path = pathlib.Path(__file__).parent.parent
output_path = base_path / "_output"
semantics_path = base_path / "data/Semantics.json"
vocabulary_path = base_path / "data/Vocabulary.json"
graph_path = base_path / "data/Graph.json"
llm_key_path = base_path / "code/llm-key.txt"

# Set up command line arguments
parser = argparse.ArgumentParser("Usd Load sample")
parser.add_argument("--usd_path", type=str, help="Path to usd file, should be relative to your default assets folder", required=True)
parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
parser.add_argument("--semantics_file", type=str, default=semantics_path, help="Path to text file in json format containing a dictionary with semantics for the prims")
parser.add_argument("--vocabulary_file", type=str, default=vocabulary_path, help="Path to text file in json format containing a vocabulary of semantic labels")
parser.add_argument("--graph_file", type=str, default=graph_path, help="Path to text file in json format containing a networkx graph of the envrionment (scene graph)")

args, unknown = parser.parse_known_args()
# Start the omniverse application
CONFIG["headless"] = args.headless
simulation_app = SimulationApp(launch_config=CONFIG)

import carb
import omni

# Locate Isaac Sim assets folder to load sample
from omni.isaac.nucleus import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
usd_path = args.usd_path

# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except:
    result = False

if result:
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {assets_root_path}"
    )
    simulation_app.close()
    sys.exit()
# Wait two frames so that stage starts loading
simulation_app.update()
simulation_app.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    simulation_app.update()
print("Loading Complete")

#####################################################################################################
from omni.isaac.core import World
import networkx as nx
import json

import preprocessing
import semanticLabeling
import clustering
import visualization
import appFunctionalRelations

# Prepare stage access
simulation_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# Preprocessing
# preprocessing.mesh_grouping(stage, prim_path="/World/plant_1F")

# # Save preprocessed usd file to new location
# new_usd_path = pathlib.Path(usd_path).stem + "_processed.usd"
# omni.usd.get_context().export_as_stage(new_usd_path)

# Semantic labeling
semantics_data, vocabulary = semanticLabeling.execute_semantic_labeling(stage, simulation_app, llm_key_path, args.semantics_file, args.vocabulary_file, output_path)
with open(args.semantics_file, "w") as outfile:
    json.dump(semantics_data, outfile)
with open(args.vocabulary_file, "w") as outfile:
    json.dump(vocabulary, outfile)

# Clustering
G_scene = clustering.mesh_clustering(stage)

# Visualization
# visualization.show_scene_graph_isaac(stage, G_scene)

# Insert semantics into graph
nx.set_node_attributes(G_scene, semantics_data)

# Functional relations
functional_labels = ["Valve Assembly", "Gauge Unit", "Pump Unit", "Tank Assembly"]
connecting_labels = ["Pipe Assembly", "Connection Assembly"]
# Give graph and returns new graph only showing the functional relations + visualization of it in different function
appFunctionalRelations.extract_functional_relations(G_scene, functional_labels, connecting_labels)
# Visualize results
visualization.compute_parent_position(G_scene)
visualization.show_connections(stage, G_scene, [n for n,d in G_scene.nodes(data=True) if "type" in d.keys() and d["type"] == "functional unit"])
visualization.insert_semantics_isaac(stage, semantics_data, attribute_label="group")

# Save scene graph
G_scene_data = nx.node_link_data(G_scene)
# with open(args.graph_file, 'w') as convert_file: 
#     convert_file.write(json.dumps(G_scene_data))
with open(args.graph_file, "w") as outfile:
    json.dump(G_scene_data, outfile)

# Let simulation editor run
from omni.isaac.core.utils.stage import is_stage_loading
omni.timeline.get_timeline_interface().play()
while simulation_app.is_running():
        # Run in realtime mode, we don't specify the step size
        simulation_app.update()
omni.timeline.get_timeline_interface().stop()
simulation_app.close()