import os
import omni
import omni.replicator.core as rep
import json 

import Camera
import Prompts
import MeshFunctions

def execute_semantic_labeling(stage, simulation_app, llm_key_path, semantics_path, vocabulary_path, output_path):
    # Load existing data
    if os.path.isfile(semantics_path):
        print("Loading previously generated information")
        semantics_data = json.load(open(semantics_path))
    else:
        print("No previously generated information found")
        semantics_data = {}
    if  os.path.isfile(vocabulary_path):
        print("Loading previously generated vocabulary")
        vocabulary = json.load(open(vocabulary_path))
        print(vocabulary)
    else:
        print("No previously generated vocabulary found")
        vocabulary = {}
    
    # Initialize llm agent
    llm_agent = Prompts.LLM_Agent(llm_key_path)

    # Execute vocabulary creation and semantic labeling
    vocabulary = create_vocabulary(stage, simulation_app, llm_agent, output_path, vocabulary)
    semantics_data, vocabulary = semantic_labeling(stage, simulation_app, llm_agent, vocabulary, semantics_data, output_path)

    return semantics_data, vocabulary

def create_vocabulary(stage, simulation_app, llm_agent, output_path, vocabulary=None):
    with rep.new_layer(): # necessary for replicator camera
        # Intiialize camera
        cam = rep.create.camera(position=(0,0,0))
        rp = rep.create.render_product(cam, (1980, 1080),force_new=False)
        ldr = rep.AnnotatorRegistry.get_annotator("LdrColor")
        cam_ldr = Camera.Camera(output_path,cam,rp,ldr,simulation_app)

        # Repeat for each hard-coded camera angle 
        for idx, (pos, rot) in enumerate(cam_ldr.pre_cam_positions):
            # Take normal RGB picture for given camera angle
            print("Picture " + str(pos) + str(rot))
            image_path = cam_ldr.shoot_pos_img(pos, rot, "Real_" + str(idx))

            # Prompt LVLM to generate or if vocabulary is given expand a vocabulary
            answer = llm_agent.generate_vocabulary_information(image_path,vocabulary)
            print(f"Prompt {idx} - Following labels were added to the vocabulary" + str(answer["addedLabels"]))
            vocabulary = json.loads(answer["vocabulary"])

    return vocabulary

def semantic_labeling(stage, simulation_app, llm_agent, vocabulary, semantics_data, output_path, force_new_images=False):
    #Collect all valid meshes in a list
    meshes = MeshFunctions.collect_xforms(stage.GetPrimAtPath("/World/Meshes"))
    meshes_path = [m.GetPath() for m in meshes]
    vis = omni.usd.commands.ToggleVisibilitySelectedPrimsCommand(meshes_path,stage,False)

    with rep.new_layer(): # necessary for replicator camera
        # Intiialize camera
        cam = rep.create.camera(position=(0,0,0))
        rp = rep.create.render_product(cam, (512, 512),force_new=False)
        ldr = rep.AnnotatorRegistry.get_annotator("LdrColor")
        my_cam = Camera.Camera(output_path,cam,rp,ldr,simulation_app)

        # Repeat semantic labeling for each mesh individually
        for idx, mesh in enumerate(meshes):
            mesh_name = str(mesh.GetPath()).replace('/','')
            mesh_obj = stage.GetPrimAtPath(mesh.GetPath())

            results = {}
            if mesh_name in semantics_data:
                #Skip this mesh as information is already available
                print(f"Mesh {idx} of {len(meshes)}" + " Already completed " + mesh_name)
                results = semantics_data[mesh_name]
                continue
            else:
                # Take images from three different (pre-defined) angles
                for i in range(3):
                    # Check if images already available or if new images flag is on
                    if (not force_new_images and os.path.isfile(output_path + mesh_name + "_" + str(i) + ".png") and 
                            os.path.isfile(output_path + mesh_name + f"_Isolated_{i}.png")):
                        
                        # RGB image (with environment rendered)
                        my_cam.shoot_prim_img(mesh_obj, mesh_name + "_", i)
                        visibility_attribute = mesh_obj.GetAttribute("visibility")

                        # RGB Mesh-isolated image
                        vis.do()
                        visibility_attribute.Set("inherited")
                        my_cam.shoot_img(mesh_name + f"_Isolated_{i}")
                        visibility_attribute.Set("invisible")
                        vis.undo()

                try:
                    results = (llm_agent.generate_semantic_information(mesh_obj, mesh_name, output_path, vocabulary))
                    print(f"Results for {mesh_name}:" + str(results))

                    # Saving generated information into vocabulary and semantics file
                    new_name = results["name"]
                    new_group = results["group"]
                    semantics_data[mesh_name] = results

                    # If new label was used, add it to the vocabulary
                    if new_group in vocabulary.keys():
                        if new_name not in vocabulary[new_group]:
                            vocabulary[new_group].append(new_name)
                    else:
                        vocabulary[new_group] = [new_name]
                
                except Exception as error:
                    print("Prompt or its results invalid for prim " + mesh_name)
                    print(error)
                    continue
                
            print(f"Mesh {idx} of {len(meshes)}" + " Completed " + mesh_name)

    return semantics_data, vocabulary