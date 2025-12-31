from openai import OpenAI
from pydantic import BaseModel
import base64
import json

import MeshFunctions

class Vocabulary(BaseModel):
    vocabulary: str
    addedLabels: list[str]
class SemanticInformation(BaseModel):
    name: str
    group: str
    material: str

class LLM_Agent:
    def __init__(self, llm_key_path):
        llm_key = open(llm_key_path)
        self.client = OpenAI(
            api_key=llm_key,
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_vocabulary_information(self, path, vocabulary_basis=None):
        base64_image = self.encode_image(path + ".png")

        if vocabulary_basis is None:
            task_intro = "Create a vocabulary of semantic labels for the elements in the image."
            task_detail = """
                            You are tasked with building a hierarchical vocabulary tree from an image of an industrial environment (e.g., CAD scene of pipe systems).

                            **Objective:**
                            Generate a JSON tree of semantic labels with exactly two levels of depth:
                            - **Depth 1**: General functional units, objects, segments, or assemblies (e.g., "Pipe Segment", "Valve Assembly", "Gauge Unit"). 
                            - **Depth 2**: Specific object/mesh elements contained in that group (e.g., "Pipe Piece", "Valve Handle", "O-Ring").

                            **Guidelines:**
                            - Break down bigger structures into its components e.g. a pipe system to parts of pipes, valves and couplings
                            - Do not exceed two levels in the tree.
                            - Be concise and consistent in naming.
                            - Do **not** distinguish based on color, orientation, minor geometric differences, or function if the objects are of the same type.
                            - Group visually and semantically similar parts under the same label to ensure generality.
                            - Avoid duplicate or redundant labels.
                            - Return:
                            - A `vocabulary_tree` in JSON format showing the group-to-part relationships.
                            - A flat list of all distinct **semantic mesh labels** used at depth 2 (leaf nodes).
                            """
            prior_vocab = ""
        else:
            task_intro = "Extend an existing vocabulary of semantic labels using the new image."
            task_detail = f"""
                            You are given a prior vocabulary tree and a new image of an industrial pipe environment.
                            **Objective:**
                            Extend if needed a JSON tree of semantic labels with exactly two levels of depth:
                            - **Depth 1**: General functional units, objects, segments, or assemblies (e.g., "Pipe Segment", "Valve Assembly", "Gauge Unit"). 
                            - **Depth 2**: Specific object/mesh elements contained in that group (e.g., "Pipe Piece", "Valve Handle", "O-Ring").

                            **Guidelines:**
                            - Break down bigger structures into its components e.g. a pipe system to parts of pipes, valves and couplings
                            - The tree must still have **exactly two levels**: general groups (depth 1) and individual mesh labels (depth 2).
                            - You may add new depth-2 labels to existing depth-1 categories if appropriate.
                            - Only add a new depth-1 group if no existing group fits.
                            - Ensure label consistency and ignore irrelevant visual variation (e.g., color or minor shape).
                            - Return:
                            - The **updated vocabulary tree** in JSON format.
                            - The **list of newly added depth-2 labels** (if any).
                            """
            prior_vocab = json.dumps(vocabulary_basis, indent=2)

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": task_intro},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_detail},
                        {"type": "text", "text": prior_vocab},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            response_format=Vocabulary,
        )

        return eval(response.choices[0].message.content)

    def generate_semantic_information(self, prim, base_file_name, output_path, vocabulary):
        # Dimensions of the object
        x, y, z = MeshFunctions.compute_bbox_dimensions(prim)

        path = output_path + base_file_name
        base64_image1a = self.encode_image(path + "_0.png")
        base64_image2a = self.encode_image(path + "_Isolated" + "_0.png")
        base64_image1b = self.encode_image(path + "_1.png")
        base64_image2b = self.encode_image(path + "_Isolated" + "_1.png")
        base64_image1c = self.encode_image(path + "_2.png")
        base64_image2c = self.encode_image(path + "_Isolated" + "_2.png")

        system_prompt = (
            "You are an expert in CAD environments and semantic labeling for complex industrial scenes. "
            "Follow the labeling rules strictly to ensure consistent and structured annotation."
        )

        task_description = f"""
            You are given 6 images of a single CAD mesh from a Fukushima robot testing environment:
            - The 1st image shows the object within its original industrial scene.
            - The 2nd image shows the object isolated (may contain nearby parts as well).
            - The 3rd and 4th show in the same fashion from a different angle.
            - The 5th and 6th show in the same fashion from a different angle.

            The mesh has the following bounding box dimensions: {x:.2f} x {y:.2f} x {z:.2f} meters.

            ### Task:
            1. Assign the object a semantic label from the given vocabulary:
                - `group`: a functional structure category (from vocabulary depth 1)
                - `name`: a specific component type within that group (from vocabulary depth 2)

            2. Reuse existing vocabulary labels wherever applicable.
            3. Only create a **new label** if:
                - No fitting label exists
                - The object clearly serves a **distinct functional role**
                - The label follows the vocabulary's naming format (short, consistent, camel case or spaces)

            4. All new labels must belong to an existing group (depth 1). Do not introduce new groups.
            5. Identify the **material** of the object in a single word (e.g., 'steel', 'rubber', 'plastic').

            ### Important Guidance:
            - Base decisions **strictly** on the object's functional and structural characteristics.
            - Do **not** consider color, texture, or small shape variations unless they indicate a distinct function.
            - Use the isolated image for geometry and structure.
            - Use the scene image for the object's role and interaction context.
            - Ensure label consistency across similar objects.
        """

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": task_description },
                        { "type": "text", "text": json.dumps(vocabulary) },
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image1a}" }},
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image2a}" }},
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image1b}" }},
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image2b}" }},
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image1c}" }},
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image2c}" }},
                    ]
                }
            ],
            response_format=SemanticInformation,
        )

        return eval(response.choices[0].message.content)