# Semantic Enrichment of CAD-Based Industrial Environments via Scene Graphs for Simulation and Reasoning
This is the repository to [cad-scenegraph.github.io](). For a detailed explanation of this work as well as exemplary results please refere to the corresponding publication.

## Ressources
Provided with the publication is also the environment and computed results which can be downloaded from this [drive folder](https://drive.google.com/drive/folders/1fFYHDF_Z97G2TYi_cAv0wrxGCfUkkvDZ?usp=sharing).

## Installation
### Requierements
- Linux environment such as Ubuntu (used system Ubuntu 24.04 LTS)
- Be able to run Isaac Sim ([requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html))
- OpenAI API key

[Install docker](https://docs.docker.com/engine/install/ubuntu/) and pull following docker image:

```bash
docker image pull nathabusz/isaac-lab
```

## Usage
Make sure that the repository is inside your home directory. Also, create a text file called llm-key.txt inside the folder .../code.
```bash
echo "OPENAI KEY" > .../cad-scenegraph/code/llm-key.txt
```

Start docker container with all necessary packages:
```bash
./DockerPrep.sh nathabusz/isaac-lab
```

Inside the container execute:
```bash
/workspace/isaaclab/_isaac_sim/python.sh /media/data/.../cad-scenegraph/code/main.py --usd_path /media/data/.../environment.usd
```
The Cad environment must be in .usd format. Isaac sim can be used to convert it.

Change the code in main.py accordingly for what scene graph you want.

## License

[MIT](https://choosealicense.com/licenses/mit/)