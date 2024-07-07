# Featurize Molecular Graphs
Code for generating molecular graphs with shallow-level node and edge embeddings. 

## Download Data
In order to save the graphs, follow the following steps.

- Clone the Repository:```git clone git@github.com:Deceptrax123/Molecular-Graph-Featuriser.git```
- Save the environment variables mentioned below in a ```.env``` file 
- Run ```main.py``` and follow the steps to effectively save the ```.pt``` files.
- You may edit the number of concurrent processes according to your system specs.

## Datasets
- Zinc Dataset: Pretraining Dataset
- Liphophilicity: Regression
- HIV Dataset: Binary Classification
- BBBP Dataset: Binary Classification
- Tox21 Dataset: Multi-label Classification
- Clintox Dataset: Binary Classification

## Train and Run
To train and evaluate GNN models, the following modifications need to be made to the ```graph_dataset.py``` script.

- Do not override the ```process()``` function
- Use the  ```processed_paths``` property in the same way as mentioned in the <a href="https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset">docs</a> by giving the absolute path to each .pt file. This ensures the entire dataset isnt processed again.

## Featurizing nodes and edges

- Both node and edge level features have been included.
- For atoms we use features such as their hybridization, atomic mass, presence in an aromatic ring, formal charge etc
- For bonds we use features such as bond type(single,double, triple), stereochemical aspects and presence in a conjugation(Resonance)

## Environment Variables

|Key|Value Description|
|-----|-----|
|tup_bins|The path to the location that saves the binary files of graphs and labels for each dataset.|
|graph_files|The root directory to where you want your graph .pt files to be stored for each dataset|
