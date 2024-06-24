import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from dotenv import load_dotenv
from Features import get_atom_features, get_bond_features
import os


def graphs(item):
    load_dotenv('.env')

    smiles = item[0]
    label = item[1]

    mol = Chem.MolFromSmiles(smiles)

    # Featurize nodes and edges
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()

    unrelated_molecule = Chem.MolFromSmiles("O=O")
    n_node_features = len(get_atom_features(
        unrelated_molecule.GetAtomWithIdx(0)))
    n_bond_features = len(get_bond_features(
        unrelated_molecule.GetBondBetweenAtoms(0, 1)))

    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
    X = torch.tensor(X, torch.float)

    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)

    edges = torch.stack([torch_rows, torch_cols], dim=0)
    edge_features = np.zeros((n_edges, n_bond_features))

    for (k, (z, j)) in enumerate(zip(rows, cols)):
        edge_features[k] = get_bond_features(
            mol.GetBondBetweenAtoms(int(z), int(j)))
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    y = torch.tensor(label)
    graph = Data(x=X, edge_index=edges,
                 edge_attr=edge_features, y=y, smiles=smiles)

    return graph
