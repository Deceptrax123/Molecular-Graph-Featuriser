import concurrent.futures
import multiprocessing
import pandas as pd
from dotenv import load_dotenv
import pickle
import os
import time
import numpy as np
from Graph_dataset import MolecularGraphDataset
from tdc.generation import MolGen


def preprocess_tox21(data, cols):
    for c in cols:
        data[c] = data[c].fillna(data[c].mode()[0])
    return data


def download_zinc():
    data = MolGen(name='ZINC')


def pretrainer_binaries():
    load_dotenv('.env')

    data = pd.read_csv("data/zinc.tab", sep='\t')
    smiles = data['smiles']

    for f_num in range(len(smiles)):
        file_name = os.path.join(os.getenv('zinc_bins'), str(f_num))

        with open(file_name, 'wb') as fp:
            pickle.dump(smiles[f_num], fp)
        print(f"File {f_num} saved.......")
    print("Binaries for Zinc saved.....")
    return


def save_data_binaries(key):
    load_dotenv('.env')
    dataset_hash = datasets[key]
    path_key, targets_key, smiles_key = dataset_hash['Path'], dataset_hash['Y'], dataset_hash['X']
    path = os.getenv(path_key)
    data = pd.read_csv(path)

    if key != 'Tox21':
        smiles, targets = data[smiles_key], data[targets_key]
        zipped = list(zip(smiles, targets))

        f_num = 0
        for i in zipped:
            file_name = os.path.join(
                os.getenv(dataset_hash['Bin_path']), str(f_num))

            with open(file_name, 'wb') as fp:
                pickle.dump(i, fp)
            print(f"File {f_num} saved....")
            f_num += 1
        print("Binaries Saved!!!!!!")
        return
    # Handle te NaN Values in the dataset
    data = preprocess_tox21(data, cols=targets_key)
    smiles = data[smiles_key]
    t = list()
    for target in targets_key:
        t.append(data[target])

    t = np.array(t).T.tolist()
    z = list(zip(smiles, t))
    for f_num in range(len(z)):
        file_name = os.path.join(
            os.getenv(dataset_hash['Bin_path']), str(f_num))

        with open(file_name, 'wb') as fp:
            pickle.dump(z[f_num], fp)
        print(f"File {f_num} saved.....")
    print("Binaries Saved!!!!!")
    return


def download_function(p, datasets, key, zinc):
    load_dotenv('.env')
    dataset_hash = datasets[key]
    dataset = MolecularGraphDataset(key=dataset_hash['Bin_path'], root=os.getenv(
        dataset_hash['graph_root'])+p[1]+'/data/', start=p[0], step=31182, zinc=zinc)


if __name__ == '__main__':
    datasets = {
        'HIV': {
            'Path': 'HIV_csv',
            'Y': 'HIV_active',
            'X': 'smiles',
            'Bin_path': 'hiv_bins',
            'graph_root': 'hiv_graph'
        },
        'Liphophilicity': {
            'Path': 'LIPHOPHILICITY_csv',
            'Y': 'exp',
            'X': 'smiles',
            'Bin_path': 'lipho_bins',
            'graph_root': 'lipho_graph'
        },
        'BBBP': {
            'Path': 'BBBP_csv',
            'Y': 'p_np',
            'X': 'smiles',
            'Bin_path': 'bbbp_bins',
            'graph_root': 'bbbp_graph'
        },
        'Clintox': {
            'Path': 'CLINTOX_csv',
            'Y': 'CT_TOX',
            'X': 'smiles',
            'Bin_path': 'clintox_bins',
            'graph_root': 'clintox_graph'
        },
        'Tox21': {
            'Path': 'TOX_csv',
            'Y': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
            'X': 'smiles',
            'Bin_path': 'tox_bins',
            'graph_root': 'tox_graph'
        },
        'Zinc': {
            'Bin_path': 'zinc_bins',
            'graph_root': 'zinc_graph'
        }
    }

    print("Press 1 to save binaries for property datasets \
          2 to download the zinc dataset \
          3 to save binaries of the zinc dataset \
          4 to save graphs")

    choice = int(input("Enter choice: "))
    if choice == 1:
        ds = ['HIV', 'Liphophilicity', 'BBBP', 'Clintox', 'Tox21']
        idx = int(input(
            "Enter the Dataset index: ['HIV', 'Liphophilicity', 'BBBP', 'Clintox', 'Tox21', 'Zinc'] "))
        key = ds[idx]
        save_data_binaries(key)
    elif choice == 2:
        download_zinc()
    elif choice == 3:
        pretrainer_binaries()
    elif choice == 4:
        ds = ['HIV', 'Liphophilicity', 'BBBP', 'Clintox', 'Tox21', 'Zinc']
        idx = int(input(
            "Enter the Dataset index: ['HIV', 'Liphophilicity', 'BBBP', 'Clintox', 'Tox21', 'Zinc'] "))
        key = ds[idx]

        params = [(0, 'Fold1'), (31182, 'Fold2'), (62364, 'Fold3'),
                  (93546, 'Fold4'), (124728, 'Fold5'), (155910, 'Fold6'),
                  (187092, 'Fold7'), (218274, 'Fold8')]
        start_time = time.perf_counter()
        processes = list()
        n_processes = 8

        zinc = False
        if key == 'Zinc':
            zinc = True

        for i in range(n_processes):
            p = multiprocessing.Process(
                target=download_function, args=(params[i], datasets, key, zinc))
            p.start()

            processes.append(p)

        for p in processes:
            p.join()
        finish_time = time.perf_counter()
        print(f"Download completed in {finish_time-start_time} seconds")
