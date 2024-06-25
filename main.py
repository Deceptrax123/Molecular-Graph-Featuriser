import concurrent.futures
import multiprocessing
import pandas as pd
from dotenv import load_dotenv
import pickle
import os
import numpy as np
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
    datasets = {
        'HIV': {
            'Path': 'HIV_csv',
            'Y': 'HIV_active',
            'X': 'smiles',
            'Bin_path': 'hiv_bins'
        },
        'Liphophilicity': {
            'Path': 'LIPHOPHILICITY_csv',
            'Y': 'exp',
            'X': 'smiles',
            'Bin_path': 'lipho_bins'
        },
        'BBBP': {
            'Path': 'BBBP_csv',
            'Y': 'p_np',
            'X': 'smiles',
            'Bin_path': 'bbbp_bins'
        },
        'Clintox': {
            'Path': 'CLINTOX_csv',
            'Y': 'CT_TOX',
            'X': 'smiles',
            'Bin_path': 'clintox_bins'
        },
        'Tox21': {
            'Path': 'TOX_csv',
            'Y': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
            'X': 'smiles',
            'Bin_path': 'tox_bins'
        }
    }
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


if __name__ == '__main__':
    print("Press 1 to save binaries for property datasets \
          2 to download the zinc dataset \
          3 to save binaries of the zinc dataset \
          4 to save graphs for the property datasets \
          5 to save graphs for the zinc dataset")

    choice = int(input("Enter choice: "))
    if choice == 1:
        ds = ['HIV', 'Liphophilicity', 'BBBP', 'Clintox', 'Tox21']
        idx = int(input("Enter the Dataset index: "))
        key = ds[idx]
        save_data_binaries(key)
    elif choice == 2:
        download_zinc()
    elif choice == 3:
        pretrainer_binaries()
