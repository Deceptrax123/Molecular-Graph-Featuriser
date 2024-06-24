import pandas as pd
from dotenv import load_dotenv
import pickle
import os
import multiprocessing
import concurrent.futures


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


if __name__ == '__main__':
    ds = ['HIV', 'Liphophilicity', 'BBBP', 'Clintox', 'Tox21']
    idx = int(input("Enter the Dataset index: "))

    key = ds[idx]
    save_data_binaries(key)
