import torch
from torch_geometric.data import Dataset
import os
import os.path as osp
from dotenv import load_dotenv
import pickle
from Molecular_graph import graphs


class MolecularGraphDataset(Dataset):
    def __init__(self, key, start, root, step, zinc, transform=None, pre_transform=None, pre_filter=None):
        self.key = key
        self.step = step
        self.start = start
        self.root = root
        self.zinc = zinc

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        load_dotenv('.env')
        rp = os.getenv(self.key)

        return [name for name in os.listdir(rp)[self.start:self.start+self.step]]

    @property
    def processed_file_names(self):
        load_dotenv('.env')

        processed_names = list()
        for n in range(self.start, self.start+self.step):
            processed_names.append(f"data_{str(n)}.pt")

        return processed_names

    @property
    def raw_paths(self):
        load_dotenv('.env')
        directory = os.getenv(self.key)

        return [osp.join(directory, file) for file in os.listdir(directory)[self.start:self.start+self.step]]

    def process(self):
        idx = self.start

        for raw_path in self.raw_paths:
            with open(raw_path, 'rb') as fp:
                bin = pickle.load(fp)
            data = graphs(bin, self.zinc)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))

        return data
