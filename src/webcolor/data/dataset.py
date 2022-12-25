import json
from collections import defaultdict
from pathlib import Path
from typing import List

import dgl
import h5py
import torch
from dgl.data import DGLDataset, load_graphs, save_graphs
from tqdm import tqdm

from webcolor.data.converter import (
    convert_color,
    convert_image,
    convert_order,
    convert_tag,
    convert_text,
)

DATASET_FILE = "webcolor_v1.0.hdf5"
SPLIT_FILE = "webcolor_split_v1.0.json"


class WebColorDataset(DGLDataset):  # type: ignore
    def __init__(self, split: str = "train") -> None:
        assert split in ["train", "val", "test", "test1", "test2"]
        self.split = split
        self.data_dir = "data"
        self.version = Path(DATASET_FILE).stem.split("_")[-1]
        super().__init__(name="webcolor")

    def process(self) -> None:
        """Process dataset to DGL graph format."""
        # check if dataset is downloaded
        dataset_path = Path(f"{self.data_dir}/{DATASET_FILE}")
        split_path = Path(f"{self.data_dir}/{SPLIT_FILE}")
        if not dataset_path.exists() or not split_path.exists():
            raise FileNotFoundError()

        # load data split
        self.data_ids = self.get_data_ids(self.split)

        # load and process dataset
        self.graphs = []
        with h5py.File(dataset_path) as f:
            for data_id in tqdm(
                self.data_ids,
                dynamic_ncols=True,
                desc=f"process {self.split}",
            ):
                src_ids, dst_ids = [], []
                feat = defaultdict(list)
                num_nodes = len(f[data_id])
                for nid in range(num_nodes):
                    nf = f[f"{data_id}/{nid}"]

                    # record parent node id
                    if nf[()] != -1:
                        src_ids.append(nid)
                        dst_ids.append(nf[()])

                    # convert features
                    feat["text_color"].append(convert_color(nf.attrs.get("text_color")))
                    feat["bg_color"].append(convert_color(nf.attrs["background_color"]))
                    feat["order"].append(convert_order(nf.attrs["sibling_order"]))
                    feat["tag"].append(convert_tag(nf.attrs["html_tag"]))
                    feat["text"].append(convert_text(nf.attrs.get("text_feat")))
                    feat["img"].append(convert_image(nf.attrs.get("img_feat")))
                    feat["bgimg"].append(convert_image(nf.attrs.get("bgimg_feat")))
                    feat["has_text"].append(torch.tensor("text_color" in nf.attrs))

                # make DGL graph (edge direction: child -> parent)
                g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
                for k, v in feat.items():
                    g.ndata[k] = torch.stack(v)
                self.graphs.append(g)

    def get_data_ids(self, split: str) -> List[str]:
        """Return data ids for the input split name."""
        data_ids: List[str]
        split_path = Path(f"{self.data_dir}/{SPLIT_FILE}")
        with split_path.open() as f:
            split2ids = json.load(f)
            if split == "test":
                data_ids = split2ids["test1"] + split2ids["test2"]
            else:
                data_ids = split2ids[split]
        return data_ids

    def get_cache_path(self, split: str) -> str:
        return f"{self.data_dir}/{split}_{self.version}.bin"

    def save(self) -> None:
        """Save graphs to cache."""
        if self.split == "test":
            len_test1 = len(self.get_data_ids("test1"))
            save_graphs(self.get_cache_path("test1"), self.graphs[:len_test1])
            save_graphs(self.get_cache_path("test2"), self.graphs[len_test1:])
        else:
            save_graphs(self.get_cache_path(self.split), self.graphs)

    def has_cache(self) -> bool:
        if self.split == "test":
            return (
                Path(self.get_cache_path("test1")).exists()
                and Path(self.get_cache_path("test2")).exists()
            )
        else:
            return Path(self.get_cache_path(self.split)).exists()

    def load(self) -> None:
        """Load graphs from cache."""
        if self.split == "test":
            self.graphs = load_graphs(self.get_cache_path("test1"))[0]
            self.graphs += load_graphs(self.get_cache_path("test2"))[0]
        else:
            self.graphs = load_graphs(self.get_cache_path(self.split))[0]

        # load data split
        self.data_ids = self.get_data_ids(self.split)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> dgl.DGLGraph:
        return self.graphs[i]
