import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import _torch_collate_batch
from torch.utils.data import Dataset, DataLoader, Subset, SequentialSampler
from typing import Optional, Union, List, Any, Dict, Tuple
from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer, T5Config
import pickle as pkl
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import math
import time
import random


class MyDataset(Dataset):
    def __init__(self, dataframe, doc_len, dataset="MSMARCO"):
        self.tokenizer = T5Tokenizer.from_pretrained("")
        if dataset == "MSMARCO100k":
            self.doc_input_ids = pkl.load(open("", "rb"))
            self.doc_attention_mask = pkl.load(open("", "rb"))
        elif dataset == "MSMARCO500k":
            self.doc_input_ids = pkl.load(open("", "rb"))
            self.doc_attention_mask = pkl.load(open("", "rb"))
        elif dataset == "MSMARCO1M":
            self.doc_input_ids = pkl.load(open("", "rb"))
            self.doc_attention_mask = pkl.load(open("", "rb"))
        elif dataset == "NQ320":
            self.doc_input_ids = pkl.load(open("", "rb"))
            self.doc_attention_mask = pkl.load(open("", "rb"))
        else:
            raise ValueError("Invalid dataset.")
        
        self.dataset = dataset

        self.doc_len = doc_len

        self.data = dataframe

        assert not self.data.isnull().values.any()

    def convert_to_features(self, example, padding="max_length", encoder_max_length=32):
        """
        没有别的special token
        """
        # if self.dataset == "NQ320":
        #     encoder_max_length = 32
        # else:
        #     encoder_max_length = 32
        ret = self.tokenizer(
            example["query"],
            padding=padding,
            truncation=True,
            max_length=32,
        )
        doc_identifier = [int(example["docid"])]

        ret["labels"] = doc_identifier

        ret["decoder_input_ids"] = [0]
        ret["decoder_attention_mask"] = [1]


        ret["doc_input_ids"] = self.doc_input_ids[int(example["docid"])][:self.doc_len]
        ret["doc_attention_mask"] = self.doc_attention_mask[int(example["docid"])][:self.doc_len]
        while len(ret["doc_input_ids"]) < self.doc_len:
            ret["doc_input_ids"] += [0]

        while len(ret["doc_attention_mask"]) < self.doc_len:
            ret["doc_attention_mask"] += [0]

        previous_label = eval(example["label"])
        ret["previous_label"] = [int(item) for item in previous_label]


        return ret


    def __getitem__(self, item):
        data = dict(self.data.iloc[item])
        data = self.convert_to_features(data)
        return data

    def __len__(self):
        return len(self.data)


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.tensor([example["input_ids"] for example in examples])
    attention_mask = torch.tensor([example["attention_mask"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    decoder_input_ids = torch.tensor([example["decoder_input_ids"] for example in examples])
    decoder_attention_mask = torch.tensor([example["decoder_attention_mask"] for example in examples])
    doc_input_ids = torch.tensor([example["doc_input_ids"] for example in examples])
    doc_attention_mask = torch.tensor([example["doc_attention_mask"] for example in examples])
    previous_label = torch.tensor([example["previous_label"] for example in examples])


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "doc_input_ids": doc_input_ids,
        "doc_attention_mask": doc_attention_mask,
        "previous_label": previous_label
    }


# class CustomDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)

#     def __iter__(self):
#         # 获取每个进程分配的样本数量
#         num_samples_per_replica = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
#         total_size = num_samples_per_replica * self.num_replicas

#         # 生成索引，按顺序划分数据集
#         indices = list(range(len(self.dataset)))


#         # 根据 rank 划分数据
#         start_index = self.rank * num_samples_per_replica
#         end_index = start_index + num_samples_per_replica
#         indices = indices[start_index:end_index]

#         return iter(indices)


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, data_source, data_len, num_replicas=None, rank=None, batch_size=80, block_size=64):
        """

        :param data_source:
        :param num_replicas:
        :param rank:
        :param batch_size:
        :param gid_labels:
        label_indices 为字典，label指向所对应的样本indices
        label_list 为列表，存有所有label
        index_pointer 为字典，label指向对应的pointer
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.num_replicas = num_replicas or 1
        self.rank = rank or 0
        self.batch_size = batch_size
        self.label_pointer = 0
        self.block_size = block_size
        self.label_indices = self.build_label_indices()
        self.label_list = [label for label in self.label_indices.keys()]
        self.index_pointer = {label: 0 for label in self.label_indices}
        self.max_indices = {label: len(self.label_indices[label]) for label in self.label_indices}
        self.label_check = {label: 0 for label in self.label_indices.keys()}
        self.device_start = None
        self.device_end = None
        self.data_len = data_len


    def build_label_indices(self):
        """
        构建label_indices，如果有的label不足一个batch size 随机抽取一些填充
        :return:
        """
        return_dict = {}
        for label, group in self.data_source.data.groupby('label'):
            indices_list = group.index.tolist()
            random.shuffle(indices_list)  # 对indices_list进行shuffle
            return_dict[label] = indices_list

        return return_dict


    def shuffle_data(self):
        """
        build_label_indices自带shuffle功能每次调用即可

        :return:
        """
        self.label_indices = self.build_label_indices()
        self.label_check = {label: 0 for label in self.label_indices.keys()}


    def end_check(self):
        """
        所有grou全部处理完就返回True

        :return:
        """
        for label in self.label_list:
            if self.label_check[label] == 0:
                return False

        return True



    def __iter__(self):
        self.shuffle_data()
        epoch_indices = []
        print(f">>>>>>>>>>>>{self.__len__()}")
        while not self.end_check():
            batch_indices = []
            if self.label_pointer >= len(self.label_list):
                self.label_pointer = 0
                random.shuffle(self.label_list)

            label = self.label_list[self.label_pointer]
            start = self.index_pointer[label]
            end = min(start + self.block_size, len(self.label_indices[label]))
            self.index_pointer[label] = end
            batch_indices = self.label_indices[label][start:end]

            if end >= (len(self.label_indices[label]) - self.block_size):
                self.label_check[label] = 1

            self.label_pointer += 1

            if len(batch_indices) < self.block_size:
                continue

            epoch_indices.extend(batch_indices)

        for index in epoch_indices:
            yield index


    def __len__(self):
        total_len = 0
        for val in self.label_indices.values():
            total_len += len(val) - (len(val) % self.block_size)
        return self.data_len



class MyDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, test_path, mode, doc_len, block_size, label_dict=None, batch_size=32, dataset="MSMARCO"):
        super().__init__()
        self.batch_size = batch_size
        self.doc_len = doc_len
        self.dataset = dataset
        self.block_size = block_size
        if mode == "train":
            self.train_df = pd.read_csv(
                train_path,
                encoding='utf-8', names=["query", "docid", "label"],
                header=None, sep='\t',
                dtype={'query': str, 'docid': int, 'label': str}
            )
            self.train_df = self.train_df.dropna()

            self.val_df = pd.read_csv(
                val_path,
                encoding='utf-8', names=["query", "docid", "label"],
                header=None, sep='\t',
                dtype={'query': str, 'docid': int, 'label': str}
            )
            self.val_df = self.val_df.dropna()
            if len(label_dict) != 0:
                self.train_df["label"] = self.train_df["docid"].apply(lambda x: str(label_dict[x]))
                self.val_df["label"] = self.val_df["docid"].apply(lambda x: str(label_dict[x]))
        else:
            self.test_df = pd.read_csv(
                test_path,
                encoding='utf-8', names=["query", "docid", "label"],
                header=None, sep='\t',
                dtype={'query': str, 'docid': int, 'label': str}
            )
            # self.test_df = self.test_df[self.test_df["docid"] < 20000]
            # self.test_df = self.test_df[self.test_df["docid"] > 70000]
            self.test_df = self.test_df.dropna()
            if len(label_dict) != 0:
                self.test_df["label"] = self.test_df["docid"].apply(lambda x: str(label_dict[x]))


    def split_dataframe(self, df, num_replicas, rank):
        total_size = len(df)
        per_device_size = int(np.ceil(total_size / num_replicas))
        start_idx = rank * per_device_size
        end_idx = min(start_idx + per_device_size, total_size)
        return df.iloc[start_idx:end_idx].reset_index(drop=True)
    

    def cal_len(self, df, num_replicas):
        total_size = len(df)
        per_device_size = int(np.ceil(total_size / num_replicas))
        min_len = 99999999
        for rank in range(num_replicas):
            start_idx = rank * per_device_size
            end_idx = min(start_idx + per_device_size, total_size)
            sub_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

            tmp_len = 0
            for label, group in sub_df.groupby('label'):
                indices_list = group.index.tolist()
                tmp_len += len(indices_list) - (len(indices_list) % self.block_size)
            min_len = min(tmp_len, min_len)
        
        return min_len



    def train_dataloader(self):
        # 每个 epoch 开始前打乱训练数据
        num_gpus = self.trainer.num_devices if self.trainer else 1
        global_rank = self.trainer.global_rank if self.trainer else 0
        split_data = self.split_dataframe(self.train_df, num_gpus, global_rank)
        train_dataset = MyDataset(split_data, doc_len=self.doc_len, dataset=self.dataset)
        data_len = self.cal_len(self.train_df, num_gpus)

        sampler = CustomDistributedSampler(train_dataset, data_len=data_len, num_replicas=num_gpus, rank=global_rank,
                                           batch_size=self.batch_size, block_size=self.block_size)


        return DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=collate_fn)
    

    # def train_dataloader(self):
    #     train_dataset = MyDataset(self.train_df, doc_len=self.doc_len, dataset=self.dataset)
    #     loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
    #     return loader



    def val_dataloader(self):
        val_dataset = MyDataset(self.val_df, doc_len=self.doc_len, dataset=self.dataset)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        return val_loader
    

    def test_dataloader(self):
        val_dataset = MyDataset(self.test_df, doc_len=self.doc_len, dataset=self.dataset)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        return val_loader



    def shuffle_df_by_label(self, df):
        # Step 1: 按 label 分组并在每个组内打乱
        start_time = time.time()
        grouped = df.groupby('label', group_keys=False)
        shuffled_within_group = grouped.apply(lambda x: x.sample(frac=1))
        end_time = time.time()
        duration = end_time - start_time
        print(f"Step1 shuffling took {duration} seconds to complete.")


        # Step 2: 获取所有唯一的标签，并打乱它们的顺序
        start_time = time.time()
        labels_shuffled = df['label'].unique()
        np.random.shuffle(labels_shuffled)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Step2 get unique label took {duration} seconds to complete.")


        pointers = {label: 0 for label in labels_shuffled}

        new_dfs = []

        while True:
            # 检查是否所有组都被完全处理
            start_time = time.time()
            all_processed = all(
                pointers[label] >= (len(shuffled_within_group[shuffled_within_group['label'] == label]) - self.batch_size) for label in
                labels_shuffled)
            if all_processed:
                break

            for label in labels_shuffled:
                current_group = shuffled_within_group[shuffled_within_group['label'] == label]
                start = pointers[label]
                end = min(start + self.batch_size, len(current_group))
                if start < end:
                    # 抽取当前组的下一个块并更新指针
                    new_dfs.append(current_group.iloc[start:end])
                    pointers[label] = end

            end_time = time.time()
            duration = end_time - start_time
            print(f"Step3 load one batch data took {duration} seconds to complete.")
        pointers = {label: 0 for label in labels_shuffled}
        # 合并所有片段
        new_df = pd.concat(new_dfs, ignore_index=True)

        # Step 3: 按照新的标签顺序重新组合 DataFrame
        # shuffled_df = pd.concat([shuffled_within_group.loc[shuffled_within_group['label'] == label] for label in labels_shuffled], ignore_index=True)
        return new_df




class IndexingDataset(Dataset):
    def __init__(self, doc_len, label_dict):
        self.data = pd.read_csv(
            "",
            encoding='utf-8', names=["docid", "input_ids", "attention_mask", "label"],
            header=None, sep='\t',
            dtype={'docid': int, 'input_ids': str, 'attention_mask': str, "label": str}).reset_index(drop=True)
        

        if len(label_dict) != 0:
            self.data["label"] = self.data["docid"].apply(lambda x: str(label_dict[x]))

        self.doc_len = doc_len



    def __getitem__(self, item):
        previous_label = eval(self.data.iloc[item]["label"])
        input_ids = eval(self.data.iloc[item]["input_ids"])[:self.doc_len]
        attention_mask = eval(self.data.iloc[item]["attention_mask"])[:self.doc_len]

        while len(input_ids) < self.doc_len:
            input_ids += [0]

        while len(attention_mask) < self.doc_len:
            attention_mask += [0]


        data = {
            "docid" : self.data.iloc[item]["docid"],
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "decoder_input_ids": torch.tensor([int(item) for item in previous_label]),
            "decoder_attention_mask": torch.ones(size=(len(previous_label), ), dtype=torch.int64),
            "previous_label": torch.tensor([int(item) for item in previous_label])
        }
        return data

    def __len__(self):
        return len(self.data)
    







class MrrDataset(Dataset):
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("")

        self.data = pd.read_csv(
                "",
                encoding='utf-8', names=["query", "docid", "label"],
                header=None, sep='\t',
                dtype={'query': str, 'docid': int, 'label': str}
            )
        


    def convert_to_features(self, example, padding="max_length", encoder_max_length=32):
        """
        没有别的special token
        """
        ret = self.tokenizer(
            example["query"],
            padding=padding,
            truncation=True,
            max_length=32,
        )

        ret["labels"] = example["docid"]

        ret["decoder_input_ids"] = [0]
        ret["decoder_attention_mask"] = [1]

        return ret


    def __getitem__(self, item):
        data = dict(self.data.iloc[item])
        data = self.convert_to_features(data)
        return data

    def __len__(self):
        return len(self.data)


def mrr_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.tensor([example["input_ids"] for example in examples])
    attention_mask = torch.tensor([example["attention_mask"] for example in examples])
    labels = [example["labels"] for example in examples]
    decoder_input_ids = torch.tensor([example["decoder_input_ids"] for example in examples])
    decoder_attention_mask = torch.tensor([example["decoder_attention_mask"] for example in examples])


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask
    }
