import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
import torch
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import precision_recall_fscore_support
import pickle as pkl
import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd
import itertools
import time

warnings.filterwarnings("ignore")



class rome(pl.LightningModule):
    def __init__(self, model, tokenizer, lr=1e-3, weight_decay=0.0, dataset="NQ320", total_step = None, generate_kwargs = None,
                 warmup_steps = None, grad_clip = 1.0, args = None, indexing_dataloader = None, stage = None, process = None,
                 vocab_range = None, cluster_w = 1.0, instance_w = 1.0, assign_w = 1.0, doc_seq_w = 1.0, query_seq_w=1.0,
                 leaf_instance_w = 1.0, save_ckpt_name = "None", prefix_allowed_tokens_fn_dict = None, every_epoch_step = None,
                 label_dict = None) -> None:
        super().__init__()
        self.model = model
        self.indexing_dataloader = indexing_dataloader
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_step = total_step
        self.grad_clip = grad_clip
        self.label_dict = label_dict
        self.doc_embedding_dict = {}
        self.warmup_steps = warmup_steps
        self.generate_kwargs = generate_kwargs
        self.stage = stage
        self.vocab_range = vocab_range
        self.cluster_w = cluster_w
        self.instance_w = instance_w
        self.assign_w = assign_w
        self.doc_seq_w = doc_seq_w
        self.query_seq_w = query_seq_w
        self.leaf_instance_w = leaf_instance_w
        self.save_ckpt_name = save_ckpt_name
        self.prefix_allowed_tokens_fn_dict = prefix_allowed_tokens_fn_dict
        self.every_epoch_step = every_epoch_step
        self.process = process
        self.candidate_emb = {}
        self.label_candidate = {}
        self.doc_embedding = None




    def my_prefix_allowed_tokens_fn(self, batch_id, input_ids):
        list_input_ids = [str(i.item()) for i in input_ids]
        return self.prefix_allowed_tokens_fn_dict[input_ids[-1].item()]

    def build_ks_list(self):
        """
        该函数用于构建metric的recall@k的列表
        :return:
        """
        # if "NQ" in self.dataset:
        #     ks_list = [i for i in [1, 10, 100] if i <= self.generate_kwargs["num_return_sequences"]]
        # elif "Trivia" in self.dataset:
        #     ks_list = [i for i in [5, 20, 100] if i <= self.generate_kwargs["num_return_sequences"]]
        # else:
        #     raise ValueError("Invalid dataset!")

        return [1, 10, 100]

    def cal_hit(self, preds, labels, ks_list):
        """
        preds是batch size*return num的二维int数组 item为预测的docid
        labels是batch size的一位数组为ground truth
        ks_list是待评估的候选集大小
        :param preds:
        :param labels:
        :return:
        """
        return_dict = {}
        # labels = [[0] + label for label in labels]
        mrr = 0.0
        for ks in ks_list:
            hit = 0.0
            for idx in range(len(labels)):
                # print(labels[idx], preds[idx][:ks])
                if labels[idx] in preds[idx][:ks]:
                    hit += 1
            return_dict[f"val_recall@{ks}"] = hit / len(labels)

        for idx in range(len(labels)):
            for rank, pred in enumerate(preds[idx][:100]):
                if pred == labels[idx]:
                    mrr += 1/(rank+1)

        return_dict["val_mrr@100"] = mrr / len(labels)


        return return_dict


    def cal_mrr(self, preds, labels):
        """
        preds是batch size*return num的二维int数组 item为预测的docid
        labels是batch size的一位数组为ground truth
        ks_list是待评估的候选集大小
        :param preds:
        :param labels:
        :return:
        """
        return_dict = {}
        # labels = [[0] + label for label in labels]
        recall10 = 0
        recall100 = 0
        mrr10 = 0
        for idx, item in enumerate(preds):
            count = sum(1 for element in labels[idx] if element in item[:10])
            recall10 += count / len(labels[idx])

            count = sum(1 for element in labels[idx] if element in item[:100])
            recall100 += count / len(labels[idx])

            first_hit = 1e5
            for idx, k in enumerate(labels[idx]):
                if k in item[:10]:
                    first_hit = min(first_hit, item[:10].index(k))

            mrr10 += 1/(first_hit+1)

            

        return_dict["recall@10"] = recall10 / len(labels)
        return_dict["recall@100"] = recall100 / len(labels)
        return_dict["mrr@10"] = mrr10 / len(labels)

        return return_dict



    def training_step(self, batch, batch_idx):
        if self.process == "tree":
            instance_loss, qd_cluster_loss, assigment_loss, query_seq_loss, doc_seq_loss= self.model.tree_forward(batch, self.stage, self.global_step, self.every_epoch_step)
            loss =  self.instance_w * instance_loss + \
                    self.cluster_w * qd_cluster_loss + \
                    self.assign_w * assigment_loss + \
                    self.doc_seq_w * doc_seq_loss + \
                    self.query_seq_w * query_seq_loss
            self.log("train_instance_loss", instance_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_cluster_loss", qd_cluster_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_query_seq_loss", query_seq_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_doc_seq_loss", doc_seq_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_assigment_loss", assigment_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.process == "leaf":
            leaf_instance_loss, query_seq_loss = self.model.leaf_forward(batch, self.stage, self.global_step, self.every_epoch_step)
            loss =  self.leaf_instance_w * leaf_instance_loss + self.query_seq_w*query_seq_loss
            self.log("train_leaf_instance_loss", leaf_instance_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_query_seq_loss", query_seq_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.process == "align":
            leaf_instance_loss, query_seq_loss = self.model.align_forward(batch, self.stage, self.global_step, self.every_epoch_step)
            loss =  self.leaf_instance_w * leaf_instance_loss + \
                    self.query_seq_w * query_seq_loss
            self.log("train_leaf_instance_loss", leaf_instance_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_query_seq_loss", query_seq_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss





    def cal_assigment_balanced(self):
        """
        用于计算每次分配label的平衡
        """
        count_dict = {}
        
        def dfs(label_list):
            label_str = "-".join([str(i) for i in label_list])
            if label_list[-1] not in self.prefix_allowed_tokens_fn_dict:
                count_dict[label_list[-1]] = 0
                return

            for next_token in self.prefix_allowed_tokens_fn_dict[label_list[-1]]:
                label_list.append(next_token)
                dfs(label_list)
                label_list.pop()

        dfs([0])
        # del count_dict["0"]
        for value_list in self.label_dict.values():
            if value_list:  # 确保列表不为空
                label_str = "-".join([str(i) for i in value_list])
                count_dict[value_list[-1]] += 1


        res = 0
        for value in count_dict.values():
            res += abs(value - float(len(self.label_dict)/len(count_dict)))

        res = 1 - res / (2*(len(self.label_dict)))

        print(f">>>>>>>> Balanced score:{res}  \n")




    def align_generate(self, batch):
        query_inputs = {'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}

        bsz = batch['input_ids'].shape[0]
        beam_size = self.generate_kwargs["num_beams"]

        query_features = self.model.query_t5.encoder_forward(**query_inputs)

        generate_kwargs = {"max_length": self.stage+1,
                           "num_beams": beam_size,
                           "num_return_sequences": beam_size}
        out = self.model.query_t5.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            **generate_kwargs,
            prefix_allowed_tokens_fn=self.my_prefix_allowed_tokens_fn
        )

        final_score, predict_docid = self.model.inference_generate(
            encoder_features_out = query_features,
            target_seq = out,
            attention_mask = batch['attention_mask'],
            bsz = bsz,
            beam_size = beam_size
        )
        rank_list = predict_docid.tolist()
        if isinstance(batch["labels"], list):
            return self.cal_mrr(rank_list, batch["labels"])
        else:
            batch_labels = batch["labels"].flatten().tolist()
            return self.cal_hit(rank_list, batch_labels, self.build_ks_list())





    def leaf_generate(self, batch):

        query_rep = self.model.query_t5.encoder_forward(input_ids=batch['input_ids'],
                                                        attention_mask=batch['attention_mask'])
        
        query_enc_features = self.model.mean_pooling(query_rep, batch['attention_mask'])

        query_enc_features = query_enc_features.cpu()
        score = query_enc_features @ self.doc_embedding.T
        _, pred = torch.topk(score, 100, dim=1)
        pred = pred.tolist()
        batch_labels = []
        for docid in batch['labels']:
            batch_labels.append(int(docid[0]))

        return self.cal_hit(pred, batch_labels, self.build_ks_list())


    def embedding_generate(self, task):
        doc_emb = []
        for data in tqdm(self.indexing_dataloader):
            input_ids = data['input_ids'].to(self.model.doc_t5.device)
            attention_mask = data['attention_mask'].to(self.model.doc_t5.device)

            doc_rep = self.model.doc_t5.encoder_forward(input_ids=input_ids,
                                                        attention_mask=attention_mask)

            doc_rep = self.model.mean_pooling(doc_rep, attention_mask)

            for idx, rep in enumerate(doc_rep):
                doc_emb.append(rep.cpu())

        doc_emb = torch.stack(doc_emb, dim=0)
        self.doc_embedding = doc_emb
        if task == "test":
            with open("", "wb") as f:
                pkl.dump(doc_emb, f)


    
    def label_generate(self, task):
        # lm_head = self.model.query_t5.lm_head.weight
        # prob_matrix = []
        doc_emb = []
        label2docid = {}
        generate_kwargs = {
             "max_length": self.stage+1,
             "num_beams": 1,
             "num_return_sequences": 1
        }
        for data in tqdm(self.indexing_dataloader):
            bsz = data['input_ids'].shape[0]

            input_ids = data['input_ids'].to(self.model.doc_t5.device)
            attention_mask = data['attention_mask'].to(self.model.doc_t5.device)
            decoder_input_ids = data['decoder_input_ids'].to(self.model.doc_t5.device)
            decoder_attention_mask = data['decoder_attention_mask'].to(self.model.doc_t5.device)
            previous_label = data['previous_label'].to(self.model.doc_t5.device)
            doc_labels = self.model.doc_t5.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                decoder_input_ids=previous_label,
                                                **generate_kwargs,
                                                prefix_allowed_tokens_fn=self.my_prefix_allowed_tokens_fn)


            for idx, label in enumerate(doc_labels):
                label = label.cpu()
                label = [int(i) for i in label]
                self.label_dict[int(data['docid'][idx])] = label

        if task == "test":
            with open("", "wb") as f:
                pkl.dump(self.label_dict, f)

        self.model.label2docid = self.model.generate_label2docid(self.label_dict)
        # self.cal_assigment_balanced()



    def on_validation_epoch_start(self) -> None:
        if self.process == "tree":
            self.label_generate("val")
        elif self.process == "leaf":
            self.embedding_generate("val")


    def on_test_epoch_start(self) -> None:
        if self.process == "tree":
            self.label_generate("test")
        elif self.process == "leaf":
            self.embedding_generate("test")






    def val_test_step(self, batch):
        if self.process == "tree":
            result_dict = self.generate_step(batch)
        elif self.process == "leaf":
            result_dict = self.leaf_generate(batch)
        elif self.process == "align":
            result_dict = self.align_generate(batch)

        self.log_dict(result_dict, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return result_dict


    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch)


    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch)


    def _filter(self, out_i):
        tail = out_i.pop()
        while tail == self.tokenizer.pad_token_id:
            tail = out_i.pop()
        out_i.append(tail)
        return out_i

    def _filter_pad(self, out):
        return [list(self._filter(out_i.cpu().numpy().tolist())) for out_i in out]


    def pad_to_max(self, out):
        for idx in range(len(out)):
            if len(out[idx]) < self.generate_kwargs["max_length"]:
                out[idx] = out[idx] + [0] * (self.generate_kwargs["max_length"] - len(out))
        return out
    

    def generate_step(self, batch):
        out = self.model.query_t5.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            **self.generate_kwargs,
            prefix_allowed_tokens_fn=self.my_prefix_allowed_tokens_fn
        )
        out = self.pad_to_max(out)
        batch_labels = []
        for docid in batch['labels']:
            batch_labels.append(self.label_dict[int(docid[0])])
        pred = [x.tolist() for x in np.array_split(out, batch['input_ids'].shape[0])]
        return self.cal_hit(pred, batch_labels, self.build_ks_list())


    def generate(self, batch):
        pred = self.model.generate(batch, 100)
        pred = pred.tolist()
        batch_labels = []
        for docid in batch['labels']:
            batch_labels.append(int(docid[0]))

        return self.cal_hit(pred, batch_labels, self.build_ks_list())


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler_ = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps
        )
        scheduler = {
            'scheduler': scheduler_,
            'interval': 'step'
        }

        return [optimizer], [scheduler]


    def configure_callbacks(self):

        checkpoint = ModelCheckpoint(dirpath="",
                                    monitor="val_recall@10",
                                    filename=self.dataset+"-"+self.save_ckpt_name + '-{epoch}-{step}-{val_recall@10:.5f}',
                                    save_top_k=1,
                                    mode='max',
                                    save_last=False,
                                    verbose=True,
                                    save_weights_only=False)
        # weight_averaging = StochasticWeightAveraging(swa_lrs=2e-4)
        return [checkpoint]


