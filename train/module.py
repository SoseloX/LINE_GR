import copy
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import math
import pickle as pkl
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Model, T5Stack, T5ForConditionalGeneration, T5LayerFF
from transformers.models.t5.configuration_t5 import T5Config
from transformers.activations import ACT2FN
import torch.distributions as dist
import numpy as np
from collections import deque
import time


class rome_model(nn.Module):
    def __init__(self, 
                 base_model_path="", 
                 temperature=1.0, 
                 prefix_allowed_tokens_fn_dict=None,
                 label_dict = None,
                 doc_embedding = None,
                 beam_size = 3,
                 process = "align"):
        super(rome_model, self).__init__()
        shared_t5 = MyT5.from_pretrained(base_model_path)
        self.query_t5 = shared_t5
        self.doc_t5 = shared_t5
        self.process = process
        new_embedding = nn.Embedding(10000, 768)
        self.query_t5.replace_decoder_embeddings_and_lm_head(new_embeddings = new_embedding)
        self.doc_t5.replace_decoder_embeddings_and_lm_head(new_embeddings = new_embedding)
        self.temperature = temperature
        self.prefix_allowed_tokens_fn_dict = prefix_allowed_tokens_fn_dict
        self.seq_alpha = nn.Parameter(torch.tensor(1.0))  # generate得到的序列的分数与双塔分数权重
        self.beam_size = beam_size
        self.doc_embedding = doc_embedding
        self.label2docid = self.generate_label2docid(label_dict)


    def my_prefix_allowed_tokens_fn(self, batch_id, input_ids):
        list_input_ids = [str(i.item()) for i in input_ids]
        return self.prefix_allowed_tokens_fn_dict[input_ids[-1].item()]


    def refresh_embedding(self, new_embedding):
        self.query_t5.replace_decoder_embeddings_and_lm_head(new_embeddings = new_embedding)
        self.doc_t5.replace_decoder_embeddings_and_lm_head(new_embeddings = new_embedding)


    def id_to_embedding(self, input_ids):
        input_emb = self.query_t5.decoder.embed_tokens(input_ids)
        input_emb = input_emb.detach()
        return input_emb
    

    def generate_label2docid(self, label_dict):
        if label_dict is None:
            return None
        label2docid = {}
        for key, val in label_dict.items():
            if val[-1] not in label2docid:
                label2docid[val[-1]] = [key]
            else:
                label2docid[val[-1]].append(key)
        return label2docid



    def mean_pooling(self, outputs, attention_mask):
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(outputs)

        masked_outputs = outputs * expanded_attention_mask

        sum_outputs = torch.sum(masked_outputs, dim=1)
        mean_pooled = sum_outputs / expanded_attention_mask.sum(dim=1)

        mean_pooled = F.normalize(mean_pooled, p=2, dim=1)

        return mean_pooled


    def forward(self, batch, stage, current_step, every_epoch_step):
        loss_fct = nn.CrossEntropyLoss()
        doc_feature = self.doc_embedding.to(batch['input_ids'].device)
        query_inputs = {'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}
        

        query_features = self.query_t5.encoder_forward(**query_inputs)

        query_enc_features = self.mean_pooling(query_features, batch['attention_mask'])

        score = query_enc_features @ doc_feature.T 
        labels = batch['labels'].squeeze(1)
        loss = loss_fct(score / 0.01, labels)
        return loss



        
    def tree_forward(self, batch, stage, current_step, every_epoch_step):
        # self.unfreeze_lm_head()

        loss_fct = nn.CrossEntropyLoss()
        query_inputs = {'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}

        doc_inputs = {'input_ids': batch['doc_input_ids'],
                        'attention_mask': batch['doc_attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}
        bsz = batch['input_ids'].shape[0]
        previous_label = batch["previous_label"] # 提取之前解码出来的label

        query_features = self.query_t5.encoder_forward(**query_inputs)
        doc_features = self.doc_t5.encoder_forward(**doc_inputs)


        previous_label_emb = self.id_to_embedding(previous_label)
        query_logits, query_outputs = self.query_t5.decoder_forward(decoder_inputs_embeds = previous_label_emb,
                                                                    hidden_states = query_features)

        doc_logits, doc_outputs = self.doc_t5.decoder_forward(decoder_inputs_embeds = previous_label_emb,
                                                              hidden_states = doc_features)

        # 文本和query计算相似度（通过query_logits计算）
        query_outputs_end = query_outputs[:, -1, :]
        doc_outputs_end = doc_outputs[:, -1, :]
        query_logits_end = query_logits[:, -1, :]
        doc_logits_end = doc_logits[:, -1, :]
        current_epoch = current_step / every_epoch_step


        doc_target_id = self.generate_target_id(batch, doc_logits_end, current_epoch)

        step_score = torch.matmul(query_outputs_end, doc_outputs_end.transpose(0, 1))
        q_labels = batch["labels"].repeat(1, bsz)
        d_labels = batch["labels"].reshape(1, -1).repeat(bsz, 1)

        q_target_id = doc_target_id.reshape(bsz, 1).repeat(1, bsz)
        d_target_id = doc_target_id.reshape(1, bsz).repeat(bsz, 1)
        target_mask = q_target_id == d_target_id
        valid_mask = d_target_id != -1

        label_mask = q_labels==d_labels
        final_mask = label_mask | (target_mask & valid_mask)
        # final_mask = label_mask

        final_mask.fill_diagonal_(False)


        step_score = step_score.masked_fill(final_mask, -1e5)

        labels = torch.arange(start=0, end=bsz).to(step_score.device)
        anneal_temp = max(0.05, math.exp(-current_epoch))
        instance_loss = loss_fct(step_score / anneal_temp, labels)  # query与doc之间的对比损失

        if stage == 1:
            qd_cluster_loss, assigment_loss = self.prefix_aware_cluster_optimization(batch, query_logits_end, doc_logits_end, doc_target_id, anneal_temp)
            query_seq_loss, doc_seq_loss = 0, 0
        elif 2 <= stage:
            qd_cluster_loss, assigment_loss = self.prefix_aware_cluster_optimization(batch, query_logits_end, doc_logits_end, doc_target_id, anneal_temp)

            query_seq_loss = self.prefix_aware_seq_optimization(batch, query_outputs)
            doc_seq_loss = self.prefix_aware_seq_optimization(batch, doc_outputs)

        return instance_loss, qd_cluster_loss, assigment_loss, query_seq_loss, doc_seq_loss


    def leaf_forward(self, batch, stage, current_step, every_epoch_step):

        loss_fct = nn.CrossEntropyLoss()
        query_inputs = {'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}

        doc_inputs = {'input_ids': batch['doc_input_ids'],
                      'attention_mask': batch['doc_attention_mask'],
                      'decoder_input_ids': batch['decoder_input_ids'],
                      'decoder_attention_mask': batch['decoder_attention_mask']}
        bsz = batch['input_ids'].shape[0]

        query_features = self.query_t5.encoder_forward(**query_inputs)
        doc_features = self.doc_t5.encoder_forward(**doc_inputs)

        query_enc_features = self.mean_pooling(query_features, batch['attention_mask'])
        doc_enc_features = self.mean_pooling(doc_features, batch['doc_attention_mask'])
        previous_label = batch["previous_label"]
        previous_label_emb = self.id_to_embedding(previous_label)
        query_logits, query_outputs = self.query_t5.decoder_forward(decoder_inputs_embeds=previous_label_emb,
                                                            hidden_states=query_features)


        leaf_score = torch.matmul(query_enc_features, doc_enc_features.transpose(0, 1))
        q_labels = batch["labels"].repeat(1, bsz)
        d_labels = batch["labels"].reshape(1, -1).repeat(bsz, 1)
        mask = q_labels == d_labels


        mask.fill_diagonal_(False)
        leaf_score = leaf_score.masked_fill(mask, -1e5)
        labels = torch.arange(start=0, end=bsz).to(leaf_score.device)
        leaf_instance_loss = loss_fct(leaf_score / self.temperature, labels)


        # query_seq_loss = self.prefix_aware_seq_optimization(batch, query_outputs)
        query_seq_loss = 0

        return leaf_instance_loss, query_seq_loss


    def align_forward(self, batch, stage, current_step, every_epoch_step):

        loss_fct = nn.CrossEntropyLoss()
        query_inputs = {'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}

        bsz = batch['input_ids'].shape[0]
        beam_size = self.beam_size
        seq_len = batch['previous_label'].shape[1]

        previous_label = batch["previous_label"]  # 提取之前解码出来的label
        previous_label_3d = previous_label.unsqueeze(1) # bsz 1 seq_len

        query_features = self.query_t5.encoder_forward(**query_inputs)
        query_enc_features = self.mean_pooling(query_features, batch['attention_mask'])
        previous_label_emb = self.id_to_embedding(previous_label)
        query_logits, query_outputs = self.query_t5.decoder_forward(decoder_inputs_embeds=previous_label_emb,
                                                                    hidden_states=query_features)

        query_seq_loss = self.prefix_aware_seq_optimization(batch, query_outputs)

        ce_label = []
        sample_docid = []
        for previous_label in batch["previous_label"]:
            leaf_label = previous_label[-1]
            sample_docid.extend(self.label2docid[leaf_label.item()])
        
        sample_docid = list(set(sample_docid))
        for label in batch["labels"]:
            ce_label.append(sample_docid.index(label[0].item()))

        doc_feature = self.doc_embedding[sample_docid]
        ce_label = torch.tensor(ce_label).to(batch['input_ids'].device)
        sample_docid = torch.tensor(sample_docid).to(batch['input_ids'].device)

        
        doc_feature = torch.tensor(doc_feature).to(batch['input_ids'].device)

        score = query_enc_features @ doc_feature.T
        leaf_loss = loss_fct(score / 0.05, ce_label)


        return leaf_loss, query_seq_loss



    def prefix_aware_cluster_optimization(self, batch, query_logits, doc_logits, doc_target_id, anneal_temp):
        """
        用于计算contrastive cluster loss 每个簇分别计算
        """
        previous_label = batch['previous_label'].cpu().tolist()
        loss_fct = nn.CrossEntropyLoss()
        # 创建一个字典来保存不同previous_label的索引
        label_to_indices = {}
        for i, label in enumerate(previous_label):
            if label[-1] not in label_to_indices:
                label_to_indices[label[-1]] = []
            label_to_indices[label[-1]].append(i)
    
        cluster_loss = 0.0
        assigment_loss = 0.0
        # 遍历每个不同的previous_label
        for label, indices in label_to_indices.items():
            # 获取每个label对应的doc_logits和doc_logits_aug
            sub_size = len(indices)
            cluster_indices = torch.tensor(self.prefix_allowed_tokens_fn_dict[label]).to(query_logits.device)
            cluster_num = len(self.prefix_allowed_tokens_fn_dict[label])

            valid_query_logits = query_logits[indices, :][:, cluster_indices]
            valid_doc_logits = doc_logits[indices, :][:, cluster_indices]

            norm_query_logits = F.softmax(valid_query_logits, dim=1)
            norm_doc_logits = F.softmax(valid_doc_logits, dim=1)
            
            uniform_prob = torch.ones(size=valid_query_logits.size()).to(valid_query_logits.device)
            smooth_query_logits = 0.99*norm_query_logits + 0.01*(uniform_prob/cluster_num)
            smooth_doc_logits = 0.99*norm_doc_logits + 0.01*(uniform_prob/cluster_num)
            score = smooth_query_logits @ smooth_doc_logits.T
            sub_doc_target_id = doc_target_id[indices]
            sub_label = batch["labels"][indices]
            q_target_id = sub_doc_target_id.reshape(sub_size, 1).repeat(1, sub_size)
            d_target_id = sub_doc_target_id.reshape(1, sub_size).repeat(sub_size, 1)
            target_mask = q_target_id == d_target_id # 计算出来的label相同要剔除
            valid_mask = d_target_id != -1 # 只有确定了才可以剔除

            q_label = sub_label.reshape(sub_size, 1).repeat(1, sub_size)
            d_label = sub_label.reshape(1, sub_size).repeat(sub_size, 1)
            label_mask = q_label == d_label  # doc相同的要剔除

            final_mask = label_mask | (target_mask & valid_mask)
            final_mask.fill_diagonal_(False)
            score = score / anneal_temp
            score = score.masked_fill(final_mask, -1e5)



            labels = torch.arange(start=0, end=score.shape[0]).to(score.device)
            cluster_loss += loss_fct(score, labels)
            assigment_loss += self.cal_assignment_loss_info(norm_doc_logits)
    
        return cluster_loss, assigment_loss


    def prefix_aware_seq_optimization(self, batch, outputs):
        """
        用于计算seq loss，只有簇内结点是有效的，不会对之前的结点更新
        :param batch:直接从dataloader取出的batch
        :param outputs:直接从decoder_forward方法取出的outputs，维度为bsz,seq len,dim，注意最后一个seq不能要

        :retrun: 只用同个簇内结点的sequence loss
        """
        lm_head_weight = self.doc_t5.lm_head.weight.clone().detach()

        labels_wo_0 = batch["previous_label"][:, 1:]
        labels_wo_end = batch['previous_label'][:, :-1].cpu().tolist()
        loss_fct = nn.CrossEntropyLoss()
        outputs = outputs[:, :-1, :]
        outputs_2d = outputs.reshape(-1, 768)
        previous_logits = torch.matmul(outputs_2d, lm_head_weight.T)
        # previous_logits = self.doc_t5.lm_head(outputs_2d)
        mask_matrix = torch.ones(previous_logits.shape[0], previous_logits.shape[1], dtype=torch.bool).to(
            previous_logits.device)

        label_wo_end_int = []
        cnt = 0
        for row in labels_wo_end:
            for node in row:
                label_wo_end_int.append(node)
                selected_indices = torch.tensor(self.prefix_allowed_tokens_fn_dict[node]).to(
                    previous_logits.device)
                mask_matrix[cnt, selected_indices] = False
                cnt += 1

        labels_wo_0 = labels_wo_0.reshape(-1)

        logits = previous_logits.masked_fill(mask_matrix, -1e5)
        loss = loss_fct(logits, labels_wo_0)
        return loss



    def inference_generate(
            self,
            encoder_features_out,
            target_seq,
            attention_mask,
            bsz,
            beam_size
    ):
        """
        给定encoder生成好的向量和Decoder的输入，生成对应的label下面所有文档的分数

        :param encoder_features_out: bsz*beam_size, query_max_len, dim 的向量
        :param target_seq: bsz*beam_size, seq_len 的向量
        :return:
        """
        doc_emb = torch.cat([self.doc_embedding, torch.zeros(size=(1, 768))], dim=0).to(encoder_features_out.device)
        pad_idx = doc_emb.shape[0] - 1

        encoder_features_out_expand = torch.repeat_interleave(encoder_features_out, beam_size, dim=0)
        decoder_inputs_embeds = self.id_to_embedding(target_seq)
        query_logits_beam, query_outputs_beam = self.query_t5.decoder_forward(decoder_inputs_embeds=decoder_inputs_embeds,
                                                                              hidden_states=encoder_features_out_expand)

        leaf_label = target_seq[:, -1]

        docid_list = []
        for leaf in leaf_label:
            docid_list.append(self.label2docid[leaf.item()])

        max_length = max(len(row) for row in docid_list)

        # 使用指定的填充值（例如0）填充每一行
        padded_docid_list = [row + [pad_idx] * (max_length - len(row)) for row in docid_list]
        padded_docid_tensor = torch.tensor(padded_docid_list) # bsz*beam_size, max_length
        padded_docid_tensor_bsz = padded_docid_tensor.reshape(bsz, -1).to(encoder_features_out.device) # bsz, beam_size*max_length

        # doc_vec = self.doc_embedding[padded_docid_tensor_3d].to(encoder_features_out.device) # bsz, beam_size*max_length, dim

        query_enc_features = self.mean_pooling(encoder_features_out, attention_mask) # bsz, dim, 1
        instance_score = query_enc_features @ doc_emb.T
        embedding_score = torch.gather(instance_score, 1, padded_docid_tensor_bsz)


        final_score = embedding_score # bsz*beam_size, max_length
        mask = padded_docid_tensor_bsz == pad_idx
        mask = mask.to(final_score.device)
        final_score = final_score.masked_fill(mask, -1e5)
        final_score = final_score.reshape(bsz, -1) # bsz, beam_size*max_length
        predict_docid = padded_docid_tensor_bsz
        sorted_scores, indices = torch.sort(final_score, descending=True, dim=1)

        # 使用获取的索引来重新排列predict_docid
        sorted_predict_docid = torch.gather(predict_docid, 1, indices)


        return sorted_scores, sorted_predict_docid




    def generate_target_id(self, batch, doc_logits, current_epoch):
        """
        在固定的区间找到最大值，生成label
        """
        max_indices = []
        max_scores = []
        threshold = {}
        instance_threshold = {}
        label_score_memory = {i: [] for i in range(3, 4500)}
        for idx in range(batch["previous_label"].shape[0]):
            parent_id = int(batch["previous_label"][idx][-1])
            previous_label = "-".join([str(i) for i in batch["previous_label"][idx].cpu().tolist()])
            logits = doc_logits[idx]
            allowed_indices = self.prefix_allowed_tokens_fn_dict.get(parent_id, [])

            # 创建一个掩码，只在允许的索引处为True
            mask = torch.zeros_like(logits, dtype=torch.bool).to(batch["labels"].device)
            mask[allowed_indices] = True

            # 应用掩码并找到最大值的索引
            masked_logits = logits.masked_select(mask)
            masked_logits = torch.softmax(masked_logits, dim=0)
            max_index = torch.argmax(masked_logits)
            max_score = masked_logits[max_index].item()

            # 将允许的索引列表转换为Tensor以便使用掩码索引
            allowed_indices_tensor = torch.tensor(allowed_indices, device=logits.device)
            max_vocab_indice = allowed_indices_tensor[max_index].item()
            max_indices.append(max_vocab_indice)
            max_scores.append(max_score)
            label_score_memory[max_vocab_indice].append(max_score)


        ratio = max(100 - current_epoch*100, 0)
        if ratio != 0:
            unique_indices = set(max_indices)
            for index in unique_indices:
                threshold[index] = np.percentile(label_score_memory[index], ratio)
        
            for idx, (index, score) in enumerate(zip(max_indices, max_scores)):
                if score <= threshold[index]:
                    max_indices[idx] = -1

        # for idx, (index, score) in enumerate(zip(max_indices, max_scores)):
        #     max_indices[idx] = -1

        
        max_indices = torch.tensor(max_indices).to(doc_logits.device)

        return max_indices


    @torch.no_grad()
    def generate(self, batch, beam_size):
        doc_feature = self.doc_embedding.to(batch['input_ids'].device)
        query_inputs = {'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'decoder_input_ids': batch['decoder_input_ids'],
                        'decoder_attention_mask': batch['decoder_attention_mask']}
        

        query_features = self.query_t5.encoder_forward(**query_inputs)
        


        query_enc_features = self.mean_pooling(query_features, batch['attention_mask'])

        score = query_enc_features @ doc_feature.T 
        _, e_index = torch.topk(score, beam_size, dim=1)

        return e_index




    @torch.no_grad()
    def doc_generate_embedding(self, 
                               input_ids,
                               attention_mask,
                               previous_label):
        """
        warmup阶段调用
        """
        doc_inputs = {'input_ids': input_ids,
                      'attention_mask': attention_mask}
        doc_features = self.doc_t5.encoder_forward(**doc_inputs)
        previous_label_emb = self.id_to_embedding(previous_label)
        doc_logits, doc_outputs = self.doc_t5.decoder_forward(decoder_inputs_embeds = previous_label_emb,
                                                              hidden_states = doc_features)
        return doc_outputs[:, -1]



    def cal_assignment_loss_info(self, logits):
        """
        :param logits:
        :return:
        logits是一个bsz cluster_num大小的向量
        依据信息熵来平衡
        """
        bsz = logits.shape[0]
        column_wise_sum = torch.sum(logits, dim=0)
        norm_1 = torch.sum(column_wise_sum)
        probability = column_wise_sum / bsz
        entropy = torch.sum(probability * torch.log(probability))

        uniform_probs = torch.full_like(probability, fill_value=1.0 / len(probability))
        const_entropy = torch.sum(uniform_probs * torch.log(uniform_probs)) # 防止负值

        return -const_entropy + entropy


















class MyT5(T5ForConditionalGeneration):

    def replace_decoder_embeddings_and_lm_head(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

        # 注意这里我们将线性层的权重转置，以使其可以作为lm_head使用
        new_lm_head = nn.Linear(new_embeddings.weight.size(1), new_embeddings.weight.size(0), bias=False)
        new_lm_head.weight = new_embeddings.weight
        self.set_output_embeddings(new_lm_head)
        # self.lm_head = new_lm_head


    def encoder_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        return hidden_states


    def decoder_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        # norms = torch.norm(sequence_output, p=2, dim=-1, keepdim=True)
        # sequence_output = sequence_output / norms

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)
        #
        # if self.config.tie_word_embeddings:
        #     # Rescale output before projecting on vocab
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #     sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        return lm_logits, sequence_output


    def end_representation_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        # norms = torch.norm(sequence_output, p=2, dim=-1, keepdim=True)
        # sequence_output = sequence_output / norms

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)
        #
        # if self.config.tie_word_embeddings:
        #     # Rescale output before projecting on vocab
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #     sequence_output = sequence_output * (self.model_dim**-0.5)

        return sequence_output[:, -1, :]


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        # norms = torch.norm(sequence_output, p=2, dim=-1, keepdim=True)
        # sequence_output = sequence_output / norms
        # print(torch.norm(sequence_output, p=2, dim=-1, keepdim=True))

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)
        #
        # if self.config.tie_word_embeddings:
        #     # Rescale output before projecting on vocab
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #     sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def encode_decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        # norms = torch.norm(sequence_output, p=2, dim=-1, keepdim=True)
        # sequence_output = sequence_output / norms
        # print(torch.norm(sequence_output, p=2, dim=-1, keepdim=True))

        return sequence_output


