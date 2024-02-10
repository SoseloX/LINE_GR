import os.path
from functools import partial
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Config
from data import MyDataModule, IndexingDataset, mrr_collate_fn, MrrDataset
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import argparse
from model import rome
from module import rome_model
import torch
import pickle as pkl

data_root_path = ""
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")


def run(args):
    if args.process not in ["tree", "leaf", "align"]:
        raise ValueError("Invalid process type specified. Must be 'tree', 'leaf', or 'align'.")

    model_path = ""
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    if args.pat_path != "0":
        with open(args.pat_path, 'rb') as file:
            prefix_allowed_tokens_fn_dict = pkl.load(file)
    else:
        prefix_allowed_tokens_fn_dict = None

    if args.label_path != "0":
        with open(args.label_path, 'rb') as file:
            label_dict = pkl.load(file)
    else:
        label_dict = {}

    if args.doc_embedding_path != "0":
        with open(args.doc_embedding_path, 'rb') as file:
            doc_embedding = pkl.load(file)
    else:
        doc_embedding = None

    # prefix_allowed_tokens_fn_dict = {0: [i for i in range(3, 11)]}

    model = rome_model(model_path, temperature=args.temperature, process=args.process,
                        prefix_allowed_tokens_fn_dict=prefix_allowed_tokens_fn_dict,
                       label_dict=label_dict, doc_embedding=doc_embedding)
    if args.ckpt_path != "0":
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    torch.cuda.empty_cache()

    hyperparams = vars(args)


    generate_kwargs = {"max_length": args.stage+1,
                       "num_beams": args.num_beams,
                       "num_return_sequences": args.num_return_sequences}

    indexing_data = IndexingDataset(doc_len=args.doc_len, label_dict=label_dict)
    indexing_dataloader = DataLoader(dataset=indexing_data, batch_size=512, shuffle=False, drop_last=False)



    if args.task == "train":
        train_dataset_path = os.path.join(data_root_path, args.train_dataset)
        val_dataset_path = os.path.join(data_root_path, args.dev_dataset)

        data_module = MyDataModule(train_path=train_dataset_path, val_path=val_dataset_path, test_path=None, label_dict=label_dict,
                                   mode = args.task, batch_size=args.batch_size, doc_len=args.doc_len, dataset=args.dataset, block_size=args.block_size)

        every_epoch_step = len(data_module.train_df) / (args.batch_size * args.accumulate_grad_batches * len(args.devices)) 
        total_step = every_epoch_step * args.epochs

        task = rome(model=model, tokenizer=tokenizer, lr=args.lr,
                    weight_decay=args.weight_decay, dataset=args.dataset,
                    total_step=total_step, generate_kwargs=generate_kwargs,
                    warmup_steps=args.num_warmup_ratio * total_step, indexing_dataloader=indexing_dataloader,
                    stage=args.stage, cluster_w=args.cluster_w, every_epoch_step=every_epoch_step,
                    instance_w=args.instance_w, assign_w=args.assign_w, doc_seq_w=args.doc_seq_w,
                    query_seq_w=args.query_seq_w, process=args.process, leaf_instance_w=args.leaf_instance_w,
                    save_ckpt_name=args.save_ckpt_name, prefix_allowed_tokens_fn_dict=prefix_allowed_tokens_fn_dict,
                    label_dict=label_dict)


        trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=args.devices, val_check_interval=1.0, precision="bf16",
                             check_val_every_n_epoch=1, accumulate_grad_batches=args.accumulate_grad_batches, num_sanity_val_steps=0,
                             reload_dataloaders_every_n_epochs=1, strategy="ddp")
        

        if args.ckpt_path != "0":
            task.load_state_dict(checkpoint['state_dict'])
        torch.cuda.empty_cache()
        
        trainer.fit(task, datamodule=data_module)
    elif args.task == "test":
        test_dataset_path = os.path.join(data_root_path, args.dev_dataset)

        data_module = MyDataModule(train_path=None, val_path=None, test_path=test_dataset_path, label_dict=label_dict,
                                   mode = args.task, doc_len=args.doc_len, dataset=args.dataset, block_size=args.block_size)

        total_step = 2024
        every_epoch_step = 2024
        task = rome(model=model, tokenizer=tokenizer, lr=args.lr,
                    weight_decay=args.weight_decay, dataset=args.dataset,
                    total_step=total_step, generate_kwargs=generate_kwargs,
                    warmup_steps=args.num_warmup_ratio * total_step, indexing_dataloader=indexing_dataloader,
                    stage=args.stage, cluster_w=args.cluster_w, every_epoch_step=every_epoch_step,
                    instance_w=args.instance_w, assign_w=args.assign_w, doc_seq_w=args.doc_seq_w,
                    query_seq_w=args.query_seq_w, process=args.process,
                    save_ckpt_name=args.save_ckpt_name, prefix_allowed_tokens_fn_dict=prefix_allowed_tokens_fn_dict,
                    label_dict=label_dict)

        task.load_state_dict(checkpoint['state_dict'])
        trainer = pl.Trainer(accelerator='gpu', devices=args.devices)
        trainer.test(model=task, datamodule=data_module)
        print(generate_kwargs)
    elif args.task == "test_mrr":
        test_dataset_path = os.path.join(data_root_path, args.dev_dataset)

        dataset = MrrDataset()
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
                                collate_fn=mrr_collate_fn)

        total_step = 2024
        every_epoch_step = 2024
        task = rome(model=model, tokenizer=tokenizer, lr=args.lr,
                    weight_decay=args.weight_decay, dataset=args.dataset,
                    total_step=total_step, generate_kwargs=generate_kwargs,
                    warmup_steps=args.num_warmup_ratio * total_step, indexing_dataloader=indexing_dataloader,
                    stage=args.stage, cluster_w=args.cluster_w, every_epoch_step=every_epoch_step,
                    instance_w=args.instance_w, assign_w=args.assign_w, doc_seq_w=args.doc_seq_w,
                    query_seq_w=args.query_seq_w, process=args.process,
                    save_ckpt_name=args.save_ckpt_name, prefix_allowed_tokens_fn_dict=prefix_allowed_tokens_fn_dict,
                    label_dict=label_dict)

        task.load_state_dict(checkpoint['state_dict'])
        trainer = pl.Trainer(accelerator='gpu', devices=args.devices)
        trainer.test(model=task, dataloaders=dataloader)
        print(generate_kwargs)
    


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument("--lr", type=float, help="", default=5e-4, required=False)
    parser.add_argument("--weight_decay", type=float, help="", default=1e-4, required=False)
    parser.add_argument("--task", type=str, help="", default="train")
    parser.add_argument("--batch_size", type=int, help="", default=128)
    parser.add_argument("--block_size", type=int, help="", default=128)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--epochs", type=int, help="", default=50)
    parser.add_argument("--dataset", type=str, help="", default='')
    parser.add_argument("--train_dataset", type=str, help="", default='')
    parser.add_argument("--dev_dataset", type=str, help="", default='')
    parser.add_argument("--doc_len", type=int, help="", default=128)
    parser.add_argument("--num_return_sequences", type=int, help="", default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, help="", default=1)
    parser.add_argument("--num_beams", type=int, help="", default=1)
    parser.add_argument("--base_model", type=str, help="sentence transformer model", default="t5-base")
    parser.add_argument("--num_warmup_ratio", type=float, help="warmup step", default=0.02)
    parser.add_argument("--stage", type=int, help="stage", default=1)
    parser.add_argument("--ckpt_path", type=str, help="ckpt_path 0则代表没有", 
                        default="")
    parser.add_argument("--pat_path", type=str, help="prefix_allowed_tokens_fn_dict的path 0则代表没有", 
                        default="")
    parser.add_argument("--label_path", type=str, help="label_dict的path 0则代表没有",
                        default="")
    parser.add_argument("--doc_embedding_path", type=str, help="doc_embedding的path 0则代表没有",
                        default="0")
    parser.add_argument("--temperature", type=float, help="temperature", default=0.05)
    parser.add_argument("--cluster_w", type=float, help="clustering loss weight", default=1.0)
    parser.add_argument("--instance_w", type=float, help="instance loss weight", default=1.0)
    parser.add_argument("--leaf_instance_w", type=float, help="leaf instance loss weight", default=1.0)
    parser.add_argument("--assign_w", type=float, help="assignment loss weight", default=1.0)
    parser.add_argument("--doc_seq_w", type=float, help="sequence loss weight", default=0.0)
    parser.add_argument("--query_seq_w", type=float, help="sequence loss weight", default=1.0)
    parser.add_argument("--save_ckpt_name", type=str, help="The file name of best result after training", default="t1")
    parser.add_argument("--process", type=str, help="tree,leaf,align", default="tree")

    args = parser.parse_args()

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")


    seed_everything(2023)
    run(args)
