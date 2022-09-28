
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import os
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from fairseq import metrics
from fairseq.tasks import register_task

from tasks.ofa_task import OFAConfig, OFATask
from data.mm_data.imagecode_dataset import ImagecodeRetrievalDataset
from data.file_dataset import FileDataset
from data import data_utils
from utils.trie import Trie

logger = logging.getLogger(__name__)


@dataclass
class ImagecodeRetrievalConfig(OFAConfig):
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )
    add_caption: bool = field(
        default=False,
        metadata={"help": "add caption to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )

class CustomDataset:
    def __init__(self, file_path, split):
        self.file_path = os.path.join(file_path, split) 
        self.dataset = self.get_data_from_filepath(file_path, split)
        self.total_row_count = len(self.dataset)

        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1
        self._init_seek_index()

        print("file {} slice_id {} row count {} total row count {}".format(
            self.file_path, self.slice_id, self.row_count, self.total_row_count)
        )

    def __len__(self):
        return len(self.dataset)

    def _init_seek_index(self):
        self._compute_start_pos_and_row_count()
        print("local datafile {} slice_id {} finished initializing row_count and line_idx-to-offset mapping".format(
            self.file_path, self.slice_id))

    def _compute_start_pos_and_row_count(self):
        self.row_count = self.total_row_count // self.slice_count
        if self.slice_id < self.total_row_count - self.row_count * self.slice_count:
            self.row_count += 1
            self.start_pos = self.row_count * self.slice_id
        else:
            self.start_pos = self.row_count * self.slice_id + (self.total_row_count - self.row_count * self.slice_count)

    def _seek(self, offset=0):
        try:
            print("slice_id {} seek offset {}".format(self.slice_id, self.start_pos + offset))
            # self._reader.seek(self.lineid_to_offset[self.start_pos + offset])
            # self.data_cnt = offset
        except Exception:
            print("slice_id {} seek offset {}".format(self.slice_id, offset))
            # self._reader.seek(self.lineid_to_offset[offset])
            # self.data_cnt = offset

    def get_data_from_filepath(self, data_dir, split):
        with open(os.path.join(data_dir, 'annotations', f'{split}_data.json')) as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{data_dir}/image-sets/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for target, text in data.items():
                for idx, image in enumerate(img_files):
                    row = (image, text, "yes" if idx == int(target) else "no")
                    dataset.append(row)
        
        return dataset

    def get_total_row_count(self):
        return self.total_row_count

    def __getitem__(self, index):
        return self.dataset[index]

@register_task("imagecode", dataclass=ImagecodeRetrievalConfig)
class ImagecodeRetrievalTask(OFATask):
    def __init__(self, cfg: ImagecodeRetrievalConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.ans2label_dict = json.loads(self.cfg.ans2label_dict)
        
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        data_dir = "/data/data_store/rabiul/codes/OFA/dataset/imagecode_data"
        # dataset = FileDataset(file_path, self.cfg.selected_cols)
        dataset = CustomDataset(data_dir, split)

        self.datasets[split] = ImagecodeRetrievalDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_caption=self.cfg.add_caption,
            constraint_trie=self.constraint_trie,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        answer_item_list = []
        self.index2ans = {}
        self.constraint_trie = Trie(self.tgt_dict.eos())
        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tgt_dict.encode_line(
                line=self.bpe.encode(' ' + answer),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            answer_item_list.append(answer_item)
            self.index2ans[i] = answer
            self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])

        constraint_mask_list = []
        for answer_item in answer_item_list:
            constraint_mask = torch.zeros((len(answer_item)+1, len(self.tgt_dict))).bool()
            for i in range(len(answer_item)+1):
                constraint_prefix_token = [self.src_dict.bos()] + answer_item[:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        self.valid_answers_list = []
        self.valid_constraint_masks_list = []
        for i in range(0, len(answer_item_list), self.cfg.valid_batch_size):
            self.valid_answers_list += [answer_item_list[i:i+self.cfg.valid_batch_size]]
            self.valid_constraint_masks_list += [constraint_mask_list[i:i+self.cfg.valid_batch_size]]

        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator


    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        model.eval()
        with torch.no_grad():
            encoder_out = model.encoder(
                sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                patch_images=sample["net_input"]["patch_images"],
                patch_masks=sample["net_input"]["patch_masks"]
            )
            device = sample["net_input"]["src_tokens"].device
            eos_item = torch.tensor([self.src_dict.eos()])
            pad = self.src_dict.pad()
            valid_result = []
            for valid_answers, valid_constraint_masks in zip(self.valid_answers_list, self.valid_constraint_masks_list):
                valid_size = len(valid_answers)
                valid_tgt_items = [
                    torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                    for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
                ]
                valid_prev_items = [
                    torch.cat([torch.tensor(decoder_prompt), valid_answer])
                    for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
                ]
                valid_constraint_mask_items = [
                    torch.cat([torch.zeros(len(decoder_prompt)-1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask], dim=0)
                    for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
                ]
                valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad, left_pad=False).to(device)
                valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad, left_pad=False).to(device)
                valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad, left_pad=False).to(device)

                new_encoder_out = {}
                new_encoder_out["encoder_out"] = [
                    encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
                ]
                new_encoder_out["encoder_padding_mask"] = [
                    encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
                ]
                new_encoder_out["position_embeddings"] = [
                    encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
                ]

                decoder_out = model.decoder(valid_prev_output, encoder_out=new_encoder_out)
                decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
                scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(valid_tgt.eq(self.tgt_dict.pad()), 0)
                scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                scores = scores.sum(1)
                scores = scores.view(-1, valid_size)
                valid_result.append(scores)

        valid_result = torch.cat(valid_result, dim=-1)
        predicts = valid_result.argmax(1).tolist()
        hyps = [self.index2ans[predict_index] for predict_index in predicts]
        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        logging_output["_imagecode_score_sum"] = sum(scores)
        logging_output["_imagecode_cnt"] = len(scores)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_imagecode_score_sum"].sum / meters["_imagecode_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_imagecode_cnt") > 0:
            metrics.log_scalar("_imagecode_score_sum", sum_logs("_imagecode_score_sum"))
            metrics.log_scalar("_imagecode_cnt", sum_logs("_imagecode_cnt"))
            metrics.log_derived("imagecode_score", compute_score)
