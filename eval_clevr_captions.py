import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import random
import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.eval import PTBTokenizer, Bleu, Meteor, Rouge, Cider
import copy

from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4IDC
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from collections import defaultdict
from skimage.transform import resize

from tabulate import tabulate
from tqdm import tqdm


def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4IDC.from_pretrained(args.cross_model, args.decoder_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model


class EvalCap(COCOEvalCap):
    def __init__(self, coco, cocoRes):
        super(EvalCap, self).__init__(coco, cocoRes)

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

def get_args(description='TAB4IDC on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=4, help='')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size eval')

    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--patch_num', type=int, default=14, help='')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="clevr", type=str, help="Point the dataset to finetune.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--intra_num_hidden_layers', type=int, default=9, help="Layer NO. of intra module")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")

    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--gt_dir", default="gt", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()

    return args

def score_generation(anno_file, result_file):
    coco = COCO(anno_file)
    coco_res = coco.loadRes(result_file)

    coco_eval = EvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()

    coco_eval.evaluate()
    return copy.deepcopy(coco_eval.eval)

def greedy_decode(args, model, tokenizer, input_ids, segment_ids, input_mask, video, video_mask, gt_left_map, gt_right_map):
    _, visual_emb, _, visual_output, left_map, right_map = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video,
                                                                      video_mask, gt_left_map[:, 0, :].squeeze(), gt_right_map[:, 0, :].squeeze())
        
    video_mask = torch.ones(visual_output.shape[0], visual_output.shape[1], device=visual_output.device).long()
    input_caption_ids = torch.zeros(visual_output.shape[0], device=visual_output.device).data.fill_(tokenizer.vocab["<|startoftext|>"])
    input_caption_ids = input_caption_ids.long().unsqueeze(1)
    decoder_mask = torch.ones_like(input_caption_ids)
    for i in range(args.max_words):
        decoder_scores = model.decoder_caption(visual_output, video_mask, input_caption_ids, decoder_mask,
                                                shaped=True, get_logits=True)
        next_words = decoder_scores[:, -1].max(1)[1].unsqueeze(1)
        input_caption_ids = torch.cat([input_caption_ids, next_words], 1)
        next_mask = torch.ones_like(next_words)
        decoder_mask = torch.cat([decoder_mask, next_mask], 1)

    return input_caption_ids[:, 1:].tolist(), left_map, right_map

def set_seed_logger(args):
    if args.datatype == "clevr" or args.datatype == "open":
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def accuracy_breakdown(caption, gt_captions, change_type):

    if "became" in caption or "turned" in caption or "changed" in caption or "different" in caption or "moved" in caption or "disappeared" in caption \
        or "missing" in caption or "gone" in caption or "added" in caption or "newly placed" in caption \
            or "has appeared" in caption or "is no longer there" in caption or "in a different location" in caption:
          predicted_type = "sc"
    else:
          predicted_type = "nsc"

    if change_type == "sc" and predicted_type == "sc":
        return 1
    
    elif change_type == "nsc" and predicted_type == "nsc":
        return 1
    else:
        return 0
    


def eval_epoch(args, model, test_dataloader, device, epoch=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()


    all_result_lists = []
    all_nc_result_lists = []


    results_loc = defaultdict(int)

    for i, batch in tqdm(enumerate(test_dataloader)):

        image_names = batch[-1]


        batch = tuple(t.to(device, non_blocking=True) for t in batch[:-1])

        input_ids, input_mask, segment_ids, bef_image, aft_image, nc_image, sup_gt_left, sup_gt_right, nc_gt, image_mask, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
        nc_pairs_input_caption_ids, nc_pairs_decoder_mask, nc_pairs_output_caption_ids = batch

        
        nc_image_pair = torch.cat([bef_image, nc_image], 1)

        image_pair = torch.cat([bef_image, aft_image], 1)

        with torch.no_grad():

            result_list, left_map, right_map = greedy_decode(args, model, tokenizer, input_ids, segment_ids, input_mask, image_pair, image_mask, sup_gt_left, sup_gt_right)

            for re_idx, (image_name, re_list) in enumerate(zip(image_names, result_list)):
                
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "<|endoftext|>" in decode_text_list:
                    SEP_index = decode_text_list.index("<|endoftext|>")
                    decode_text_list = decode_text_list[:SEP_index]
                if "!" in decode_text_list:
                    PAD_index = decode_text_list.index("!")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = decode_text_list.strip()
                new_decode_item = {"caption": decode_text, "image_id": image_name}
                all_result_lists.append(new_decode_item)

            if args.datatype == "clevr" or args.datatype == "open":
                nc_result_list, left_map, right_map = greedy_decode(args, model, tokenizer, input_ids, segment_ids, input_mask, nc_image_pair, image_mask, nc_gt, nc_gt)

                for re_idx, (image_name, re_list) in enumerate(zip(image_names, nc_result_list)):
                    decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                    if "<|endoftext|>" in decode_text_list:
                        SEP_index = decode_text_list.index("<|endoftext|>")
                        decode_text_list = decode_text_list[:SEP_index]
                    if "!" in decode_text_list:
                        PAD_index = decode_text_list.index("!")
                        decode_text_list = decode_text_list[:PAD_index]
                    decode_text = decode_text_list.strip()
                    new_decode_item = {"caption": decode_text, "image_id": image_name + "_n"}
                    all_nc_result_lists.append(new_decode_item)

                        
                        
    total_results = all_result_lists + all_nc_result_lists
    json.dump(total_results, open(os.path.join(args.output_dir, "%s_predictions.json" % args.datatype), "w"))
    assert os.path.exists(os.path.join(args.output_dir, "%s_predictions.json" % args.datatype))

    #### Evaluate
    
    metrics_nlg = score_generation(os.path.join(args.gt_dir, "%s_total_change_captions_reformat.json" % args.datatype),
                                os.path.join(args.output_dir, "%s_predictions.json" % args.datatype))

    print(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    print(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))

    CIDEr = metrics_nlg["CIDEr"]


if __name__ == '__main__':
    args = get_args()

    set_seed_logger(args)

    args.data_path = "/clevr_data"
    args.features_path = "/clevr_data"

    args.datatype = "clevr"
    args.init_model = "ckpts/pytorch.model.bin.clevr"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ClipTokenizer()
    model = init_model(args, device, n_gpu=1, local_rank=0)

    if args.pretrained_clip_name == "ViT-B/32":
        patch_num = 7
    else:
        patch_num = 14


    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, patch_num, subset="test")

    with torch.no_grad():
        eval_epoch(args, model, test_dataloader, device)
