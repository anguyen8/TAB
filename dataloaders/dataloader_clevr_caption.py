from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import random
from dataloaders.rawimage_util_clevr import RawImageExtractor
from collections import defaultdict
import itertools
from PIL import Image


class CLEVR_DataLoader(Dataset):
    """CLEVR dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            patch_n=14,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.default_features_path = os.path.join(self.data_path, 'images')
        self.nsc_features_path = os.path.join(self.data_path, 'nsc_images')
        self.sc_features_path = os.path.join(self.data_path, 'sc_images')
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.patch_N = 
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        image_id_path = os.path.join(self.data_path, 'splits.json')
        change_caption_file = os.path.join(self.data_path, "change_captions.json")
        no_change_caption_file = os.path.join(self.data_path, "no_change_captions.json")

        with open(image_id_path, 'r') as fp:
            image_ids = json.load(fp)[subset]
            
        with open(change_caption_file, 'r') as fp:
            change_captions = json.load(fp)

        with open(no_change_caption_file, 'r') as fp:
            no_change_captions = json.load(fp)

        self.no_change_captions = no_change_captions

        with open("/home/pooyan/clevr_data/change_captions_with_bbox.json", 'r') as fp:
            self.bboxes = json.load(fp)

        image_dict = {}
        image_files = os.listdir(self.default_features_path)
        for image_file in image_files:
            image_id_ = image_file.split(".")[0].split('_')[-1]
            if int(image_id_) not in image_ids:
                continue
            # file_path_ = os.path.join(self.default_features_path, image_file)
            image_dict[int(image_id_)] = image_id_
        self.image_dict = image_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for image_id in image_ids:
            image_id_name = "CLEVR_default_%s.png" % self.image_dict[image_id]
            assert image_id_name in change_captions
            self.sentences_dict[len(self.sentences_dict)] = (image_id, change_captions[image_id_name])
            # for cap_txt in change_captions[image_id_name]:
            #     self.sentences_dict[len(self.sentences_dict)] = (image_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        # self.nc_sentences_dict = defaultdict(list)
        # for image_id in image_ids:
        #     image_id_name = "CLEVR_default_%s.png" % self.image_dict[image_id]
        #     assert image_id_name in no_change_captions
        #     for cap_txt in no_change_captions[image_id_name]:
        #         self.nc_sentences_dict[image_id].append(cap_txt)

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.image_num: used to cut the image pair representation
        self.multi_sentence_per_pair = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.image_num = len(image_ids)
            assert len(self.cut_off_points) == self.image_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, image number: {}".format(self.subset, self.image_num))

        print("Image number: {}".format(len(self.image_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawImageExtractor = RawImageExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, image_id, caption):
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, image_id in enumerate(choice_image_ids):
            words = []

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words
            output_caption_words = caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids

    def _get_rawimage(self, image_path):
        choice_image_path = [image_path]
        # Pair x L x T x 3 x H x W
        image = np.zeros((len(choice_image_path), 3, self.rawImageExtractor.size,
                          self.rawImageExtractor.size), dtype=np.float)

        for i, image_path in enumerate(choice_image_path):

            raw_image_data = self.rawImageExtractor.get_image_data(image_path)
            raw_image_data = raw_image_data['image']

            image[i] = raw_image_data

        return image

    def create_anno_from_bbox(self, bbox, img_size):
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        if x1 >= 480:
            x1 = 470
        if y1 >= 320:
            y1 = 310

        if x2 > 480:
            x2 = 480
        if y2 > 320:
            y2 = 320

        anno = np.zeros(img_size)
        anno[y1:y2, x1:x2] = 1.0
        mask = np.array(Image.fromarray(anno).convert('L').resize((224, 224)))
        bbox_xy = np.nonzero(mask)
        return mask, bbox_xy
    
    def __getitem__(self, idx):
        image_id, caption = self.sentences_dict[idx]
        caption = random.choice(caption)
        image_name = "CLEVR_default_%s.png" % self.image_dict[image_id]
        image_idx_name = "%s.png" % self.image_dict[image_id]

        b_before = self.bboxes[str(int(self.image_dict[image_id]))]["bbox_before"]
        b_after = self.bboxes[str(int(self.image_dict[image_id]))]["bbox_after"]


        left_gt_patch = np.reshape(np.zeros((1, self.patch_N * self.patch_N)), (self.patch_N, self.patch_N))
        right_gt_patch = np.reshape(np.zeros((1, self.patch_N * self.patch_N)), (self.patch_N, self.patch_N))

        patch_size = int(224/self.patch_N)

        if b_before:
            _, bbox_bef = self.create_anno_from_bbox(b_before, (320, 480))
            y1 = min(bbox_bef[1])
            y2 = max(bbox_bef[1])
            x1 = min(bbox_bef[0])
            x2 = max(bbox_bef[0])
            start_x = int(x1 / patch_size)
            start_y = int(y1 / patch_size)
            end_x = int(x2 / patch_size)
            end_y = int(y2 / patch_size)
            xs = np.arange(start_x, end_x+1)
            ys = np.arange(start_y, end_y+1)
            unique_com = list(itertools.product(xs, ys))
            interArea = []
            for (x, y) in unique_com:
                s_x = x * patch_size
                e_x = (x + 1) * patch_size
                s_y = y * patch_size
                e_y = (y + 1) * patch_size
                xA = max(x1, s_x)
                yA = max(y1, s_y)
                xB = min(x2, e_x)
                yB = min(y2, e_y)
                interArea.append(abs(max((xB - xA, 0)) * max((yB - yA), 0)))

            max_intesected_index = np.argmax(interArea)
            x, y = unique_com[max_intesected_index]
            left_gt_patch[x, y] = 1.
            gt_left_map = np.zeros((1, self.patch_N * self.patch_N + 1))
            gt_left_map[0, 1:] = np.reshape(left_gt_patch, (1, self.patch_N * self.patch_N ))
        else:
            gt_left_map = np.zeros((1, self.patch_N * self.patch_N  + 1))
            gt_left_map[0, 0] = 1.0

        if b_after:
            _, bbox_aft = self.create_anno_from_bbox(b_after, (320, 480))
            y1 = min(bbox_aft[1])
            y2 = max(bbox_aft[1])
            x1 = min(bbox_aft[0])
            x2 = max(bbox_aft[0])
            start_x = int(x1 / patch_size)
            start_y = int(y1 / patch_size)
            end_x = int(x2 / patch_size)
            end_y = int(y2 / patch_size)
            xs = np.arange(start_x, end_x+1)
            ys = np.arange(start_y, end_y+1)
            unique_com = list(itertools.product(xs, ys))
            interArea = []
            for (x, y) in unique_com:
                s_x = x * patch_size
                e_x = (x + 1) * patch_size
                s_y = y * patch_size
                e_y = (y + 1) * patch_size
                xA = max(x1, s_x)
                yA = max(y1, s_y)
                xB = min(x2, e_x)
                yB = min(y2, e_y)
                interArea.append(abs  (max((xB - xA, 0)) * max((yB - yA), 0)))

            max_intesected_index = np.argmax(interArea)
            x, y = unique_com[max_intesected_index]
            right_gt_patch[x, y] = 1.
            gt_right_map = np.zeros((1, self.patch_N * self.patch_N  + 1))
            gt_right_map[0, 1:] = np.reshape(right_gt_patch, (1, self.patch_N * self.patch_N ))
        else:
            gt_right_map = np.zeros((1, self.patch_N * self.patch_N  + 1))
            gt_right_map[0, 0] = 1.0


        bef_image_path = os.path.join(self.default_features_path, image_name)
        aft_image_path = os.path.join(self.sc_features_path, image_name.replace('default', 'semantic'))
        no_image_path = os.path.join(self.nsc_features_path, image_name.replace('default', 'nonsemantic'))

        nsc_map = np.zeros((1, self.patch_N * self.patch_N  + 1))
        nsc_map[0, 0] = 1.0

        pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = self._get_text(image_id, caption)

        no_caption = self.no_change_captions[image_name]
        no_caption = random.choice(no_caption)

        _, _, _, no_pairs_input_caption_ids, no_pairs_decoder_mask, no_pairs_output_caption_ids = self._get_text(image_id, no_caption)

        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        no_image = self._get_rawimage(no_image_path)

        image_mask = np.ones(2, dtype=np.long)
        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, no_image, gt_left_map, gt_right_map, nsc_map, image_mask, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               no_pairs_input_caption_ids, no_pairs_decoder_mask, no_pairs_output_caption_ids, image_idx_name