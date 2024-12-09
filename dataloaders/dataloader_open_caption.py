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
from dataloaders.rawimage_util import RawImageExtractor
from collections import defaultdict
import itertools
from PIL import Image


class OPEN_DataLoader(Dataset):
    """OPEN-IMAGES-I dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            patch_n = 14,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.patch_N = patch_n
        self.default_features_path = os.path.join(self.data_path, 'images_and_masks')
        self.nsc_features_path = os.path.join(self.data_path, 'inpainted')
        self.sc_features_path = os.path.join(self.data_path, 'inpainted')
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        metadata_path = os.path.join(self.data_path, 'metadata.json')
        with open(metadata_path, 'r') as fp:
            self.annotations = json.load(fp)

        image_ids = self.create_splits(self.annotations)

        image_ids = image_ids[self.subset]

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []

        add_template = ['the <c> has appeared', 'the <c> has been newly placed', 'the <c> has been added']
        drop_template = ['the <c> has disappeared', 'the <c> is missing', 'the <c> is gone', 'the <c> is no longer there']
        
        self.no_change_captions = ['no change was made', 'there is no change', 'the two scenes seem identical', 'the scene is the same as before', 'the scene remains the same', 'nothing has changed', 'nothing was modified', 'no change has occurred', 'there is no difference']

        with open(os.path.join(self.data_path, self.subset + "_type_mapping.json"), "r") as gp:
            type_mapping = json.load(gp)

        for image_id in image_ids:
            
            change_type = type_mapping[image_id]

            if change_type == 'add':
                change_captions = add_template
            else:
                change_captions = drop_template

            self.sentences_dict[len(self.sentences_dict)] = (image_id, change_type, change_captions)
            self.cut_off_points.append(len(self.sentences_dict))


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

        print("Image number: {}".format(len(image_ids)))
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

    def create_splits(self, metadata):

        imgs_ids_list = list(metadata.keys())

        if not os.path.exists(os.path.join(self.data_path, 'splits.json')):
            random.shuffle(imgs_ids_list) 

            ids = dict()

            ids['train'] = list(imgs_ids_list[:-20000])
            ids['val'] = list(imgs_ids_list[-20000: -15000])
            ids['test'] = list(imgs_ids_list[-15000:])

            with open(os.path.join(self.data_path, 'splits.json'), 'w') as fp:
                json.dump(ids, fp)
                fp.flush()
                os.fsync(fp.fileno())

        else:
            with open(os.path.join(self.data_path, 'splits.json'), 'r') as fp:
                ids = json.load(fp)

        return ids
    
    def create_anno(self, bbox, n_patch, im_size):

        x, y = im_size[0], im_size[1]

        anno = np.zeros((x, y))
        anno[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1.0
        

        mask = np.array(Image.fromarray(anno).convert('L').resize((224, 224)))
        bbox_xy = np.nonzero(mask)

        gt_patch = np.reshape(np.zeros((1, n_patch * n_patch)), (n_patch, n_patch))

        patch_size = 224/n_patch

        y1 = min(bbox_xy[1])
        y2 = max(bbox_xy[1])
        x1 = min(bbox_xy[0])
        x2 = max(bbox_xy[0])
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
        gt_patch[x, y] = 1.
        
        gt_map = np.zeros((1, n_patch * n_patch + 1))
        gt_map[0, 1:] = np.reshape(gt_patch.T, (1, n_patch * n_patch))

        return gt_map
    

    def get_box(self, bbox, im_size):
        
        x, y = im_size[0], im_size[1]

        anno = np.zeros((x, y))
        anno[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1.0
        

        mask = np.array(Image.fromarray(anno).convert('L').resize((224, 224)))
        bbox_xy = np.nonzero(mask)

        y1 = min(bbox_xy[1])
        y2 = max(bbox_xy[1])
        x1 = min(bbox_xy[0])
        x2 = max(bbox_xy[0])

        return [x1, x2, y1, y2]
    

    def __getitem__(self, idx):
        image_id, change_type, caption = self.sentences_dict[idx]
        caption = random.choice(caption)
        caption = caption.replace('<c>', self.annotations[image_id]['label'].lower())

        if change_type == 'drop':
            if random.random() < 0.5:
                bef_image_path = self.annotations[image_id]['path_bef']
                b_before = self.annotations[image_id]['bbox_original']
                bef_size = self.annotations[image_id]['original_size'] 
                aft_image_path = self.annotations[image_id]['crop_path_aft']
                b_after = self.annotations[image_id]['bbox_crop_aft']
                aft_size = self.annotations[image_id]['crop_aft_size']
            else:
                bef_image_path = self.annotations[image_id]['crop_path_bef']
                b_before = self.annotations[image_id]['bbox_crop_bef']
                bef_size = self.annotations[image_id]['crop_bef_size'] 
                aft_image_path = self.annotations[image_id]['path_aft']
                b_after = self.annotations[image_id]['bbox_original']  
                aft_size = self.annotations[image_id]['original_size']  
                
        else:
            if random.random() < 0.5:
                bef_image_path = self.annotations[image_id]['path_aft']
                b_before = self.annotations[image_id]['bbox_original']
                bef_size = self.annotations[image_id]['original_size']
                aft_image_path = self.annotations[image_id]['crop_path_bef']
                b_after = self.annotations[image_id]['bbox_crop_bef']
                aft_size = self.annotations[image_id]['crop_bef_size'] 
            else:
                bef_image_path = self.annotations[image_id]['crop_path_aft']
                b_before = self.annotations[image_id]['bbox_crop_aft']
                bef_size = self.annotations[image_id]['crop_aft_size']
                aft_image_path = self.annotations[image_id]['path_bef']
                b_after = self.annotations[image_id]['bbox_original']
                aft_size = self.annotations[image_id]['original_size'] 

        no_image_path = bef_image_path

        gt_left_map = self.create_anno(b_before, self.patch_N, bef_size)
        gt_right_map = self.create_anno(b_after, self.patch_N, aft_size)

        left_box, right_box = self.get_box(b_before, bef_size), self.get_box(b_after, aft_size)

        nsc_map = np.zeros((1, self.patch_N * self.patch_N + 1))
        nsc_map[0, 0] = 1.0

        pairs_text, pairs_mask, pairs_segment, pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = self._get_text(image_id, caption)

        no_caption = random.choice(self.no_change_captions)

        _, _, _, no_pairs_input_caption_ids, no_pairs_decoder_mask, no_pairs_output_caption_ids = self._get_text(image_id, no_caption)

        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        no_image = self._get_rawimage(no_image_path)
        sc_target = 1.0
        nsc_target = 0.0

        image_mask = np.ones(2, dtype=np.long)
        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, no_image, sc_target, nsc_target, gt_left_map, gt_right_map, nsc_map, image_mask, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
               no_pairs_input_caption_ids, no_pairs_decoder_mask, no_pairs_output_caption_ids, image_id, left_box, right_box, bef_image_path, aft_image_path, no_image_path
