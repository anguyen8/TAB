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
            patch_n=14,
            max_words=30,
            image_resolution=224,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.default_features_path = os.path.join(self.data_path, 'images_and_masks')
        self.nsc_features_path = os.path.join(self.data_path, 'inpainted')
        self.sc_features_path = os.path.join(self.data_path, 'inpainted')
        
        self.max_words = max_words
        self.tokenizer = tokenizer

        metadata_path = os.path.join(self.data_path, 'metadata.json')
        with open(metadata_path, 'r') as fp:
            self.annotations = json.load(fp)

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        image_ids = self.create_splits(self.annotations)

        image_ids = image_ids[self.subset]

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []

        add_template = ['the <c> has appeared', 'the <c> has been newly placed', 'the <c> has been added']
        drop_template = ['the <c> has disappeared', 'the <c> is missing', 'the <c> is gone', 'the <c> is no longer there']
        
        self.no_change_captions = ['no change was made', 'there is no change', 'the two scenes seem identical', 'the scence is the same as before', 'the scene remains the same', 'nothing has changed', 'nothing was modified', 'no change has occurred', 'there is no difference']


        if not os.path.exists(os.path.join(self.data_path, self.subset + "_type_mapping.json")):
            type_mapping = dict()
            for image_id in image_ids:
                change_type = 'add'

                if random.random() < 0.5:
                    change_type = 'drop'
                type_mapping.update({str(image_id):str(change_type)})

            with open(os.path.join(self.data_path, self.subset + "_type_mapping.json"), "w") as gp:
                json.dump(type_mapping, gp)
                gp.flush()
                os.fsync(gp.fileno())
   
        else:
            with open(os.path.join(self.data_path, self.subset + "_type_mapping.json"), "r") as gp:
                type_mapping = json.load(gp)

        for image_id in image_ids:
                        
            change_type = type_mapping[image_id]

            if change_type == 'add':
                change_captions = add_template
            else:
                change_captions = drop_template

            for cap_txt in change_captions:
                self.sentences_dict[len(self.sentences_dict)] = (image_id, change_type, cap_txt)
             
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


    def create_splits(self, metadata):

        imgs_ids_list = list(metadata.keys())

        if not os.path.exists(os.path.join(self.data_path, 'splits.json')):
            random.shuffle(imgs_ids_list) 

            ids = dict()

            ids['train'] = list(imgs_ids_list[:-20000])
            ids['val'] = list(imgs_ids_list[-20000:-15000])
            ids['test'] = list(imgs_ids_list[-15000:])

            with open(os.path.join(self.data_path, 'splits.json'), 'w') as fp:
                json.dump(ids, fp)
                fp.flush()
                os.fsync(fp.fileno())

        else:
            with open(os.path.join(self.data_path, 'splits.json'), 'r') as fp:
                ids = json.load(fp)

        return ids


    def _get_text(self, image_id, caption):
        k = 1
        choice_image_ids = [image_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, image_id in enumerate(choice_image_ids):
            words = self.tokenizer.tokenize(caption)

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

        return pairs_text, pairs_mask, pairs_segment

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

    def __getitem__(self, idx):
        image_id, change_type, caption = self.sentences_dict[idx]
        caption = caption.replace('<c>', self.annotations[image_id]['label'].lower())
        pairs_text, pairs_mask, pairs_segment = self._get_text(image_id, caption)
       
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


        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        image_mask = np.ones(2, dtype=np.long)
        
        return pairs_text, pairs_mask, pairs_segment, bef_image, aft_image, image_mask
