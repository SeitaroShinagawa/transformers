#/usr/bin/env python

import os
import sys
sys.path.append("transformers/examples/research_projects/lxmert/")
import io
import json
import pickle

import torch
import numpy as np
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
from PIL import Image

from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils

class LXMERT_server:

    def __init__(self, gga_or_vqa="vqa"):
        # load object, attribute, and answer labels
        OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
        ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
        GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
        VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
        self.objids = utils.get_data(OBJ_URL)
        self.attrids = utils.get_data(ATTR_URL)
        if gga_or_vqa == "gga":
            self.gqa_answers = utils.get_data(GQA_URL)
        else:
            self.vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda:0"
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        if gga_or_vqa == "gga":
            self.lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
        else:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
        
        self.gga_or_vqa = gga_or_vqa

    def encode_img(self, img):
        """
        img is array (H, W, 3) or list of images
        """
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(img)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        # add boxes and labels to the image
        return output_dict

    def visualize_img(self, img):
        """
        img is array (H, W, 3) or URL(str)
        """
        frcnn_visualizer = SingleImageViz(img, id2obj=objids, id2attr=attrids)
      
        output_dict = self.encode_img(img)

        frcnn_visualizer.draw_boxes(
            output_dict.get("boxes"),
            output_dict.pop("obj_ids"),
            output_dict.pop("obj_probs"),
            output_dict.pop("attr_ids"),
            output_dict.pop("attr_probs"),
        )
        x = frcnn_visualizer._get_buffer()

        x = np.uint8(np.clip(x, 0, 255))
        f = io.BytesIO()
        return Image.fromarray(x)


    def __call__(self, imgs, questions, return_dict=True, output_attentions=False):        
        
        # Encode images
        output_dict = self.encode_img(imgs)

        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        if not isinstance(questions, list):
            questions = [questions]

        inputs = self.lxmert_tokenizer(
            questions,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # run lxmert(s)
        if self.gga_or_vqa == "gga":
            output_gqa = self.lxmert_gqa(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=inputs.token_type_ids,
                return_dict=return_dict,
                output_attentions=output_attentions,
            )
            preds = output_gqa["question_answering_score"].argmax(-1)
            answer_texts = [self.gqa_answers[pred] for pred in preds]
        
        else:
            output_vqa = self.lxmert_vqa(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=inputs.token_type_ids,
                return_dict=return_dict,
                output_attentions=output_attentions,
            )
            preds = output_vqa["question_answering_score"].argmax(-1)
            answer_texts = [self.vqa_answers[pred] for pred in preds]
        
        return preds, answer_texts

        
if __name__ == "__main__":
    print("Testing LXMERT prediction")
    testimg_path = "input.jpg"
    #testimg_path = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"


    lxmert = LXMERT_server(gga_or_vqa="vqa")
    questions = [
        "Where is this scene?",
        "what is the man riding?",
        "What is the man wearing?",
        "What is the color of the horse?"]

    preds, answers = lxmert([testimg_path]*4, questions)
    
    for pred, answer in zip(preds, answers):
        print(pred)
        print(answer)
