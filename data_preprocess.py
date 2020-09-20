#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : data_preprocess.py
@Author: gaoyongwei
@Date  : 2020/9/20 14:41
@Desc  : 数据预处理，三个文件train.conll  dev.conll test.conll
"""
import sys
import os
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from config import bert_base_chinese_model_dir


def preprocess_data(data_dir, model_name_or_path=bert_base_chinese_model_dir, max_len=512):
    """
    将原始输入的conll文件针对预训练模型的编码方式进行清洗
    :param data_dir: 原始输入文件的
    :param model_name_or_path: 模型路径
    :param max_len: 最大长度
    :return:
    """


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_len -= tokenizer.num_special_tokens_to_add()

    def handle_one(mode):

        dataset = os.path.join(data_dir, mode + ".conll")
        all_sequence = []
        all_labels = []
        subword_len_counter = 0
        with open(dataset, "r", encoding='utf8') as f_p:
            words = []
            labels = []
            for line in f_p:
                line = line.rstrip()
                if not line:
                    # print(line)
                    if words and labels:
                        all_sequence.append(words)
                        all_labels.append(labels)
                    words = []
                    labels = []
                    subword_len_counter = 0
                    continue

                token = line.split()[0]
                label = line.split()[1]

                current_subwords_len = len(tokenizer.tokenize(token))

                # Token contains strange control characters like \x96 or \x95
                # Just filter out the complete line
                if current_subwords_len == 0:
                    continue

                if (subword_len_counter + current_subwords_len) > max_len:
                    # print("")
                    # print(line)
                    if words and labels:
                        all_sequence.append(words)
                        all_labels.append(labels)

                    words = []
                    labels = []
                    words.append(token)
                    labels.append(label)
                    labels_set.add(label)
                    subword_len_counter = current_subwords_len
                    continue

                subword_len_counter += current_subwords_len

                # print(line)
                words.append(token)
                labels.append(label)
                labels_set.add(label)

            if words and labels:
                all_sequence.append(words)
                all_labels.append(labels)
        return all_sequence, all_labels

    def save_one(sequence, seq_labels, mode):
        with open(os.path.join(data_dir, mode + ".txt"), 'w', encoding='utf8') as writer:
            for words, labels in zip(sequence, seq_labels):
                for word, label in zip(words, labels):
                    writer.write(word + "\t" + label + "\n")
                writer.write("\n")

    train_dataset = os.path.join(data_dir, "train.conll")
    dev_dataset = os.path.join(data_dir, "dev.conll")
    test_dataset = os.path.join(data_dir, "test.conll")
    inference_dataset = os.path.join(data_dir, "inference.conll")
    labels_set = set()
    if not os.path.exists(train_dataset):
        print("no train data found")
        train_seq, train_labels = [], []
        exit(-1)
    else:
        train_seq, train_labels = handle_one("train")
    if os.path.exists(dev_dataset):
        dev_seq, dev_labels = handle_one("dev")
    else:
        train_seq, dev_seq, train_labels, dev_labels = train_test_split(train_seq, train_labels, test_size=0.1,
                                                                        random_state=0)
    if os.path.exists(test_dataset):
        test_seq, test_labels = handle_one("test")
    else:
        test_seq, test_labels = dev_seq, dev_labels

    if os.path.exists(inference_dataset):
        inference_seq, inference_labels = handle_one("inference")
        save_one(inference_seq, inference_labels, "inference")

    save_one(train_seq, train_labels, "train")
    save_one(dev_seq, dev_labels, "dev")
    save_one(test_seq, test_labels, "test")

    labels_set = list(filter(None, sorted(labels_set)))
    with open(os.path.join(data_dir, "label.txt"), 'w', encoding="utf8") as writer:
        writer.write("\n".join(labels_set))


if __name__ == '__main__':
    preprocess_data("data", bert_base_chinese_model_dir, 512)
