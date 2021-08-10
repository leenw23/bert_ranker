import json
import csv
import json
from collections import defaultdict
import os, glob
import random
import numpy as np
import shutil

def json_to_txt(data_dir, target_path, file_name):
    num = 0

    with open(data_dir, 'r', encoding='UTF8') as json_file:
        json_data = json.load(json_file)
        f = open(target_path+file_name, 'w', encoding='UTF8')

        for data in json_data:
            temp_dialog = []

            temp_dialog.append(data['free_turker_utterance'])
            temp_dialog.append(data['guided_turker_utterance'])

            for idx, utt in data['dialog']:
                temp_dialog.append(utt)

            temp_txt = ' __eou__ '.join(temp_dialog)
            f.write(temp_txt + '\n')

            num += 1
        
        print(num)

json_to_txt('./test.json', './test/', 'test.txt')
json_to_txt('./train.json', './train/', 'train.txt')
json_to_txt('./valid.json', './valid/', 'valid.txt')