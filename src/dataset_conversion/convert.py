#!/usr/bin/env python3
'''
Converts the given dataset into the required format for w266_final_project
'''

import argparse
import json
import csv
from nltk.tokenize import word_tokenize


DATA_DIR = '../../../data/'
SQUAD_DIR = DATA_DIR + 'squad/'
MARCO_DIR = DATA_DIR + 'marco/'
CSV_HEADER = ['story_id', 'story_text', 'question', 'answer_token_ranges']


def squad_parse_convert(input_dict, csv_out):

    with open(SQUAD_DIR + csv_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        writer.writeheader()

        row_dict = {
            'story_id': '',
            'story_text': '',
            'question': '',
            'answer_token_ranges': ''}
        count=0
        # Parse file and convert
        for entry in input_dict['data']:
            for paragraph in entry['paragraphs']:
                story_text = paragraph['context']
                tokens = word_tokenize(story_text)
                for qa in paragraph['qas']:
                    story_id = qa['id']
                    question = qa['question']
                    # Get answer token ranges
                    prev_ans = ''
                    # Iterate through answers, ignore duplicates
                    for answer in qa['answers']:
                        answ_tokens = word_tokenize(answer['text'])
                        if answer['text'] != prev_ans:
                            if count < 10:
                                print("answer['text']", answer['text'], "prev_ans", prev_ans, answer['text'] != prev_ans)
                                count += 1
                            first_idx = next(
                                (i for i, t in enumerate(tokens) if answ_tokens[0] == t),
                                None)  # Get index of answer token or None
                            # Check index
                            if first_idx is not None:
                                # Found exact match
                                answer_token_ranges = (
                                    str(first_idx) + ":" + str(first_idx + len(answ_tokens)))
                            else:
                                # Couldn't find exact match, check if answer is substring
                                # of token
                                for i, token in enumerate(tokens):
                                    if answ_tokens[0] in token:
                                        first_idx = i
                                if first_idx:
                                    answer_token_ranges = (
                                        str(first_idx) + ":" + str(first_idx + len(answ_tokens)))

                            row_dict['story_id'] = story_id
                            row_dict['story_text'] = story_text
                            row_dict['question'] = question
                            row_dict['answer_token_ranges'] = answer_token_ranges
                            writer.writerow(row_dict)

                        prev_ans = answer['text']

def convert_squad():
    '''
    Convert data from the squad dataset to the correct format
    '''
    dev_in = 'dev-v1.1.json'
    train_in = 'train-v1.1.json'
    dev_out = 'dev_converted.csv'
    train_out = 'train_converted.csv'

    # Open the dataset files
    try:
        dev_f = open(SQUAD_DIR + dev_in, 'r')
        train_f = open(SQUAD_DIR + train_in, 'r')
    except IOError as e:
        print(e)
        exit()
    with dev_f:
        dev_dict = json.load(dev_f)
    with train_f:
        train_dict = json.load(train_f)

    print("Converting", dev_in, "to", dev_out)
    squad_parse_convert(dev_dict, dev_out)
    print("Converting", train_in, "to", train_out)
    squad_parse_convert(train_dict, train_out)


def convert_marco():
    '''
    Convert data from the marco dataset to the correct format
    '''
    pass


def main(**kwargs):
    dataset = kwargs.get('dataset', None)
    print("Converting ", dataset, "dataset")

    if dataset:
        if dataset == 'squad':
            convert_squad()
        elif dataset == 'marco':
            convert_marco()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset',
        help=(
            'Dataset to convert, one of '
            '(squad|marco)'),
        default='squad')
    args = parser.parse_args()

    main(**vars(args))
