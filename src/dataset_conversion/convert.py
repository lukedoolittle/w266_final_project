#!/usr/bin/env python3
'''
Converts the given dataset into the required format for w266_final_project
'''

import argparse
import json
import csv
from nltk.tokenize import word_tokenize
from string import ascii_uppercase as letters


DATA_DIR = '../../../data/'
SQUAD_DIR = DATA_DIR + 'squad/'
MARCO_DIR = DATA_DIR + 'marco/'
CSV_HEADER = ['story_id', 'story_text', 'question', 'answer_token_ranges']
row_dict = {
    'story_id': '',
    'story_text': '',
    'question': '',
    'answer_token_ranges': ''}


def get_answer_token_ranges(para_tokens, answ_tokens):
    '''
    Returns the start and end position of answer tokens within a tokenized
    input paragraph. Formatted as "start_pos:end_pos"

    Returns None if the answer token does not exist in the input
    '''
    answer_token_ranges = ''
    first_idx = next(
        (i for i, t in enumerate(para_tokens) if answ_tokens[0] == t),
        None)  # Get index of answer token or None
    # Check index
    if first_idx is not None:
        # Found exact match
        answer_token_ranges = (
            str(first_idx) + ":" + str(first_idx + len(answ_tokens)))
    else:
        # Couldn't find exact match, check if answer is substring
        # of token
        for i, token in enumerate(para_tokens):
            if answ_tokens[0] in token:
                first_idx = i
        if first_idx:
            answer_token_ranges = (
                str(first_idx) + ":" + str(first_idx + len(answ_tokens)))
    if not answer_token_ranges:
        pass
        # Ignore answers that don't appear in input tokens
    return answer_token_ranges


def squad_parse_convert(input_dict, csv_out):
    with open(SQUAD_DIR + csv_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        writer.writeheader()

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
                            answer_token_ranges = get_answer_token_ranges(
                                tokens,
                                answ_tokens)
                            if answer_token_ranges:
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


def marco_parse_convert(input_dict, csv_out):
    with open(MARCO_DIR + csv_out, 'w') as csvfile:
        output = {'entries': 0, 'e_written': 0, 'err_count': 0}
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        writer.writeheader()

        for query in input_dict:
            story_id = query.get('query_id', None)
            question = query.get('query', None)
            if not story_id or not question:
                output['err_count'] += 1
                continue  # Skip ahead if data is missing

            sel_passages = []
            for passage in query['passages']:  # Ignore non-selected passages
                if passage.get('is_selected', 0) == 1:
                    passage_text = passage.get('passage_text', None)
                    if passage_text:
                        sel_passages.append(passage_text)

            if not sel_passages:
                output['err_count'] += 1
                continue  # Skip ahead if data is missing

            for answer in query['answers']:
                answ_tokens = word_tokenize(answer)
                for i, passage in enumerate(sel_passages):
                    answer_token_ranges = get_answer_token_ranges(
                        word_tokenize(passage),
                        answ_tokens)

                    if answer_token_ranges:
                        row_dict['story_id'] = str(story_id) + letters[i]
                        row_dict['story_text'] = passage
                        row_dict['question'] = question
                        row_dict['answer_token_ranges'] = answer_token_ranges
                        writer.writerow(row_dict)
                        output['e_written'] += 1
                    else:
                        output['err_count'] += 1
                    output['entries'] += 1
    return output


def convert_marco():
    '''
    Convert data from the marco dataset to the correct format
    '''
    dev_in = 'dev_v2.0_well_formed.json'
    train_in = 'train_v2.0_well_formed.json'

    dev_out = 'dev_converted.csv'
    train_out = 'train_converted.csv'

    # Open the dataset files
    try:
        dev_f = open(MARCO_DIR + dev_in, 'r')
        train_f = open(MARCO_DIR + train_in, 'r')
    except IOError as e:
        print(e)
        exit()
    with dev_f:
        dev_dict = json.load(dev_f)
    with train_f:
        train_dict = json.load(train_f)

    # Parse dataset files
    print("Converting", dev_in, "to", dev_out)
    print(marco_parse_convert(dev_dict, dev_out))
    print("Converting", train_in, "to", train_out)
    print(marco_parse_convert(train_dict, train_out))


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
