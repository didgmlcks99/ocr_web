import pandas as pd
import random
import numpy as np

def cut(sentence):
    words = sentence.split()
    lim = len(words)

    if lim == 1: cut = lim
    else: cut = random.randint(1, lim-1)

    cut_first = ' '.join(words[:cut])
    cut_second = ' '.join(words[cut:])

    return cut_first, cut_second

def makeData(class_cnt, cut_cnt, first, second):
    if class_cnt < 5000:
        label = 0

        if cut_cnt < 2500:
            c_first, c_second = cut(first)
            
            first = c_first
            second = ' '.join([c_second, second])

            cut_cnt += 1
        else:
            c_first, c_second = cut(second)

            first = ' '.join([first, c_first])
            second = c_second
        
        class_cnt += 1
    else: label = 1

    return [first, second, label]

def getInitData():
    tot_data = []
    
    tot_data += getChatBot()
    print('got ChatBot data!')

    tot_data += getKCC()
    print('got KCC data!')

    tot_data += getKo()
    print('got KO data!')

    random.shuffle(tot_data)
    print('shuffled data!')

    df = pd.DataFrame(tot_data, columns=['first', 'second', 'label'])
    print('saved to dataframe!')

    return df


def getChatBot():
    chatBot_df = pd.read_csv('../data/raw/ChatbotData.csv').sample(frac=1)
    
    data_cnt = 0
    class_cnt = 0
    cut_cnt = 0
    label = 0

    total_data = []

    for index, row in chatBot_df.iterrows():
        
        first = row['Q']
        second = row['A']

        if len(first.split()) == 1 or len(second.split()) == 1: continue

        cuts = makeData(class_cnt, cut_cnt, first, second)

        total_data.append(cuts)

        cut_cnt += 1
        class_cnt += 1
        data_cnt += 1

        if data_cnt == 10000: break

    return total_data

def getKCC():
    kccN_txt = '../data/raw/KCC150_Korean_sentences_UTF8.txt'
    kccQ_txt = '../data/raw/KCCq28_Korean_sentences_UTF8_v2.txt'

    kccN_df = pd.read_csv(kccN_txt, sep='\t', header=None)
    kccQ_df = pd.read_csv(kccQ_txt, sep='\t', header=None)
    
    sentence_list = []

    type_cnt = 0
    point_cnt = 0

    for index, row in kccN_df.iterrows():

        sentence = row[0]

        if point_cnt < 5000:
            sentence = sentence[:len(sentence)-1]
            point_cnt += 1

        sentence_list.append(sentence)

        type_cnt += 1

        if type_cnt == 10000: break
    
    type_cnt = 0
    point_cnt = 0

    for index, row in kccQ_df.iterrows():

        sentence = row[0]

        if point_cnt < 5000:
            sentence = sentence[:len(sentence)-1]
            point_cnt += 1

        sentence_list.append(sentence)

        type_cnt += 1

        if type_cnt == 10000: break
    
    random.shuffle(sentence_list)

    class_cnt = 0
    cut_cnt = 0

    label = 0

    total_data = []

    for i in range(0, len(sentence_list), 2):
        
        first = sentence_list[i]
        second = sentence_list[i+1]

        cuts = makeData(class_cnt, cut_cnt, first, second)

        total_data.append(cuts)

        cut_cnt += 1
        class_cnt += 1
    
    return total_data

def getKo():
    chatBot_df = pd.read_csv('../data/raw/Ko_persona_train_corrected.csv', low_memory=False).sample(frac=1)
    
    data_cnt = 0
    class_cnt = 0
    cut_cnt = 0
    label = 0

    total_data = []

    for index, row in chatBot_df.iterrows():
        
        first = row['dialogue/0/1']
        second = row['dialogue/1/0']

        if len(first.split()) == 1 or len(second.split()) == 1: continue

        cuts = makeData(class_cnt, cut_cnt, first, second)

        total_data.append(cuts)

        cut_cnt += 1
        class_cnt += 1
        data_cnt += 1

        if data_cnt == 10000: break

    return total_data

def mk_initData(df):
    # df.to_csv("../data/processed/data.csv", index=False, header=False)
    # df.to_excel('../data/processed/data.xlsx', index=False, header=False, sheet_name='sheet1')
    df.to_csv("../data/processed/data", index=False, header=False)

    sentences_df = df[['first', 'second']]
    label = df['label']

    sentences_list = []
    label_list = []

    for index, row in df.iterrows():
        sentences_list.append(row['first'])
        sentences_list.append(row['second'])

        label_list.append(row['label'])
    
    sentences_df = pd.DataFrame(sentences_list, columns=['sentence'])
    label_df = pd.DataFrame(label_list, columns=['label'])

    # sentences_df.to_csv("../data/processed/sentence.csv", index=False, header=False)
    # sentences_df.to_excel('../data/processed/sentence.xlsx', index=False, header=False, sheet_name='sheet1')
    # sentences_df.to_csv("../data/processed/sentence", index=False, header=False)
    to_file(sentences_list, "../data/processed/sentence")

    # label_df.to_csv("../data/processed/label.csv", index=False, header=False)
    # label_df.to_excel('../data/processed/label.xlsx', index=False, header=False, sheet_name='sheet1')
    label_df.to_csv("../data/processed/label", index=False, header=False)

    sentences_df.to_csv("../data/train_tokenizer.txt", index=False, header=False)
    to_file(sentences_list, "../data/train_tokenizer.txt")

def to_file(ls, fn):

    with open(fn, 'w') as f:
        f.write('\n'.join(ls))

def getData():
    
    first = []
    second = []
    labels = []
    
    with open('../data/processed/sentence', 'r') as f:
        sentences = f.read().splitlines()

        for i in range(0, len(sentences), 2):
            
            first.append(str(sentences[i]))
            second.append(str(sentences[i+1]))
    
    with open('../data/processed/label', 'r') as f:
        ls = f.read().splitlines()

        for d in ls:
            labels.append(int(d))
    
    return np.array(first), np.array(second), np.array(labels)

def getCh2idx():

    ch2idx = {}

    with open('records/ch2idx', 'r') as f:
        data = f.read().splitlines()

        for d in data:
            r = d.split(': ')
            c = r[0]
            i = int(r[1])

            ch2idx[c] = i
    
    print('successfully got ch2idx!')

    return ch2idx