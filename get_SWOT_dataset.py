import argparse
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_companies', type=int, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_load_path', type=str,
                        default='./Dataset/concat_data_jobkorea_catch_marketing91.csv')
    parser.add_argument('--dataset_save_path', type=str,
                        default='./Dataset/')

    args = parser.parse_args()
    return args

def _get_SWOT_NLI(dataframe) :
    result = []
    labels = ["SW", "SO", "ST", "WO", "WT", "OT"]
    for i in range(len(dataframe)):
        s, w, o, t = dataframe.iloc[i][1:]
        comb = list(combinations([s, w, o, t], 2))
        for data in zip(comb, labels):
            sentence1, sentence2 = data[0]
            label = data[1]
            result.append([sentence1, sentence2, label])

    df = pd.DataFrame(result, columns=['sentence1', 'sentence2', 'label'])
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle data

    train, tmp = train_test_split(df, test_size=0.3, shuffle=False)
    valid, test = train_test_split(tmp, test_size=0.5, shuffle=False)
    return train, valid, test


def _get_SWOT_quad(dataframe) :
    labeled_data = []
    for row in tqdm(dataframe.iterrows()):
        labeled_data.append([row[1].Strength, "S"])
        labeled_data.append([row[1].Weakness, "W"])
        labeled_data.append([row[1].Opportunity, "O"])
        labeled_data.append([row[1].Threat, "T"])

    df = pd.DataFrame(labeled_data, columns=['sentence', 'label'])
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle data
    df = df.drop_duplicates()

    train, tmp = train_test_split(df, test_size=0.3, shuffle=False)
    valid, test = train_test_split(tmp, test_size=0.5, shuffle=False)

    return train, valid, test


def main(args):
    DATASET_LOAD_PATH = args.dataset_load_path
    DATASET_SAVE_PATH = args.dataset_save_path
    N_COMPANIES = args.n_companies
    DATASET_NAME = args.dataset_name
    assert DATASET_NAME in ["SWOT_quad", "SWOT_NLI"] , "dataset_name은 SWOT_quad 또는 SWOT_NLI 중 하나여야 함"

    f = pd.read_csv(DATASET_LOAD_PATH)
    f_ = f.sample(N_COMPANIES, random_state=0)

    if DATASET_NAME == "SWOT_quad" :
        train, valid, test = _get_SWOT_quad(f_)
    else :
        train, valid, test = _get_SWOT_NLI(f_)

    print("TRAIN : {}".format(len(train)))
    print("VALID : {}".format(len(valid)))
    print("TEST : {}".format(len(test)))

    Path(DATASET_SAVE_PATH + DATASET_NAME + '/{}'.format(str(N_COMPANIES).zfill(4))).mkdir(parents=True, exist_ok=True)

    train.to_csv(DATASET_SAVE_PATH + DATASET_NAME + '/{}/train.tsv'.format(str(N_COMPANIES).zfill(4)), sep='\t',
                 header=False, index=False)
    valid.to_csv(DATASET_SAVE_PATH + DATASET_NAME + '/{}/valid.tsv'.format(str(N_COMPANIES).zfill(4)), sep='\t',
                 header=False, index=False)
    test.to_csv(DATASET_SAVE_PATH + DATASET_NAME + '/{}/test.tsv'.format(str(N_COMPANIES).zfill(4)), sep='\t',
                header=False, index=False)

    print("completed.")

if __name__ == '__main__':
    args = define_argparser()
    main(args)
