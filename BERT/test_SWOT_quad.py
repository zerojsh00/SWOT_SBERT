import torch
import argparse
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification

def define_argparser() :
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--n_companies', type=int, required=True)
    p.add_argument('--model_dir', type=str, default='./output')
    p.add_argument('--dataset_dir', type=str, default='../Dataset/SWOT_quad')
    config = p.parse_args()

    return config

def main(config) :
    N_COMPANIES = config.n_companies
    DATASET_DIR = config.dataset_dir
    MODEL_DIR = config.model_dir
    DATASET_PATH = DATASET_DIR + '/{}/test.tsv'.format(str(N_COMPANIES).zfill(4))
    MODEL_NAME = MODEL_DIR + '/SWOT_BERT_{}.pt'.format(str(N_COMPANIES).zfill(4))

    DATASET = pd.read_csv(DATASET_PATH, header=None, sep='\t') # test dataset

    saved_data = torch.load(
        MODEL_NAME,
        map_location='cuda:0'
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    model_loader = BertForSequenceClassification
    model = model_loader.from_pretrained(
        train_config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    model.load_state_dict(bert_best)

    # inference
    top_k = 1
    batch_size = 64
    lines = DATASET[0].to_list()

    with torch.no_grad():
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)

        device = 'cuda:0'
        model.cuda(0)

        model.eval()

        y_hats = []
        for idx in range(0, len(lines), batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1)
            y_hats += [y_hat]

        y_hats = torch.cat(y_hats, dim=0)

        probs, indice = y_hats.cpu().topk(top_k)

    y_pred = [index_to_label[int(ind)] for ind in indice]
    y_true = DATASET[1].to_list()
    print(classification_report(y_true, y_pred, output_dict=False))
   
    # with open('../output_for_paper/BERT_output/BERT_report_{}.txt'.format(str(N_COMPANIES).zfill(4)) , "w") as text_file:
    with open(MODEL_DIR+'/BERT_report_{}.txt'.format(str(N_COMPANIES).zfill(4)) , "w") as text_file:
        print(classification_report(y_true, y_pred, digits=4), file=text_file)


   # report = classification_report(y_true, y_pred, output_dict=True)
   # pd.DataFrame(report).transpose().to_csv(MODEL_DIR+'/report_{}.csv'.format(str(N_COMPANIES).zfill(4)), index = False)
    print("completed.")


if __name__ == '__main__':
    config = define_argparser()
    main(config)
