from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, SentenceTransformer, InputExample
import logging
import argparse

def define_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model', type=str, default='klue/bert-base')
    parser.add_argument('--model_save_dir', type=str, default='./output')
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/SWOT_NLI')
    parser.add_argument('-n', '--n_companies', type=int, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    # parser.add_argument('--evaluation_steps', type=int, default=1000) # To Do

    args = parser.parse_args()
    return args


def main(args):

    BERT_MODEL = args.bert_model
    TRAIN_BATCH_SIZE = args.batch
    N_COMPANIES = args.n_companies
    MODEL_SAVE_DIR = args.model_save_dir
    MODEL_SAVE_PATH = MODEL_SAVE_DIR + "/SWOT_SBERT_{}".format(str(N_COMPANIES).zfill(4))
    DATASET_DIR = args.dataset_dir
    word_embedding_model = models.Transformer(BERT_MODEL)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Read All SWOT NLI train dataset")

    # SWOT 임베딩 학습을 위해서 Strength, Weakness, Opportunity, Threat로 레이블링된 문단(문장) 데이터를
    # 강점-약점(S-W), 강점-기회(S-O), 강점-위협(S-T), 약점-기회(W-O), 약점-위협(W-T), 기회-위협(O-T)의 총 여섯 가지 관계 중
    # 어떤 범주인지를 예측하는 자연어 추론(Natural Language Inference, NLI) 태스크로 SentenceBERT 학습함
    label2int = {"SW": 0, "SO": 1, "ST":2, "WO":3, "WT":4, "OT":5}
    train_samples = []

    with open(DATASET_DIR+'/{}/train.tsv'.format(str(N_COMPANIES).zfill(4)), "rt", encoding="utf-8") as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, label = line.split('\t')
            label = label2int[label.strip()]
            train_samples.append(InputExample(texts=[s1, s2], label=label))

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))


    num_epochs = args.epochs

    warmup_steps = math.ceil(len(train_dataset) * num_epochs / TRAIN_BATCH_SIZE * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              # evaluator=dev_evaluator, # To Do : Evaluator with Devset
              epochs=num_epochs,
              # evaluation_steps=args.evaluation_steps, # To Do
              warmup_steps=warmup_steps,
              output_path=MODEL_SAVE_PATH
              )

    # model = SentenceTransformer(MODEL_SAVE_PATH)
    print(f"model save path > {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    args = define_argparser()
    main(args)