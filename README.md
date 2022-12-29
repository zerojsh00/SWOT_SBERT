# SWOT 자동 분류를 위한 SentenceBERT 및 BERT

---
🍭 본 프로젝트는 SWOT 분석을 함에 있어 `Strength`, `Weakness`, `Opportunity`, `Threat`을 자동으로 분류하는 연구이며, 다음의 두 방법론이 적용되었다.

- SentenceBERT
  - `Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터를 `강점-약점(S-W)`, `강점-기회(S-O)`, `강점-위협(S-T)`, `약점-기회(W-O)`, `약점-위협(W-T)`, `기회-위협(O-T)`의 총 여섯 가지 관계 중 어떤 범주인지를 예측하는 자연어 추론(Natural Language Inference, NLI) 태스크로 SentenceBERT 학습함
  - 학습된 SentenceBERT로 `Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터를 임베딩 하고, 이를 바탕으로 K-Means Clustering 하여 군집을 생성함
    - 이때, 학습용 데이터셋으로 구축한 군집 내에서 가장 많은 빈도의 실제 데이터 값을 해당 군집의 레이블로 간주함
  - 학습된 SentenceBERT로 테스트용 데이터셋을 임베딩 하고, 이들이 어떤 군집에 소속하는지 파악한 후, 앞서 정의한 해당 군집의 레이블을 예측된 레이블로 정의함
  - 예측된 레이블과 실제 테스트용 데이터셋의 레이블을 비교함

- BERT
  - `Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터를 `Strength`, `Weakness`, `Opportunity`, `Threat`로 직접 예측함

# Structure

---

```
.
├── BERT
│   ├── output
│   │   ├── SWOT_BERT_1358.pt
│   │   ├── report_1358.csv
│   └── simple_ntc
│   │   ├── __pycache__
│   │   │   ├── bert_dataset.cpython-37.pyc
│   │   │   ├── bert_trainer.cpython-37.pyc
│   │   │   ├── bert_trainer.cpython-39.pyc
│   │   │   ├── trainer.cpython-37.pyc
│   │   │   └── utils.cpython-37.pyc
│   │   ├── bert_dataset.py
│   │   ├── bert_trainer.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── run_experiments.sh
│   ├── test_SWOT_quad.py
│   └── training_SWOT_quad.py
├── Dataset
│   ├── SWOT_NLI
│   │   ├── 1358
│   │   │   ├── test.tsv
│   │   │   ├── train.tsv
│   │   │   └── valid.tsv
…
│   └── SWOT_quad
│   │   ├── 1358
│   │   │   ├── test.tsv
│   │   │   ├── train.tsv
│   │   │   └── valid.tsv
…
├── SBERT
│   └── output
│       ├── SWOT_SBERT_1358
│       │   ├── 1_Pooling
│   │   │   │   └── config.json
│   │   │   ├── Inertia_1358.png
│   │   │   ├── KMeans_Clusters_1358.png
│   │   │   ├── README.md
│   │   │   ├── SWOT_data_points_1358.png
│   │   │   ├── config.json
│   │   │   ├── config_sentence_transformers.json
│   │   │   ├── eval
│   │   │   ├── modules.json
│   │   │   ├── pytorch_model.bin
│   │   │   ├── sentence_bert_config.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   ├── run_experiments.sh
│   ├── test_SWOT_NLI.py
│   └── training_SWOT_NLI.py
```
- 참고, `1358`은 `--n_companies(데이터로 사용한 기업의 수) = 1358`을 의미한다.

# Quick Start

---
## Package Installation
``` bash
conda create -n SWOT_SentenceBERT python=3.7
conda activate SWOT_SentenceBERT

pip install -r requirements.txt
```

## Make SWOT dataset
논문 실험을 위해서 아래의 스크립트를 이용하면 기업의 수를 다양하게 변형해가며 데이터셋을 자동 구축할 수 있다.
``` bash
bash make_SWOT_dataset.sh
```
### NLI 형태의 SWOT 데이터셋 
`강점-약점(S-W)`, `강점-기회(S-O)`, `강점-위협(S-T)`, `약점-기회(W-O)`, `약점-위협(W-T)`, `기회-위협(O-T)` 형태의 SWOT NLI 데이터셋을 `Dataset/SWOT_NLI` 폴더에 만든다. `--n_companies` 파라미터를 통해 사용할 기업의 수를 설정한다.

``` python
python get_SWOT_dataset.py --n_companies 1358 --dataset_name "SWOT_NLI"
```


### Multi-class 형태의 SWOT 데이터셋
`Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터셋을 `Dataset/SWOT_quad` 폴더에 저장한다.  `--n_companies` 파라미터를 통해 사용할 기업의 수를 설정한다.

``` python
cd SBERT
python get_SWOT_dataset.py --n_companies 1358 --dataset_name "SWOT_quad"
```
## SWOT Sentence BERT
논문 실험을 위해서 아래의 학습 및 테스트를 자동화 했던 코드로 실험을 돌려볼 수 있다. 상용화 시에는 사용되지 않는다.
``` bash
cd SBERT
bash run_experiments.sh
```
### Train SWOT Sentence BERT
``` python
cd SBERT
python training_SWOT_NLI --n_companies 1358
```

### Test SWOT Sentence BERT
``` python
cd BERT
python test_SWOT_NLI --n_companies 1358
```
## SWOT BERT
논문 실험을 위해서 아래의 학습 및 테스트를 자동화 했던 코드로 실험을 돌려볼 수 있다. 상용화 시에는 사용되지 않는다.
``` bash
cd BERT
bash run_experiments.sh
```
### Train SWOT BERT
``` python
cd BERT
python training_SWOT_quad --n_companies 1358
```
### Test SWOT BERT
``` python
cd BERT
python test_SWOT_quad --n_companies 1358
```


# Original SentenceBERT 

---

- 🤗 [Original Model](https://github.com/BM-K/Sentence-Embedding-is-all-you-need/tree/main/KoSBERT)
- Dataset
    - Training: snli_1.0_train.ko.tsv, sts-train.tsv (multi-task)
      - Performance can be further improved by adding multinli data to training.
    - Validation: sts-dev.tsv
    - Test: sts-test.tsv


# License

---

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />



# References

---


```bibtex
@misc{park2021klue,
    title={KLUE: Korean Language Understanding Evaluation},
    author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
    year={2021},
    eprint={2105.09680},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```
