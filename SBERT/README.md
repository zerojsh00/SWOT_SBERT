# SWOT Sentence BERT

---

- SentenceBERT
  - `Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터를 `강점-약점(S-W)`, `강점-기회(S-O)`, `강점-위협(S-T)`, `약점-기회(W-O)`, `약점-위협(W-T)`, `기회-위협(O-T)`의 총 여섯 가지 관계 중 어떤 범주인지를 예측하는 자연어 추론(Natural Language Inference, NLI) 태스크로 SentenceBERT 학습함
  - 학습된 SentenceBERT로 `Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터를 임베딩 하고, 이를 바탕으로 K-Means Clustering 하여 군집을 생성함
    - 이때, 학습용 데이터셋으로 구축한 군집 내에서 가장 많은 빈도의 실제 데이터 값을 해당 군집의 레이블로 간주함
  - 학습된 SentenceBERT로 테스트용 데이터셋을 임베딩 하고, 이들이 어떤 군집에 소속하는지 파악한 후, 앞서 정의한 해당 군집의 레이블을 예측된 레이블로 정의함
  - 예측된 레이블과 실제 테스트용 데이터셋의 레이블을 비교함

# How to Run

---

논문 실험을 위해서 아래의 학습 및 테스트를 자동화 했던 코드로 실험을 돌려볼 수 있다. 상용화 시에는 사용되지 않는다.
``` bash
bash run_experiments.sh
```
### Train SWOT Sentence BERT
``` python
python training_SWOT_NLI --n_companies 1358
```
--n_companies 파라미터는 데이터로 사용할 기업의 수를 의미하며, `../get_SWOT_dataset.py` 파일을 통해서 사전에 데이터를 구축해두어야 한다.
### Test SWOT Sentence BERT
``` python
python test_SWOT_NLI --n_companies 1358
```
--n_companies 파라미터는 데이터로 사용할 기업의 수를 의미하며, `../get_SWOT_dataset.py` 파일을 통해서 사전에 데이터를 구축해두어야 한다.