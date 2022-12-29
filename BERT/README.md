# SWOT BERT
  - `Strength`, `Weakness`, `Opportunity`, `Threat`로 레이블링된 문단(문장) 데이터를 `Strength`, `Weakness`, `Opportunity`, `Threat`로 직접 예측함

---

# How to Run

---

논문 실험을 위해서 아래의 학습 및 테스트를 자동화 했던 코드로 실험을 돌려볼 수 있다. 상용화 시에는 사용되지 않는다.
``` bash
bash run_experiments.sh
```
### Train SWOT BERT
``` python
python training_SWOT_quad --n_companies 1358
```
--n_companies 파라미터는 데이터로 사용할 기업의 수를 의미하며, `../get_SWOT_dataset.py` 파일을 통해서 사전에 데이터를 구축해두어야 한다.
### Test SWOT BERT
``` python
python test_SWOT_quad --n_companies 1358
```
--n_companies 파라미터는 데이터로 사용할 기업의 수를 의미하며, `../get_SWOT_dataset.py` 파일을 통해서 사전에 데이터를 구축해두어야 한다.