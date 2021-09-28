# Tune: Scalable Hyperparameter Tuning [¶](https://docs.ray.io/en/latest/tune/index.html#tune-scalable-hyperparameter-tuning)



[![../_images/tune.png](https://docs.ray.io/en/latest/_images/tune.png)](https://docs.ray.io/en/latest/_images/tune.png)

Tune은 모든 규모에서 실험 실행 및 하이퍼파라미터 조정을 위한 Python 라이브러리입니다.

핵심 기능:

- 10줄 미만의 코드 로 다중 노드 [distributed hyperparameter sweep](https://docs.ray.io/en/latest/tune/tutorials/tune-distributed.html#tune-distributed)을 시작합니다.
- [PyTorch, XGBoost, MXNet 및 Keras를 포함한](https://docs.ray.io/en/latest/tune/tutorials/overview.html#tune-guides) 모든 기계 학습 프레임워크를 지원합니다.
- 자동으로 [체크포인트를 관리](https://docs.ray.io/en/latest/tune/user-guide.html#tune-checkpoint)하고 [TensorBoard에 로깅](https://docs.ray.io/en/latest/tune/user-guide.html#tune-logging)합니다.
- [Population Based Training(PBT)](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-pbt) , [BayesOptSearch](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bayesopt) , [HyperBand/ASHA](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-hyperband) 와 같은 최신 알고리즘 중에서 선택하십시오.
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)를 사용하여 동일한 인프라에서 학습에서 서비스로 모델을 이동합니다.

**시작하시겠습니까?** [주요 개념 페이지](https://docs.ray.io/en/latest/tune/key-concepts.html).

## Quick Start [¶](https://docs.ray.io/en/latest/tune/index.html#quick-start)

이 예를 실행하려면 다음을 설치하십시오. `pip install 'ray[tune]'`

이 예제에서는 예제 목적 함수를 최적화하기 위해 병렬 그리드 검색을 실행합니다.

```
from ray import tune


def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100)**(-1) + beta * 0.1


def training_function(config):
    # Hyperparameters
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)


analysis = tune.run(
    training_function,
    config={
        "alpha": tune.grid_search([0.001, 0.01, 0.1]),
        "beta": tune.choice([1, 2, 3])
    })

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
```



TensorBoard가 설치된 경우 모든 시도 결과를 자동으로 시각화합니다.

```
tensorboard --logdir ~/ray_results
```



[![../_images/tune-start-tb.png](https://docs.ray.io/en/latest/_images/tune-start-tb.png)](https://docs.ray.io/en/latest/_images/tune-start-tb.png)

TF2 및 TensorBoard를 사용하는 경우 Tune은 TensorBoard HParams 출력도 자동으로 생성합니다.

[![../_images/tune-hparams-coord.png](https://docs.ray.io/en/latest/_images/tune-hparams-coord.png)](https://docs.ray.io/en/latest/_images/tune-hparams-coord.png)

## 왜 Tune을 선택합니까? [¶](https://docs.ray.io/en/latest/tune/index.html#why-choose-tune)

다른 많은 하이퍼파라미터 최적화 라이브러리가 있습니다. Tune을 처음 사용하는 경우 "Tune이 다른 점은 무엇입니까?"

### 최첨단 최적화 알고리즘 [¶](https://docs.ray.io/en/latest/tune/index.html#cutting-edge-optimization-algorithms)

사용자는 모델 성능을 빠르게 향상시키기 위해 하이퍼파라미터 최적화를 검토하고 있을 것입니다.

Tune을 사용하면 이러한 최첨단 최적화 알고리즘을 다양하게 활용하여 [잘못된 하이퍼파라미터 평가를 적극적으로 종료](https://docs.ray.io/en/latest/tune/tune-scheduler-hyperband)하고, [평가할 더 나은 파라미터를 지능적으로 선택](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg)하거나, [훈련 중에 하이퍼 ](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-pbt)[파라미터를 변경](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg)하여 하이퍼파라미터 일정을 최적화 함으로써 튜닝 비용을 줄일 수 있습니다.

### 최고 수준의 개발자 생산성 [¶](https://docs.ray.io/en/latest/tune/index.html#first-class-developer-productivity)

기계 학습 프레임워크의 주요 문제는 프레임워크에 맞게 모든 코드를 재구성해야 한다는 것입니다.

Tune을 사용하면 [몇 가지 코드 조각을 추가](https://docs.ray.io/en/latest/tune/tutorials/tune-tutorial.html#tune-tutorial)하여 모델을 최적화할 수 있습니다 .

또한 Tune은 실제로 코드 교육 워크플로에서 상용구를 제거하여 자동으로 [체크포인트를 관리](https://docs.ray.io/en/latest/tune/user-guide.html#tune-checkpoint) 하고 MLflow 및 TensorBoard와 같은 [도구에 결과를 로깅](https://docs.ray.io/en/latest/tune/user-guide.html#tune-logging)합니다.

### 즉시 사용 가능한 다중 GPU 및 분산 교육 [¶](https://docs.ray.io/en/latest/tune/index.html#multi-gpu-distributed-training-out-of-the-box)

하이퍼파라미터 조정은 시간이 많이 소요되는 것으로 알려져 있어 프로세스를 병렬화해야 하는 경우가 많습니다. 대부분의 다른 조정 프레임워크에서는 고유한 다중 프로세스 프레임워크를 구현하거나 고유한 분산 시스템을 구축하여 하이퍼파라미터 조정 속도를 높여야 합니다.

그러나 Tune을 사용하면 [여러 GPU와 여러 노드에서 투명하게 병렬화](https://docs.ray.io/en/latest/tune/user-guide.html#tune-parallelism)할 수 있습니다 . Tune은 원활한 [내결함성과 클라우드 지원](https://docs.ray.io/en/latest/tune/tutorials/tune-distributed.html#tune-distributed)을 제공하므로 저렴한 선점형 인스턴스를 사용하여 비용을 최대 10배까지 절감하면서 하이퍼파라미터 검색을 100배 확장할 수 있습니다.

### 이미 하이퍼파라미터 조정을 하고 있는 경우에는 어떻게 합니까? [¶](https://docs.ray.io/en/latest/tune/index.html#what-if-i-m-already-doing-hyperparameter-tuning)

HyperOpt 또는 Bayesian 최적화와 같은 기존 하이퍼파라미터 조정 도구를 이미 사용하고 있을 수 있습니다.

이 상황에서 Tune을 사용하면 실제로 기존 워크플로를 강화할 수 있습니다. Tune의 [검색 알고리즘](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg)은 널리 사용되는 다양한 하이퍼파라미터 튜닝 라이브러리(예: Nevergrad 또는 HyperOpt)와 통합되어 성능 저하 없이 최적화 프로세스를 원활하게 확장할 수 있습니다.

## 참고 자료 [¶](https://docs.ray.io/en/latest/tune/index.html#reference-materials)

다음은 Tune에 대한 몇 가지 참고 자료입니다.

- [사용자 가이드 및 Tune 구성](https://docs.ray.io/en/latest/tune/user-guide.html)
- [자주 묻는 질문](https://docs.ray.io/en/latest/tune/tutorials/overview.html#tune-faq)
- [코드](https://github.com/ray-project/ray/tree/master/python/ray/tune) : Tune용 GitHub 리포지토리

다음은 Tune에 대한 몇 가지 블로그 게시물과 이야기입니다.

- [블로그] [Tune: 모든 규모에서 빠른 하이퍼파라미터 튜닝을 위한 Python 라이브러리](https://towardsdatascience.com/fast-hyperparameter-tuning-at-scale-d428223b081c)
- [블로그] [Ray Tune을 통한 최첨단 하이퍼파라미터 튜닝](https://medium.com/riselab/cutting-edge-hyperparameter-tuning-with-ray-tune-be6c0447afdf)
- [블로그] [Ray Tune을 사용한 tensorflow의 간단한 하이퍼파라미터 및 아키텍처 검색](http://louiskirsch.com/ai/ray-tune)
- [슬라이드] [RISECamp 2019에서 발표된 강연](https://docs.google.com/presentation/d/1v3IldXWrFNMK-vuONlSdEuM82fuGTrNUDuwtfx4axsQ/edit?usp=sharing)
- [동영상] [RISECamp 2018 강연](https://www.youtube.com/watch?v=38Yd_dXW51Q)
- [동영상] [최신 초매개변수 최적화 가이드(PyData LA 2019)](https://www.youtube.com/watch?v=10uz5U3Gy6E) ( [슬라이드](https://speakerdeck.com/richardliaw/a-modern-guide-to-hyperparameter-optimization) )

## Tune 인용 [¶](https://docs.ray.io/en/latest/tune/index.html#citing-tune)

Tune이 학술 연구에 도움이 된다면 [우리 논문](https://arxiv.org/abs/1807.05118)을 인용하는 것이 좋습니다 . 다음은 bibtex의 예입니다.

```
@article{liaw2018tune,
    title={Tune: A Research Platform for Distributed Model Selection and Training},
    author={Liaw, Richard and Liang, Eric and Nishihara, Robert
            and Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
    journal={arXiv preprint arXiv:1807.05118},
    year={2018}
}
```

