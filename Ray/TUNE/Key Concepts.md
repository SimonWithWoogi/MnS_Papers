# Key Concepts[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#key-concepts)

Tune을 사용하기 위해 알아야 할 주요 개념을 빠르게 살펴보겠습니다. 이 가이드에서는 다음을 다룹니다.

- [Trainables](https://docs.ray.io/en/latest/tune/key-concepts.html#trainables)
- [tune.run and Trials](https://docs.ray.io/en/latest/tune/key-concepts.html#tune-run-and-trials)
- [Search spaces](https://docs.ray.io/en/latest/tune/key-concepts.html#search-spaces)
- [Search Algorithms](https://docs.ray.io/en/latest/tune/key-concepts.html#search-algorithms)
- [Trial Schedulers](https://docs.ray.io/en/latest/tune/key-concepts.html#trial-schedulers)
- [Analysis](https://docs.ray.io/en/latest/tune/key-concepts.html#analysis)
- [What’s Next?](https://docs.ray.io/en/latest/tune/key-concepts.html#what-s-next)

![../_images/tune-workflow.png](https://docs.ray.io/en/latest/_images/tune-workflow.png)

## [Trainables](https://docs.ray.io/en/latest/tune/key-concepts.html#id1)

시작으로, 이 목적 함수를 최대화해 보겠습니다.

```python
def objective(x, a, b):
    return a * (x ** 0.5) + b
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Tune을 사용하려면 이 함수를 [학습 가능한 경량 API](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-docs)로 래핑해야 합니다 . [함수 기반 버전](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-function-api) 또는 [클래스 ](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-class-api)[기반 버전을](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-function-api) 사용할 수 있습니다 .

다음은 [함수 기반 Trainable API](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-function-api)를 사용하여 목적 함수를 지정하는 예입니다.

```python
def trainable(config):
    # config (dict): A dict of hyperparameters.

    for x in range(20):
        score = objective(x, config["a"], config["b"])

        tune.report(score=score)  # This sends the score to Tune.
```

다음은 [클래스 기반 API](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-class-api)를 사용하여 목적 함수를 지정하는 예입니다.

```python
from ray import tune

class Trainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]

    def step(self):  # This is called iteratively.
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}
```

Tip : `Trainable` 클래스 내에서 `tune.report`를 사용하지 마십시오.

[교육(tune.Trainable, tune.report)](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-docs) 및 [예제](https://docs.ray.io/en/latest/tune/examples/index.html#tune-general-examples) 문서를 참조하세요.



## [tune.run and Trials ](https://docs.ray.io/en/latest/tune/key-concepts.html#id2)[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#tune-run-and-trials)

[tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run-ref) 을 사용 하여 하이퍼파라미터 튜닝을 실행합니다. 이 기능은 실험을 관리하고 [로깅](https://docs.ray.io/en/latest/tune/user-guide.html#tune-logging) , [체크포인트](https://docs.ray.io/en/latest/tune/user-guide.html#tune-checkpoint) 및 [조기 중지](https://docs.ray.io/en/latest/tune/user-guide.html#tune-stopping) 와 같은 많은 기능을 제공합니다 .

```python
# Pass in a Trainable class or function to tune.run.
tune.run(trainable)
```



`tune.run`인수에서 몇 가지 하이퍼파라미터 구성을 생성하고 이를 [Trial 객체](https://docs.ray.io/en/latest/tune/api_docs/internals.html#trial-docstring) 로 래핑합니다 .

각 trial은 다음을 갖습니다:

- 하이퍼파라미터 구성( `trial.config`), id( `trial.trial_id`)
- 리소스 사양( `resources_per_trial`또는 `trial.resources`)
- 및 기타 구성 값.

각 trial은 또한 [Trainable](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-docs)의 하나의 인스턴스와 연결됩니다 . `tune.run` 이 종료된 후 제공되는 [Analysis 객체](https://docs.ray.io/en/latest/tune/key-concepts.html#tune-concepts-analysis)를 통해 trial 개체에 접근할 수 있습니다 

모든 trial이 중지되거나 오류가 발생할 때까지 `tune.run` 이 실행됩니다.

```python
== Status ==
Memory usage on this node: 11.4/16.0 GiB
Using FIFO scheduling algorithm.
Resources requested: 1/12 CPUs, 0/0 GPUs, 0.0/3.17 GiB heap, 0.0/1.07 GiB objects
Result logdir: /Users/foo/ray_results/myexp
Number of trials: 1 (1 RUNNING)
+----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+
| Trial name           | status   | loc                 |         a |      b |  score | total time (s) |  iter |
|----------------------+----------+---------------------+-----------+--------+--------+----------------+-------|
| MyTrainable_a826033a | RUNNING  | 10.234.98.164:31115 | 0.303706  | 0.0761 | 0.1289 |        7.54952 |    15 |
+----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+
```



10개의 trial을 쉽게 실행할 수도 있습니다. Tune은 [병렬로 실행할 시도 횟수를 자동으로 결정](https://docs.ray.io/en/latest/tune/user-guide.html#tune-parallelism)합니다.

```python
tune.run(trainable, num_samples=10)
```



마지막으로 Tune의 [search space API](https://docs.ray.io/en/latest/tune/user-guide.html#tune-default-search-space)를 통해 검색 하이퍼파라미터를 무작위로 샘플링하거나 그리드 검색할 수 있습니다 .

```python
space = {"x": tune.uniform(0, 1)}
tune.run(my_trainable, config=space, num_samples=10)
```

더 많은 문서를 참조하십시오: [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run-ref).



## [Search spaces](https://docs.ray.io/en/latest/tune/key-concepts.html#id3)[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#search-spaces)

하이퍼파라미터를 최적화하려면 *search space* 을 정의해야 합니다. Search space은 하이퍼파라미터에 대한 유효한 값을 정의하고 이러한 값이 샘플링되는 방법(예: 균일 분포 또는 정규 분포)을 지정할 수 있습니다.

Tune은 search space와 샘플링 방법을 정의하는 다양한 기능을 제공합니다. [여기에서 이러한 search space 정의에 대한 문서를 찾을 수 있습니다](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) .

일반적으로 `tune.run()`의 *config* 매개변수로 search space 정의를 전달합니다.

다음은 모든 search space 기능을 다루는 예입니다. 다시 말하지만, [여기에 이러한 모든 기능에 대한 전체 설명이 있습니다](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) .

```python
config = {
    "uniform": tune.uniform(-5, -1),  # Uniform float between -5 and -1
    "quniform": tune.quniform(3.2, 5.4, 0.2),  # Round to increments of 0.2
    "loguniform": tune.loguniform(1e-4, 1e-1),  # Uniform float in log space
    "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-5),  # Round to increments of 0.00005
    "randn": tune.randn(10, 2),  # Normal distribution with mean 10 and sd 2
    "qrandn": tune.qrandn(10, 2, 0.2),  # Round to increments of 0.2
    "randint": tune.randint(-9, 15),  # Random integer between -9 and 15
    "qrandint": tune.qrandint(-21, 12, 3),  # Round to increments of 3 (includes 12)
    "lograndint": tune.lograndint(1, 10),  # Random integer in log space
    "qlograndint": tune.qlograndint(1, 10, 2),  # Round to increments of 2
    "choice": tune.choice(["a", "b", "c"]),  # Choose one of these options uniformly
    "func": tune.sample_from(lambda spec: spec.config.uniform * 0.01), # Depends on other value
    "grid": tune.grid_search([32, 64, 128])  # Search over all these values
}
```



## [Search Algorithms](https://docs.ray.io/en/latest/tune/key-concepts.html#id4)[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#search-algorithms)

훈련 과정의 하이퍼파라미터를 최적화하려면 더 나은 하이퍼파라미터를 제안하는 데 도움이 되는 [Search Algorithm](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg)을 사용하는 것이 좋습니다.

```python
# Be sure to first run `pip install bayesian-optimization`

from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

# Define the search space
config = {
    "a": tune.uniform(0, 1),
    "b": tune.uniform(0, 20)
}

# Execute 20 trials using BayesOpt and stop after 20 iterations
tune.run(
    trainable,
    config=config,
    metric="score",
    mode="max",
    # Limit to two concurrent trials (otherwise we end up with random search)
    search_alg=ConcurrencyLimiter(
        BayesOptSearch(random_search_steps=4),
        max_concurrent=2),
    num_samples=20,
    stop={"training_iteration": 20},
    verbose=2)
```



Tune에는 [Nevergrad](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#nevergrad) 및 [Hyperopt](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-hyperopt) 와 같은 많은 인기 있는 **최적화** 라이브러리와 통합되는 SearchAlgorithms가 있습니다. Tune은 제공된 search space를 search algorithms/기본 라이브러리가 기대하는 search space로 자동 변환합니다.

```
Note:
현재 모든 search algorithms에 대해 자동 search space 변환을 구현하는 과정에 있습니다. 현재 이것은 AxSearch, BayesOpt, Hyperopt 및 Optuna에서 작동합니다. 다른 search algorithms이 곧 추가되겠지만 현재로서는 각각의 search space으로 인스턴스화해야 합니다.
```

문서를 참조하십시오. [검색 알고리즘(tune.suggest)](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg).



## [Trial Schedulers](https://docs.ray.io/en/latest/tune/key-concepts.html#id5)[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#trial-schedulers)

또한 [Trial Scheduler](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-schedulers)를 사용하여 학습과정을 보다 효율적으로 만들 수 있습니다 .

Trial Schedulers는 실행 중인 Trial의 하이퍼파라미터를 중지/일시 중지/조정하여 하이퍼파라미터 조정 프로세스를 훨씬 빠르게 만듭니다.

```python
from ray.tune.schedulers import HyperBandScheduler

# Create HyperBand scheduler and maximize score
hyperband = HyperBandScheduler(metric="score", mode="max")

# Execute 20 trials using HyperBand using a search space
configs = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 1)}

tune.run(
    MyTrainableClass,
    config=configs,
    num_samples=20,
    scheduler=hyperband
)
```



[Population-based Training](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-pbt) 및[ HyperBand](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-scheduler-hyperband)는 Trial Scheduler로 구현된 인기 있는 최적화 알고리즘의 예입니다.

**Search Algorithms**과 달리 , [Trial Scheduler](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-schedulers)는 어떤 하이퍼파라미터 구성을 평가할지 선택하지 않습니다. 그러나 이 둘을 동시에 사용할 수 있습니다.

문서를 참조하십시오: [Summary](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#schedulers-ref)



## [Analysis](https://docs.ray.io/en/latest/tune/key-concepts.html#id6)[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#analysis)

`tune.run`은 학습 분석에 사용할 수 있는 메서드가 있는 [Analysis](https://docs.ray.io/en/latest/tune/api_docs/analysis.html#tune-analysis-docs) 개체를 반환합니다 .

```python
analysis = tune.run(trainable, search_alg=algo, stop={"training_iteration": 20})

best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
```



이 개체는 또한 모든 학습실행을 데이터 프레임으로 검색할 수 있으므로 결과에 대해 임시 데이터 분석을 수행할 수 있습니다.

```python
# Get a dataframe with the last results for each trial
df_results = analysis.results_df

# Get a dataframe of results for a specific score or mode
df = analysis.dataframe(metric="score", mode="max")
```



## [What's Next? ](https://docs.ray.io/en/latest/tune/key-concepts.html#id7)[¶](https://docs.ray.io/en/latest/tune/key-concepts.html#what-s-next)

이제 Tune에 대해 제대로 이해했으므로 다음을 확인하세요.

- [사용자 가이드 및 Tune 구성](https://docs.ray.io/en/latest/tune/user-guide.html) : Tune의 기능에 대한 포괄적인 개요입니다.
- [자습서 및 FAQ](https://docs.ray.io/en/latest/tune/tutorials/overview.html#tune-guides) : 선호하는 기계 학습 라이브러리와 함께 Tune을 사용하기 위한 자습서입니다.
- [예제](https://docs.ray.io/en/latest/tune/examples/index.html) : 선호하는 기계 학습 라이브러리와 함께 Tune을 사용하기 위한 종단 간 예제 및 템플릿.
- [A Basic Tune Tutorial](https://docs.ray.io/en/latest/tune/tutorials/tune-tutorial.html#tune-tutorial) : Tune 실험을 설정하는 과정을 안내하는 간단한 튜토리얼입니다.

### 추가 질문이나 문제가 있습니까? [¶](https://docs.ray.io/en/latest/tune/key-concepts.html#further-questions-or-issues)

다음 채널을 통해 질문이나 문제 또는 피드백을 게시할 수 있습니다.

1. [토론 게시판](https://discuss.ray.io/) : **Ray 사용** 또는 **기능 요청** **에** 대한 **질문** 입니다.
2. [GitHub 문제](https://github.com/ray-project/ray/issues) : **버그 보고용** .
3. [Ray Slack](https://forms.gle/9TSdDYUgxYs8SA9e8) : Ray 유지 관리자와 **연락** 하기 위한 것입니다.
4. [StackOverflow](https://stackoverflow.com/questions/tagged/ray) : **Ray 에 대한 질문**에 [ray] 태그를 사용하십시오.

