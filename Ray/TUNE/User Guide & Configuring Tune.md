# User Guide & Configuring Tune [¶](https://docs.ray.io/en/latest/tune/user-guide.html#user-guide-configuring-tune)

이 페이지는 Tune의 다양한 기능과 구성을 보여줍니다.

---

팁

계속하기 전에 [주요 개념](https://docs.ray.io/en/latest/tune/key-concepts.html#tune-60-seconds) 을 읽어 보십시오 .

---

이 문서는 Tune 실행을 위한 일부 구성과 핵심 개념에 대한 개요를 제공합니다.



## Resources(Parallelism, GPUs, Distributed)[¶](https://docs.ray.io/en/latest/tune/user-guide.html#resources-parallelism-gpus-distributed)

---

팁

모든 것을 순차적으로 실행하려면 [Ray Local Mode](https://docs.ray.io/en/latest/tune/user-guide.html#tune-debugging)를 사용하십시오 .

---



병렬도는 `resources_per_trial`(기본값은 1 CPU, 0 GPU per trial) 및 Tune에 사용 가능한 리소스( `ray.cluster_resources()`)에 의해 결정됩니다 .

기본적으로 Tune은 자동으로 N개의 동시 Trial을 실행합니다. 여기서 N은 컴퓨터의 CPU(코어) 수입니다.

```python
# If you have 4 CPUs on your machine, this will run 4 concurrent trials at a time.
tune.run(trainable, num_samples=10)
```



 `resources_per_trial`로 이 병렬 처리를 재정의할 수 있습니다. 여기에서 사전이나 [`PlacementGroupFactory`](https://docs.ray.io/en/latest/tune/api_docs/internals.html#ray.tune.utils.placement_groups.PlacementGroupFactory) 개체를 사용하여 리소스 요청을 지정할 수 있습니다 . 어쨌든 Ray Tune은 각 trial에 대해 배치 그룹을 시작하려고 시도합니다.

```python
# If you have 4 CPUs on your machine, this will run 2 concurrent trials at a time.
tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 2})

# If you have 4 CPUs on your machine, this will run 1 trial at a time.
tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 4})

# Fractional values are also supported, (i.e., {"cpu": 0.5}).
tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 0.5})
```



Tune은 GPU와 CPU를 `resources_per_trial`에서 각 Trial로 할당합니다 . 지금 Trial을 예약할 수 없더라도 Ray Tune은 여전히 해당 배치 그룹을 시작하려고 시도합니다. 사용 가능한 리소스가 충분하지 않은 경우, Ray 클러스터 시작 관리자를 사용하고 있다면 [자동 크기 조정 동작](https://docs.ray.io/en/latest/cluster/index.html#cluster-index) 이 트리거됩니다 .

훈련 가능한 함수가 더 많은 원격 작업자를 시작하는 경우, 이러한 리소스를 요청하기 위해 배치 그룹 팩토리 객체를 전달해야 합니다. 자세한 내용은 [`PlacementGroupFactory documentation`](https://docs.ray.io/en/latest/tune/api_docs/internals.html#ray.tune.utils.placement_groups.PlacementGroupFactory)를 참조하십시오.

### GPU 사용 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#using-gpus)

GPU를 활용하려면 `tune.run(resources_per_trial)`의 `gpu`에서 설정해야 합니다. 이것은 각 Trial에 대해 자동으로 `CUDA_VISIBLE_DEVICES` 설정됩니다 .

```python
# If you have 8 GPUs, this will run 8 trials at once.
tune.run(trainable, num_samples=10, resources_per_trial={"gpu": 1})

# If you have 4 CPUs on your machine and 1 GPU, this will run 1 trial at a time.
tune.run(trainable, num_samples=10, resources_per_trial={"cpu": 2, "gpu": 1})
```

[Keras MNIST 예제](https://docs.ray.io/en/latest/tune/examples/tune_mnist_keras.html) 에서 이에 대한 예제를 찾을 수 있습니다 .

---

**경고**

'gpu'가 설정되지 않으면 `CUDA_VISIBLE_DEVICES`환경 변수가 비어 있게 설정되어 GPU 액세스가 허용되지 않습니다.

**문제 해결** : 경우에 따라 새 Trial을 실행할 때 GPU 메모리 문제가 발생할 수 있습니다. 이는 이전 Trial이 GPU 상태를 충분히 빠르게 정리하지 않았기 때문일 수 있습니다. 이를 피하기 위해 `tune.utils.wait_for_gpu`를 사용할 수 있습니다 - [docstring을](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-util-ref) 참고하세요.

---

### Concurrent samples [¶](https://docs.ray.io/en/latest/tune/user-guide.html#concurrent-samples)

[Search algorithm](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#tune-search-alg)을 사용하는 경우, 평가되는 Trial 횟수를 제한할 수 있습니다. 예를 들어, 순차 최적화를 수행하기 위해 시도 평가를 직렬화할 수 있습니다.

다음의 경우 `ray.tune.suggest.ConcurrencyLimiter`가 동시성 양을 제한합니다.

```python
algo = BayesOptSearch(utility_kwargs={
    "kind": "ucb",
    "kappa": 2.5,
    "xi": 0.0
})
algo = ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = AsyncHyperBandScheduler()
```

자세한 내용은 [ConcurrencyLimiter(tune.suggest.ConcurrencyLimiter)](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#limiter) 를 참조하세요.



### Distributed Tuning[¶](https://docs.ray.io/en/latest/tune/user-guide.html#distributed-tuning)

---

팁

이 섹션에서는 여러 컴퓨터에서 Tune을 실행하는 방법을 다룹니다. 분산 교육 작업 조정에 대한 지침 은 [분산 교육](https://docs.ray.io/en/latest/tune/user-guide.html#tune-dist-training) 을 참조하세요.

---



Ray 클러스터에 `tune.run`을 실행하기전에 `ray.init`을 실행하십시오.

`ray.init`에 대한 자세한 내용은 [CLI(ray start)를 통한 Ray 시작](https://docs.ray.io/en/latest/starting-ray.html#start-ray-cli)을 참조하십시오.

```python
# Connect to an existing distributed Ray cluster
ray.init(address=<ray_address>)
tune.run(trainable, num_samples=100, resources_per_trial=tune.PlacementGroupFactory([{"CPU": 2, "GPU": 1}]))
```

Tune [분산 실험 가이드](https://docs.ray.io/en/latest/tune/tutorials/tune-distributed.html#tune-distributed) 를 읽어보세요.



### Tune Distributed Training [¶](https://docs.ray.io/en/latest/tune/user-guide.html#tune-distributed-training)

분산 학습 작업을 조정하기 위해 Tune은 다양한 교육 프레임워크를 위한 `DistributedTrainableCreator` 세트를 제공합니다 . 다음은 분산 TensorFlow 작업을 조정하는 예입니다.

```python
# Please refer to full example in tf_distributed_keras_example.py
from ray.tune.integration.tensorflow import DistributedTrainableCreator
tf_trainable = DistributedTrainableCreator(
    train_mnist,
    use_gpu=args.use_gpu,
    num_workers=2)
tune.run(tf_trainable,
         num_samples=1)
```

[Distributed PyTorch](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-ddp-doc) , [TensorFlow](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#tune-dist-tf-doc) 및 [Horovod](https://docs.ray.io/en/latest/tune/api_docs/integration.html#tune-integration-horovod) 작업 조정에 대해 자세히 알아보세요.



## Search Space(Grid/Random) [¶](https://docs.ray.io/en/latest/tune/user-guide.html#search-space-grid-random)

`tune.run(config=)`에 전달된 dict를 통해 그리드 검색 또는 샘플링 분포를 지정할 수 있습니다.

```python
parameters = {
    "qux": tune.sample_from(lambda spec: 2 + 2),
    "bar": tune.grid_search([True, False]),
    "foo": tune.grid_search([1, 2, 3]),
    "baz": "asd",  # a constant value
}

tune.run(trainable, config=parameters)
```

기본적으로 각 랜덤 변수와 그리드 검색 포인트는 한 번 샘플링됩니다. 여러 무작위 샘플을 가져오려면 `num_samples: N`실험 구성에 추가 하세요. *grid_search*이 인수로 제공된 경우, 그리드는 `num_samples` 만큼 반복됩니다.

```python
 # num_samples=10 repeats the 3x3 grid search 10 times, for a total of 90 trials
 tune.run(
     my_trainable,
     name="my_trainable",
     config={
         "alpha": tune.uniform(100),
         "beta": tune.sample_from(lambda spec: spec.config.alpha * np.random.normal()),
         "nn_layers": [
             tune.grid_search([16, 64, 256]),
             tune.grid_search([16, 64, 256]),
         ],
     },
     num_samples=10
 )
```

Search space는 다른 search algorithm에서 상호 운용되지 않을 수 있습니다. 예를 들어, 많은 검색 알고리즘의 경우 `grid_search`매개변수 를 사용할 수 없을 수도 있습니다.

이에 대해 [Search Space API](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-search-space)를 읽어 보세요.



## Auto-filled Metrics [¶](https://docs.ray.io/en/latest/tune/user-guide.html#auto-filled-metrics)

두 학습 API에서 임의의 값과 metrics를 기록할 수 있습니다.

```python
def trainable(config):
    for i in range(num_epochs):
        ...
        tune.report(acc=accuracy, metric_foo=random_metric_1, bar=metric_2)

class Trainable(tune.Trainable):
    def step(self):
        ...
        # don't call report here!
        return dict(acc=accuracy, metric_foo=random_metric_1, bar=metric_2)
```

학습하는 동안 Tune은 사용자가 제공한 값과 함께 아래 지표를 자동으로 기록합니다. 이 모든 것은 중지 조건으로 사용되거나 Trial Schedulers/Search Algorithms에 매개변수로 전달될 수 있습니다.

- `config`: 하이퍼파라미터 구성
- `date`: 결과가 처리된 문자열 형식의 날짜 및 시간
- `done`: Trial이 완료되면 True, 그렇지 않으면 False
- `episodes_total`: 총 에피소드 수(RLLib 학습 가능 항목의 경우)
- `experiment_id`: 고유한 실험 ID
- `experiment_tag`: 고유한 실험 태그(매개변수 값 포함)
- `hostname`: 작업자의 호스트 이름
- `iterations_since_restore`:  체크포인트에서 작업자를 복원한 후`tune.report()/trainable.train()`가 호출된 횟수
- `node_ip`: 작업자의 호스트 IP
- `pid`: 작업자 프로세스의 프로세스 ID(PID)
- `time_since_restore`: 체크포인트에서 복원한 후 경과된 시간(초)입니다.
- `time_this_iter_s`: 현재 훈련 반복의 런타임(초)입니다(즉, 훈련 가능한 함수 또는 `_train()`클래스 API에 대한 한 번의 호출 .
- `time_total_s`: 총 실행 시간(초)입니다.
- `timestamp`: 결과가 처리된 타임스탬프
- `timesteps_since_restore`: 체크포인트에서 복원한 이후의 시간 단계 수
- `timesteps_total`: 총 타임스텝 수
- `training_iteration`: `tune.report()`가 호출된 횟수
- `trial_id`: 고유한 평가판 ID

이러한 모든 metrics는 `Trial.last_result`사전에서 볼 수 있습니다 .



## Checkpointing[¶](https://docs.ray.io/en/latest/tune/user-guide.html#checkpointing)

하이퍼파라미터 검색을 실행할 때 Tune은 모델을 자동으로 주기적으로 저장/체크포인트할 수 있습니다. 이를 통해 다음을 수행할 수 있습니다.

> - 훈련 전반에 걸쳐 중간 모델 저장
> - 선점형 머신 사용(마지막 체크포인트에서 자동으로 복원)
> - HyperBand 및 PBT와 같은 Trial Scheduler를 사용할 때 Trial을 일시 중지합니다.

Tune의 checkpointing 기능을 사용하려면 `checkpoint_dir`함수 서명에 인수를 노출 하고 `tune.checkpoint_dir` 호출해야 합니다.

```python
import os
import time
from ray import tune

def train_func(config, checkpoint_dir=None):
    start = 0
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            state = json.loads(f.read())
            start = state["step"] + 1

    for step in range(start, 100):
        time.sleep(1)

        # Obtain a checkpoint directory
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            with open(path, "w") as f:
                f.write(json.dumps({"step": step}))

        tune.report(hello="world", ray="tune")

tune.run(train_func)
```

이 예제에서 체크포인트는 반복 학습을 통해 `local_dir/exp_name/trial_name/checkpoint_<step>`에 저장됩니다 .

`tune.run(restore=<checkpoint_dir>)`를사용하여 단일 trial 체크포인트를 복원할 수 있습니다. 이렇게 하면 실험 이름과 같은 실험 구성을 변경할 수 있습니다.

```python
# Restored previous trial from the given checkpoint
tune.run(
    "PG",
    name="RestoredExp", # The name can be different.
    stop={"training_iteration": 10}, # train 5 more iterations than previous
    restore="~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
    config={"env": "CartPole-v0"},
)
```





### Distributed Checkpointing [¶](https://docs.ray.io/en/latest/tune/user-guide.html#distributed-checkpointing)

다중 노드 클러스터에서 Tune은 헤드 노드에서 모든 Trial 체크포인트의 복사본을 자동으로 생성합니다. 이를 위해서는 [클러스터 런처로](https://docs.ray.io/en/latest/cluster/cloud.html#cluster-cloud) Ray 클러스터를 시작 해야 하며 rsync도 설치해야 합니다.

동기화를 트리거 하려면 `tune.checkpoint_dir`API를 사용해야 합니다 .

Kubernetes에서 Ray Tune을 실행하는 경우 일반적으로 체크포인트 공유를 위해  [`DurableTrainable`](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#ray.tune.durable) 또는 공유 파일 시스템을 사용해야 합니다 . Kubernetes <tune-kubernetes>에서 Tune을 실행하기 위한 모범 사례는 여기를 참조하세요.

클러스터 시작 관리자를 사용하지 않는 경우 NFS 또는 전역 파일 시스템을 설정하고 노드 간 동기화를 비활성화해야 합니다.

```python
sync_config = tune.SyncConfig(sync_to_driver=False)
tune.run(func, sync_config=sync_config)
```



## 튜닝 실행 중지 및 재개 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#stopping-and-resuming-a-tuning-run)

Ray Tune은 실험 상태를 주기적으로 체크포인트하여 실패하거나 중지할 때 다시 시작할 수 있습니다. 체크포인트 기간은 훈련 결과 및 일정을 처리하는 데 시간의 최소 95%가 사용되도록 동적으로 조정됩니다.

 `tune.run()`을 실행 중인 프로세스에 SIGINT 신호를 보내면(대개 콘솔에서 Ctrl+C를 누를 때 발생) Ray Tune은 훈련을 정상적으로 종료하고 최종 실험 수준 체크포인트를 저장합니다. 그런 다음 `resume=True` 설정과 함께 `tune.run()`을 호출하여 앞으로 이 실행을 계속할 수 있습니다.

```python
tune.run(
    train,
    # ...
    name="my_experiment"
)

# This is interrupted e.g. by sending a SIGINT signal
# Next time, continue the run like so:

tune.run(
    train,
    # ...
    name="my_experiment",
    resume=True
)
```



`resume=True` 를 사용할 때 Ray Tune이 실험 폴더(보통 `~/ray_results/my_experiment` 에 저장됨)를 감지할 수 있도록  `name`을 전달해야 합니다. 첫 번째 호출에서 `name`을 전달하는 것을 잊은 경우 실행을 재개할 때 `name`을 전달할 수 있습니다. 이 경우 실험 이름에 날짜 접미사가 붙으므로  `tune.run(my_trainable)`를 실행할 때 `name`은 `my_trainable_2021-01-29_10-16-44`의 형태로 될 것입니다.

원래 튜닝 실행의 결과 테이블을 보면 전달해야 하는 이름을 확인할 수 있습니다.

```python
== Status ==
Memory usage on this node: 11.0/16.0 GiB
Using FIFO scheduling algorithm.
Resources requested: 1/16 CPUs, 0/0 GPUs, 0.0/4.69 GiB heap, 0.0/1.61 GiB objects
Result logdir: /Users/ray/ray_results/my_trainable_2021-01-29_10-16-44
Number of trials: 1/1 (1 RUNNING)
```



## Handling Large Datasets [¶](https://docs.ray.io/en/latest/tune/user-guide.html#handling-large-datasets)

드라이버에서 큰 개체(예: 학습 데이터, 모델 가중치)를 계산하고 각 trial에서 해당 개체를 사용하려는 경우가 많습니다.

Tune은 큰 개체를 훈련 가능 개체로 브로드캐스트할 수 있는 래퍼 기능 `tune.with_parameters()`를 제공합니다 . 이 래퍼와 함께 전달된 개체는 [Ray 개체 저장소에 저장](https://docs.ray.io/en/latest/walkthrough.html#objects-in-ray) 되고 자동으로 가져와서 매개변수로 훈련 가능 개체에 전달됩니다.

---

팁

개체의 크기가 작거나 [Ray 개체 저장소](https://docs.ray.io/en/latest/walkthrough.html#objects-in-ray) 에 이미 존재 하는 경우 `tune.with_parameters()` 를 사용할 필요가 없습니다 . [부분](https://docs.python.org/3/library/functools.html#functools.partial)을 사용 하거나 `config`대신 직접 전달할 수 있습니다 .

---



```python
from ray import tune

import numpy as np

def f(config, data=None):
    pass
    # use data

data = np.random.random(size=100000000)

tune.run(tune.with_parameters(f, data=data))
```





## Stopping Trials [¶](https://docs.ray.io/en/latest/tune/user-guide.html#stopping-trials)

`stop`인수를 에 전달하여 시도가 일찍 중지되는 시점을 제어할 수 있습니다 `tune.run`. 이 인수는 사전, 함수 또는 `Stopper`클래스를 인수로 취 합니다.

사전이 전달되는 경우 키는 `tune.report`Function API 의 반환 결과 또는 `step()`(의 결과 `step`및 자동 완성 메트릭 포함)의 모든 필드일 수 있습니다 .

아래 예에서 각 시도는 10회 반복을 완료하거나 평균 정확도가 0.98에 도달하면 중지됩니다. 이러한 메트릭은 **증가** 하는 것으로 가정됩니다 .

```python
# training_iteration is an auto-filled metric by Tune.
tune.run(
    my_trainable,
    stop={"training_iteration": 10, "mean_accuracy": 0.98}
)
```



더 많은 유연성을 위해 대신 함수를 전달할 수 있습니다. 함수가 전달 되면 인수로 취하여 부울 값을 반환 해야 합니다 ( 시행을 중지해야 하는 경우 및 그렇지 않은 경우).`(trial_id, result)``True``False`

```python
def stopper(trial_id, result):
    return result["mean_accuracy"] / result["training_iteration"] > 5

tune.run(my_trainable, stop=stopper)
```



마지막으로 `Stopper`전체 실험을 중지 하기 위해 추상 클래스를 구현할 수 있습니다 . 예를 들어 다음 예에서는 개별 시도에서 기준이 충족된 후 모든 시도를 중지하고 새 시도가 시작되지 않도록 합니다.

```python
from ray.tune import Stopper

class CustomStopper(Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        if not self.should_stop and result['foo'] > 10:
            self.should_stop = True
        return self.should_stop

    def stop_all(self):
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop

stopper = CustomStopper()
tune.run(my_trainable, stop=stopper)
```



위의 예에서 현재 실행 중인 시도는 즉시 중지되지 않지만 현재 반복이 완료되면 중지됩니다.

Ray Tune은 즉시 사용 가능한 스토퍼 클래스 세트와 함께 제공됩니다. [스토퍼](https://docs.ray.io/en/latest/tune/api_docs/stoppers.html#tune-stoppers) 문서를 참조하십시오 .



## 로깅 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#logging)

기본적으로 Tune은 Tensorboard, CSV 및 JSON 형식에 대한 결과를 기록합니다. 모델 가중치 또는 기울기와 같은 더 낮은 수준을 기록해야 하는 경우 [훈련 가능한 기록을](https://docs.ray.io/en/latest/tune/api_docs/logging.html#trainable-logging) 참조하십시오 .

**여기에서 로깅 및 사용자 정의에 대해 자세히 알아보십시오** . [로거(tune.logger)](https://docs.ray.io/en/latest/tune/api_docs/logging.html#loggers-docstring) .

Tune은 각 시도의 결과를 지정된 로컬 디렉토리 아래의 하위 폴더에 기록합니다(기본값은 `~/ray_results`.

```python
# This logs to 2 different trial folders:
# ~/ray_results/trainable_name/trial_name_1 and ~/ray_results/trainable_name/trial_name_2
# trainable_name and trial_name are autogenerated.
tune.run(trainable, num_samples=2)
```



당신은 지정할 수 있습니다 `local_dir`과 `trainable_name`:

```python
# This logs to 2 different trial folders:
# ./results/test_experiment/trial_name_1 and ./results/test_experiment/trial_name_2
# Only trial_name is autogenerated.
tune.run(trainable, num_samples=2, local_dir="./results", name="test_experiment")
```



사용자 지정 평가판 폴더 이름을 지정하려면 use `trial_name_creator`인수를 tune.run 에 전달할 수 있습니다 . 이것은 다음 서명이 있는 함수를 사용합니다.

```python
def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    return str(trial)

tune.run(
    MyTrainableClass,
    name="example-experiment",
    num_samples=1,
    trial_name_creator=trial_name_string
)
```



평가판: [평가판](https://docs.ray.io/en/latest/tune/api_docs/internals.html#trial-docstring) 문서를 참조하십시오 .



## 텐서보드(로깅) [¶](https://docs.ray.io/en/latest/tune/user-guide.html#tensorboard-logging)

Tune은 텐서보드 파일을 자동으로 출력합니다 `tune.run`. 텐서보드에서 학습을 시각화하려면 tensorboardX를 설치하세요.

```python
$ pip install tensorboardX
```



그런 다음 실험을 실행한 후 결과의 출력 디렉토리를 지정하여 TensorBoard로 실험을 시각화할 수 있습니다.

```python
$ tensorboard --logdir=~/ray_results/my_experiment
```



sudo 액세스 권한이 없는 원격 다중 사용자 클러스터에서 Ray를 실행하는 경우 다음 명령을 실행하여 텐서보드가 tmp 디렉토리에 쓸 수 있는지 확인할 수 있습니다.

```python
$ export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir=~/ray_results
```



![../_images/ray-tune-tensorboard.png](https://docs.ray.io/en/latest/_images/ray-tune-tensorboard.png)

TF2를 사용하는 경우 Tune은 아래와 같이 TensorBoard HParams 출력도 자동으로 생성합니다.

```python
tune.run(
    ...,
    config={
        "lr": tune.grid_search([1e-5, 1e-4]),
        "momentum": tune.grid_search([0, 0.9])
    }
)
```



![../_images/tune-hparams.png](https://docs.ray.io/en/latest/_images/tune-hparams.png)

## 콘솔 출력 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#console-output)

사용자가 제공한 필드는 최선을 다해 자동으로 출력됩니다. [Reporter](https://docs.ray.io/en/latest/tune/api_docs/reporters.html#tune-reporter-doc) 개체를 사용 하여 콘솔 출력을 사용자 지정할 수 있습니다 .

```python
== Status ==
Memory usage on this node: 11.4/16.0 GiB
Using FIFO scheduling algorithm.
Resources requested: 4/12 CPUs, 0/0 GPUs, 0.0/3.17 GiB heap, 0.0/1.07 GiB objects
Result logdir: /Users/foo/ray_results/myexp
Number of trials: 4 (4 RUNNING)
+----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+
| Trial name           | status   | loc                 |    param1 | param2 |    acc | total time (s) |  iter |
|----------------------+----------+---------------------+-----------+--------+--------+----------------+-------|
| MyTrainable_a826033a | RUNNING  | 10.234.98.164:31115 | 0.303706  | 0.0761 | 0.1289 |        7.54952 |    15 |
| MyTrainable_a8263fc6 | RUNNING  | 10.234.98.164:31117 | 0.929276  | 0.158  | 0.4865 |        7.0501  |    14 |
| MyTrainable_a8267914 | RUNNING  | 10.234.98.164:31111 | 0.068426  | 0.0319 | 0.9585 |        7.0477  |    14 |
| MyTrainable_a826b7bc | RUNNING  | 10.234.98.164:31112 | 0.729127  | 0.0748 | 0.1797 |        7.05715 |    14 |
+----------------------+----------+---------------------+-----------+--------+--------+----------------+-------+
```



## 결과 업로드 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#uploading-results)

업로드 디렉터리가 제공되면 Tune은 `local_dir`기본적으로 표준 S3/gsutil/HDFS URI를 지원하는 지정된 디렉터리로 결과를 자동으로 동기화합니다 .

```python
tune.run(
    MyTrainableClass,
    local_dir="~/ray_results",
    sync_config=tune.SyncConfig(upload_dir="s3://my-log-dir")
)
```



의 `sync_to_cloud`인수를 사용하여 임의의 저장소를 지정하도록 이를 사용자 지정할 수 있습니다 `tune.SyncConfig`. 이 인수는 대체 필드가 동일한 문자열 또는 임의의 함수를 지원합니다.

```python
tune.run(
    MyTrainableClass,
    sync_config=tune.SyncConfig(
        upload_dir="s3://my-log-dir",
        sync_to_cloud=custom_sync_str_or_func
    )
)
```



문자열이 제공되면 대체 필드 `{source}`및 `{target}`와 같은 등이 포함되어야 합니다 . 또는 다음 서명과 함께 함수를 제공할 수 있습니다.`s3 sync {source} {target}`

```python
def custom_sync_func(source, target):
    # do arbitrary things inside
    sync_cmd = "s3 {source} {target}".format(
        source=source,
        target=target)
    sync_process = subprocess.Popen(sync_cmd, shell=True)
    sync_process.wait()
```



기본적으로 동기화는 300초마다 발생합니다. 동기화 빈도를 변경하려면 `TUNE_CLOUD_SYNC_S`드라이버 의 환경 변수를 원하는 동기화 주기로 설정하세요.

업로드는 전역 실험 상태가 수집될 때만 발생하며, 이 빈도는 `TUNE_GLOBAL_CHECKPOINT_S`환경 변수에 의해 결정됩니다 . 따라서 실제 업로드 기간은 로 지정됩니다 .`max(TUNE_CLOUD_SYNC_S, TUNE_GLOBAL_CHECKPOINT_S)`



## Docker로 튜닝 사용하기 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#using-tune-with-docker)

Tune은 필요에 따라 서로 다른 원격 컨테이너 간에 파일과 검사점을 자동으로 동기화합니다.

Docker 클러스터에서 이 작업을 수행하려면, 예를 들어 도커 컨테이너와 함께 Ray 자동 확장 처리를 사용하는 경우 `DockerSyncer`의 `sync_to_driver`인수에 a 를 전달해야 합니다 `tune.SyncConfig`.

```python
from ray.tune.integration.docker import DockerSyncer
sync_config = tune.SyncConfig(
    sync_to_driver=DockerSyncer)

tune.run(train, sync_config=sync_config)
```





## Kubernetes로 Tune 사용하기 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#using-tune-with-kubernetes)

Ray Tune은 필요에 따라 서로 다른 원격 노드 간에 파일과 체크포인트를 자동으로 동기화합니다. 이것은 일반적으로 SSH를 통해 발생하지만 특히 많은 시도를 병렬로 실행할 때 [성능 병목 현상](https://docs.ray.io/en/latest/tune/tutorials/overview.html#tune-bottlenecks) 이 발생할 수 있습니다 .

대신 노드 간 추가 동기화가 필요하지 않도록 체크포인트에 공유 스토리지를 사용해야 합니다. 두 가지 주요 옵션이 있습니다.

먼저 다음을 사용하여 [`DurableTrainable`](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#ray.tune.durable)AWS S3 또는 Google Cloud Storage와 같은 클라우드 스토리지에 로그 및 체크포인트를 저장할 수 있습니다 .

```python
from ray import tune

tune.run(
    tune.durable(train_fn),
    # ...,
    sync_config=tune.SyncConfig(
        sync_to_driver=False,
        upload_dir="s3://your-s3-bucket/durable-trial/"
    )
)
```



둘째, NFS와 같은 공유 파일 시스템을 설정할 수 있습니다. 이렇게 하면 자동 평가판 동기화를 비활성화합니다.

```python
from ray import tune

tune.run(
    train_fn,
    # ...,
    local_dir="/path/to/shared/storage",
    sync_config=tune.SyncConfig(
        # Do not sync to driver because we are on shared storage
        sync_to_driver=False
    )
)
```



마지막으로, 시험 동기화를 위해 여전히 ssh를 사용하고 싶지만 Ray 클러스터 런처에서 실행되고 있지 않다면 `KubernetesSyncer`의 `sync_to_driver`인수에 전달해야 할 수도 있습니다 `tune.SyncConfig`. Kubernetes 네임스페이스를 명시적으로 지정해야 합니다.

```python
from ray.tune.integration.kubernetes import NamespacedKubernetesSyncer
sync_config = tune.SyncConfig(
    sync_to_driver=NamespacedKubernetesSyncer("ray")
)

tune.run(train, sync_config=sync_config)
```



다른 두 옵션 중 하나를 대신 사용하는 것이 좋습니다. 그러면 오버헤드가 줄어들고 포드가 SSH를 통해 서로 연결할 필요가 없기 때문입니다.



## stdout 및 stderr을 파일로 리디렉션 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#redirecting-stdout-and-stderr-to-files)

stdout 및 stderr 스트림은 일반적으로 콘솔에 인쇄됩니다. 원격 액터의 경우 Ray는 이러한 로그를 수집하여 헤드 프로세스에 인쇄합니다.

그러나 나중에 분석 또는 문제 해결을 위해 스트림 출력을 파일로 수집하려는 경우 Tune은 이를 위한 유틸리티 매개변수 를 제공합니다 `log_to_file`.

에 전달 `log_to_file=True`하면 `tune.run()`stdout 및 stderr이 각각 `trial_logdir/stdout`및 에 기록됩니다 `trial_logdir/stderr`.

```python
tune.run(
    trainable,
    log_to_file=True)
```



출력 파일을 지정하려면 결합된 출력이 저장될 하나의 파일 이름을 전달하거나 stdout 및 stderr에 대해 각각 두 개의 파일 이름을 전달할 수 있습니다.

```python
tune.run(
    trainable,
    log_to_file="std_combined.log")

tune.run(
    trainable,
    log_to_file=("my_stdout.log", "my_stderr.log"))
```



파일 이름은 평가판의 logdir에 상대적입니다. 절대 경로도 전달할 수 있습니다.

경우 `log_to_file`설정, 조정은 자동으로 레이의 기본 로거를위한 새로운 로깅 핸들러를 등록하고 지정된 표준 에러 출력 파일에 출력을 기록합니다.



## 콜백 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#callbacks)

Ray Tune은 훈련 과정의 다양한 시간 동안 호출되는 콜백을 지원합니다. 콜백은 매개변수로 전달될 수 있으며 `tune.run()`하위 메소드가 자동으로 호출됩니다.

이 간단한 콜백은 결과가 수신될 때마다 메트릭을 인쇄합니다.

```python
from ray import tune
from ray.tune import Callback


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['metric']}")


def train(config):
    for i in range(10):
        tune.report(metric=i)


tune.run(
    train,
    callbacks=[MyCallback()])
```



자세한 내용과 사용 가능한 후크 [는 Ray Tune 콜백용 API 문서](https://docs.ray.io/en/latest/tune/api_docs/internals.html#tune-callbacks-docs) 를 [참조](https://docs.ray.io/en/latest/tune/api_docs/internals.html#tune-callbacks-docs) 하세요 .



## 디버깅 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#debugging)

기본적으로 Tune은 여러 프로세스에서 하이퍼파라미터 평가를 실행합니다. 그러나 훈련 프로세스를 디버그해야 하는 경우 단일 프로세스에서 모든 작업을 수행하는 것이 더 쉬울 수 있습니다. `local_mode`전에 다음을 호출하여 모든 Ray 함수가 단일 프로세스에서 발생하도록 할 수 있습니다 `tune.run`.

```python
ray.init(local_mode=True)
```



다중 구성 평가가 있는 로컬 모드는 계산을 인터리브하므로 단일 구성 평가를 실행할 때 가장 자연스럽게 사용됩니다.

참고 `local_mode`몇 가지 알려진 문제가 있습니다, 그래서 읽어 보시기 바랍니다 [이 끝을](https://docs.ray.io/en/latest/auto_examples/testing-tips.html#local-mode-tips) 추가 정보를 원하시면.

## 첫 번째 실패 후 중지 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#stopping-after-the-first-failure)

기본적으로 `tune.run`모든 시도가 종료되거나 오류가 발생할 때까지 계속 실행됩니다. 시행 착오가 발생 **하는** 즉시 전체 Tune 실행을 중지하려면 :

```python
tune.run(trainable, fail_fast=True)
```



이것은 대규모 하이퍼파라미터 실험을 설정하려고 할 때 유용합니다.



## 환경 변수 [¶](https://docs.ray.io/en/latest/tune/user-guide.html#environment-variables)

Ray Tune의 일부 동작은 환경 변수를 사용하여 구성할 수 있습니다. Ray Tune이 현재 고려하는 환경 변수는 다음과 같습니다.

- **TUNE_CLUSTER_SSH_KEY** : Tune 드라이버 프로세스에서 체크포인트 동기화를 위해 원격 클러스터 시스템에 연결하는 데 사용하는 SSH 키입니다. 이것이 설정되지 않은 경우 `~/ray_bootstrap_key.pem`사용됩니다.
- **TUNE_DISABLE_AUTO_CALLBACK_LOGGERS** : Ray Tune은 CSV 및 JSON 로거 콜백이 전달되지 않은 경우 자동으로 추가합니다. 이 변수를 1로 설정 하면 이 자동 생성이 비활성화됩니다. 이것은 튜닝 실행 후 결과 분석에 영향을 미칠 가능성이 가장 높다는 점에 유의하십시오.
- **TUNE_DISABLE_AUTO_CALLBACK_SYNCER** : Ray Tune은 동기화 로그 및 체크포인트가 전달되지 않은 경우 서로 다른 노드 간의 체크포인트를 동기화하기 위해 자동으로 Syncer 콜백을 추가합니다. 이 변수를 1로 설정 하면 이 자동 생성이 비활성화됩니다. 이는 PopulationBasedTraining과 같은 고급 일정 알고리즘에 영향을 미칠 가능성이 높습니다.
- **TUNE_DISABLE_AUTO_INIT** : `ray.init()`Ray 세션에 연결되지 않은 경우 자동 호출을 비활성화합니다 .
- **TUNE_DISABLE_DATED_SUBDIR** : Ray Tune은 이름이 명시적으로 지정되지 않았거나 학습 가능 항목이 문자열로 전달되지 않은 경우 실험 디렉토리에 날짜 문자열을 자동으로 추가합니다. 이 환경 변수를 설정 `1`하면 이러한 날짜 문자열을 추가할 수 없습니다.
- **TUNE_DISABLE_STRICT_METRIC_CHECKING** : , 스케줄러 또는 검색 알고리즘 을 통해 Tune에 메트릭을 보고 `tune.report()`하고 `metric`매개변수를 전달할 `tune.run()`때 메트릭이 결과에 보고되지 않으면 Tune에 오류가 발생합니다. 이 환경 변수를 `1`로 설정하면 이 검사가 비활성화됩니다.
- **TUNE_DISABLE_SIGINT_HANDLER** : Ray Tune은 SIGINT 신호(예: Ctrl+C로 전송)를 포착하여 정상적으로 종료하고 최종 체크포인트를 수행합니다. 이 변수를 로 설정하면 `1`신호 처리가 비활성화되고 실행이 즉시 중지됩니다. 기본값은 `0`입니다.
- **TUNE_FUNCTION_THREAD_TIMEOUT_S** : 함수 API가 스레드에 완료를 지시한 후 스레드가 완료될 때까지 기다리는 시간(초). 기본값은 `2`입니다.
- **TUNE_GLOBAL_CHECKPOINT_S** : Tune의 실험 상태가 체크포인트되는 빈도를 제한하는 시간(초)입니다. 설정하지 않으면 기본값이 로 설정됩니다 `10`.
- **TUNE_MAX_LEN_IDENTIFIER** : 시도 하위 디렉토리 이름의 최대 길이(매개변수 값이 있는 이름)
- **TUNE_MAX_PENDING_TRIALS_PG** : 배치 그룹이 사용될 때 보류 중인 시도의 최대 수입니다. 기본값 `auto`은 로 설정 되며 임의/격자 검색 및 기타 검색 알고리즘 에 대해 로 업데이트됩니다 .`max(16, cluster_cpus * 1.1)``1`
- **TUNE_PLACEMENT_GROUP_AUTO_DISABLED** : Ray Tune은 레거시 리소스 요청 대신 배치 그룹을 자동으로 사용합니다. 이것을 1로 설정하면 레거시 배치가 활성화됩니다.
- **TUNE_PLACEMENT_GROUP_CLEANUP_DISABLED** : Ray Tune `_tune__`은 실행을 시작하기 전에 이름에 접두사가 있는 기존 배치 그룹을 정리합니다 . 이는 `tune.run()`동일한 스크립트에서 에 대한 여러 호출 이 수행 될 때 예약된 배치 그룹이 제거되었는지 확인하는 데 사용됩니다 . 다른 스크립트에서 여러 Tune 실행을 병렬로 실행하는 경우 이 기능을 비활성화할 수 있습니다. 비활성화하려면 1로 설정합니다.
- **TUNE_PLACEMENT_GROUP_PREFIX** : Ray Tune에 의해 생성된 배치 그룹의 접두사. 이 접두사는 예를 들어 튜닝 실행 시작/중지 시 정리해야 하는 배치 그룹을 식별하는 데 사용됩니다. 이것은 첫 번째 실행이 시작될 때 고유한 이름으로 초기화됩니다.
- **TUNE_PLACEMENT_GROUP_RECON_INTERVAL** : 배치 그룹을 조정하는 빈도입니다. 조정은 요청된 배치 그룹 수와 보류/실행 시도가 동기화되었는지 확인하는 데 사용됩니다. 정상적인 상황에서는 어쨌든 다르지 않아야 하지만 조정을 통해 배치 그룹이 수동으로 제거되는 경우를 캡처해야 합니다. 조정에는 많은 시간이 걸리지 않지만 많은 수의 짧은 시도를 실행할 때 추가될 수 있습니다. 기본값은 매 `5`(초)입니다.
- **TUNE_PLACEMENT_GROUP_WAIT_S** : 시험 실행기가 조정 루프를 계속하기 전에 배치 그룹이 배치되기를 기다리는 기본 시간입니다. 이것을 float로 설정하면 몇 초 동안 차단됩니다. 이것은 주로 테스트 목적으로 사용됩니다. 기본값은 -1이며 차단을 비활성화합니다.
- **TUNE_RESULT_DIR** : Ray Tune 시도 결과가 저장되는 디렉토리. 이것이 설정되지 않은 경우 `~/ray_results`사용됩니다.
- **TUNE_RESULT_BUFFER_LENGTH** : Ray Tune은 드라이버에게 전달되기 전에 훈련 가능한 결과를 버퍼링할 수 있습니다. 이를 활성화하면 훈련 가능 항목이 추측에 따라 계속되기 때문에 일정 결정이 지연될 수 있습니다. `0`결과 버퍼링 을 비활성화 하도록 설정합니다 . 기본값은 1000(결과) 또는 와 함께 사용되는 경우 1(버퍼링 없음)입니다 `checkpoint_at_end`.
- **TUNE_RESULT_BUFFER_MAX_TIME_S** : 마찬가지로, Ray Tune 버퍼는 최대 `number_of_trial/10`초 까지 결과를 생성 하지만 이 값보다 길지는 않습니다. 기본값은 100(초)입니다.
- **TUNE_RESULT_BUFFER_MIN_TIME_S** : 또한 결과를 버퍼링하는 최소 시간을 지정할 수 있습니다. 기본값은 0입니다.
- **TUNE_SYNCER_VERBOSITY** : Tune with Docker Syncer 사용 시 명령어 출력량. 기본값은 0입니다.
- **TUNE_TRIAL_STARTUP_GRACE_PERIOD** : 평가판 시작 후 Ray Tune이 성공적인 평가판 시작을 확인하는 시간입니다. 유예 기간이 지나면 Tune은 실행 중인 평가판 결과가 수신될 때까지 차단됩니다. 이것을 0 이하로 설정하여 비활성화할 수 있습니다.
- **TUNE_WARN_THRESHOLD_S** : Tune 이벤트 루프 작업이 너무 오래 걸리는 경우 로깅을 위한 임계값입니다. 기본값은 0.5(초)입니다.
- **TUNE_STATE_REFRESH_PERIOD** : Ray에서 추적하는 리소스를 업데이트하는 빈도입니다. 기본값은 10(초)입니다.

주로 통합 라이브러리와 관련된 몇 가지 환경 변수가 있습니다.

- **SIGOPT_KEY** : SigOpt API 액세스 키.
- **WANDB_API_KEY** : 가중치 및 **편향** API 키. 대신 사용할 수도 있습니다 .`wandb login`
