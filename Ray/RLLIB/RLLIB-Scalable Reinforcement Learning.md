# RLlib: Scalable Reinforcement Learning[¶](https://docs.ray.io/en/latest/rllib.html#rllib-scalable-reinforcement-learning)

`RLlib` 는 강화학습을 위한 오픈소스 라이브러리로 높은 `scalability(확장성)` 와 `다양한 어플리케이션의 통합된 API` 를 제공합니다. `RLlib` 는 자연스럽게 `TensorFlow, Tensorflow Eager, PyTorch` 를 지원하고 있으나, 대부분의 프레임 워크를 지원합니다.

![_images/rllib-stack.svg](https://docs.ray.io/en/latest/_images/rllib-stack.svg)

시작하며, [custom env example](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py) 여기서는 환경 예제를, [API documentation](https://docs.ray.io/en/latest/rllib-toc.html) 여기서는 API 설명을 확인할 수 있습니다.

만약 `RLlib` 로 개발중인 자체 알고리즘을 테스트하고자 한다면, [concepts and custom algorithms](https://docs.ray.io/en/latest/rllib-concepts.html) 여기서 볼 수 있습니다.

### Running RLlib[¶](https://docs.ray.io/en/latest/rllib.html#running-rllib)

`RLlib` 는 `ray` 에 종속관계입니다. 먼저, `PyTorch, TensorFlow` 에서 `RLlib` 를 다운받습니다.

```python
pip install 'ray[rllib]'
```

그리고, 아래의 동일한 방법으로 학습을 시도할 수 있습니다.

```
rllib train --run=PPO --env=CartPole-v0  # -v [-vv] for verbose,
                                         # --config='{"framework": "tf2", "eager_tracing": True}' for eager,
                                         # --torch to use PyTorch OR --config='{"framework": "torch"}'
```

```python
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
tune.run(PPOTrainer, config={"env": "CartPole-v0"})  # "log_level": "INFO" for verbose,
                                                     # "framework": "tfe"/"tf2" for eager,
                                                     # "framework": "torch" for PyTorch
```

다음은, 세 가지의 중요 컨셉들을 설명하겠습니다. `Plicies, Samples, Trainers`

### Policies[¶](https://docs.ray.io/en/latest/rllib.html#policies)

`Policies` 는 `RLlib` 에서 핵심 컨셉입니다. 요약하자면, `Policies` 는 환경 속에서 `agents` 가 어떤 행동을 하는 지 정의하는 파이썬 클래스입니다.

`상호작용하는 워커들(Rollout workers)` 는 에이전트 액션을 결정하기위해 정책을 얻어옵니다.  [gym](https://docs.ray.io/en/latest/rllib-env.html#openai-gym) 환경에서는, 단일 에이전트와 정책이 존재합니다. [vector envs](https://docs.ray.io/en/latest/rllib-env.html#vectorized) 에서는 다중 에이전트와 `policy` 간 인터페이스를 합니다. 그리고 [multi-agent](https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)는 다중의 `policies` 가 있을 수 있고, 에이전트를 단일에서 다중으로 제어합니다.

![_images/multi-flat.svg](https://docs.ray.io/en/latest/_images/multi-flat.svg)

`Policies` 는 [any framework](https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py) 를 이용하여 구현할 수 있습니다. 하지만, `TensorFlow, PyTorch` 의 `RLlib` 는 [build_tf_policy](https://docs.ray.io/en/latest/rllib-concepts.html#building-policies-in-tensorflow) and [build_torch_policy](https://docs.ray.io/en/latest/rllib-concepts.html#building-policies-in-pytorch) 를 가지고 있습니다. 학습가능한 `policy` 를 정의하기 위해 도와주는 함수들은 아래를 참고할 수 있습니다.

```python
def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return -tf.reduce_mean(
        action_dist.logp(train_batch["actions"]) * train_batch["rewards"])

# <class 'ray.rllib.policy.tf_policy_template.MyTFPolicy'>
MyTFPolicy = build_tf_policy(
    name="MyTFPolicy",
    loss_fn=policy_gradient_loss)
```

### Sample Batches[¶](https://docs.ray.io/en/latest/rllib.html#sample-batches)

단일 프로세서나 [대규모 클러스터](https://docs.ray.io/en/latest/rllib-training.html#specifying-resources)에서든, `RLlib` 의 [sample batches](https://github.com/ray-project/ray/blob/master/rllib/policy/sample_batch.py) 안에서 모든 데이터들은 서로 주고받습니다. `Sample batches` 는 학습동안 지나온 히스토리(trajectory)의 파편들을 하나 이상으로 전환해줍니다. 일반적으로 `RLlib` 는 `rollout workers` 로부터 `rollout_fragment_lenth` 의 파라미터를 통해 배치사이즈를  가져옵니다. 그리고 하나 이상의 배치들을 `train_batch_size` 를 기준으로 합칩니다.

일반적인 샘플 배치는 요약된 정보를 제공합니다. 대체적으로 값들은 행렬로 구성되어 있습니다만, 효과적인 인코딩과 네트웍 연산에 용이합니다.

```python
{ 'action_logp': np.ndarray((200,), dtype=float32, min=-0.701, max=-0.685, mean=-0.694),
  'actions': np.ndarray((200,), dtype=int64, min=0.0, max=1.0, mean=0.495),
  'dones': np.ndarray((200,), dtype=bool, min=0.0, max=1.0, mean=0.055),
  'infos': np.ndarray((200,), dtype=object, head={}),
  'new_obs': np.ndarray((200, 4), dtype=float32, min=-2.46, max=2.259, mean=0.018),
  'obs': np.ndarray((200, 4), dtype=float32, min=-2.46, max=2.259, mean=0.016),
  'rewards': np.ndarray((200,), dtype=float32, min=1.0, max=1.0, mean=1.0),
  't': np.ndarray((200,), dtype=int64, min=0.0, max=34.0, mean=9.14)}
```

[multi-agent mode](https://docs.ray.io/en/latest/rllib-concepts.html#policies-in-multi-agent) 에서는 각 `policy` 들이 개별로 나뉘어 샘플 배치들이 들어갑니다.

### Training[¶](https://docs.ray.io/en/latest/rllib.html#training)

주어진 입력으로 들어가는 샘플 배치를 통해 `Policy` 를 개선하는 `learn_on_batch()` 함수를 정의합니다. `Tensor Flow, Pytorch` 의 `policy` 에는 입력 샘플 배치 텐서와 아웃풋 스칼라 손실 기능이 있는 `loss function` 이 구현되어 있습니다.

- Simple [policy gradient loss](https://github.com/ray-project/ray/blob/master/rllib/agents/pg/pg_tf_policy.py)
- Simple [Q-function loss](https://github.com/ray-project/ray/blob/a1d2e1762325cd34e14dc411666d63bb15d6eaf0/rllib/agents/dqn/simple_q_policy.py#L136)
- Importance-weighted [APPO surrogate loss](https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/appo_torch_policy.py)

`RLlib` 의  [Trainer classes](https://docs.ray.io/en/latest/rllib-concepts.html#trainers) 는 `Rollout` 실행과 `Policy` 최적화의 분산 워크플로우를 조정합니다. 이를 위해서 `Ray` 의 [병렬](https://docs.ray.io/en/latest/iter.html) `Iterator` 가 원하는 컴퓨테이션 패턴을 구현합니다. 아래 그림은 동기화된 샘플링을 보여주며, [여러 패턴](https://docs.ray.io/en/latest/rllib-algorithms.html)중에서도 가장 간단한 패턴에 속합니다.

![_images/a2c-arch.svg](https://docs.ray.io/en/latest/_images/a2c-arch.svg)

Synchronous Sampling (e.g., A2C, PG, PPO)[¶](https://docs.ray.io/en/latest/rllib.html#id1)

`RLlib` 는 싱글 코어에서 클러스터의 수 천많은 코어들로 스케일에 맞춘 학습을 위해 [Ray actors](https://docs.ray.io/en/latest/actors.html) 를 사용합니다. 여러분은 학습을 위해 `num_workers` 를 바꿔가면서 [병렬화에 대한 설정](https://docs.ray.io/en/latest/rllib-training.html#specifying-resources)이 가능합니다. 상세한 설명은 [scaling guide](https://docs.ray.io/en/latest/rllib-training.html#scaling-guide) 해당 가이드에서 설명합니다.

### Application Support[¶](https://docs.ray.io/en/latest/rllib.html#application-support)

파이썬에서 정의된 환경너머로는 `RLlib` 가 [데이터셋](https://docs.ray.io/en/latest/rllib-offline.html)을 제공하고 있습니다. 그리고 다양한 통합된 전략들이 [외부 어플리케이션](https://docs.ray.io/en/latest/rllib-env.html#external-agents-and-applications)으로 있습니다.

### Customization[¶](https://docs.ray.io/en/latest/rllib.html#customization)

`RLlib` 는 커스텀 방법을 제공합니다. 강화학습의 대부분 파트의 [neural network models](https://docs.ray.io/en/latest/rllib-models.html#tensorflow-models), [action distributions](https://docs.ray.io/en/latest/rllib-models.html#custom-action-distributions),[policy definitions](https://docs.ray.io/en/latest/rllib-concepts.html#policies): the [environment](https://docs.ray.io/en/latest/rllib-env.html#configuring-environments), and the [sample collection process](https://docs.ray.io/en/latest/rllib-sample-collection.html) 입니다.

![_images/rllib-components.svg](https://docs.ray.io/en/latest/_images/rllib-components.svg)