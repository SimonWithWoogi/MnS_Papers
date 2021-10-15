# RLlib Table of Contents[¶](https://docs.ray.io/en/latest/rllib-toc.html#rllib-table-of-contents)

## Training APIs[¶](https://docs.ray.io/en/latest/rllib-toc.html#training-apis)

- [Command-line](https://docs.ray.io/en/latest/rllib-training.html)
  - [Evaluating Trained Policies](https://docs.ray.io/en/latest/rllib-training.html#evaluating-trained-policies)
- [Configuration](https://docs.ray.io/en/latest/rllib-training.html#configuration)
  - [Specifying Parameters](https://docs.ray.io/en/latest/rllib-training.html#specifying-parameters)
  - [Specifying Resources](https://docs.ray.io/en/latest/rllib-training.html#specifying-resources)
  - [Common Parameters](https://docs.ray.io/en/latest/rllib-training.html#common-parameters)
  - [Scaling Guide](https://docs.ray.io/en/latest/rllib-training.html#scaling-guide)
  - [Tuned Examples](https://docs.ray.io/en/latest/rllib-training.html#tuned-examples)
- [Basic Python API](https://docs.ray.io/en/latest/rllib-training.html#basic-python-api)
  - [Computing Actions](https://docs.ray.io/en/latest/rllib-training.html#computing-actions)
  - [Accessing Policy State](https://docs.ray.io/en/latest/rllib-training.html#accessing-policy-state)
  - [Accessing Model State](https://docs.ray.io/en/latest/rllib-training.html#accessing-model-state)
- [Advanced Python APIs](https://docs.ray.io/en/latest/rllib-training.html#advanced-python-apis)
  - [Custom Training Workflows](https://docs.ray.io/en/latest/rllib-training.html#custom-training-workflows)
  - [Global Coordination](https://docs.ray.io/en/latest/rllib-training.html#global-coordination)
  - [Callbacks and Custom Metrics](https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics)
  - [Customizing Exploration Behavior](https://docs.ray.io/en/latest/rllib-training.html#customizing-exploration-behavior)
  - [Customized Evaluation During Training](https://docs.ray.io/en/latest/rllib-training.html#customized-evaluation-during-training)
  - [Rewriting Trajectories](https://docs.ray.io/en/latest/rllib-training.html#rewriting-trajectories)
  - [Curriculum Learning](https://docs.ray.io/en/latest/rllib-training.html#curriculum-learning)
- [Debugging](https://docs.ray.io/en/latest/rllib-training.html#debugging)
  - [Gym Monitor](https://docs.ray.io/en/latest/rllib-training.html#gym-monitor)
  - [Eager Mode](https://docs.ray.io/en/latest/rllib-training.html#eager-mode)
  - [Episode Traces](https://docs.ray.io/en/latest/rllib-training.html#episode-traces)
  - [Log Verbosity](https://docs.ray.io/en/latest/rllib-training.html#log-verbosity)
  - [Stack Traces](https://docs.ray.io/en/latest/rllib-training.html#stack-traces)
- [External Application API](https://docs.ray.io/en/latest/rllib-training.html#external-application-api)

## Environments[¶](https://docs.ray.io/en/latest/rllib-toc.html#environments)

- [RLlib Environments Overview](https://docs.ray.io/en/latest/rllib-env.html)
- [OpenAI Gym](https://docs.ray.io/en/latest/rllib-env.html#openai-gym)
- [Vectorized](https://docs.ray.io/en/latest/rllib-env.html#vectorized)
- [Multi-Agent and Hierarchical](https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
- [External Agents and Applications](https://docs.ray.io/en/latest/rllib-env.html#external-agents-and-applications)
  - [External Application Clients](https://docs.ray.io/en/latest/rllib-env.html#external-application-clients)
- [Advanced Integrations](https://docs.ray.io/en/latest/rllib-env.html#advanced-integrations)

## Models, Preprocessors, and Action Distributions[¶](https://docs.ray.io/en/latest/rllib-toc.html#models-preprocessors-and-action-distributions)

- [RLlib Models, Preprocessors, and Action Distributions Overview](https://docs.ray.io/en/latest/rllib-models.html)
- [TensorFlow Models](https://docs.ray.io/en/latest/rllib-models.html#tensorflow-models)
- [PyTorch Models](https://docs.ray.io/en/latest/rllib-models.html#pytorch-models)
- [Custom Preprocessors](https://docs.ray.io/en/latest/rllib-models.html#custom-preprocessors)
- [Custom Action Distributions](https://docs.ray.io/en/latest/rllib-models.html#custom-action-distributions)
- [Supervised Model Losses](https://docs.ray.io/en/latest/rllib-models.html#supervised-model-losses)
- [Self-Supervised Model Losses](https://docs.ray.io/en/latest/rllib-models.html#self-supervised-model-losses)
- [Variable-length / Complex Observation Spaces](https://docs.ray.io/en/latest/rllib-models.html#variable-length-complex-observation-spaces)
- [Variable-length / Parametric Action Spaces](https://docs.ray.io/en/latest/rllib-models.html#variable-length-parametric-action-spaces)
- [Autoregressive Action Distributions](https://docs.ray.io/en/latest/rllib-models.html#autoregressive-action-distributions)

## Algorithms[¶](https://docs.ray.io/en/latest/rllib-toc.html#algorithms)

- High-throughput architectures
  - [Distributed Prioritized Experience Replay (Ape-X)](https://docs.ray.io/en/latest/rllib-algorithms.html#apex)
  - [Importance Weighted Actor-Learner Architecture (IMPALA)](https://docs.ray.io/en/latest/rllib-algorithms.html#impala)
  -   [Asynchronous Proximal Policy Optimization (APPO)](https://docs.ray.io/en/latest/rllib-algorithms.html#appo)
  -  [Decentralized Distributed Proximal Policy Optimization (DD-PPO)](https://docs.ray.io/en/latest/rllib-algorithms.html#ddppo)
- Gradient-based
  -  [Advantage Actor-Critic (A2C, A3C)](https://docs.ray.io/en/latest/rllib-algorithms.html#a3c)
  -  [Deep Deterministic Policy Gradients (DDPG, TD3)](https://docs.ray.io/en/latest/rllib-algorithms.html#ddpg)
  -  [Deep Q Networks (DQN, Rainbow, Parametric DQN)](https://docs.ray.io/en/latest/rllib-algorithms.html#dqn)
  -  [Policy Gradients](https://docs.ray.io/en/latest/rllib-algorithms.html#pg)
  -  [Proximal Policy Optimization (PPO)](https://docs.ray.io/en/latest/rllib-algorithms.html#ppo)
  -   [Soft Actor Critic (SAC)](https://docs.ray.io/en/latest/rllib-algorithms.html#sac)
  -  [Slate Q-Learning (SlateQ)](https://docs.ray.io/en/latest/rllib-algorithms.html#slateq)
- Derivative-free
  -   [Augmented Random Search (ARS)](https://docs.ray.io/en/latest/rllib-algorithms.html#ars)
  -   [Evolution Strategies](https://docs.ray.io/en/latest/rllib-algorithms.html#es)
- Model-based / Meta-learning / Offline
  -  [Single-Player AlphaZero (contrib/AlphaZero)](https://docs.ray.io/en/latest/rllib-algorithms.html#alphazero)
  -   [Model-Agnostic Meta-Learning (MAML)](https://docs.ray.io/en/latest/rllib-algorithms.html#maml)
  -  [Model-Based Meta-Policy-Optimization (MBMPO)](https://docs.ray.io/en/latest/rllib-algorithms.html#mbmpo)
  -  [Dreamer (DREAMER)](https://docs.ray.io/en/latest/rllib-algorithms.html#dreamer)
  -  [Conservative Q-Learning (CQL)](https://docs.ray.io/en/latest/rllib-algorithms.html#cql)
- Multi-agent
  -  [QMIX Monotonic Value Factorisation (QMIX, VDN, IQN)](https://docs.ray.io/en/latest/rllib-algorithms.html#qmix)
  -  [Multi-Agent Deep Deterministic Policy Gradient (contrib/MADDPG)](https://docs.ray.io/en/latest/rllib-algorithms.html#maddpg)
- Offline
  -   [Advantage Re-Weighted Imitation Learning (MARWIL)](https://docs.ray.io/en/latest/rllib-algorithms.html#marwil)
- Contextual bandits
  -  [Linear Upper Confidence Bound (contrib/LinUCB)](https://docs.ray.io/en/latest/rllib-algorithms.html#linucb)
  -  [Linear Thompson Sampling (contrib/LinTS)](https://docs.ray.io/en/latest/rllib-algorithms.html#lints)
- Exploration-based plug-ins (can be combined with any algo)
  -  [Curiosity (ICM: Intrinsic Curiosity Module)](https://docs.ray.io/en/latest/rllib-algorithms.html#curiosity)

## Sample Collection[¶](https://docs.ray.io/en/latest/rllib-toc.html#sample-collection)

- [The SampleCollector Class is Used to Store and Retrieve Temporary Data](https://docs.ray.io/en/latest/rllib-sample-collection.html#the-samplecollector-class-is-used-to-store-and-retrieve-temporary-data)
- [Trajectory View API](https://docs.ray.io/en/latest/rllib-sample-collection.html#trajectory-view-api)

## Offline Datasets[¶](https://docs.ray.io/en/latest/rllib-toc.html#offline-datasets)

- [Working with Offline Datasets](https://docs.ray.io/en/latest/rllib-offline.html)
- [Input Pipeline for Supervised Losses](https://docs.ray.io/en/latest/rllib-offline.html#input-pipeline-for-supervised-losses)
- [Input API](https://docs.ray.io/en/latest/rllib-offline.html#input-api)
- [Output API](https://docs.ray.io/en/latest/rllib-offline.html#output-api)

## Concepts and Custom Algorithms[¶](https://docs.ray.io/en/latest/rllib-toc.html#concepts-and-custom-algorithms)

- [Policies](https://docs.ray.io/en/latest/rllib-concepts.html)
  - [Policies in Multi-Agent](https://docs.ray.io/en/latest/rllib-concepts.html#policies-in-multi-agent)
  - [Building Policies in TensorFlow](https://docs.ray.io/en/latest/rllib-concepts.html#building-policies-in-tensorflow)
  - [Building Policies in TensorFlow Eager](https://docs.ray.io/en/latest/rllib-concepts.html#building-policies-in-tensorflow-eager)
  - [Building Policies in PyTorch](https://docs.ray.io/en/latest/rllib-concepts.html#building-policies-in-pytorch)
  - [Extending Existing Policies](https://docs.ray.io/en/latest/rllib-concepts.html#extending-existing-policies)
- [Policy Evaluation](https://docs.ray.io/en/latest/rllib-concepts.html#policy-evaluation)
- [Execution Plans](https://docs.ray.io/en/latest/rllib-concepts.html#execution-plans)
- [Trainers](https://docs.ray.io/en/latest/rllib-concepts.html#trainers)

## Examples[¶](https://docs.ray.io/en/latest/rllib-toc.html#examples)

- [Tuned Examples](https://docs.ray.io/en/latest/rllib-examples.html#tuned-examples)
- [Training Workflows](https://docs.ray.io/en/latest/rllib-examples.html#training-workflows)
- [Custom Envs and Models](https://docs.ray.io/en/latest/rllib-examples.html#custom-envs-and-models)
- [Serving and Offline](https://docs.ray.io/en/latest/rllib-examples.html#serving-and-offline)
- [Multi-Agent and Hierarchical](https://docs.ray.io/en/latest/rllib-examples.html#multi-agent-and-hierarchical)
- [Community Examples](https://docs.ray.io/en/latest/rllib-examples.html#community-examples)

## Development[¶](https://docs.ray.io/en/latest/rllib-toc.html#development)

- [Development Install](https://docs.ray.io/en/latest/rllib-dev.html#development-install)
- [API Stability](https://docs.ray.io/en/latest/rllib-dev.html#api-stability)
- [Features](https://docs.ray.io/en/latest/rllib-dev.html#feature-development)
- [Benchmarks](https://docs.ray.io/en/latest/rllib-dev.html#benchmarks)
- [Contributing Algorithms](https://docs.ray.io/en/latest/rllib-dev.html#contributing-algorithms)

## Package Reference[¶](https://docs.ray.io/en/latest/rllib-toc.html#package-reference)

- [ray.rllib.agents](https://docs.ray.io/en/latest/rllib-package-ref.html#module-ray.rllib.agents)
- [ray.rllib.env](https://docs.ray.io/en/latest/rllib-package-ref.html#module-ray.rllib.env)
- [ray.rllib.evaluation](https://docs.ray.io/en/latest/rllib-package-ref.html#module-ray.rllib.evaluation)
- [ray.rllib.execution](https://docs.ray.io/en/latest/rllib-package-ref.html#module-ray.rllib.execution)
- [ray.rllib.models](https://docs.ray.io/en/latest/rllib-package-ref.html#module-ray.rllib.models)
- [ray.rllib.utils](https://docs.ray.io/en/latest/rllib-package-ref.html#module-ray.rllib.utils)

## Troubleshooting[¶](https://docs.ray.io/en/latest/rllib-toc.html#troubleshooting)

`blas_thread_init: pthread_create` 오류가 발생하는 경우엔 일시적으로 많은 `Worker` 가 리소스를 사용할 수 없는 경우입니다. `OMP_NUM_THREADS=1` 로 설정해주세요. 또한 `ulimit - a` 를 통해 시스템 리소스 제한이 있는지 확인할 수 있습니다.

병렬 콜스택 기능으로 `ray stack` 이 있습니다. 현재 노드에 있는 `Worker` 들의 스택 추적이 가능하며 `ray timeline` 은 시각화를 맡고있고 `ray memory` 는 클러스터의 모든 객체 참조들을 리스트업할 수 있습니다.

### TensorFlow 2.0[¶](https://docs.ray.io/en/latest/rllib-toc.html#tensorflow-2-0)

`RLlib`는 `tf2.x` 모드와 `tf.compat.v1` 모드를 모두 지원합니다. `Tensor Flow`을 가져오려면 항상 `ray.rllib.utils.framework.try_import_tf()` 유틸리티 함수를 사용하세요. 이는 세 가지 값을 반환합니다. `'tf1: tf.compat.v1` 모듈 또는 설치된 `tf1.x` 패키지(버전이 < 2.0인 경우). `'tf: 설치된 Tensor Flow 모듈 그대로. tfv: 값이 1 또는 2인 편의 버전` int입니다.