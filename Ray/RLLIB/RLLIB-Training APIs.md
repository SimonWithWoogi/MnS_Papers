# RLlib Training APIs[¶](https://docs.ray.io/en/latest/rllib-training.html#rllib-training-apis)

## Getting Started[¶](https://docs.ray.io/en/latest/rllib-training.html#getting-started)

인간이 이해하기 쉬운 수준에서,`RLlib` 는 환경의 상화작용들에 대한 정책을 잡아주는 `Trainer class` 를 제공합니다. `Trainer` 를 통해서 `policy` 를 훈련하고 체크포인트를 찍거나 `action` 을 계산할 수 있습니다. `multi agnet` 학습에서는 `trainer` 가 여러 `Policy` 의 쿼리, 최적화를 한 번에 관리합니다.

![_images/rllib-api.svg](https://docs.ray.io/en/latest/_images/rllib-api.svg)

간단한 `DQN(Depp Q Learning)` 에 대한 예제 소스입니다.

```python
rllib train --run DQN --env CartPole-v0  # --config '{"framework": "tf2", "eager_tracing": True}' for eager execution
```

기본적으로, 훈련 결과는 `subdirectory(~/ray_results)` 에 로그로서 남습니다. `subdirectory` 는 하이퍼파라미터가 담겨져 있는 `params.json` 파일을 포함하고 에피소드에 대한 학습의 전반적인 요약인 `result.json` 을 포함합니다. 긑으로, `TensorBoard` 에서 시각화할 수 있는 `Tensorboard` v파일도 있습니다.

```python
tensorboard --logdir=~/ray_results
```

The `rllib train` command (same as the `train.py` script in the repo) has a number of options you can show by running:

```python
rllib train --help
-or-
python ray/rllib/train.py --help
```

가장 중요한 것은 `--env`(사용자가 등록하고 쓸 수 있는 OpenAI gym 환경)을 이용할 수 있고 알고리즘 선택 시 `SAC` `PPO` `PG`` A2C` `A3C` `IMPALA` `ES` `DDPG` `DQN` `MARW` 등을 `--run` 에서 이용할 수 있습니다.

### Evaluating Trained Policies[¶](https://docs.ray.io/en/latest/rllib-training.html#evaluating-trained-policies)

In order to save checkpoints from which to evaluate policies, set `--checkpoint-freq` (number of training iterations between checkpoints) when running `rllib train`.

An example of evaluating a previously trained DQN policy is as follows:

```
rllib rollout \
    ~/ray_results/default/DQN_CartPole-v0_0upjmdgr0/checkpoint_1/checkpoint-1 \
    --run DQN --env CartPole-v0 --steps 10000
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

The `rollout.py` helper script reconstructs a DQN policy from the checkpoint located at `~/ray_results/default/DQN_CartPole-v0_0upjmdgr0/checkpoint_1/checkpoint-1` and renders its behavior in the environment specified by `--env`.

(Type `rllib rollout --help` to see the available evaluation options.)

For more advanced evaluation functionality, refer to [Customized Evaluation During Training](https://docs.ray.io/en/latest/rllib-training.html#customized-evaluation-during-training).

## Configuration[¶](https://docs.ray.io/en/latest/rllib-training.html#configuration)

### Specifying Parameters[¶](https://docs.ray.io/en/latest/rllib-training.html#specifying-parameters)

Each algorithm has specific hyperparameters that can be set with `--config`, in addition to a number of [common hyperparameters](https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py). See the [algorithms documentation](https://docs.ray.io/en/latest/rllib-algorithms.html) for more information.

In an example below, we train A2C by specifying 8 workers through the config flag.

```
rllib train --env=PongDeterministic-v4 --run=A2C --config '{"num_workers": 8}'
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Specifying Resources[¶](https://docs.ray.io/en/latest/rllib-training.html#specifying-resources)

You can control the degree of parallelism used by setting the `num_workers` hyperparameter for most algorithms. The Trainer will construct that many “remote worker” instances ([see RolloutWorker class](https://github.com/ray-project/ray/blob/master/rllib/evaluation/rollout_worker.py)) that are constructed as ray.remote actors, plus exactly one “local worker”, a `RolloutWorker` object that is not a ray actor, but lives directly inside the Trainer. For most algorithms, learning updates are performed on the local worker and sample collection from one or more environments is performed by the remote workers (in parallel). For example, setting `num_workers=0` will only create the local worker, in which case both sample collection and training will be done by the local worker. On the other hand, setting `num_workers=5` will create the local worker (responsible for training updates) and 5 remote workers (responsible for sample collection).

Since learning is most of the time done on the local worker, it may help to provide one or more GPUs to that worker via the `num_gpus` setting. Similarly, the resource allocation to remote workers can be controlled via `num_cpus_per_worker`, `num_gpus_per_worker`, and `custom_resources_per_worker`.

The number of GPUs can be fractional quantities (e.g. 0.5) to allocate only a fraction of a GPU. For example, with DQN you can pack five trainers onto one GPU by setting `num_gpus: 0.2`. Check out [this fractional GPU example here](https://github.com/ray-project/ray/blob/master/rllib/examples/fractional_gpus.py) as well that also demonstrates how environments (running on the remote workers) that require a GPU can benefit from the `num_gpus_per_worker` setting.

For synchronous algorithms like PPO and A2C, the driver and workers can make use of the same GPU. To do this for an amount of `n` GPUS:

```
gpu_count = n
num_gpus = 0.0001 # Driver GPU
num_gpus_per_worker = (gpu_count - num_gpus) / num_workers
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

![_images/rllib-config.svg](https://docs.ray.io/en/latest/_images/rllib-config.svg)

If you specify `num_gpus` and your machine does not have the required number of GPUs available, a RuntimeError will be thrown by the respective worker. On the other hand, if you set `num_gpus=0`, your policies will be built solely on the CPU, even if GPUs are available on the machine.

### Scaling Guide[¶](https://docs.ray.io/en/latest/rllib-training.html#scaling-guide)

Here are some rules of thumb for scaling training with RLlib.

1. If the environment is slow and cannot be replicated (e.g., since it requires interaction with physical systems), then you should use a sample-efficient off-policy algorithm such as [DQN](https://docs.ray.io/en/latest/rllib-algorithms.html#dqn) or [SAC](https://docs.ray.io/en/latest/rllib-algorithms.html#sac). These algorithms default to `num_workers: 0` for single-process operation. Make sure to set `num_gpus: 1` if you want to use a GPU. Consider also batch RL training with the [offline data](https://docs.ray.io/en/latest/rllib-offline.html) API.
2. If the environment is fast and the model is small (most models for RL are), use time-efficient algorithms such as [PPO](https://docs.ray.io/en/latest/rllib-algorithms.html#ppo), [IMPALA](https://docs.ray.io/en/latest/rllib-algorithms.html#impala), or [APEX](https://docs.ray.io/en/latest/rllib-algorithms.html#apex). These can be scaled by increasing `num_workers` to add rollout workers. It may also make sense to enable [vectorization](https://docs.ray.io/en/latest/rllib-env.html#vectorized) for inference. Make sure to set `num_gpus: 1` if you want to use a GPU. If the learner becomes a bottleneck, multiple GPUs can be used for learning by setting `num_gpus > 1`.
3. If the model is compute intensive (e.g., a large deep residual network) and inference is the bottleneck, consider allocating GPUs to workers by setting `num_gpus_per_worker: 1`. If you only have a single GPU, consider `num_workers: 0` to use the learner GPU for inference. For efficient use of GPU time, use a small number of GPU workers and a large number of [envs per worker](https://docs.ray.io/en/latest/rllib-env.html#vectorized).
4. Finally, if both model and environment are compute intensive, then enable [remote worker envs](https://docs.ray.io/en/latest/rllib-env.html#vectorized) with [async batching](https://docs.ray.io/en/latest/rllib-env.html#vectorized) by setting `remote_worker_envs: True` and optionally `remote_env_batch_wait_ms`. This batches inference on GPUs in the rollout workers while letting envs run asynchronously in separate actors, similar to the [SEED](https://ai.googleblog.com/2020/03/massively-scaling-reinforcement.html) architecture. The number of workers and number of envs per worker should be tuned to maximize GPU utilization. If your env requires GPUs to function, or if multi-node SGD is needed, then also consider [DD-PPO](https://docs.ray.io/en/latest/rllib-algorithms.html#ddppo).

### Common Parameters[¶](https://docs.ray.io/en/latest/rllib-training.html#common-parameters)

The following is a list of the common algorithm hyperparameters:

```
COMMON_CONFIG: TrainerConfigDict = {
    # === Settings for Rollout Worker processes ===
    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    "num_workers": 2,
    # Number of environments to evaluate vector-wise per worker. This enables
    # model inference batching, which can improve performance for inference
    # bottlenecked workloads.
    "num_envs_per_worker": 1,
    # When `num_workers` > 0, the driver (local_worker; worker-idx=0) does not
    # need an environment. This is because it doesn't have to sample (done by
    # remote_workers; worker_indices > 0) nor evaluate (done by evaluation
    # workers; see below).
    "create_env_on_driver": False,
    # Divide episodes into fragments of this many steps each during rollouts.
    # Sample batches of this size are collected from rollout workers and
    # combined into a larger batch of `train_batch_size` for learning.
    #
    # For example, given rollout_fragment_length=100 and train_batch_size=1000:
    #   1. RLlib collects 10 fragments of 100 steps each from rollout workers.
    #   2. These fragments are concatenated and we perform an epoch of SGD.
    #
    # When using multiple envs per worker, the fragment size is multiplied by
    # `num_envs_per_worker`. This is since we are collecting steps from
    # multiple envs in parallel. For example, if num_envs_per_worker=5, then
    # rollout workers will return experiences in chunks of 5*100 = 500 steps.
    #
    # The dataflow here can vary per algorithm. For example, PPO further
    # divides the train batch into minibatches for multi-epoch SGD.
    "rollout_fragment_length": 200,
    # How to build per-Sampler (RolloutWorker) batches, which are then
    # usually concat'd to form the train batch. Note that "steps" below can
    # mean different things (either env- or agent-steps) and depends on the
    # `count_steps_by` (multiagent) setting below.
    # truncate_episodes: Each produced batch (when calling
    #   RolloutWorker.sample()) will contain exactly `rollout_fragment_length`
    #   steps. This mode guarantees evenly sized batches, but increases
    #   variance as the future return must now be estimated at truncation
    #   boundaries.
    # complete_episodes: Each unroll happens exactly over one episode, from
    #   beginning to end. Data collection will not stop unless the episode
    #   terminates or a configured horizon (hard or soft) is hit.
    "batch_mode": "truncate_episodes",

    # === Settings for the Trainer process ===
    # Discount factor of the MDP.
    "gamma": 0.99,
    # The default learning rate.
    "lr": 0.0001,
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    "train_batch_size": 200,
    # Arguments to pass to the policy model. See models/catalog.py for a full
    # list of the available model options.
    "model": MODEL_DEFAULTS,
    # Arguments to pass to the policy optimizer. These vary by optimizer.
    "optimizer": {},

    # === Environment Settings ===
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": None,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": False,
    # Don't set 'done' at the end of the episode.
    # In combination with `soft_horizon`, this works as follows:
    # - no_done_at_end=False soft_horizon=False:
    #   Reset env and add `done=True` at end of each episode.
    # - no_done_at_end=True soft_horizon=False:
    #   Reset env, but do NOT add `done=True` at end of the episode.
    # - no_done_at_end=False soft_horizon=True:
    #   Do NOT reset env at horizon, but add `done=True` at the horizon
    #   (pretending the episode has terminated).
    # - no_done_at_end=True soft_horizon=True:
    #   Do NOT reset env at horizon and do NOT add `done=True` at the horizon.
    "no_done_at_end": False,
    # The environment specifier:
    # This can either be a tune-registered env, via
    # `tune.register_env([name], lambda env_ctx: [env object])`,
    # or a string specifier of an RLlib supported type. In the latter case,
    # RLlib will try to interpret the specifier as either an openAI gym env,
    # a PyBullet env, a ViZDoomGym env, or a fully qualified classpath to an
    # Env class, e.g. "ray.rllib.examples.env.random_env.RandomEnv".
    "env": None,
    # The observation- and action spaces for the Policies of this Trainer.
    # Use None for automatically inferring these from the given env.
    "observation_space": None,
    "action_space": None,
    # Arguments dict passed to the env creator as an EnvContext object (which
    # is a dict plus the properties: num_workers, worker_index, vector_index,
    # and remote).
    "env_config": {},
    # A callable taking the last train results, the base env and the env
    # context as args and returning a new task to set the env to.
    # The env must be a `TaskSettableEnv` sub-class for this to work.
    # See `examples/curriculum_learning.py` for an example.
    "env_task_fn": None,
    # If True, try to render the environment on the local worker or on worker
    # 1 (if num_workers > 0). For vectorized envs, this usually means that only
    # the first sub-environment will be rendered.
    # In order for this to work, your env will have to implement the
    # `render()` method which either:
    # a) handles window generation and rendering itself (returning True) or
    # b) returns a numpy uint8 image of shape [height x width x 3 (RGB)].
    "render_env": False,
    # If True, stores videos in this relative directory inside the default
    # output dir (~/ray_results/...). Alternatively, you can specify an
    # absolute path (str), in which the env recordings should be
    # stored instead.
    # Set to False for not recording anything.
    # Note: This setting replaces the deprecated `monitor` key.
    "record_env": False,
    # Whether to clip rewards during Policy's postprocessing.
    # None (default): Clip for Atari only (r=sign(r)).
    # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    # False: Never clip.
    # [float value]: Clip at -value and + value.
    # Tuple[value1, value2]: Clip at value1 and value2.
    "clip_rewards": None,
    # If True, RLlib will learn entirely inside a normalized action space
    # (0.0 centered with small stddev; only affecting Box components) and
    # only unsquash actions (and clip just in case) to the bounds of
    # env's action space before sending actions back to the env.
    "normalize_actions": True,
    # If True, RLlib will clip actions according to the env's bounds
    # before sending them back to the env.
    # TODO: (sven) This option should be obsoleted and always be False.
    "clip_actions": False,
    # Whether to use "rllib" or "deepmind" preprocessors by default
    "preprocessor_pref": "deepmind",

    # === Debug Settings ===
    # Set the ray.rllib.* log level for the agent process and its workers.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level). When using the
    # `rllib train` command, you can also use the `-v` and `-vv` flags as
    # shorthand for INFO and DEBUG.
    "log_level": "WARN",
    # Callbacks that will be run during various phases of training. See the
    # `DefaultCallbacks` class and `examples/custom_metrics_and_callbacks.py`
    # for more usage information.
    "callbacks": DefaultCallbacks,
    # Whether to attempt to continue training if a worker crashes. The number
    # of currently healthy workers is reported as the "num_healthy_workers"
    # metric.
    "ignore_worker_failures": False,
    # Log system resource metrics to results. This requires `psutil` to be
    # installed for sys stats, and `gputil` for GPU metrics.
    "log_sys_usage": True,
    # Use fake (infinite speed) sampler. For testing only.
    "fake_sampler": False,

    # === Deep Learning Framework Settings ===
    # tf: TensorFlow (static-graph)
    # tf2: TensorFlow 2.x (eager)
    # tfe: TensorFlow eager
    # torch: PyTorch
    "framework": "tf",
    # Enable tracing in eager mode. This greatly improves performance, but
    # makes it slightly harder to debug since Python code won't be evaluated
    # after the initial eager pass. Only possible if framework=tfe.
    "eager_tracing": False,

    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    "explore": True,
    # Provide a dict specifying the Exploration object's config.
    "exploration_config": {
        # The Exploration class to use. In the simplest case, this is the name
        # (str) of any class present in the `rllib.utils.exploration` package.
        # You can also provide the python class directly or the full location
        # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
        # EpsilonGreedy").
        "type": "StochasticSampling",
        # Add constructor kwargs here (if any).
    },
    # === Evaluation Settings ===
    # Evaluate with every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period. If using multiple
    # evaluation workers, we will run at least this many episodes total.
    "evaluation_num_episodes": 10,
    # Whether to run evaluation in parallel to a Trainer.train() call
    # using threading. Default=False.
    # E.g. evaluation_interval=2 -> For every other training iteration,
    # the Trainer.train() and Trainer.evaluate() calls run in parallel.
    # Note: This is experimental. Possible pitfalls could be race conditions
    # for weight synching at the beginning of the evaluation loop.
    "evaluation_parallel_to_training": False,
    # Internal flag that is set to True for evaluation workers.
    "in_evaluation": False,
    # Typical usage is to pass extra args to evaluation env creator
    # and to disable exploration by computing deterministic actions.
    # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
    # policy, even if this is a stochastic one. Setting "explore=False" here
    # will result in the evaluation workers not using this optimal policy!
    "evaluation_config": {
        # Example: overriding env_config, exploration, etc:
        # "env_config": {...},
        # "explore": False
    },
    # Number of parallel workers to use for evaluation. Note that this is set
    # to zero by default, which means evaluation will be run in the trainer
    # process (only if evaluation_interval is not None). If you increase this,
    # it will increase the Ray resource usage of the trainer since evaluation
    # workers are created separately from rollout workers (used to sample data
    # for training).
    "evaluation_num_workers": 0,
    # Customize the evaluation method. This must be a function of signature
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # Trainer.evaluate() method to see the default implementation. The
    # trainer guarantees all eval workers have the latest policy state before
    # this function is called.
    "custom_eval_function": None,

    # === Advanced Rollout Settings ===
    # Use a background thread for sampling (slightly off-policy, usually not
    # advisable to turn on unless your env specifically requires it).
    "sample_async": False,

    # The SampleCollector class to be used to collect and retrieve
    # environment-, model-, and sampler data. Override the SampleCollector base
    # class to implement your own collection/buffering/retrieval logic.
    "sample_collector": SimpleListCollector,

    # Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    "observation_filter": "NoFilter",
    # Whether to synchronize the statistics of remote filters.
    "synchronize_filters": True,
    # Configures TF for single-process operation by default.
    "tf_session_args": {
        # note: overridden by `local_tf_session_args`
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {
            "allow_growth": True,
        },
        "log_device_placement": False,
        "device_count": {
            "CPU": 1
        },
        "allow_soft_placement": True,  # required by PPO multi-gpu
    },
    # Override the following tf session args on the local worker
    "local_tf_session_args": {
        # Allow a higher level of parallelism by default, but not unlimited
        # since that can cause crashes with many concurrent drivers.
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
    # Whether to LZ4 compress individual observations
    "compress_observations": False,
    # Wait for metric batches for at most this many seconds. Those that
    # have not returned in time will be collected in the next train iteration.
    "collect_metrics_timeout": 180,
    # Smooth metrics over this many episodes.
    "metrics_smoothing_episodes": 100,
    # If using num_envs_per_worker > 1, whether to create those new envs in
    # remote processes instead of in the same worker. This adds overheads, but
    # can make sense if your envs can take much time to step / reset
    # (e.g., for StarCraft). Use this cautiously; overheads are significant.
    "remote_worker_envs": False,
    # Timeout that remote workers are waiting when polling environments.
    # 0 (continue when at least one env is ready) is a reasonable default,
    # but optimal value could be obtained by measuring your environment
    # step / reset and model inference perf.
    "remote_env_batch_wait_ms": 0,
    # Minimum time per train iteration (frequency of metrics reporting).
    "min_iter_time_s": 0,
    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of train iterations.
    "timesteps_per_iteration": 0,
    # This argument, in conjunction with worker_index, sets the random seed of
    # each worker, so that identically configured trials will have identical
    # results. This makes experiments reproducible.
    "seed": None,
    # Any extra python env vars to set in the trainer process, e.g.,
    # {"OMP_NUM_THREADS": "16"}
    "extra_python_environs_for_driver": {},
    # The extra python environments need to set for worker processes.
    "extra_python_environs_for_worker": {},

    # === Resource Settings ===
    # Number of GPUs to allocate to the trainer process. Note that not all
    # algorithms can take advantage of trainer GPUs. Support for multi-GPU
    # is currently only available for tf-[PPO/IMPALA/DQN/PG].
    # This can be fractional (e.g., 0.3 GPUs).
    "num_gpus": 0,
    # Set to True for debugging (multi-)?GPU funcitonality on a CPU machine.
    # GPU towers will be simulated by graphs located on CPUs in this case.
    # Use `num_gpus` to test for different numbers of fake GPUs.
    "_fake_gpus": False,
    # Number of CPUs to allocate per worker.
    "num_cpus_per_worker": 1,
    # Number of GPUs to allocate per worker. This can be fractional. This is
    # usually needed only if your env itself requires a GPU (i.e., it is a
    # GPU-intensive video game), or model inference is unusually expensive.
    "num_gpus_per_worker": 0,
    # Any custom Ray resources to allocate per worker.
    "custom_resources_per_worker": {},
    # Number of CPUs to allocate for the trainer. Note: this only takes effect
    # when running in Tune. Otherwise, the trainer runs in the main program.
    "num_cpus_for_driver": 1,
    # The strategy for the placement group factory returned by
    # `Trainer.default_resource_request()`. A PlacementGroup defines, which
    # devices (resources) should always be co-located on the same node.
    # For example, a Trainer with 2 rollout workers, running with
    # num_gpus=1 will request a placement group with the bundles:
    # [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], where the first bundle is
    # for the driver and the other 2 bundles are for the two workers.
    # These bundles can now be "placed" on the same or different
    # nodes depending on the value of `placement_strategy`:
    # "PACK": Packs bundles into as few nodes as possible.
    # "SPREAD": Places bundles across distinct nodes as even as possible.
    # "STRICT_PACK": Packs bundles into one node. The group is not allowed
    #   to span multiple nodes.
    # "STRICT_SPREAD": Packs bundles across distinct nodes.
    "placement_strategy": "PACK",

    # === Offline Datasets ===
    # Specify how to generate experiences:
    #  - "sampler": Generate experiences via online (env) simulation (default).
    #  - A local directory or file glob expression (e.g., "/tmp/*.json").
    #  - A list of individual file paths/URIs (e.g., ["/tmp/1.json",
    #    "s3://bucket/2.json"]).
    #  - A dict with string keys and sampling probabilities as values (e.g.,
    #    {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2}).
    #  - A callable that returns a ray.rllib.offline.InputReader.
    #  - A string key that indexes a callable with tune.registry.register_input
    "input": "sampler",
    # Arguments accessible from the IOContext for configuring custom input
    "input_config": {},
    # True, if the actions in a given offline "input" are already normalized
    # (between -1.0 and 1.0). This is usually the case when the offline
    # file has been generated by another RLlib algorithm (e.g. PPO or SAC),
    # while "normalize_actions" was set to True.
    "actions_in_input_normalized": False,
    # Specify how to evaluate the current policy. This only has an effect when
    # reading offline experiences ("input" is not "sampler").
    # Available options:
    #  - "wis": the weighted step-wise importance sampling estimator.
    #  - "is": the step-wise importance sampling estimator.
    #  - "simulation": run the environment in the background, but use
    #    this data for evaluation only and not for learning.
    "input_evaluation": ["is", "wis"],
    # Whether to run postprocess_trajectory() on the trajectory fragments from
    # offline inputs. Note that postprocessing will be done using the *current*
    # policy, not the *behavior* policy, which is typically undesirable for
    # on-policy algorithms.
    "postprocess_inputs": False,
    # If positive, input batches will be shuffled via a sliding window buffer
    # of this number of batches. Use this if the input data is not in random
    # enough order. Input is delayed until the shuffle buffer is filled.
    "shuffle_buffer_size": 0,
    # Specify where experiences should be saved:
    #  - None: don't save any experiences
    #  - "logdir" to save to the agent log dir
    #  - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
    #  - a function that returns a rllib.offline.OutputWriter
    "output": None,
    # What sample batch columns to LZ4 compress in the output data.
    "output_compress_columns": ["obs", "new_obs"],
    # Max output file size before rolling over to a new file.
    "output_max_file_size": 64 * 1024 * 1024,

    # === Settings for Multi-Agent Environments ===
    "multiagent": {
        # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
        # of (policy_cls, obs_space, act_space, config). This defines the
        # observation and action spaces of the policies and any extra config.
        "policies": {},
        # Keep this many policies in the "policy_map" (before writing
        # least-recently used ones to disk/S3).
        "policy_map_capacity": 100,
        # Where to store overflowing (least-recently used) policies?
        # Could be a directory (str) or an S3 location. None for using
        # the default output dir.
        "policy_map_cache": None,
        # Function mapping agent ids to policy ids.
        "policy_mapping_fn": None,
        # Optional list of policies to train, or None for all policies.
        "policies_to_train": None,
        # Optional function that can be used to enhance the local agent
        # observations to include more state.
        # See rllib/evaluation/observation_function.py for more info.
        "observation_fn": None,
        # When replay_mode=lockstep, RLlib will replay all the agent
        # transitions at a particular timestep together in a batch. This allows
        # the policy to implement differentiable shared computations between
        # agents it controls at that timestep. When replay_mode=independent,
        # transitions are replayed independently per policy.
        "replay_mode": "independent",
        # Which metric to use as the "batch size" when building a
        # MultiAgentBatch. The two supported values are:
        # env_steps: Count each time the env is "stepped" (no matter how many
        #   multi-agent actions are passed/how many multi-agent observations
        #   have been returned in the previous step).
        # agent_steps: Count each individual agent step as one step.
        "count_steps_by": "env_steps",
    },

    # === Logger ===
    # Define logger-specific configuration to be used inside Logger
    # Default value None allows overwriting with nested dicts
    "logger_config": None,

    # === Deprecated keys ===
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    # This will be set automatically from now on.
    "simple_optimizer": DEPRECATED_VALUE,
    # Whether to write episode stats and videos to the agent log dir. This is
    # typically located in ~/ray_results.
    "monitor": DEPRECATED_VALUE,
}
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Tuned Examples[¶](https://docs.ray.io/en/latest/rllib-training.html#tuned-examples)

Some good hyperparameters and settings are available in [the repository](https://github.com/ray-project/ray/blob/master/rllib/tuned_examples) (some of them are tuned to run on GPUs). If you find better settings or tune an algorithm on a different domain, consider submitting a Pull Request!

You can run these with the `rllib train` command as follows:

```
rllib train -f /path/to/tuned/example.yaml
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

## Basic Python API[¶](https://docs.ray.io/en/latest/rllib-training.html#basic-python-api)

The Python API provides the needed flexibility for applying RLlib to new problems. You will need to use this API if you wish to use [custom environments, preprocessors, or models](https://docs.ray.io/en/latest/rllib-models.html) with RLlib.

Here is an example of the basic usage (for a more complete example, see [custom_env.py](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py)):

```
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Trainer's Policy's ModelV2
# (tf or torch) by doing:
trainer.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch trainers/models.
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Note

It’s recommended that you run RLlib trainers with [Tune](https://docs.ray.io/en/latest/tune/index.html), for easy experiment management and visualization of results. Just set `"run": ALG_NAME, "env": ENV_NAME` in the experiment config.

All RLlib trainers are compatible with the [Tune API](https://docs.ray.io/en/latest/tune/key-concepts.html#tune-60-seconds). This enables them to be easily used in experiments with [Tune](https://docs.ray.io/en/latest/tune/index.html). For example, the following code performs a simple hyperparam sweep of PPO:

```
import ray
from ray import tune

ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
)
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Tune will schedule the trials to run in parallel on your Ray cluster:

```
== Status ==
Using FIFO scheduling algorithm.
Resources requested: 4/4 CPUs, 0/0 GPUs
Result logdir: ~/ray_results/my_experiment
PENDING trials:
 - PPO_CartPole-v0_2_lr=0.0001:     PENDING
RUNNING trials:
 - PPO_CartPole-v0_0_lr=0.01:       RUNNING [pid=21940], 16 s, 4013 ts, 22 rew
 - PPO_CartPole-v0_1_lr=0.001:      RUNNING [pid=21942], 27 s, 8111 ts, 54.7 rew
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

`tune.run()` returns an ExperimentAnalysis object that allows further analysis of the training results and retrieving the checkpoint(s) of the trained agent. It also simplifies saving the trained agent. For example:

```
# tune.run() allows setting a custom log directory (other than ``~/ray-results``)
# and automatically saving the trained agent
analysis = ray.tune.run(
    ppo.PPOTrainer,
    config=config,
    local_dir=log_dir,
    stop=stop_criteria,
    checkpoint_at_end=True)

# list of lists: one list per checkpoint; each checkpoint list contains
# 1st the path, 2nd the metric value
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")

# or simply get the last checkpoint (with highest "training_iteration")
last_checkpoint = analysis.get_last_checkpoint()
# if there are multiple trials, select a specific trial or automatically
# choose the best one according to a given metric
last_checkpoint = analysis.get_last_checkpoint(
    metric="episode_reward_mean", mode="max"
)
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Loading and restoring a trained agent from a checkpoint is simple:

```
agent = ppo.PPOTrainer(config=config, env=env_class)
agent.restore(checkpoint_path)
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Computing Actions[¶](https://docs.ray.io/en/latest/rllib-training.html#computing-actions)

The simplest way to programmatically compute actions from a trained agent is to use `trainer.compute_action()`. This method preprocesses and filters the observation before passing it to the agent policy. Here is a simple example of testing a trained agent for one episode:

```
# instantiate env class
env = env_class(env_config)

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

For more advanced usage, you can access the `workers` and policies held by the trainer directly as `compute_action()`does:

```
class Trainer(Trainable):

  @PublicAPI
  def compute_action(self,
                     observation,
                     state=None,
                     prev_action=None,
                     prev_reward=None,
                     info=None,
                     policy_id=DEFAULT_POLICY_ID,
                     full_fetch=False):
      """Computes an action for the specified policy.

      Note that you can also access the policy object through
      self.get_policy(policy_id) and call compute_actions() on it directly.

      Arguments:
          observation (obj): observation from the environment.
          state (list): RNN hidden state, if any. If state is not None,
                        then all of compute_single_action(...) is returned
                        (computed action, rnn state, logits dictionary).
                        Otherwise compute_single_action(...)[0] is
                        returned (computed action).
          prev_action (obj): previous action value, if any
          prev_reward (int): previous reward, if any
          info (dict): info object, if any
          policy_id (str): policy to query (only applies to multi-agent).
          full_fetch (bool): whether to return extra action fetch results.
              This is always set to true if RNN state is specified.

      Returns:
          Just the computed action if full_fetch=False, or the full output
          of policy.compute_actions() otherwise.
      """

      if state is None:
          state = []
      preprocessed = self.workers.local_worker().preprocessors[
          policy_id].transform(observation)
      filtered_obs = self.workers.local_worker().filters[policy_id](
          preprocessed, update=False)
      if state:
          return self.get_policy(policy_id).compute_single_action(
              filtered_obs,
              state,
              prev_action,
              prev_reward,
              info,
              clip_actions=self.config["clip_actions"])
      res = self.get_policy(policy_id).compute_single_action(
          filtered_obs,
          state,
          prev_action,
          prev_reward,
          info,
          clip_actions=self.config["clip_actions"])
      if full_fetch:
          return res
      else:
          return res[0]  # backwards compatibility
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Accessing Policy State[¶](https://docs.ray.io/en/latest/rllib-training.html#accessing-policy-state)

It is common to need to access a trainer’s internal state, e.g., to set or get internal weights. In RLlib trainer state is replicated across multiple *rollout workers* (Ray actors) in the cluster. However, you can easily get and update this state between calls to `train()` via `trainer.workers.foreach_worker()` or `trainer.workers.foreach_worker_with_index()`. These functions take a lambda function that is applied with the worker as an arg. You can also return values from these functions and those will be returned as a list.

You can also access just the “master” copy of the trainer state through `trainer.get_policy()` or `trainer.workers.local_worker()`, but note that updates here may not be immediately reflected in remote replicas if you have configured `num_workers > 0`. For example, to access the weights of a local TF policy, you can run `trainer.get_policy().get_weights()`. This is also equivalent to `trainer.workers.local_worker().policy_map["default_policy"].get_weights()`:

```
# Get weights of the default local policy
trainer.get_policy().get_weights()

# Same as above
trainer.workers.local_worker().policy_map["default_policy"].get_weights()

# Get list of weights of each worker, including remote replicas
trainer.workers.foreach_worker(lambda ev: ev.get_policy().get_weights())

# Same as above
trainer.workers.foreach_worker_with_index(lambda ev, i: ev.get_policy().get_weights())
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Accessing Model State[¶](https://docs.ray.io/en/latest/rllib-training.html#accessing-model-state)

Similar to accessing policy state, you may want to get a reference to the underlying neural network model being trained. For example, you may want to pre-train it separately, or otherwise update its weights outside of RLlib. This can be done by accessing the `model` of the policy:

**Example: Preprocessing observations for feeding into a model**

```
>>> import gym
>>> env = gym.make("Pong-v0")

# RLlib uses preprocessors to implement transforms such as one-hot encoding
# and flattening of tuple and dict observations.
>>> from ray.rllib.models.preprocessors import get_preprocessor
>>> prep = get_preprocessor(env.observation_space)(env.observation_space)
<ray.rllib.models.preprocessors.GenericPixelPreprocessor object at 0x7fc4d049de80>

# Observations should be preprocessed prior to feeding into a model
>>> env.reset().shape
(210, 160, 3)
>>> prep.transform(env.reset()).shape
(84, 84, 3)
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

**Example: Querying a policy’s action distribution**

```
# Get a reference to the policy
>>> from ray.rllib.agents.ppo import PPOTrainer
>>> trainer = PPOTrainer(env="CartPole-v0", config={"framework": "tf2", "num_workers": 0})
>>> policy = trainer.get_policy()
<ray.rllib.policy.eager_tf_policy.PPOTFPolicy_eager object at 0x7fd020165470>

# Run a forward pass to get model output logits. Note that complex observations
# must be preprocessed as in the above code block.
>>> logits, _ = policy.model.from_batch({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
(<tf.Tensor: id=1274, shape=(1, 2), dtype=float32, numpy=...>, [])

# Compute action distribution given logits
>>> policy.dist_class
<class_object 'ray.rllib.models.tf.tf_action_dist.Categorical'>
>>> dist = policy.dist_class(logits, policy.model)
<ray.rllib.models.tf.tf_action_dist.Categorical object at 0x7fd02301d710>

# Query the distribution for samples, sample logps
>>> dist.sample()
<tf.Tensor: id=661, shape=(1,), dtype=int64, numpy=..>
>>> dist.logp([1])
<tf.Tensor: id=1298, shape=(1,), dtype=float32, numpy=...>

# Get the estimated values for the most recent forward pass
>>> policy.model.value_function()
<tf.Tensor: id=670, shape=(1,), dtype=float32, numpy=...>

>>> policy.model.base_model.summary()
Model: "model"
_____________________________________________________________________
Layer (type)               Output Shape  Param #  Connected to
=====================================================================
observations (InputLayer)  [(None, 4)]   0
_____________________________________________________________________
fc_1 (Dense)               (None, 256)   1280     observations[0][0]
_____________________________________________________________________
fc_value_1 (Dense)         (None, 256)   1280     observations[0][0]
_____________________________________________________________________
fc_2 (Dense)               (None, 256)   65792    fc_1[0][0]
_____________________________________________________________________
fc_value_2 (Dense)         (None, 256)   65792    fc_value_1[0][0]
_____________________________________________________________________
fc_out (Dense)             (None, 2)     514      fc_2[0][0]
_____________________________________________________________________
value_out (Dense)          (None, 1)     257      fc_value_2[0][0]
=====================================================================
Total params: 134,915
Trainable params: 134,915
Non-trainable params: 0
_____________________________________________________________________
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

**Example: Getting Q values from a DQN model**

```
# Get a reference to the model through the policy
>>> from ray.rllib.agents.dqn import DQNTrainer
>>> trainer = DQNTrainer(env="CartPole-v0", config={"framework": "tf2"})
>>> model = trainer.get_policy().model
<ray.rllib.models.catalog.FullyConnectedNetwork_as_DistributionalQModel ...>

# List of all model variables
>>> model.variables()
[<tf.Variable 'default_policy/fc_1/kernel:0' shape=(4, 256) dtype=float32>, ...]

# Run a forward pass to get base model output. Note that complex observations
# must be preprocessed. An example of preprocessing is examples/saving_experiences.py
>>> model_out = model.from_batch({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
(<tf.Tensor: id=832, shape=(1, 256), dtype=float32, numpy=...)

# Access the base Keras models (all default models have a base)
>>> model.base_model.summary()
Model: "model"
_______________________________________________________________________
Layer (type)                Output Shape    Param #  Connected to
=======================================================================
observations (InputLayer)   [(None, 4)]     0
_______________________________________________________________________
fc_1 (Dense)                (None, 256)     1280     observations[0][0]
_______________________________________________________________________
fc_out (Dense)              (None, 256)     65792    fc_1[0][0]
_______________________________________________________________________
value_out (Dense)           (None, 1)       257      fc_1[0][0]
=======================================================================
Total params: 67,329
Trainable params: 67,329
Non-trainable params: 0
______________________________________________________________________________

# Access the Q value model (specific to DQN)
>>> model.get_q_value_distributions(model_out)
[<tf.Tensor: id=891, shape=(1, 2)>, <tf.Tensor: id=896, shape=(1, 2, 1)>]

>>> model.q_value_head.summary()
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
model_out (InputLayer)       [(None, 256)]             0
_________________________________________________________________
lambda (Lambda)              [(None, 2), (None, 2, 1), 66306
=================================================================
Total params: 66,306
Trainable params: 66,306
Non-trainable params: 0
_________________________________________________________________

# Access the state value model (specific to DQN)
>>> model.get_state_value(model_out)
<tf.Tensor: id=913, shape=(1, 1), dtype=float32>

>>> model.state_value_head.summary()
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
model_out (InputLayer)       [(None, 256)]             0
_________________________________________________________________
lambda_1 (Lambda)            (None, 1)                 66049
=================================================================
Total params: 66,049
Trainable params: 66,049
Non-trainable params: 0
_________________________________________________________________
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

This is especially useful when used with [custom model classes](https://docs.ray.io/en/latest/rllib-models.html).

## Advanced Python APIs[¶](https://docs.ray.io/en/latest/rllib-training.html#advanced-python-apis)

### Custom Training Workflows[¶](https://docs.ray.io/en/latest/rllib-training.html#custom-training-workflows)

In the [basic training example](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py), Tune will call `train()` on your trainer once per training iteration and report the new training results. Sometimes, it is desirable to have full control over training, but still run inside Tune. Tune supports [custom trainable functions](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#trainable-docs) that can be used to implement [custom training workflows (example)](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py).

For even finer-grained control over training, you can use RLlib’s lower-level [building blocks](https://docs.ray.io/en/latest/rllib-concepts.html) directly to implement [fully customized training workflows](https://github.com/ray-project/ray/blob/master/rllib/examples/rollout_worker_custom_workflow.py).

### Global Coordination[¶](https://docs.ray.io/en/latest/rllib-training.html#global-coordination)

Sometimes, it is necessary to coordinate between pieces of code that live in different processes managed by RLlib. For example, it can be useful to maintain a global average of a certain variable, or centrally control a hyperparameter used by policies. Ray provides a general way to achieve this through *named actors* (learn more about Ray actors [here](https://docs.ray.io/en/latest/actors.html)). These actors are assigned a global name and handles to them can be retrieved using these names. As an example, consider maintaining a shared global counter that is incremented by environments and read periodically from your driver program:

```
@ray.remote
class Counter:
   def __init__(self):
      self.count = 0
   def inc(self, n):
      self.count += n
   def get(self):
      return self.count

# on the driver
counter = Counter.options(name="global_counter").remote()
print(ray.get(counter.get.remote()))  # get the latest count

# in your envs
counter = ray.get_actor("global_counter")
counter.inc.remote(1)  # async call to increment the global count
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Ray actors provide high levels of performance, so in more complex cases they can be used implement communication patterns such as parameter servers and allreduce.

### Callbacks and Custom Metrics[¶](https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics)

You can provide callbacks to be called at points during policy evaluation. These callbacks have access to state for the current [episode](https://github.com/ray-project/ray/blob/master/rllib/evaluation/episode.py). Certain callbacks such as `on_postprocess_trajectory`, `on_sample_end`, and `on_train_result` are also places where custom postprocessing can be applied to intermediate data or results.

User-defined state can be stored for the [episode](https://github.com/ray-project/ray/blob/master/rllib/evaluation/episode.py) in the `episode.user_data` dict, and custom scalar metrics reported by saving values to the `episode.custom_metrics` dict. These custom metrics will be aggregated and reported as part of training results. For a full example, see [custom_metrics_and_callbacks.py](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py).

- *class* `ray.rllib.agents.callbacks.``DefaultCallbacks`(*legacy_callbacks_dict: Dict[str, callable] = None*)[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks)

  Abstract base class for RLlib callbacks (similar to Keras callbacks).These callbacks can be used for custom metrics and custom postprocessing.By default, all of these callbacks are no-ops. To configure custom training callbacks, subclass DefaultCallbacks and then set {“callbacks”: YourCallbacksClass} in the trainer config.`on_episode_start`(***, *worker: RolloutWorker*, *base_env: ray.rllib.env.base_env.BaseEnv*, *policies: Dict[str, ray.rllib.policy.policy.Policy]*, *episode: ray.rllib.evaluation.episode.MultiAgentEpisode*, *env_index: Optional[int] = None*, ***kwargs*)→ None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_episode_start)Callback run on the rollout worker before each episode starts.Parameters**worker** ([*RolloutWorker*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.RolloutWorker)) – Reference to the current rollout worker.**base_env** ([*BaseEnv*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.env.BaseEnv)) – BaseEnv running the episode. The underlying env object can be gotten by calling base_env.get_unwrapped().**policies** (*dict*) – Mapping of policy id to policy objects. In single agent mode there will only be a single “default” policy.**episode** ([*MultiAgentEpisode*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.MultiAgentEpisode)) – Episode object which contains episode state. You can use the episode.user_data dict to store temporary data, and episode.custom_metrics to store custom metrics for the episode.**env_index** (*EnvID*) – Obsoleted: The ID of the environment, which the episode belongs to.**kwargs** – Forward compatibility placeholder.`on_episode_step`(***, *worker: RolloutWorker*, *base_env: ray.rllib.env.base_env.BaseEnv*, *episode:ray.rllib.evaluation.episode.MultiAgentEpisode*, *env_index: Optional[int] = None*, ***kwargs*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_episode_step)Runs on each episode step.Parameters**worker** ([*RolloutWorker*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.RolloutWorker)) – Reference to the current rollout worker.**base_env** ([*BaseEnv*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.env.BaseEnv)) – BaseEnv running the episode. The underlying env object can be gotten by calling base_env.get_unwrapped().**episode** ([*MultiAgentEpisode*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.MultiAgentEpisode)) – Episode object which contains episode state. You can use the episode.user_data dict to store temporary data, and episode.custom_metrics to store custom metrics for the episode.**env_index** (*EnvID*) – Obsoleted: The ID of the environment, which the episode belongs to.**kwargs** – Forward compatibility placeholder.`on_episode_end`(***, *worker: RolloutWorker*, *base_env: ray.rllib.env.base_env.BaseEnv*, *policies: Dict[str, ray.rllib.policy.policy.Policy]*, *episode: ray.rllib.evaluation.episode.MultiAgentEpisode*, *env_index: Optional[int] = None*, ***kwargs*)→ None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_episode_end)Runs when an episode is done.Parameters**worker** ([*RolloutWorker*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.RolloutWorker)) – Reference to the current rollout worker.**base_env** ([*BaseEnv*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.env.BaseEnv)) – BaseEnv running the episode. The underlying env object can be gotten by calling base_env.get_unwrapped().**policies** (*dict*) – Mapping of policy id to policy objects. In single agent mode there will only be a single “default” policy.**episode** ([*MultiAgentEpisode*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.MultiAgentEpisode)) – Episode object which contains episode state. You can use the episode.user_data dict to store temporary data, and episode.custom_metrics to store custom metrics for the episode.**env_index** (*EnvID*) – Obsoleted: The ID of the environment, which the episode belongs to.**kwargs** – Forward compatibility placeholder.`on_postprocess_trajectory`(***, *worker: RolloutWorker*, *episode: ray.rllib.evaluation.episode.MultiAgentEpisode*, *agent_id: Any*, *policy_id: str*, *policies: Dict[str, ray.rllib.policy.policy.Policy]*, *postprocessed_batch:ray.rllib.policy.sample_batch.SampleBatch*, *original_batches: Dict[Any, ray.rllib.policy.sample_batch.SampleBatch]*, ***kwargs*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_postprocess_trajectory)Called immediately after a policy’s postprocess_fn is called.You can use this callback to do additional postprocessing for a policy, including looking at the trajectory data of other agents in multi-agent settings.Parameters**worker** ([*RolloutWorker*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.RolloutWorker)) – Reference to the current rollout worker.**episode** ([*MultiAgentEpisode*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.MultiAgentEpisode)) – Episode object.**agent_id** (*str*) – Id of the current agent.**policy_id** (*str*) – Id of the current policy for the agent.**policies** (*dict*) – Mapping of policy id to policy objects. In single agent mode there will only be a single “default” policy.**postprocessed_batch** ([*SampleBatch*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.SampleBatch)) – The postprocessed sample batch for this agent. You can mutate this object to apply your own trajectory postprocessing.**original_batches** (*dict*) – Mapping of agents to their unpostprocessed trajectory data. You should not mutate this object.**kwargs** – Forward compatibility placeholder.`on_sample_end`(***, *worker: RolloutWorker*, *samples: ray.rllib.policy.sample_batch.SampleBatch*, ***kwargs*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_sample_end)Called at the end of RolloutWorker.sample().Parameters**worker** ([*RolloutWorker*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.RolloutWorker)) – Reference to the current rollout worker.**samples** ([*SampleBatch*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.SampleBatch)) – Batch to be returned. You can mutate this object to modify the samples generated.**kwargs** – Forward compatibility placeholder.`on_learn_on_batch`(***, *policy: ray.rllib.policy.policy.Policy*, *train_batch: ray.rllib.policy.sample_batch.SampleBatch*, *result:dict*, ***kwargs*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_learn_on_batch)Called at the beginning of Policy.learn_on_batch().Note: This is called before 0-padding via pad_batch_to_sequences_of_same_size.Parameters**policy** ([*Policy*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.policy.Policy)) – Reference to the current Policy object.**train_batch** ([*SampleBatch*](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.SampleBatch)) – SampleBatch to be trained on. You can mutate this object to modify the samples generated.**result** (*dict*) – A results dict to add custom metrics to.**kwargs** – Forward compatibility placeholder.`on_train_result`(***, *trainer*, *result: dict*, ***kwargs*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_train_result)Called at the end of Trainable.train().Parameters**trainer** ([*Trainer*](https://docs.ray.io/en/latest/raysgd/v2/api.html#ray.util.sgd.v2.Trainer)) – Current trainer instance.**result** (*dict*) – Dict of results returned from trainer.train() call. You can mutate this object to add additional metrics.**kwargs** – Forward compatibility placeholder.

### Chaining Callbacks[¶](https://docs.ray.io/en/latest/rllib-training.html#chaining-callbacks)

Use the `MultiCallbacks` class to chaim multiple callbacks together.

- *class* `ray.rllib.agents.callbacks.``MultiCallbacks`(*callback_class_list*)[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#MultiCallbacks)

  MultiCallbacks allows multiple callbacks to be registered at the same time in the config of the environment.Example`'callbacks': MultiCallbacks([    MyCustomStatsCallbacks,    MyCustomVideoCallbacks,    MyCustomTraceCallbacks,    .... ]) `![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Visualizing Custom Metrics[¶](https://docs.ray.io/en/latest/rllib-training.html#visualizing-custom-metrics)

Custom metrics can be accessed and visualized like any other training result:

![_images/custom_metric.png](https://docs.ray.io/en/latest/_images/custom_metric.png)

### Customizing Exploration Behavior[¶](https://docs.ray.io/en/latest/rllib-training.html#customizing-exploration-behavior)

RLlib offers a unified top-level API to configure and customize an agent’s exploration behavior, including the decisions (how and whether) to sample actions from distributions (stochastically or deterministically). The setup can be done via using built-in Exploration classes (see [this package](https://github.com/ray-project/ray/blob/master/rllib/utils/exploration/)), which are specified (and further configured) inside `Trainer.config["exploration_config"]`. Besides using one of the available classes, one can sub-class any of these built-ins, add custom behavior to it, and use that new class in the config instead.

Every policy has-an Exploration object, which is created from the Trainer’s `config[“exploration_config”]` dict, which specifies the class to use via the special “type” key, as well as constructor arguments via all other keys, e.g.:

```
# in Trainer.config:
"exploration_config": {
    "type": "StochasticSampling",  # <- Special `type` key provides class information
    "[c'tor arg]" : "[value]",  # <- Add any needed constructor args here.
    # etc
}
# ...
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

The following table lists all built-in Exploration sub-classes and the agents that currently use these by default:

![_images/rllib-exploration-api-table.svg](https://docs.ray.io/en/latest/_images/rllib-exploration-api-table.svg)

An Exploration class implements the `get_exploration_action` method, in which the exact exploratory behavior is defined. It takes the model’s output, the action distribution class, the model itself, a timestep (the global env-sampling steps already taken), and an `explore` switch and outputs a tuple of a) action and b) log-likelihood:

```
    @DeveloperAPI
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[TensorType, int],
                               explore: bool = True):
        """Returns a (possibly) exploratory action and its log-likelihood.

        Given the Model's logits outputs and action distribution, returns an
        exploratory action.

        Args:
            action_distribution (ActionDistribution): The instantiated
                ActionDistribution object to work with when creating
                exploration actions.
            timestep (Union[TensorType, int]): The current sampling time step.
                It can be a tensor for TF graph mode, otherwise an integer.
            explore (Union[TensorType, bool]): True: "Normal" exploration
                behavior. False: Suppress all exploratory behavior and return
                a deterministic action.

        Returns:
            Tuple:
            - The chosen exploration action or a tf-op to fetch the exploration
              action from the graph.
            - The log-likelihood of the exploration action.
        """
        pass
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

On the highest level, the `Trainer.compute_action` and `Policy.compute_action(s)` methods have a boolean `explore`switch, which is passed into `Exploration.get_exploration_action`. If `explore=None`, the value of`Trainer.config[“explore”]` is used, which thus serves as a main switch for exploratory behavior, allowing e.g. turning off any exploration easily for evaluation purposes (see [Customized Evaluation During Training](https://docs.ray.io/en/latest/rllib-training.html#customevaluation)).

The following are example excerpts from different Trainers’ configs (see rllib/agents/trainer.py) to setup different exploration behaviors:

```
# All of the following configs go into Trainer.config.

# 1) Switching *off* exploration by default.
# Behavior: Calling `compute_action(s)` without explicitly setting its `explore`
# param will result in no exploration.
# However, explicitly calling `compute_action(s)` with `explore=True` will
# still(!) result in exploration (per-call overrides default).
"explore": False,

# 2) Switching *on* exploration by default.
# Behavior: Calling `compute_action(s)` without explicitly setting its
# explore param will result in exploration.
# However, explicitly calling `compute_action(s)` with `explore=False`
# will result in no(!) exploration (per-call overrides default).
"explore": True,

# 3) Example exploration_config usages:
# a) DQN: see rllib/agents/dqn/dqn.py
"explore": True,
"exploration_config": {
   # Exploration sub-class by name or full path to module+class
   # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
   "type": "EpsilonGreedy",
   # Parameters for the Exploration class' constructor:
   "initial_epsilon": 1.0,
   "final_epsilon": 0.02,
   "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
},

# b) DQN Soft-Q: In order to switch to Soft-Q exploration, do instead:
"explore": True,
"exploration_config": {
   "type": "SoftQ",
   # Parameters for the Exploration class' constructor:
   "temperature": 1.0,
},

# c) All policy-gradient algos and SAC: see rllib/agents/trainer.py
# Behavior: The algo samples stochastically from the
# model-parameterized distribution. This is the global Trainer default
# setting defined in trainer.py and used by all PG-type algos (plus SAC).
"explore": True,
"exploration_config": {
   "type": "StochasticSampling",
   "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
},
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)



### Customized Evaluation During Training[¶](https://docs.ray.io/en/latest/rllib-training.html#customized-evaluation-during-training)

RLlib will report online training rewards, however in some cases you may want to compute rewards with different settings (e.g., with exploration turned off, or on a specific set of environment configurations). You can evaluate policies during training by setting the `evaluation_interval` config, and optionally also `evaluation_num_episodes`,`evaluation_config`, `evaluation_num_workers`, and `custom_eval_function` (see [trainer.py](https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py) for further documentation).

By default, exploration is left as-is within `evaluation_config`. However, you can switch off any exploration behavior for the evaluation workers via:

```
# Switching off exploration behavior for evaluation workers
# (see rllib/agents/trainer.py)
"evaluation_config": {
   "explore": False
}
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Note

Policy gradient algorithms are able to find the optimal policy, even if this is a stochastic one. Setting “explore=False” above will result in the evaluation workers not using this stochastic policy.

There is an end to end example of how to set up custom online evaluation in [custom_eval.py](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py). Note that if you only want to eval your policy at the end of training, you can set `evaluation_interval: N`, where `N` is the number of training iterations before stopping.

Below are some examples of how the custom evaluation metrics are reported nested under the `evaluation` key of normal training results:

```
------------------------------------------------------------------------
Sample output for `python custom_eval.py`
------------------------------------------------------------------------

INFO trainer.py:623 -- Evaluating current policy for 10 episodes.
INFO trainer.py:650 -- Running round 0 of parallel evaluation (2/10 episodes)
INFO trainer.py:650 -- Running round 1 of parallel evaluation (4/10 episodes)
INFO trainer.py:650 -- Running round 2 of parallel evaluation (6/10 episodes)
INFO trainer.py:650 -- Running round 3 of parallel evaluation (8/10 episodes)
INFO trainer.py:650 -- Running round 4 of parallel evaluation (10/10 episodes)

Result for PG_SimpleCorridor_2c6b27dc:
  ...
  evaluation:
    custom_metrics: {}
    episode_len_mean: 15.864661654135338
    episode_reward_max: 1.0
    episode_reward_mean: 0.49624060150375937
    episode_reward_min: 0.0
    episodes_this_iter: 133
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

```
------------------------------------------------------------------------
Sample output for `python custom_eval.py --custom-eval`
------------------------------------------------------------------------

INFO trainer.py:631 -- Running custom eval function <function ...>
Update corridor length to 4
Update corridor length to 7
Custom evaluation round 1
Custom evaluation round 2
Custom evaluation round 3
Custom evaluation round 4

Result for PG_SimpleCorridor_0de4e686:
  ...
  evaluation:
    custom_metrics: {}
    episode_len_mean: 9.15695067264574
    episode_reward_max: 1.0
    episode_reward_mean: 0.9596412556053812
    episode_reward_min: 0.0
    episodes_this_iter: 223
    foo: 1
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Rewriting Trajectories[¶](https://docs.ray.io/en/latest/rllib-training.html#rewriting-trajectories)

Note that in the `on_postprocess_traj` callback you have full access to the trajectory batch (`post_batch`) and other training state. This can be used to rewrite the trajectory, which has a number of uses including:

> - Backdating rewards to previous time steps (e.g., based on values in `info`).
> - Adding model-based curiosity bonuses to rewards (you can train the model with a [custom model supervised loss](https://docs.ray.io/en/latest/rllib-models.html#supervised-model-losses)).

To access the policy / model (`policy.model`) in the callbacks, note that `info['pre_batch']` returns a tuple where the first element is a policy and the second one is the batch itself. You can also access all the rollout worker state using the following call:

```
from ray.rllib.evaluation.rollout_worker import get_global_worker

# You can use this from any callback to get a reference to the
# RolloutWorker running in the process, which in turn has references to
# all the policies, etc: see rollout_worker.py for more info.
rollout_worker = get_global_worker()
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

Policy losses are defined over the `post_batch` data, so you can mutate that in the callbacks to change what data the policy loss function sees.

### Curriculum Learning[¶](https://docs.ray.io/en/latest/rllib-training.html#curriculum-learning)

In Curriculum learning, the environment can be set to different difficulties (or “tasks”) to allow for learning to progress through controlled phases (from easy to more difficult). RLlib comes with a basic curriculum learning API utilizing the[TaskSettableEnv](https://github.com/ray-project/ray/blob/master/rllib/env/apis/task_settable_env.py) environment API. Your environment only needs to implement the set_task and get_task methods for this to work. You can then define an env_task_fn in your config, which receives the last training results and returns a new task for the env to be set to:

```
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

class MyEnv(TaskSettableEnv):
    def get_task(self):
        return self.current_difficulty

    def set_task(self, task):
        self.current_difficulty = task

def curriculum_fn(train_results, task_settable_env, env_ctx):
    # Very simple curriculum function.
    current_task = task_settable_env.get_task()
    new_task = current_task + 1
    return new_task

# Setup your Trainer's config like so:
config = {
    "env": MyEnv,
    "env_task_fn": curriculum_fn,
}
# Train using `tune.run` or `Trainer.train()` and the above config stub.
# ...
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

There are two more ways to use the RLlib’s other APIs to implement [curriculum learning](https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/).

Use the Trainer API and update the environment between calls to `train()`. This example shows the trainer being run inside a Tune function. This is basically the same as what the built-in env_task_fn API described above already does under the hood, but allows you to do even more customizations to your training loop.

```
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

def train(config, reporter):
    trainer = PPOTrainer(config=config, env=YourEnv)
    while True:
        result = trainer.train()
        reporter(**result)
        if result["episode_reward_mean"] > 200:
            task = 2
        elif result["episode_reward_mean"] > 100:
            task = 1
        else:
            task = 0
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_task(task)))

num_gpus = 0
num_workers = 2

ray.init()
tune.run(
    train,
    config={
        "num_gpus": num_gpus,
        "num_workers": num_workers,
    },
    resources_per_trial=tune.PlacementGroupFactory(
        [{"CPU": 1}, {"GPU": num_gpus}] + [{"CPU": 1}] * num_workers
    ),
)
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

You could also use RLlib’s callbacks API to update the environment on new training results:

```
import ray
from ray import tune

def on_train_result(info):
    result = info["result"]
    if result["episode_reward_mean"] > 200:
        task = 2
    elif result["episode_reward_mean"] > 100:
        task = 1
    else:
        task = 0
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_task(task)))

ray.init()
tune.run(
    "PPO",
    config={
        "env": YourEnv,
        "callbacks": {
            "on_train_result": on_train_result,
        },
    },
)
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

## Debugging[¶](https://docs.ray.io/en/latest/rllib-training.html#debugging)

### Gym Monitor[¶](https://docs.ray.io/en/latest/rllib-training.html#gym-monitor)

The `"monitor": true` config can be used to save Gym episode videos to the result dir. For example:

```
rllib train --env=PongDeterministic-v4 \
    --run=A2C --config '{"num_workers": 2, "monitor": true}'

# videos will be saved in the ~/ray_results/<experiment> dir, for example
openaigym.video.0.31401.video000000.meta.json
openaigym.video.0.31401.video000000.mp4
openaigym.video.0.31403.video000000.meta.json
openaigym.video.0.31403.video000000.mp4
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Eager Mode[¶](https://docs.ray.io/en/latest/rllib-training.html#eager-mode)

Policies built with `build_tf_policy` (most of the reference algorithms are) can be run in eager mode by setting the`"framework": "[tf2|tfe]"` / `"eager_tracing": True` config options or using `rllib train --config'{"framework": "tf2"}' [--trace]`. This will tell RLlib to execute the model forward pass, action distribution, loss, and stats functions in eager mode.

Eager mode makes debugging much easier, since you can now use line-by-line debugging with breakpoints or Python `print()` to inspect intermediate tensor values. However, eager can be slower than graph mode unless tracing is enabled.

### Using PyTorch[¶](https://docs.ray.io/en/latest/rllib-training.html#using-pytorch)

Trainers that have an implemented TorchPolicy, will allow you to run rllib train using the command line `--torch` flag. Algorithms that do not have a torch version yet will complain with an error in this case.

### Episode Traces[¶](https://docs.ray.io/en/latest/rllib-training.html#episode-traces)

You can use the [data output API](https://docs.ray.io/en/latest/rllib-offline.html) to save episode traces for debugging. For example, the following command will run PPO while saving episode traces to `/tmp/debug`.

```
rllib train --run=PPO --env=CartPole-v0 \
    --config='{"output": "/tmp/debug", "output_compress_columns": []}'

# episode traces will be saved in /tmp/debug, for example
output-2019-02-23_12-02-03_worker-2_0.json
output-2019-02-23_12-02-04_worker-1_0.json
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

### Log Verbosity[¶](https://docs.ray.io/en/latest/rllib-training.html#log-verbosity)

You can control the trainer log level via the `"log_level"` flag. Valid values are “DEBUG”, “INFO”, “WARN” (default), and “ERROR”. This can be used to increase or decrease the verbosity of internal logging. You can also use the `-v` and `-vv` flags. For example, the following two commands are about equivalent:

```
rllib train --env=PongDeterministic-v4 \
    --run=A2C --config '{"num_workers": 2, "log_level": "DEBUG"}'

rllib train --env=PongDeterministic-v4 \
    --run=A2C --config '{"num_workers": 2}' -vv
```

![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)

The default log level is `WARN`. We strongly recommend using at least `INFO` level logging for development.

### Stack Traces[¶](https://docs.ray.io/en/latest/rllib-training.html#stack-traces)

You can use the `ray stack` command to dump the stack traces of all the Python workers on a single node. This can be useful for debugging unexpected hangs or performance issues.

## External Application API[¶](https://docs.ray.io/en/latest/rllib-training.html#external-application-api)

In some cases (i.e., when interacting with an externally hosted simulator or production environment) it makes more sense to interact with RLlib as if it were an independently running service, rather than RLlib hosting the simulations itself. This is possible via RLlib’s external applications interface [(full documentation)](https://docs.ray.io/en/latest/rllib-env.html#external-agents-and-applications).

- *class* `ray.rllib.env.policy_client.``PolicyClient`(*address: str*, *inference_mode: str = 'local'*, *update_interval: float =10.0*)[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient)

  REST client to interact with a RLlib policy server.`start_episode`(*episode_id: Optional[str] = None*, *training_enabled: bool = True*) → str[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient.start_episode)Record the start of one or more episode(s).Parameters**episode_id** (*Optional**[**str**]*) – Unique string id for the episode or None for it to be auto-assigned.**training_enabled** (*bool*) – Whether to use experiences for this episode to improve the policy.ReturnsUnique string id for the episode.Return typeepisode_id (str)`get_action`(*episode_id: str*, *observation: Union[Any, Dict[Any, Any]]*) → Union[Any, Dict[Any, Any]][[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient.get_action)Record an observation and get the on-policy action.Parameters**episode_id** (*str*) – Episode id returned from start_episode().**observation** (*obj*) – Current environment observation.ReturnsAction from the env action space.Return typeaction (obj)`log_action`(*episode_id: str*, *observation: Union[Any, Dict[Any, Any]]*, *action: Union[Any, Dict[Any, Any]]*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient.log_action)Record an observation and (off-policy) action taken.Parameters**episode_id** (*str*) – Episode id returned from start_episode().**observation** (*obj*) – Current environment observation.**action** (*obj*) – Action for the observation.`log_returns`(*episode_id: str*, *reward: int*, *info: Union[dict, Dict[Any, Any]] = None*, *multiagent_done_dict: Optional[Dict[Any, Any]] = None*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient.log_returns)Record returns from the environment.The reward will be attributed to the previous action taken by the episode. Rewards accumulate until the next action. If no reward is logged before the next action, a reward of 0.0 is assumed.Parameters**episode_id** (*str*) – Episode id returned from start_episode().**reward** (*float*) – Reward from the environment.**info** (*dict*) – Extra info dict.**multiagent_done_dict** (*dict*) – Multi-agent done information.`end_episode`(*episode_id: str*, *observation: Union[Any, Dict[Any, Any]]*) → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient.end_episode)Record the end of an episode.Parameters**episode_id** (*str*) – Episode id returned from start_episode().**observation** (*obj*) – Current environment observation.`update_policy_weights`() → None[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_client.html#PolicyClient.update_policy_weights)Query the server for new policy weights, if local inference is enabled.

- *class* `ray.rllib.env.policy_server_input.``PolicyServerInput`(*ioctx*, *address*, *port*, *idle_timeout=3.0*)[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_server_input.html#PolicyServerInput)

  REST policy server that acts as an offline data source.This launches a multi-threaded server that listens on the specified host and port to serve policy requests and forward experiences to RLlib. For high performance experience collection, it implements InputReader.For an example, run examples/cartpole_server.py along with examples/cartpole_client.py –inference-mode=local|remote.Examples`>>> pg = PGTrainer( ...     env="CartPole-v0", config={ ...         "input": lambda ioctx: ...             PolicyServerInput(ioctx, addr, port), ...         "num_workers": 0,  # Run just 1 server, in the trainer. ...     } >>> while True: >>>     pg.train() `![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)`>>> client = PolicyClient("localhost:9900", inference_mode="local") >>> eps_id = client.start_episode() >>> action = client.get_action(eps_id, obs) >>> ... >>> client.log_returns(eps_id, reward) >>> ... >>> client.log_returns(eps_id, reward) `![Copy to clipboard](https://docs.ray.io/en/latest/_static/copy-button.svg)`next`()[[source\]](https://docs.ray.io/en/latest/_modules/ray/rllib/env/policy_server_input.html#PolicyServerInput.next)Returns the next batch of experiences read.ReturnsThe experience read.Return typeUnion[[SampleBatch](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.SampleBatch), [MultiAgentBatch](https://docs.ray.io/en/latest/rllib-package-ref.html#ray.rllib.evaluation.MultiAgentBatch)]