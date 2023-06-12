# POME-implementation

self implementation of RL method [Policy Optimization with Model-based Explorations](https://arxiv.org/abs/1811.07350)

## Environment Installation Using Anaconda

```python
    conda create --f environment.yml
```

## Visualization

Tensorboard logged datas will be located below runs/ directory, to visualize data after a pome run

```python
    tensorboard --logdir runs or
    python -m tensorboard.main --logdir=./experiments/{algo}/runs
```

## Todos

1. Currently, one file implementation [pome.py](pome.py) can run without error, while algorithm is not tested
2. For reproduction of [OpenAI Baselines](https://github.com/openai/baselines), there are many addtional implementation details, according to [this site](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
3. Visualization
4. code for other environments

## Current Implementation

1. fixed-length trajectory segments
2. Orthogonal Initialization of Weights and Constant Initialization of biases
3. Mini-batch Updates
4. Skip Frame
5. Resize images
6. Scaling the Images to Range [0, 1]
7. minibatch standardization
8. remove reward estimation
