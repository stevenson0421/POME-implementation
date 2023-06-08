# POME-implementation

self implementation of RL method [Policy Optimization with Model-based Explorations](https://arxiv.org/abs/1811.07350)

## Environment Installation Using Anaconda

```python
    conda create --name {env name} python=3.10.11
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install scipy opencv-python
    pip install gymnasium[atari, accept_rom_license]
```

## Todos

1. Currently, one file implementation [pome.py](pome.py) can run without error, while algorithm is not tested
2. For reproduction of [OpenAI Baselines](https://github.com/openai/baselines), there are many addtional implementation details, according to [this site](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
3. Visualization
4. code for other environments
