# physics_world_rl
Training RL algorithms for active physical inference

Humans exhibited rich and informative strategies that selectively revealed objects' properties of interest while minimizing causal "noise". We would like to explore how a variety of artificial agents solve this task, and what components are required to achieve human levels of performance, and whether similar or different strategies emerge.

## On-going first phase
Develop a python port of simulation environment based on pybox2D to test control algorithms. The idea is to collect the state space and the informational rewards by utilizing pre-defined action space. Simulator design and reward formulation are adapted from this [paper](https://reader.elsevier.com/reader/sd/pii/S001002851730347X?token=99E4A9B2B68F6724DCCBA56A91667C5C0F8DA9436EEB3E021D5B50318A71897D73E995F08741D5B6837F742F89DCD71B). We have packaged all the source code into a OpenAI gym-like environment for convenient simulating and training purpose.

## On-going second phase
Once the python-versioned environment simulator set up, we will try to formulate an appropriate reinforcement learning modeling framework to learn and optimize human actions behind the scenes. Since the state transition matrix is unknown, and the state-space and the action space are continous, we tend to start with a DQN to find optimal actions. We also want to utilize several generative functions/models to mimic human-level behaviors. We will mainly follow the ideas presented in these papers: [paper1](https://arxiv.org/pdf/1802.07442.pdf), [paper2](http://people.idsia.ch/~juergen/ieeecreative.pdf), with our adaptive modifications.
