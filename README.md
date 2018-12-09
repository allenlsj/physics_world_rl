# Physics_World_RL
Training RL algorithms for active physical inference

Humans exhibited rich and informative strategies that selectively revealed objects' properties of interest while minimizing causal "noise". We would like to explore how a variety of artificial agents solve this task, and what components are required to achieve human levels of performance, and whether similar or different strategies emerge.

## First phase
Develop a python port of simulation environment based on pybox2D to test control algorithms. The idea is to collect the state space and the intrinstic rewards by utilizing pre-defined action space. Simulator design and reward formulation are adapted from this [paper](https://reader.elsevier.com/reader/sd/pii/S001002851730347X?token=99E4A9B2B68F6724DCCBA56A91667C5C0F8DA9436EEB3E021D5B50318A71897D73E995F08741D5B6837F742F89DCD71B). We have packaged all the source code into a OpenAI gym-like environment for convenient simulating and training purpose.

## Second phase
Once the python-versioned environment simulator set up, we will try to formulate an appropriate reinforcement learning modeling framework to learn and optimize human actions behind the scenes. Since the state transition matrix is unknown, and the state-space and the action space are continuous, we tend to start with a model-free based RL model(controller) to output optimal actions. Noticed that our goal is to explore whether people’s actions are generally effective at reducing uncertainty about the specific parameters of the scene they are asked about. To approximate those structured learning behaviours within an unstructured environment, we may add an adversarial network to challenge our controller. We also want to utilize several generative functions/models to mimic human-level behaviors. We will mainly follow the ideas presented in these papers: [paper1](https://arxiv.org/pdf/1802.07442.pdf), [paper2](http://people.idsia.ch/~juergen/ieeecreative.pdf), [paper3](https://web.mit.edu/cocosci/Papers/Science-2015-Lake-1332-8.pdf), [paper4](http://papers.nips.cc/paper/6705-question-asking-as-program-generation.pdf) with our adaptive modifications.

### Dependencies
* **Python 2.7**
* **numpy/scipy**
* **tensorflow 1.12.0+**
* **keras 2.2.4+**
* **[pybox2d](https://github.com/pybox2d/pybox2d)**

### Training
If you'd like to train your models, you will first need to download this repository and move to the agent directory:
```bash
$ git clone https://github.com/allenlsj/physics_world_rl
$ cd physics_world_rl/src/code/agent
```
To train a q-value function approximator with standard semi-gradient update, simply execute the following commend:
```bash
$ python2 Qlearning.py --mode 1
```
for 10 epochs and 10 iteration of games per epoch. The `mode` arguement stands for the type of intinsic reward returned by the simulator, where 1 is for `mass` and 2 is for `force`. Feel free to use `python2 Qlearning.py --h` for more information regarding the other input arguments.

Similarly, if you want to train your q-agent with a target network to check if the instability issue is reduced, use:
```bash
$ python2 Qagent.py --mode [1 or 2 depends on your need]
```
Additionally, we also provide a recurrent q-network (RQN) that predicts q-values based on the hidden states of the objects:
```bash
$ python2 RQN.py --mode [1 or 2 depends on your need]
```
You can modifiy the default time frame range per action and the total time frames per game in `simulator/config.py`.

### Prediction
A sample code illustrating how to apply the trained model to explore the environment and how to generate the video regarding the exploration has been provided under the `agent/` folder. Simply execute the following command:
```bash
$ python2 record_video.py --mode [1 or 2 depends on your need]
```
