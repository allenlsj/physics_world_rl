# Physics_World_RL
Humans exhibited rich and informative strategies that selectively revealed objects' properties of interest while minimizing causal "noise". We would like to explore how a variety of artificial agents solve this task, and what components are required to achieve human levels of performance, and whether similar or different strategies emerge.

For parallelized version (faster simulation), please switch to the 'simulator-opt' branch.

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
The `mode` arguement stands for the type of intinsic reward returned by the simulator, where 1 is for `mass` and 2 is for `force`. Feel free to use `python2 Qlearning.py --h` for more information regarding the other input arguments.

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
