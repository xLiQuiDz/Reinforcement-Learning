# Deep Q-learning
RL on its own perforce very well in small environments. This is because we could seize all states and actions in a small Q-table. It becomes a more challenging problem when we increase an agent's state and action space in the environment. The algorithms' performance, like Q-learning, will drop increasingly in more complex and sophisticated environments (Mnih et al.,2015). The iterative computing process for updating all Q-values for each state-action pair in an environment becomes computationally inefficient and possibly infeasible due to the computational resources. We combine Q-learning with artificial neural networks (ANNs) to overcome this exponential overload in deep reinforcement learning. While using Q-learning, we iteratively update our Q-values using the Bellman equation to converge to the optimal Q-value. In Deep reinforcement learning, we use the Bellman equation to approximate the optimal Q-values for each state-action pair, while using an artificial neural network (Sutton & Barto, 2018).

This Github repository replicates tree novel papers released by Deepmind, representing a single line of research. We first implement a **naive deep Q-learning** algorithm using Q-learning combined with a DQN to approximate the optimal Q-value for every state-action pair. We see that a naive deep Q-network approach applied to a continuous state space does not work for several reasons. It turns out Q-learning combined with a nonlinear function approximator such as a neural network can be unstable or even diverge as mentioned in (Mnih et al.,2015). We overcome this instability by introducing a replay memory and a second neural network called the target network. It turns out, as explained in (Van Hasselt, Guez, & Silver, 2016) that some Q-values tempt to be an overestimation under certain conditions. We address these overestimations by introducing a double Q-learning algorithm, which was originally introduced for a tabular setting, but generalize it for large-scale function approximators.

## Image Preprocessing
We are working directly with raw frames, which are 640 × 480 pixel images with a 128 color palette, which can be computationally demanding, so we apply an essential preprocessing step to reduce the input dimensionality. Preprocessing the modules is optional but speeds up the performance of the agent. The following bullets summarise what needs to be done in order to preprocess the raw frames. 

1) Go from three channels to one channel
    - Screen images have three channels, while our agent only needs one channel. We convert the image to grayscale.
2) Downscale to 84 x 84
    - Images are realy large, with makes training slow. We resize the image to 84 x 84 to improve learning.
3) Take the maximum of previous two frames
    - We keep track of the two most recent frames and we take the maximum over the two. This is needed to overcome flickering in some Atari games.  
4) Repeat each action four times
    - We repeat the same action four times for every skipped frame. This allows us to play 4 times more.
5) Swap channels to first possition
    - Pytorch expects that images have channels first, while the OpenAI Gym returns images with channels last. We fix this by swapping the axis of the NumPy array. 
6) Scale output
    - Scale output, since they are integers from 0 to 255. We can deal with this by dividing the image by 255. 
7) Stack four most recent frames

## Neural Network Architecture
### Naive Deep Q-learning
We use a neural network with two linear layers that take a low-level representation of the environment as input and output the Q-values, corresponding to the actions an agent can take from that state.

### Human-Level Control Through Deep Reinforcement Learning
This GitHub is a replica of the work done in the paper (Mnih et al., 2015 & an Hasselt) so we use the same network architecture as proposed. The neural network input consists of an 8 by 84 x 84 image produced by first preprocessing the observation returned by the OpenAI Atari environment. The first hidden layer convolves 32 filters of 8 x 8 with stride 4, followed by a rectified nonlinearity. The second hidden layer convolves 64 filters of 4 x 4, with side 2, again followed by rectified nonlinearity. This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier. The final hidden layer is fully connected and consists of 512 rectified units. The output layer is a fully-connected layer with a single output for each valid action.

### Deep Reinforcement Learning with Double Q-learning
This GitHub is a replica of the work done in the paper (Guez, & Silver, 2016) so we use the same network architecture as proposed. The neural network input consists of an 8 by 84 x 84 image produced by first preprocessing the observation returned by the OpenAI Atari environment. The first hidden layer convolves 32 filters of 8 x 8 with stride 4, followed by a rectified nonlinearity. The second hidden layer convolves 64 filters of 4 x 4, with side 2, again followed by rectified nonlinearity. This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier. The final hidden layer is fully connected and consists of 512 rectified units. The output layer is a fully-connected layer with a single output for each valid action.

## List of Hyperparameters
### Q-learning
| Hyperparamter | Value | Description |
| --- | --- | --- |
| Learning rate | 0.1 | The learning rate used by the Q-function. |
| Discount rate | 0.99 | The Discount rate gamma used in Q-Learning update. |
| Initial exploration rate | 1.0 | The initial value of the exploration rate. |
| Maximum exploration rate | 1.0 | The maximum value of the exploration rate. |
| Minimum exploration rate | 0.01 | The minimum value of the exploration rate. |
| Exploration decay rate | 0.001 | The rate at which the exploration rate decays. |

### Naive Deep Q-learning
| Hyperparamter | Value | Description |
| --- | --- | --- |
| Learning rate | 0.001 | The learning rate used by the Q-function. |
| Discount rate | 0.99 | The Discount rate gamma used in Q-Learning update. |
| Initial exploration rate | 1.0 | The initial value of the exploration rate. |
| Maximum exploration rate | 1.0 | The maximum value of the exploration rate. |
| Minimum exploration rate | 0.01 | The minimum value of the exploration rate. |
| Exploration decay rate | 0.001 | The rate at which the exploration rate decays. |

### Human-Level Control Through Deep Reinforcement Learning
| Hyperparamter | Value | Description |
| --- | --- | --- |
| Learning rate | 0.0001 | The learning rate used by the SGD optimizer. |
| Discount rate | 0.99 | The Discount rate gamma used in Q-Learning update. |
| Initial exploration rate | 1.0 | The initial value of the exploration rate. |
| Maximum exploration rate | 1.0 | The maximum value of the exploration rate. |
| Minimum exploration rate | 0.1 | The minimum value of the exploration rate. |
| Exploration decay rate | 0.01 | The rate at which the exploration rate decays. |
| Batch size | 32 | The number of training cases over which each SGD update is computed. |
| Replace | 1.000 | The number of steps after the target network is replaced by the policy network. |
| Replay memory capacity | 50.000 | SGD updates are sampled from this number of most recent frames.|

### Deep Reinforcement Learning With Double Q-learning
| Hyperparamter | Value | Description |
| --- | --- | --- |
| Learning rate | 0.0001 | The learning rate used by the SGD optimizer. |
| Discount rate | 0.99 | The Discount rate gamma used in Q-Learning update. |
| Initial exploration rate | 1.0 | The initial value of the exploration rate. |
| Maximum exploration rate | 1.0 | The maximum value of the exploration rate. |
| Minimum exploration rate | 0.1 | The minimum value of the exploration rate. |
| Exploration decay rate | 0.01 | The rate at which the exploration rate decays. |
| Batch size | 32 | The number of training cases over which each SGD update is computed. |
| Replace | 1.000 | The number of steps after the target network is replaced by the policy network. |
| Replay memory capacity | 50.000 | SGD updates are sampled from this number of most recent frames.|

### Dueling Network Architectures for Deep Reinforcement Learning
| Hyperparamter | Value | Description |
| --- | --- | --- |
| Learning rate | 0.0001 | The learning rate used by the SGD optimizer. |
| Discount rate | 0.99 | The Discount rate gamma used in Q-Learning update. |
| Initial exploration rate | 1.0 | The initial value of the exploration rate. |
| Maximum exploration rate | 1.0 | The maximum value of the exploration rate. |
| Minimum exploration rate | 0.1 | The minimum value of the exploration rate. |
| Exploration decay rate | 0.01 | The rate at which the exploration rate decays. |
| Batch size | 32 | The number of training cases over which each SGD update is computed. |
| Replace | 1.000 | The number of steps after the target network is replaced by the policy network. |
| Replay memory capacity | 50.000 | SGD updates are sampled from this number of most recent frames.|

## Installation Dependencies
Download PyTorch from https://pytorch.org
- python 3.5+
- PyTorch 1.8.0
- Gym 0.18.0
- pygame 1.9.6
- opencv-python 4.2.0

## References
1) Sutton, R. S., & Barto, A. G.  (2018).Reinforcement learning:  An introduction.  MIT press. https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
2) Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel, Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski, Georg, et al. https://www.nature.com/articles/nature14236
3) Van  Hasselt,  H.,  Guez,  A.,  &  Silver,  D.   (2016).   Deep  Reinforcement  Learning  with  Double  Q-learning.  InProceedings of the aaai conference on artificial intelligence(Vol. 30). https://arxiv.org/abs/1509.06461
4) Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling networkarchitectures  for  deep  reinforcement  learning.   InInternational  conference  on  machinelearning(pp. 1995–2003). https://arxiv.org/pdf/1511.06581.pdf
