# Reinforcement-Learning
While Q-learning has shown some promising results while dealing in a discrete number of states and actions, many real-life environments deal with an exponential number of states and actions. Trying to extend the tabular representation of the action-value function to a large number of states, we encounter several problems. Visiting every state and trying out each of the actions multiple times can be computationally expensive and extremely time-limited; furthermore, what happens when we move from a discrete state-space to a continuous state-space (Mnih et al., 2015). Even the interval from zero to one is infinite if one allows arbitrary precision. Clearly, a tabular representation is not going to work for continuous state-spaces. In the past several years, we have seen an outbreak in the field known as deep learning, which is particularly well suited to address this problem. Deep learning is the application of deep neural networks to a broad collection of problems. Neural networks are function approximators that can approximate any continuous function. This will be useful for us since the action-value function Q is a continuous function, and the state space can be approximated with enough precision to achieve real learning.

## GitHub repository
This GitHub repository represents my learning attempt to learn this "beast" called reinforcement learning. We first start with the simplest case of reinforcement learning, where we implement the simplest form of the Q-learning algorithm in a discrete environment where every state and action can be captured in a tabular form, called the Q-table. This is illustrated in notebook **Q-learning**. Afterward, we move to a continuous state-action space where we introduce a neural network to approximate the Q-values in every state and replicate two novel papers released by Deepmind, representing a single line of research. We implement a naive deep Q-learning algorithm using Q-learning combined with a DQN to approximate the optimal Q-value for every state-action pair in notebook **Naive Deep Q-learning**. We see that a naive deep Q-network approach applied to a continuous state space does not work for several reasons. It turns out Q-learning combined with a nonlinear function approximator such as a neural network can be unstable or even diverge, as mentioned in (Mnih et al., 2015). We overcome this instability by introducing a replay memory and a second neural network called the target network in notebook **Human-level Control through Deep Reinforcement Learning**. It turns out, as explained in (Van Hasselt, Guez, & Silver, 2016) that some Q-values tempt to be overestimated under certain conditions. We address these overestimations by introducing a double Q-learning algorithm, which was originally introduced for a tabular setting, but generalize it for large-scale function approximators in notebook **Deep Reinforcement Learning with Double Q-learning**. 

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

## Installation Dependencies
Download PyTorch from https://pytorch.org
- python 3.5+
- PyTorch 1.8.0
- Gym 0.18.0
- pygame 1.9.6
- opencv-python 4.2.0

## References
1) Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel, Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski, Georg, et al. https://www.nature.com/articles/nature14236
2) Van  Hasselt,  H.,  Guez,  A.,  &  Silver,  D.   (2016).   Deep  Reinforcement  Learning  with  Double  Q-learning.  InProceedings of the aaai conference on artificial intelligence(Vol. 30). https://arxiv.org/abs/1509.06461
