# Reinforcement-Learning

## Introduction
While Q-learning has shown some promising results while dealing in a discrete number of states and actions, many real-life environments deal with an exponential number of states and actions. Trying to extend the tabular representation of the action-value function to a large number of states, we encounter several problems. Visiting every state and trying out each of the actions multiple times can be computationally expensive and extremely time-limited; furthermore, what happens when we move from a discrete state-space to a continuous state-space \cite{mnih2015human}. Even the interval from zero to one is infinite if one allows arbitrary precision. Clearly, a tabular representation is not going to work for continuous state-spaces. In the past several years, we have seen an outbreak in the field known as deep learning, which is particularly well suited to address this problem. Deep learning is the application of deep neural networks to a broad collection of problems. Neural networks are function approximators that can approximate any continuous function. This will be useful for us since the action-value function Q is a continuous function, and the state space can be approximated with enough precision to achieve real learning.

This assignment replicates two novel papers released by Deepmind, representing a single line of research. We first implement a naive deep Q-learning algorithm using Q-learning combined with a DQN to approximate the optimal Q-value for every state-action pair in section \ref{section:Naive}. We see that a naive deep Q-network approach applied to a continuous state space does not work for several reasons. It turns out Q-learning combined with a nonlinear function approximator such as a neural network can be unstable or even diverge as mentioned in \cite{mnih2015human}. We overcome this instability by introducing a replay memory and a second neural network called the target network in section \ref{section:Human}. It turns out, as explained in \cite{van2016deep} that some Q-values tempt to be an overestimation under certain conditions. We address these overestimations by introducing a double Q-learning algorithm, which was originally introduced for a tabular setting, but generalize it for large-scale function approximators in section \ref{section:Double}.


## Image Preprocessing
We are working directly with raw pong frames, which are 640 × 480 pixel images with a 128 color palette, which can be computationally demanding, so we apply an essential preprocessing step to reduce the input dimensionality. The Atari's raw frames are preprocessed by first converting their RGB representation to gray-scale and down-scaling it to an 80 × 80 image. We overcome flickering, which is present in some Atari games, by taking the two previous two observations' max value. We stack four frames as input to our neural network.  

## Model Architecture
![alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.nature.com%2Farticles%2Fnature14236&psig=AOvVaw2pJPyK54LSdfxE7nHJ2Niv&ust=1615280264723000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPCMxYWqoO8CFQAAAAAdAAAAABAJ)


## List of Hyperparameters and their values
All the hyperparameters' values were selected by performing an informal search on Breakout and Pong. We did not perform a regular grid search owing to the high computational cost, although it is likely that even better results could be obtained by regularly tuning the hyperparameter values.
| Hyperparamter | Value | Value |
| --- | --- |
| Minibatch size | List all new or modified files |
| Replay memory size | Show file differences that haven't been staged |
| Agent history length | Show file differences that haven't been staged |
| Discount factor | Show file differences that haven't been staged |
| Learning rate | Show file differences that haven't been staged |
| Initial exploration | Show file differences that haven't been staged |
| Final exploration | Show file differences that haven't been staged |
| Learning rate | Show file differences that haven't been staged |

## Installation Dependencies
Download PyTorch from https://pytorch.org
- python 2.7.16
- PyTorch 1.8.0
- pygame 1.9.6
- opencv-python 4.2.0



## References
1) Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel, Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski, Georg, et al. https://www.nature.com/articles/nature14236
2) Van  Hasselt,  H.,  Guez,  A.,  &  Silver,  D.   (2016).   Deep  Reinforcement  Learning  with  Double  Q-learning.  InProceedings of the aaai conference on artificial intelligence(Vol. 30). https://arxiv.org/abs/1509.06461

