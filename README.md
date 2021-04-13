# Reinforcement-Learning

## GitHub repository
This GitHub repository represents my learning attempt to learn this "beast" called reinforcement learning. We first start with the simplest case of reinforcement learning, where we implement the simplest form of the Q-learning algorithm in a discrete environment where every state and action can be captured in a tabular form, called the Q-table. This is illustrated in notebook **Q-learning**. Afterward, we move to a continuous state-action space where we introduce a neural network to approximate the Q-values in every state and replicate two novel papers released by Deepmind, representing a single line of research. We implement a naive deep Q-learning algorithm using Q-learning combined with a DQN to approximate the optimal Q-value for every state-action pair in notebook **Naive Deep Q-learning**. We see that a naive deep Q-network approach applied to a continuous state space does not work for several reasons. It turns out Q-learning combined with a nonlinear function approximator such as a neural network can be unstable or even diverge, as mentioned in (Mnih et al., 2015). We overcome this instability by introducing a replay memory and a second neural network called the target network in notebook **Human-level Control through Deep Reinforcement Learning**. It turns out, as explained in (Van Hasselt, Guez, & Silver, 2016) that some Q-values tempt to be overestimated under certain conditions. We address these overestimations by introducing a double Q-learning algorithm, which was originally introduced for a tabular setting, but generalize it for large-scale function approximators in notebook **Deep Reinforcement Learning with Double Q-learning**. 
