# Background:

## Learning what is rewarding:
It is likely that biological learning systems are capable of evolving or learning new reward functions over time for use in learning effective behavioral policies. An example of this is our sense of taste: starting with an initial, outcome-oriented reward function which simply rewarded organisms that consume enough nutrients and penalizes (kills off) those that don't, organisms such as humans developed immediate reward functions that provide short term positive feedback in the presence of certain (until recently) rare chemicals which contain large amounts of calories, e.g. sugar and fat. The process of evolving an immediate reward as a proxy for the delayed reward of survival vs. starvation can be thought of as a form of meta reinforcement learning over a flexible architecture capable of factoring out some elements of the state variable into their own proxy reward system. 

## Analogy to model-based RL:
The problem is thus decomposed into two questions: what in the environment is rewarding? and how does an agent find a policy that maximizes that reward? One can think of this decomposition as a way to implement model-based RL over an otherwise model-agnostic learning algorithm. The inner learning loop may reflect action choices based solely on cached Q-values for state, action pairs, but the outer system, which learns which elements of the observable state relate to the underlying reward function (survival vs. starvation), essentially encodes causal knowledge relating observables such as sugar content to likelihood of survival.

## Computational motivation:
From a computational perspective, a big motivating question is: what is the point of factoring out certain state variables into a proxy-reward function when we could maintain the original reward function as is and simply let the causal relationship between state and outcome be encoded in the weights of some neural value function approximator? I believe there are four potential benefits of factoring out observables into a proxy reward:

1. A proxy reward function may be simpler to evaluate and its immediate evaluation may provide better feedback over long training episodes.
2. Using a proxy reward function may reduce generalization error and increase stability of Q-learning
3. A proxy reward function may show greater capacity for transfer learning.
4. Factoring out the question "what in the environment is rewarding?" into a separate module may reduce training time in the same way that using deeper neural networks as opposed to shallower ones can exponentially reduce training complexity for highly compositional problem spaces.


My final hypothesis is that the reward function may learn to incorporate information from the environment in addition to just the immediate observable information for the choice items. For example, the learned reward function might come to both reflect information about the nutritional content of various foods, as well as information on the scarcity or relative difficulty of obtaining certain nutrients - a variable which is mitigated through the distribution of food items high in that nutrient.
# Experimental set-up
The experiment will consist of an agent navigating two environments with identical reward structure. The first environment will be simpler and will be used to illustrate transfer learning into the second, more complex environment.

The two environments have a number of similarities. In particular, the agent must acquire **A** grams of protein and **B** total calories within a time horizon of **T**. There are **k** foods each with a number of macro nutrients: fiber, fat, sugar, and protein. Total calories will be computed from the macros. 

The true reward **R** will be a 2 if both **A** and **B** are met, 1 if only **B** is met, and 0 if neither are met.

## Environment #1: Multi-armed bandit
In the first environment, there will be one lever for each of the **k** foods. With some probability **p_{i}**, the **i^{th}** lever dispenses food **i** and yields the nutritional value for that food. The agent has **T** pulls per episode before it receives **R**

We will generate a number of different variants of this environment with different multinomial distributions for the **k** foods. These environments may demonstrate that under different scarcity conditions different policies may evolve.
## Environment #2: Grid world
In this environment the agent navigates an **NxN** grid containing clusteres of food from the **k** categories distributed across the grid with rarity proportional to **p'_{i}**. Each move Up, Down, Left, Right takes some time. After time **T** the agent receives **R**. 

One of the uses of this task is to test transfer abilities from **Environment 1** as it captures the intuition that while the bandit teaches problem solving skills related to scarcity and nutritional content, the grid world scenario merely adds an extra layer of environment navigation, enabling the scarcity and content-related info to transfer from the bandit set up. This hypothesis would be especially proven if the reward functions in **Environment 1** tend to reflect the underlying multinomial distribution as well. In that case, we would find that bandit environments will transfer better to grid worlds with similar multinomials over **k**.

A variant of this environment will also contain **j > k** different foods. This will likely enhance transfer because an agent with a good learned-reward function will no longer need to relearn how every single food relates to nutritional success, instead relying primarily on the immediate reward provided by its macro nutrients.
## Architecture 
### Control
**NOTE: UNSURE ABOUT GOOD STATE REPRESENTATION, ACTUAL RL ALG TBD, MAYBE DEEP-Q LEARNING? OR STANDARD Q-LEARNING**
In both environments the control will have a state vector **S** describing the amount of macro nutrients consumed so far, a state vector **S'** describing the amount of macro nutrients consumed in the last time step, and a time counter **t**. 

In addition to **S + S' + t**, the grid world agent will also have access to its grid location **(x,y)** and a map of the 8 squares surrounding it describing each grid location as containing no food: **0** or a food indicated by its category **[1,k]**. We will refer to the environmental state variables (**t**, **(x,y)**, and the proximate grid values) collectively **V**, and the two nutritional state vectors **Ŝ**.

### Experimental condition
The experimental agent will have the exact same architecture as in the control, except this time it doesn't receive any of the **Ŝ** variables as input, receiving only the **V** variables. 

At the same time the policy is being learned, the proxy reward function is being learned by a separate module. That module will either be a.) simple linear regression from **Ŝ** to **R**, or b.) a neural network trained to map **Ŝ** to **R**. In either case, the proxy reward will be called **r(Ŝ)**. 

**r(Ŝ)** will be used to train the policy of the experimental agent. It will also be learned over time - though in the transfer learning experments, it will be initialized on the basis of the results from the multi armed bandit trials.

We will measure the true reward of the experimental agent as being **R**, thus the proxy reward trains the policy, but is not used to evaluae it.

## Questions
We would like to answer the following questions:

1. Does the proxy-reward trained agent still converge to policy which is optimal over **R**
2. Does the proxy-reward trained agent consume fewer computational resources?
3. Does the proxy-reward trained agent converge to a good policy after fewer training episodes than the control?
4. Do the proxy-reward functions differ according to the changing parameters of the environment?
5. Does transfer learning of the proxy-reward function provide faster convergence for the experimental agent?
6. Does transfer learning have an especially large impact in the **j > k** food environment? 
7. Do multi armed bandits transfer knowledge notably better to grid world agents which exposed to more similar multinomial distributions in their environments? (this is a follow up to q4, does the reward function incorporate information about the environmental parameters)

Additional directions: instead of providing the structure of partitioning **Ŝ** and **V** can the agent learn to partition the observation space on its own? 

