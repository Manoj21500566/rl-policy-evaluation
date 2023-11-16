# EXP : 2


# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

1)Two Terminal States: G: The goal state & H: A hole state.

2)Five Transition states / Non-terminal States including S: The starting state.
### Action:
The agent can take two actions:

R: Move right.

L: Move left.
### Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.
### Graphical Representation


![R1](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/406422e8-2ef8-418b-b083-25a244f4dd28)


## POLICY EVALUATION FUNCTION
### Formula


![R2](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/1b3de065-7486-4f52-8f88-279298fa56f0)


### program:
~~~

Manoj M
212221240027

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
   	'''Initialize 1st Iteration estimates of state-value function(V) to zero'''
    prev_V = np.zeros(len(P), dtype=np.float64)

    while True:
        '''Initialize the current iteration estimates to zero'''
        V=np.zeros(len(P),dtype=np.float64)
        
        for s in range(len(P)):
        
            '''Update the value function for each state'''
            for prob,next_state,reward,done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                
            '''Check for convergence'''
            if np.max(np.abs(prev_V-V))<theta:
                break
                
            '''Update the previous state-value function'''
            prev_V=V.copy()
        return V
~~~


## OUTPUT:


### Policy 1

![o11](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/ebd7763a-4b7f-4f6d-a7f6-137eaf8f5c9d)

![o12](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/a2d66c36-fc5f-402e-b2a0-c8631f0147fe)

![o13](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/64ca2ff7-9f9a-40fd-bf24-0e4949fdab61)


### Policy 2

![o21](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/c8a361fd-9068-4000-9411-6b1a6f685c19)

![o23](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/29634d64-a6d5-47e8-989c-0a0fbab4a671)

![o22](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/cd9bf230-0a88-4459-acda-acd9c8104428)



### Comparison

![o3](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/c147c3e2-b3aa-4386-8691-957bbb79a550)



### Conclusion

![o4](https://github.com/Manoj21500566/rl-policy-evaluation/assets/94588708/59824c60-144c-4bb1-8f3f-2a7a2b783bcf)

## RESULT:

Thus, a Python program is developed to evaluate the given policy.
