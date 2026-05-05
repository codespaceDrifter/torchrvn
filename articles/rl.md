# reinforcement learning  

(s , a, R(s) , s'): state, action, reward, next state  
return is the discounted sum of rewards from a state onward: G = R1 + γR2 + γ^(i-1)Ri  
policy maps states to actions to maximize return: π(s) = a  
Q(s,a) is the return assuming agent act optimally after action a
the optimal action gives max a Q(s,a)  


Q-learning iteratively updates a table of Q(s,a) values: start with zeros, take actions in environment, and on each step update Q(s,a) ← r + γ max_a' Q(s',a'); values propagate backward from terminal states over many episodes until convergence.


