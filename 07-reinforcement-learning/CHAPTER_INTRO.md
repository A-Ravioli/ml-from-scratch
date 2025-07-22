# Chapter 7: Reinforcement Learning - Learning Approach Guide

## Overview
This chapter explores reinforcement learning - the paradigm where agents learn optimal behavior through interaction with environments. From classical dynamic programming to modern deep RL algorithms, you'll master the mathematical foundations and implement cutting-edge methods that enable autonomous decision-making.

## Prerequisites
- **Chapter 4**: Deep learning fundamentals (neural networks, optimization)
- **Chapter 5**: Neural architectures (CNNs, RNNs, attention for complex state/action spaces)
- **Chapter 0**: Probability theory, optimization theory, dynamic programming
- **Mathematical Background**: Markov processes, game theory basics, control theory

## Learning Philosophy
Reinforcement learning combines **control theory**, **statistics**, and **optimization** in a unique framework where learning happens through interaction. This chapter emphasizes:
1. **Mathematical Rigor**: Deep understanding of MDP theory, optimality conditions, and convergence guarantees
2. **Algorithmic Mastery**: Implement RL algorithms from first principles with proper theoretical grounding
3. **Practical Understanding**: Know when and why different RL approaches work in practice
4. **Modern Connections**: Bridge classical RL theory with deep learning and contemporary applications

## The RL Algorithm Taxonomy

```
Setting         → Algorithm Family    → Key Innovation
─────────────────────────────────────────────────────
Tabular         → DP/Monte Carlo     → Exact solutions for small MDPs
Value-Based     → TD Learning        → Bootstrapping and temporal differences  
Policy-Based    → Policy Gradients   → Direct policy optimization
Actor-Critic    → Hybrid Methods     → Combine value estimation and policy optimization
Model-Based     → Planning Methods   → Learn environment model for planning
Multi-Agent     → Game Theory        → Strategic interaction between agents
```

## Section-by-Section Mastery Plan

### 01. Foundations
**Core Question**: What is the mathematical framework that governs sequential decision-making?

#### Week 1: Markov Decision Processes
**Mathematical Foundation**:

**MDP Definition**: Tuple (S, A, P, R, γ)
- **S**: State space
- **A**: Action space  
- **P**: Transition function P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor

**Key Theoretical Results**:
```python
class MDP:
    """Markov Decision Process implementation"""
    def __init__(self, states, actions, transitions, rewards, discount=0.9):
        self.states = states
        self.actions = actions
        self.P = transitions  # P[s][a] = [(prob, next_state, reward, done), ...]
        self.R = rewards      # R[s][a] = expected reward
        self.gamma = discount
    
    def bellman_equation_v(self, policy, V):
        """Bellman equation for state-value function"""
        V_new = np.zeros_like(V)
        for s in range(len(self.states)):
            v = 0
            for a in range(len(self.actions)):
                action_prob = policy[s][a]
                for prob, next_state, reward, done in self.P[s][a]:
                    v += action_prob * prob * (reward + self.gamma * V[next_state] * (1 - done))
            V_new[s] = v
        return V_new
    
    def bellman_optimality_equation(self, V):
        """Bellman optimality equation"""
        V_new = np.zeros_like(V)
        policy = np.zeros((len(self.states), len(self.actions)))
        
        for s in range(len(self.states)):
            action_values = []
            for a in range(len(self.actions)):
                q_value = 0
                for prob, next_state, reward, done in self.P[s][a]:
                    q_value += prob * (reward + self.gamma * V[next_state] * (1 - done))
                action_values.append(q_value)
            
            best_action = np.argmax(action_values)
            V_new[s] = action_values[best_action]
            policy[s][best_action] = 1.0
            
        return V_new, policy
```

**Optimality Theory**:
- **Principle of Optimality**: Optimal substructure in sequential decisions
- **Bellman Equations**: Recursive relationship for optimal value functions
- **Policy Iteration vs Value Iteration**: Two fundamental solution approaches

#### Week 2: Dynamic Programming Solutions
**Classical Algorithms**:

**Policy Iteration**:
```python
class PolicyIteration:
    """Policy iteration for solving MDPs"""
    def __init__(self, mdp):
        self.mdp = mdp
        self.num_states = len(mdp.states)
        self.num_actions = len(mdp.actions)
    
    def policy_evaluation(self, policy, theta=1e-6):
        """Evaluate a given policy"""
        V = np.zeros(self.num_states)
        
        while True:
            delta = 0
            for s in range(self.num_states):
                v = V[s]
                V[s] = sum(policy[s][a] * self.expected_return(s, a, V) 
                          for a in range(self.num_actions))
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        return V
    
    def policy_improvement(self, V):
        """Improve policy using current value function"""
        policy = np.zeros((self.num_states, self.num_actions))
        policy_stable = True
        
        for s in range(self.num_states):
            old_action = np.argmax(policy[s])
            
            # Find best action
            action_values = [self.expected_return(s, a, V) 
                           for a in range(self.num_actions)]
            best_action = np.argmax(action_values)
            
            # Update policy (deterministic)
            policy[s] = 0
            policy[s][best_action] = 1
            
            if best_action != old_action:
                policy_stable = False
        
        return policy, policy_stable
    
    def solve(self):
        """Solve MDP using policy iteration"""
        # Initialize random policy
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        
        while True:
            # Policy evaluation
            V = self.policy_evaluation(policy)
            
            # Policy improvement
            policy, policy_stable = self.policy_improvement(V)
            
            if policy_stable:
                break
        
        return policy, V
```

**Value Iteration**:
```python
class ValueIteration:
    """Value iteration for solving MDPs"""
    def __init__(self, mdp):
        self.mdp = mdp
        self.num_states = len(mdp.states)
        self.num_actions = len(mdp.actions)
    
    def solve(self, theta=1e-6):
        """Solve MDP using value iteration"""
        V = np.zeros(self.num_states)
        
        while True:
            delta = 0
            V_new = np.zeros(self.num_states)
            
            for s in range(self.num_states):
                # Compute value for each action
                action_values = [self.expected_return(s, a, V) 
                               for a in range(self.num_actions)]
                
                V_new[s] = max(action_values)
                delta = max(delta, abs(V[s] - V_new[s]))
            
            V = V_new
            
            if delta < theta:
                break
        
        # Extract optimal policy
        policy = self.extract_policy(V)
        
        return policy, V
    
    def extract_policy(self, V):
        """Extract optimal policy from value function"""
        policy = np.zeros((self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            action_values = [self.expected_return(s, a, V) 
                           for a in range(self.num_actions)]
            best_action = np.argmax(action_values)
            policy[s][best_action] = 1.0
        
        return policy
```

### 02. Tabular Methods
**Core Question**: How do we learn when we don't know the MDP transition and reward functions?

#### Week 3: Monte Carlo Methods
**Sample-Based Learning**:

**First-Visit Monte Carlo**:
```python
class MonteCarloControl:
    """Monte Carlo control for policy optimization"""
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize Q-values and policy
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(list)
        self.policy = self.make_epsilon_greedy_policy()
    
    def make_epsilon_greedy_policy(self):
        """Create epsilon-greedy policy"""
        def policy_fn(state):
            A = np.ones(self.env.action_space.n, dtype=float) * self.epsilon / self.env.action_space.n
            best_action = max(self.Q[state].keys(), key=lambda a: self.Q[state][a]) if self.Q[state] else 0
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn
    
    def generate_episode(self):
        """Generate episode using current policy"""
        episode = []
        state = self.env.reset()
        
        while True:
            action_probs = self.policy(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, done, _ = self.env.step(action)
            
            episode.append((state, action, reward))
            
            if done:
                break
            
            state = next_state
        
        return episode
    
    def update_q_values(self, episode):
        """Update Q-values using Monte Carlo returns"""
        G = 0
        visited_sa_pairs = set()
        
        # Process episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # First-visit MC
            if (state, action) not in visited_sa_pairs:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                visited_sa_pairs.add((state, action))
    
    def train(self, num_episodes=10000):
        """Train using Monte Carlo control"""
        for episode_num in range(num_episodes):
            episode = self.generate_episode()
            self.update_q_values(episode)
            
            # Update policy
            self.policy = self.make_epsilon_greedy_policy()
        
        return self.Q, self.policy
```

#### Week 4: Temporal Difference Learning
**Bootstrapping Methods**:

**Q-Learning (Off-Policy TD Control)**:
```python
class QLearning:
    """Q-Learning algorithm"""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table
        self.Q = defaultdict(lambda: defaultdict(float))
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state not in self.Q:
                return self.env.action_space.sample()
            return max(self.Q[state].keys(), key=lambda a: self.Q[state][a])
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule"""
        if next_state not in self.Q or not self.Q[next_state]:
            max_next_q = 0
        else:
            max_next_q = max(self.Q[next_state].values())
        
        # Q-learning update
        target = reward + (self.gamma * max_next_q * (1 - done))
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def train(self, num_episodes=5000):
        """Training loop"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Decay exploration
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return episode_rewards
```

**SARSA (On-Policy TD Control)**:
```python
class SARSA:
    """SARSA algorithm (on-policy)"""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state not in self.Q or not self.Q[state]:
                return self.env.action_space.sample()
            return max(self.Q[state].keys(), key=lambda a: self.Q[state][a])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update rule"""
        if next_state not in self.Q:
            next_q = 0
        else:
            next_q = self.Q[next_state][next_action]
        
        # SARSA update
        target = reward + (self.gamma * next_q * (1 - done))
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def train(self, num_episodes=5000):
        """Training loop"""
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.get_action(state)
            
            while True:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.get_action(next_state) if not done else None
                
                self.update(state, action, reward, next_state, next_action, done)
                
                if done:
                    break
                
                state, action = next_state, next_action
```

### 03. Function Approximation
**Core Question**: How do we scale RL to large state spaces using neural networks?

#### Week 5: Deep Q-Networks (DQN)
**Value Function Approximation**:

**Basic DQN Implementation**:
```python
class DQN:
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Neural networks
        self.q_network = self.build_network(state_dim, action_dim)
        self.target_network = self.build_network(state_dim, action_dim)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update_freq = 1000
        self.train_step = 0
    
    def build_network(self, input_dim, output_dim):
        """Build Q-network"""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self, batch_size=32):
        """Single training step"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
```

**Advanced DQN Variants**:

**Double DQN**:
```python
class DoubleDQN(DQN):
    """Double DQN to address overestimation bias"""
    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select action, target network to evaluate
        next_actions = self.q_network(next_states).argmax(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and optimize
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
```

#### Week 6: Advanced Value-Based Methods
**Distributional RL and Rainbow DQN**:

**Categorical DQN (C51)**:
```python
class CategoricalDQN:
    """Categorical DQN for distributional RL"""
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10):
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms)
        
        # Build network that outputs distributions
        self.q_network = self.build_distributional_network(state_dim, action_dim, num_atoms)
        self.target_network = self.build_distributional_network(state_dim, action_dim, num_atoms)
    
    def build_distributional_network(self, input_dim, action_dim, num_atoms):
        """Build network that outputs probability distributions"""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * num_atoms)
        )
    
    def get_action(self, state):
        """Select action using expected values"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist = self.q_network(state_tensor)
        dist = F.softmax(dist.view(-1, self.action_dim, self.num_atoms), dim=2)
        
        # Compute expected values
        q_values = (dist * self.support).sum(2)
        return q_values.argmax().item()
    
    def train_step(self, batch_size=32):
        """Distributional Bellman update"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current distributions
        current_dist = self.q_network(states)
        current_dist = F.softmax(current_dist.view(batch_size, self.action_dim, self.num_atoms), dim=2)
        current_dist = current_dist[range(batch_size), actions]
        
        # Target distributions
        next_dist = self.target_network(next_states)
        next_dist = F.softmax(next_dist.view(batch_size, self.action_dim, self.num_atoms), dim=2)
        
        # Select best actions using main network
        next_q_values = (next_dist * self.support).sum(2)
        next_actions = next_q_values.argmax(1)
        next_dist = next_dist[range(batch_size), next_actions]
        
        # Compute target support
        target_support = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
        target_support = torch.clamp(target_support, self.v_min, self.v_max)
        
        # Project onto support
        target_dist = self.project_distribution(target_support, next_dist)
        
        # Cross-entropy loss
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(1).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 04. Policy Gradient Methods
**Core Question**: How can we directly optimize policies without value function estimation?

#### Week 7: REINFORCE and Actor-Critic
**Policy Gradient Theory**:

**REINFORCE Algorithm**:
```python
class REINFORCE:
    """REINFORCE policy gradient algorithm"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Policy network
        self.policy_network = self.build_policy_network(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # Storage for episode
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def build_policy_network(self, input_dim, output_dim):
        """Build policy network"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def get_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store for training
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(action_dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def train_episode(self):
        """Train on completed episode"""
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # Normalize returns (optional, often helps)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        
        return loss.item()
```

**Actor-Critic Architecture**:
```python
class ActorCritic:
    """Actor-Critic with shared network"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Shared network with actor and critic heads
        self.network = self.build_network(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def build_network(self, input_dim, action_dim):
        """Build actor-critic network"""
        class ActorCriticNet(nn.Module):
            def __init__(self, input_dim, action_dim):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
                
                # Actor head (policy)
                self.actor = nn.Sequential(
                    nn.Linear(128, action_dim),
                    nn.Softmax(dim=-1)
                )
                
                # Critic head (value function)
                self.critic = nn.Linear(128, 1)
            
            def forward(self, x):
                shared_features = self.shared(x)
                action_probs = self.actor(shared_features)
                state_value = self.critic(shared_features)
                return action_probs, state_value
        
        return ActorCriticNet(input_dim, action_dim)
    
    def get_action(self, state):
        """Sample action and get value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.network(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action), state_value
    
    def train_step(self, state, action_log_prob, reward, next_state, done):
        """Single step actor-critic update"""
        # Current state value
        _, current_value = self.network(torch.FloatTensor(state).unsqueeze(0))
        
        # Next state value (0 if terminal)
        if done:
            next_value = 0
        else:
            _, next_value = self.network(torch.FloatTensor(next_state).unsqueeze(0))
            next_value = next_value.detach()
        
        # TD error (advantage)
        td_target = reward + self.gamma * next_value
        advantage = td_target - current_value
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -action_log_prob * advantage.detach()
        
        # Critic loss (value function)
        critic_loss = advantage.pow(2)
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

#### Week 8: Advanced Policy Methods
**PPO and A2C/A3C**:

**Proximal Policy Optimization (PPO)**:
```python
class PPO:
    """Proximal Policy Optimization"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        
        # Actor-critic network
        self.network = self.build_network(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # Storage for trajectories
        self.reset_storage()
    
    def reset_storage(self):
        """Reset trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_action(self, state):
        """Sample action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(action_dist.log_prob(action).item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_transition(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self, next_state=None):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        # Get final value if episode didn't end
        if next_state is not None and not self.dones[-1]:
            with torch.no_grad():
                _, final_value = self.network(torch.FloatTensor(next_state).unsqueeze(0))
                self.values.append(final_value.item())
        else:
            self.values.append(0)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * self.values[t + 1] * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * 0.95 * (1 - self.dones[t]) * gae  # λ = 0.95 for GAE
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)
    
    def train_step(self, next_state=None):
        """PPO training step"""
        if len(self.rewards) == 0:
            return
        
        # Compute advantages
        returns, advantages = self.compute_advantages(next_state)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # PPO epochs
        for _ in range(4):  # Multiple epochs
            # Forward pass
            action_probs, values = self.network(states)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # New log probabilities
            new_log_probs = action_dist.log_prob(actions)
            
            # Probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -action_dist.entropy().mean()
            
            # Total loss
            total_loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Reset storage
        self.reset_storage()
        
        return total_loss.item()
```

### 05. Model-Based RL
**Core Question**: How can we learn environment models to improve sample efficiency?

#### Week 9: Planning with Learned Models
**Model-Based Planning Approaches**:

**Dyna-Q Integration**:
```python
class DynaQ:
    """Dyna-Q: Integrating planning and learning"""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, planning_steps=50):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        # Q-values
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Model: (state, action) -> (reward, next_state)
        self.model = {}
        
        # Track visited state-action pairs
        self.visited_sa = set()
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state not in self.Q:
                return self.env.action_space.sample()
            return max(self.Q[state].keys(), key=lambda a: self.Q[state][a])
    
    def direct_rl_update(self, state, action, reward, next_state, done):
        """Direct RL update (Q-learning)"""
        if next_state not in self.Q or not self.Q[next_state]:
            max_next_q = 0
        else:
            max_next_q = max(self.Q[next_state].values())
        
        target = reward + (self.gamma * max_next_q * (1 - done))
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
    
    def update_model(self, state, action, reward, next_state):
        """Update environment model"""
        self.model[(state, action)] = (reward, next_state)
        self.visited_sa.add((state, action))
    
    def planning_update(self):
        """Perform planning updates using learned model"""
        for _ in range(self.planning_steps):
            if not self.visited_sa:
                break
            
            # Random previously experienced state-action pair
            state, action = random.choice(list(self.visited_sa))
            reward, next_state = self.model[(state, action)]
            
            # Simulated experience update
            self.direct_rl_update(state, action, reward, next_state, False)
    
    def train_episode(self):
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        
        while True:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            # Direct RL update
            self.direct_rl_update(state, action, reward, next_state, done)
            
            # Model update
            self.update_model(state, action, reward, next_state)
            
            # Planning updates
            self.planning_update()
            
            episode_reward += reward
            
            if done:
                break
            
            state = next_state
        
        return episode_reward
```

**Model-Based Policy Optimization (MBPO)**:
```python
class MBPO:
    """Model-Based Policy Optimization"""
    def __init__(self, state_dim, action_dim, model_ensemble_size=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Environment model ensemble
        self.model_ensemble = [
            self.build_dynamics_model(state_dim, action_dim) 
            for _ in range(model_ensemble_size)
        ]
        
        # Policy (SAC in this case)
        self.policy = SAC(state_dim, action_dim)
        
        # Model buffer for real data
        self.model_buffer = ReplayBuffer(capacity=1000000)
        
        # Policy buffer (includes synthetic data)
        self.policy_buffer = ReplayBuffer(capacity=1000000)
    
    def build_dynamics_model(self, state_dim, action_dim):
        """Build dynamics model: (s, a) -> (s', r, done)"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim + 2)  # next_state + reward + done
        )
    
    def train_model(self, batch_size=256):
        """Train dynamics model on real data"""
        if len(self.model_buffer) < batch_size:
            return
        
        # Sample real transitions
        states, actions, rewards, next_states, dones = self.model_buffer.sample(batch_size)
        
        inputs = torch.cat([states, actions], dim=1)
        targets = torch.cat([next_states, rewards.unsqueeze(1), dones.unsqueeze(1)], dim=1)
        
        # Train each model in ensemble
        for model in self.model_ensemble:
            predictions = model(inputs)
            loss = F.mse_loss(predictions, targets)
            
            # Optimize model
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
    
    def generate_synthetic_data(self, num_steps=1000):
        """Generate synthetic rollouts using learned models"""
        # Sample starting states from real data
        if len(self.model_buffer) == 0:
            return
        
        states, _, _, _, _ = self.model_buffer.sample(100)
        
        for _ in range(num_steps):
            # Sample actions from current policy
            actions = self.policy.get_action(states, deterministic=False)
            
            # Predict next states using random model from ensemble
            model = random.choice(self.model_ensemble)
            inputs = torch.cat([states, actions], dim=1)
            
            with torch.no_grad():
                predictions = model(inputs)
                next_states = predictions[:, :self.state_dim]
                rewards = predictions[:, self.state_dim]
                dones = predictions[:, self.state_dim + 1] > 0.5
            
            # Add to policy buffer
            for i in range(len(states)):
                self.policy_buffer.add(
                    states[i].numpy(),
                    actions[i].numpy(), 
                    rewards[i].item(),
                    next_states[i].numpy(),
                    dones[i].item()
                )
            
            # Continue rollout from next states (unless done)
            states = next_states[~dones]
            if len(states) == 0:
                break
    
    def train_policy(self):
        """Train policy using both real and synthetic data"""
        return self.policy.train_step(self.policy_buffer)
```

### 06. Multi-Agent RL
**Core Question**: How do we handle strategic interaction between multiple learning agents?

#### Week 10: Game Theory and Multi-Agent Learning
**Nash Equilibria and Solution Concepts**:

**Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**:
```python
class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient"""
    def __init__(self, num_agents, state_dims, action_dims, lr_actor=1e-4, lr_critic=1e-3):
        self.num_agents = num_agents
        
        # Each agent has its own actor and critic
        self.actors = [self.build_actor(state_dims[i], action_dims[i]) for i in range(num_agents)]
        self.critics = [self.build_critic(sum(state_dims), sum(action_dims)) for i in range(num_agents)]
        
        # Target networks
        self.target_actors = [self.build_actor(state_dims[i], action_dims[i]) for i in range(num_agents)]
        self.target_critics = [self.build_critic(sum(state_dims), sum(action_dims)) for i in range(num_agents)]
        
        # Optimizers
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=lr_critic) for critic in self.critics]
        
        # Replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(capacity=100000)
    
    def get_actions(self, states, add_noise=True):
        """Get actions for all agents"""
        actions = []
        for i, (actor, state) in enumerate(zip(self.actors, states)):
            action = actor(torch.FloatTensor(state).unsqueeze(0))
            if add_noise:
                action += torch.normal(0, 0.1, size=action.shape)
            actions.append(action.squeeze(0).detach().numpy())
        return actions
    
    def train_step(self, batch_size=1024):
        """Train all agents"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
            self.replay_buffer.sample(batch_size)
        
        # Train each agent
        for agent_idx in range(self.num_agents):
            # Critic update
            # Get next actions from target actors
            next_actions = []
            for i, (target_actor, next_state) in enumerate(zip(self.target_actors, next_states_batch)):
                next_action = target_actor(next_state[:, i])
                next_actions.append(next_action)
            next_actions = torch.cat(next_actions, dim=1)
            
            # Q-target
            next_states_all = torch.cat(next_states_batch, dim=1)
            target_q = self.target_critics[agent_idx](next_states_all, next_actions)
            y = rewards_batch[:, agent_idx].unsqueeze(1) + 0.99 * target_q * (1 - dones_batch[:, agent_idx].unsqueeze(1))
            
            # Current Q
            states_all = torch.cat(states_batch, dim=1)
            actions_all = torch.cat(actions_batch, dim=1)
            current_q = self.critics[agent_idx](states_all, actions_all)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q, y.detach())
            
            # Update critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()
            
            # Actor update
            # Get actions from current actors
            current_actions = []
            for i, (actor, state) in enumerate(zip(self.actors, states_batch)):
                if i == agent_idx:
                    action = actor(state[:, i])
                else:
                    action = actions_batch[i].detach()
                current_actions.append(action)
            current_actions = torch.cat(current_actions, dim=1)
            
            # Actor loss (maximize Q)
            actor_loss = -self.critics[agent_idx](states_all, current_actions).mean()
            
            # Update actor
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()
        
        # Soft update target networks
        self.soft_update_targets()
    
    def soft_update_targets(self, tau=0.01):
        """Soft update target networks"""
        for i in range(self.num_agents):
            # Update target actor
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            # Update target critic
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## Integration and Advanced Topics

### Week 11-12: Modern RL Frontiers
**Cutting-Edge Developments**:

1. **Offline RL**: Learning from fixed datasets
2. **Meta-RL**: Learning to learn across tasks
3. **Hierarchical RL**: Temporal abstractions and options
4. **Safe RL**: Constraint satisfaction and risk-aware learning
5. **Continual RL**: Learning without forgetting in non-stationary environments

## Assessment and Mastery Framework

### Theoretical Mastery Checkpoints

**Week 4**:
- [ ] Understands MDP formulation and Bellman equations
- [ ] Can derive policy iteration and value iteration algorithms
- [ ] Masters Monte Carlo and temporal difference learning theory

**Week 8**:
- [ ] Understands policy gradient theorem and its derivations
- [ ] Can implement and analyze actor-critic methods
- [ ] Masters advanced policy optimization (PPO, TRPO)

**Week 12**:
- [ ] Understands model-based RL and planning integration
- [ ] Masters multi-agent RL and game-theoretic concepts
- [ ] Can analyze convergence and sample complexity of RL algorithms

### Implementation Mastery Checkpoints

**Week 6**:
- [ ] Complete tabular RL implementations (Q-learning, SARSA, MC)
- [ ] Working DQN with experience replay and target networks
- [ ] Proper exploration strategies and hyperparameter tuning

**Week 10**:
- [ ] Advanced value-based methods (Double DQN, Distributional RL)
- [ ] Policy gradient methods (REINFORCE, A2C, PPO)
- [ ] Actor-critic architectures with proper advantage estimation

**Week 12**:
- [ ] Model-based RL with learned dynamics
- [ ] Multi-agent RL algorithms
- [ ] Integration of different RL paradigms

### Integration Mastery Checkpoints
- [ ] Can select appropriate RL algorithms for different problem types
- [ ] Understands exploration-exploitation tradeoffs deeply
- [ ] Can debug and optimize RL training procedures
- [ ] Can implement RL algorithms from research papers

## Time Investment Strategy

### Intensive Track (10-12 weeks full-time)
- **Weeks 1-3**: MDP theory and tabular methods
- **Weeks 4-6**: Function approximation and deep RL
- **Weeks 7-9**: Policy methods and advanced algorithms
- **Weeks 10-12**: Model-based and multi-agent RL

### Standard Track (16-20 weeks part-time)
- **Weeks 1-5**: Solid foundations in MDP theory and classical methods
- **Weeks 6-12**: Deep RL with proper theoretical understanding
- **Weeks 13-16**: Advanced topics and integration projects
- **Weeks 17-20**: Cutting-edge methods and applications

### Research Track (20+ weeks)
- Include implementation of recent RL advances
- Original research projects in RL theory or applications
- Deep dives into specialized areas (safe RL, meta-RL, etc.)

## Integration with ML-from-Scratch Journey

### Applications and Impact
- **Robotics**: Control and manipulation tasks
- **Game Playing**: Strategic decision making
- **Autonomous Systems**: Self-driving cars, drones
- **Resource Management**: Cloud computing, finance
- **Healthcare**: Treatment optimization, drug discovery

### Advanced Connections
- **Generative Models**: RL for training GANs, fine-tuning language models
- **Meta-Learning**: Learning to learn across tasks and domains
- **Continual Learning**: Learning without catastrophic forgetting

## Success Metrics

By the end of this chapter, you should:
- **Understand the mathematical foundations** of sequential decision making
- **Implement any RL algorithm** from mathematical descriptions in papers
- **Select appropriate methods** for different types of RL problems
- **Debug and optimize** RL training procedures effectively
- **Contribute to RL research** by extending existing methods or developing novel approaches

Remember: Reinforcement learning represents the **decision-making frontier** of AI, where agents learn to act optimally through experience. Master these techniques to build systems that can learn, adapt, and optimize their behavior in complex, dynamic environments.