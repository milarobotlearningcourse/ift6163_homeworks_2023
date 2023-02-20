import numpy as np
import copy

from hw3.roble.infrastructure.replay_buffer import ReplayBuffer
from hw3.roble.policies.MLP_policy import MLPPolicyDeterministic
from hw3.roble.critics.td3_critic import TD3Critic
from hw3.roble.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.q_fun = TD3Critic(self.actor, 
                               agent_params, 
                               self.optimizer_spec)
        