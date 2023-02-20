import numpy as np
import copy

from hw3.roble.infrastructure.replay_buffer import ReplayBuffer
from hw3.roble.policies.MLP_policy import MLPPolicyStochastic
from hw3.roble.critics.sac_critic import SACCritic
from hw3.roble.agents.ddpg_agent import DDPGAgent

class SACAgent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.actor = MLPPolicyStochastic(
            self.agent_params['alg']['sac_entropy_coeff'],
            self.agent_params['alg']['ac_dim'],
            self.agent_params['alg']['ob_dim'],
            self.agent_params['alg']['n_layers'],
            self.agent_params['alg']['size'],
            discrete=self.agent_params['alg']['discrete'],
            learning_rate=self.agent_params['alg']['learning_rate'],
            nn_baseline=False,
        )

        self.q_fun = SACCritic(self.actor, 
                               agent_params, 
                               self.optimizer_spec)
        
    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO: Take the code from DDPG Agent and make sure the remove the exploration noise
        return