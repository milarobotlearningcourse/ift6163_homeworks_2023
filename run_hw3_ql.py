import os
import time
import sys
print(sys.path)
import hydra, json

from hw3.roble.agents.dqn_agent import DQNAgent
from hw3.roble.agents.ddpg_agent import DDPGAgent
from hw3.roble.agents.td3_agent import TD3Agent
from hw3.roble.agents.sac_agent import SACAgent
from hw3.roble.infrastructure.rl_trainer import RL_Trainer
from omegaconf import DictConfig, OmegaConf
from hw3.roble.infrastructure.dqn_utils import get_env_kwargs

class offpolicy_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        self.params = params
        self.params['alg']['batch_size_initial'] = self.params['alg']['batch_size']

        if self.params['alg']['rl_alg'] == 'dqn':
            agent = DQNAgent
        elif self.params['alg']['rl_alg'] == 'ddpg':
            agent = DDPGAgent
        elif self.params['alg']['rl_alg'] == 'td3':
            agent = TD3Agent    
        elif self.params['alg']['rl_alg'] == 'sac':
            agent = SACAgent
        else:
            print("Pick a rl_alg first")
            sys.exit()
        print(self.params)
        print(self.params['alg']['train_batch_size'])

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params , agent_class =  agent)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['alg']['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


@hydra.main(config_path="conf", config_name="config_hw3")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    # print ("params: ", json.dumps(params, indent=4))
    if cfg['env']['env_name']=='reacher-roble-v0':
        cfg['env']['ep_len']=200
    if cfg['env']['env_name']=='cheetah-roble-v0':
        cfg['env']['ep_len']=500
    if cfg['env']['env_name']=='obstacles-roble-v0':
        cfg['env']['ep_len']=100
        
    params = vars(cfg)
    # params.extend(env_args)
    for key, value in cfg.items():
        params[key] = value
    if cfg['env']['atari']:
        env_args = get_env_kwargs(cfg['env']['env_name'])
        for key, value in env_args.items():
            params[key] = value
    print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw3_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################
    # cfg = OmegaConf.merge(cfg, params)
    trainer = offpolicy_Trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    import os
    print("Command Dir:", os.getcwd())
    my_main()
