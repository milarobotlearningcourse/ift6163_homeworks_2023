import os
import time

from hw1.roble.infrastructure.rl_trainer import RL_Trainer
from hw1.roble.agents.bc_agent import BCAgent
from hw1.roble.policies.loaded_gaussian_policy import LoadedGaussianPolicy
#test2s
class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################
        print ("params2: ", params)
        print ("params: ", params["alg"]['n_layers'])
        self.params = params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params, agent_class=BCAgent) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################
        ### Correcting for hydra logging folder. 
        print('Loading expert policy from...',  self.params["env"]['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy( self.params["env"]['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['alg']['n_iter'],
            initial_expertdata=self.params['env']['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['alg']['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )

import hydra, json
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config_hw1")
def my_main(cfg: DictConfig):
    
    returns = my_app(cfg)
    print ("returns: ", returns)
    
    
def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    params = vars(cfg)
    # print ("params: ", json.dumps(params, indent=4))
    print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if cfg.alg.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert cfg.alg.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert cfg.alg.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.logging.logdir = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(cfg)
    out = trainer.run_training_loop()

if __name__ == "__main__":
    my_main()
