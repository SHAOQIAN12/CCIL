# Imitating Cost Constrained Behaviors in Reinforcement Learning

The code is built on  OpenAI Baselines library, firstly need to install [baselines](https://github.com/openai/baselines), [mujoco](https://github.com/openai/mujoco-py) and [safety gym](https://github.com/openai/safety-gym).


Our implementation is in [/baselines/gail](/baselines/gail/). 

###  Download Expert data 
Download link https://drive.google.com/open?id=1U_-YDuWuDEI8e_f6Kp5cM-E3c9mEePTb&usp=drive_fs

Store expert data in `./baselines/gail/dataset/`


###  Example 

###  Run CCIL/CVAG
python run_mujoco.py --env_id Swimmer-v3 --expert_path .../gail/dataset/Swimmer-v3.npz --save_per_iter 500 --num_timesteps 1e7 --traj_limitation 10 --penalty 0.01 --batch_size=2000 --cost_t 1 -add_cost True --cost_method lagrangian

 
###  Run MALM
python run_mujoco_meta.py --env_id Swimmer-v3 --expert_path .../gail/dataset/Swimmer-v3.npz --save_per_iter 500 --num_timesteps 1e7 --traj_limitation 10 --penalty 0.01 --batch_size=2000 --cost_t 1




To cite this repository in publications:

    @inproceedings{shao2024imitating,
    title={Imitating Cost Constrained Behaviors in Reinforcement Learning},
    author={Shao, Qian and Varakantham, Pradeep and Cheng, Shih-Fen},
    booktitle={34th International Conference on Automated Planning and Scheduling},
    year={2024}
    }
