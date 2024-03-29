import numpy as np



def get_cost_rwd(info, rew,cost_t,env_name):
    safety_gym_env = ['Safexp-PointGoal1-v0', 'Safexp-CarGoal1-v0', 'Safexp-DoggoGoal1-v0', 'Safexp-CarButton1-v0',
                      'Safexp-CarPush1-v0', 'Safexp-PointPush1-v0', 'Safexp-PointButton1-v0', 'Safexp-DoggoButton1-v0']
    if env_name not in safety_gym_env:
        if 'y_velocity' not in info:
            curcost = np.abs(info['x_velocity'])
        else:
            curcost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)

    if env_name in ['HalfCheetah-v3', 'Hopper-v3']:
        curcost = np.abs(info['reward_ctrl'])
        rew = np.array([info['reward_run']])
    if env_name in ['Ant-v3','Humanoid-v3']:
        curcost = np.abs(info['reward_ctrl']) + np.abs(info['reward_contact'])
        rew = np.array([info['reward_run']])
    if env_name not in safety_gym_env:
        if env_name == 'Hopper-v3':
            if curcost * 1000 > cost_t:  # 1.0:
                curcost_ = 1.0
            else:
                curcost_ = 0.0
        else:
            if curcost > cost_t:  # 0.5: # change 0.5 to 0.2
                curcost_ = 1.0
            else:
                curcost_ = 0.0

    if env_name in safety_gym_env:
        curcost_ = info.get('cost', 0)

    return rew, curcost_
