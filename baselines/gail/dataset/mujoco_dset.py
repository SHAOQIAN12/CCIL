'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, accosts,costs,randomize):
        self.inputs = inputs
        self.labels = labels
        self.accosts = accosts
        self.cost_labels = costs
        assert len(self.inputs) == len(self.labels) == len(self.accosts) == len(self.cost_labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]
            self.accosts = self.accosts[idx, :]
            self.cost_labels = self.cost_labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels,self.accosts, self.cost_labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        costs = self.accosts[self.pointer:end, :]
        costs_labels = self.cost_labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels, costs, costs_labels



class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        traj_data = np.load(expert_path,allow_pickle=True)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        length_idx = range(0,traj_limitation)
        self.cost = []
        self.accucost = []
        self.cost_labels = []
        obs = []
        acs = []
        rwd = []
        cost = []
        accucost = []
        cost_labels= []
        for i in length_idx:
            obs.append(traj_data['obs'][i])
            acs.append(traj_data['acs'][i])
            rwd.append(traj_data['ep_rets'][i])
            cost.append(traj_data['costs'][i])
            accucost.append(np.expand_dims(traj_data['accucost'][i],axis=1))
            cost_idx_ = []
            for j in range(len(traj_data['accucost'][i])):
                if j == 0:
                    cost_idx_.append(traj_data['accucost'][i][j])
                else:
                    cost_idx_.append(traj_data['accucost'][i][j] - traj_data['accucost'][i][j - 1])
            cost_labels.append(np.expand_dims(cost_idx_,axis=1))
        obs = np.array(obs)
        acs = np.array(acs)
        accucost = np.array(accucost)
        cost_labels = np.array(cost_labels)

        if len(obs.shape) > 2:
            self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
            self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
            self.accucost = np.reshape(accucost, [-1, np.prod(accucost.shape[2:])])
            self.cost_labels = np.reshape(cost_labels, [-1, np.prod(cost_labels.shape[2:])])
        else:
            self.obs = np.vstack(obs)
            self.acs = np.vstack(acs)
            self.accucost = np.vstack(accucost)
            self.cost_labels = np.vstack(cost_labels)

        self.rets = np.array(rwd)
        self.cost = np.array(cost)
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs,self.accucost,self.cost_labels, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.accucost[:int(self.num_transition * train_fraction), :],
                              self.cost_labels[:int(self.num_transition * train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.accucost[int(self.num_transition*train_fraction):, :],
                            self.cost_labels[int(self.num_transition * train_fraction):, :],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %.2f" % self.avg_ret)
        logger.log("Returns: ", np.round(self.rets,2))
        logger.log("Std for returns: %.2f" % self.std_ret)
        logger.log('Average costs: %.2f' % np.mean(self.cost))
        logger.log('Costs: ' , np.round(self.cost,2))
        logger.log('Std for costs: %.2f' %np.std(np.array(self.cost)))

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="/.../")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    dset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    dset.plot()
    #test(args.expert_path, args.traj_limitation, args.plot)
