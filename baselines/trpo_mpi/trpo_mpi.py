from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from contextlib import contextmanager
from baselines.gail import mlp_policy
from baselines.gail.mpi_tf import MpiAdamOptimizer

# add by SQ
from baselines.common.get_cost_rwd import get_cost_rwd_ijcai,get_cost_rwd_iqlearn

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
import torch

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('There exists the folder.')


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_cost = 0
    ep_rets = []
    ep_lens = []
    ep_costs = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_costs":ep_costs}
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_costs = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        #cur_ep_cost += abs(ob[0][1]) + 0.01 * np.square(ac[0]).sum()
        ob, rew, new, info = env.step(ac)
        if 'y_velocity' not in info:
            curcost= np.abs(info[0]['x_velocity'])
        else:
            curcost = np.sqrt(info[0]['x_velocity'] ** 2 + info[0]['y_velocity'] ** 2)
        if curcost > 1.0:
            curcost_ = 1.0
        else:
            curcost_ = 0.0
        cur_ep_cost += curcost_

        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_costs.append(cur_ep_cost)

            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_cost = 0
            ob = env.reset()
        t += 1


def traj_segment_generator_cost(pi, env, horizon, env_id,cost_t,stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_cost = 0
    ep_rets = []
    ep_lens = []
    ep_costs = []

    # Initialize history arrays
    obs = np.array([ob[0] for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    cpreds = np.zeros(horizon, 'float32')
    accucosts = np.zeros(horizon, 'float32')
    costs = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred,cpred = pi.act(stochastic,ob)   # here needs to add cost value
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "cpred":cpreds, "accucost":accucosts,"cost":costs,
                   "new" : news,"ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                   "nextcpred": cpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens,"ep_costs":ep_costs}
            _, vpred, cpred = pi.act(stochastic,ob) # here needs to add cost value
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_costs = []
        i = t % horizon
        obs[i] = ob[0]
        vpreds[i] = vpred
        cpreds[i] = cpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        #curcost = abs(ob[0][1]) + 0.01 * np.square(ac).sum()
        #cur_ep_cost += curcost
        ob, rew, new, info = env.step(ac)

        '''
        if 'y_velocity' not in info:
            curcost= np.abs(info[0]['x_velocity'])
        else:
            curcost = np.sqrt(info[0]['x_velocity'] ** 2 + info[0]['y_velocity'] ** 2)
        if env_id in ['HalfCheetah-v3','Hopper-v3']:
            curcost = np.abs(info[0]['reward_ctrl'])

        if env_id == 'Humanoid-v3':
            curcost = np.abs(info[0]['reward_quadctrl']) + np.abs(info[0]['reward_impact'])

        if env_id == 'Humanoid-v3':
            if curcost > cost_t:#0.2:
                curcost_ = 1.0
            else:
                curcost_ = 0.0
        elif env_id == 'Hopper-v3':
            if curcost*1000 > cost_t:#1.0:
                curcost_ = 1.0
            else:
                curcost_ = 0.0
        else:
            if curcost > cost_t:#0.5: # change from 0.5 to 0.2,针对halfcheetah
                curcost_ = 1.0
            else:
                curcost_ = 0.0

        cur_ep_cost += curcost_
        if env_id in ['HalfCheetah-v3','Hopper-v3']:
            rew = info[0]['reward_run']

        if env_id == 'Humanoid-v3':
            rew = info[0]['reward_linvel'] + info[0]['reward_alive']
        '''
        rew,curcost_ = get_cost_rwd_iqlearn(info[0],rew,cost_t,env_id)
        cur_ep_cost += curcost_

        rews[i] = rew

        costs[i] = curcost_
        accucosts[i] = cur_ep_cost

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_costs.append(cur_ep_cost)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_cost = 0
            ob = env.reset()
        t += 1


def traj_segment_generator_cost_r(pi, env, horizon, stochastic):

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0

    ob = env.reset()
    start_flag = 1

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_cost = 0
    ep_rets = []
    ep_lens = []
    ep_costs = []


    # Initialize history arrays
    obs = np.array([ob[0] for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    cpreds = np.zeros(horizon,'float32')
    cpreds_r = np.zeros(horizon, 'float32')
    accucosts = np.zeros(horizon, 'float32')
    #violation = np.zeros(horizon, 'float32')
    costs = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    start_news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, cpred, cpred_r = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            if start_news[0] == 1:
                cpred_r = 0
            else:
                cpred_r = c_review
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "cpred":cpreds,"cpredr":cpreds_r, "accucost":accucosts,"cost":costs,
                  "new": news,"snew":start_news,"ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),"nextcpred": cpred * (1 - new),
                   "nextcpredr": cpred_r,"ep_rets": ep_rets, "ep_lens": ep_lens, "ep_costs":ep_costs}
            c_review = cpreds_r[-1]
            _, vpred,cpred ,cpred_r= pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_costs= []
        i = t % horizon
        if start_flag==1:
            start_news[i] = 1
        else:
            start_news[i] = 0
        obs[i] = ob[0]
        vpreds[i] = vpred
        cpreds[i] = cpred
        cpreds_r[i] = cpred_r

        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac


        ob, rew, new, info = env.step(ac)
        start_flag = 0
        if 'y_velocity' not in info:
            curcost = np.abs(info[0]['x_velocity'])
        else:
            curcost = np.sqrt(info[0]['x_velocity'] ** 2 + info[0]['y_velocity'] ** 2)
        if curcost > 1.0:
            curcost_ = 1.0
        else:
            curcost_ = 0.0
        cur_ep_cost += curcost_
        rews[i] = rew

        costs[i] = curcost_
        accucosts[i] = cur_ep_cost

        cur_ep_ret += rew

        cur_ep_len += 1

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_costs.append(cur_ep_cost)

            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_cost = 0
            ob = env.reset()
            start_flag = 1
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def add_ctarg_and_cadv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    cpred = np.append(seg["cpred"], seg["nextcpred"])
    T = len(seg["cost"])
    seg["cadv"] = gaelam = np.empty(T, 'float32')
    cost = seg["cost"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = cost[t] + gamma * cpred[t+1] * nonterminal - cpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamcostret"] = seg["cadv"] + seg["cpred"]   #tdlamret = returns


def add_ctarg_and_cadv_r(seg, gamma, lam):
    new = np.append(seg["snew"],0)   # last element is only used for last vtarg, but we already zeroed it if last new = 1
    cpred = np.append(seg["cpredr"],seg["nextcpredr"])
    T = len(seg["cost"])
    seg["cadvr"] = gaelam = np.empty(T, 'float32')
    cost = seg["cost"]
    lastgaelam = 0
    for t in range(T):
        nonterminal = 1-new[t-1]
        delta = cost[t] + gamma * cpred[t-1] * nonterminal - cpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamcostretr"] = seg["cadvr"] + seg["cpredr"]   #tdlamret = returns

def check_violation(seg,max_cost):
    violation = np.ones(len(seg['new']), 'float32')
    end_idx = []
    for i in range(len(seg['new'])):
        if i != 0 and seg['new'][i] == 1:
            end_idx.append(i)
    if len(end_idx) == 0:
        if seg['accucost'][-1] > max_cost:
            violation = np.zeros(len(seg['new']), 'float32')
            return  violation
    for i in range(len(end_idx)):
        if seg['accucost'][end_idx[i]-1] >max_cost:
            if i == 0:
                violation[0:end_idx[i]] = [0 for j in range(end_idx[i])]
            else:
                violation[end_idx[i-1]:end_idx[i]] = [0 for j in range(end_idx[i-1],end_idx[i])]

    if seg['accucost'][len(seg['new'])-1] > max_cost:
        violation[end_idx[-1]:len(seg['new'])] = [0 for j in range(end_idx[-1],len(seg['new']))]
    vio = 0
    for i in violation:
        if i == 0:
            vio+=1
    violation_rate = round(vio/len(violation),2)
    return violation, violation_rate

def learn(*,
        network,
        env,
        total_timesteps,
        env_id,
        timesteps_per_batch=2000,#1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))


    policy = build_policy(env, network, value_network='copy', **network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    U.initialize()
    if load_path is not None:
        pi.load(load_path)

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    costbuffer = deque(maxlen=40) # rolling buffer for episode costs

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'
    result = {}
    result['EpLenMean'] = []
    result['EpRewMean'] = []
    result['EpCostMean'] = []
    result_file = '/home/shaoqian/CMDP/baselines/baselines/models/trpo_mpi/' + env_id + '/basic/'
    mkdir(result_file)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            torch.save(result, os.path.join(result_file, 'results_dict_' + str(iters_so_far) + '.pt'))
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"],seg["ep_costs"] ) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews, tra_costs = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        costbuffer.extend(tra_costs)


        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpCostMean", np.mean(costbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        result['EpLenMean'].append(np.mean(lenbuffer))
        result['EpRewMean'].append(np.mean(rewbuffer))
        result['EpCostMean'].append(np.mean(costbuffer))

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)



        if rank==0:
            logger.dump_tabular()

        if iters_so_far % 1000 == 0:
            torch.save(result, os.path.join(result_file, 'results_dict_' + str(iters_so_far) + '.pt'))
    return pi



def learn_cost(*,
        network,
        env,
        total_timesteps,
        max_cost,  # add maximum cost
        env_id,
        cost_t,
        #cost_method,
        #penalty,
        timesteps_per_batch=2000,#1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy_cost(name=name, ob_space=ob_space, ac_space=ac_space,
                                         reuse=reuse, hid_size=100, num_hid_layers=2)

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    # ADD for cost
    cost_atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Cost target advantage function
    cost_ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical cost

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    cur_cost_ph = tf.placeholder(tf.float32, shape=())

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))  # (v_prdict-return）^2

    # add for cost
    cost_vferr = tf.reduce_mean(tf.square(pi.cost_vpred - cost_ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus

    cost_method = network_kwargs['method']
    if cost_method == 'lagrangian':
        penalty = network_kwargs['penalty']
    if cost_method == 'lagrangian':
        learn_penalty = True
        penalty_param_loss = True
        objective_penalized = True
    else:
        learn_penalty = False
        penalty_param_loss = False
        objective_penalized = False
    if cost_method == 'lagrangian':
        penalty_lr = 0.05
        penalty_init = penalty

    if cost_method == 'lagrangian':
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init) - 1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                            initializer=float(param_init),
                                            trainable=learn_penalty,
                                            dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty_ = tf.nn.softplus(penalty_param)

    if learn_penalty:
        if penalty_param_loss:
            penalty_loss = -penalty_param * (cur_cost_ph - max_cost)
            #penalty_loss = -penalty_param * (cur_cost_ph)
        else:
            penalty_loss = -penalty_ * (cur_cost_ph - max_cost)
        train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cost_atarg)
    if objective_penalized:
        optimgain -= penalty_ * surr_cost
        optimgain /= (1 + penalty_)


    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    cost_var_list = [v for v in all_var_list if v.name.startswith("pi/cost")]
    assert len(var_list) == len(vf_var_list) + 1
    assert len(var_list) == len(cost_var_list) + 1

    vfadam = MpiAdam(vf_var_list)
    costadam = MpiAdam(cost_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    if cost_method == 'lagrangian':
        compute_losses = U.function([ob, ac, atarg, cost_atarg], losses)
        compute_lossandgrad = U.function([ob, ac, atarg,cost_atarg], losses + [U.flatgrad(optimgain, var_list)])
        compute_fvp = U.function([flat_tangent, ob, ac, atarg,cost_atarg], fvp)
    else:
        compute_losses = U.function([ob, ac, atarg], losses)
        compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)


    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
    compute_cflossandgrad = U.function([ob, cost_ret], U.flatgrad(cost_vferr, cost_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    costadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_cost(pi, env, timesteps_per_batch,env_id,cost_t, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    costbuffer = deque(maxlen=40) # rolling buffer for episode costs

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'
    result = {}
    result['EpLenMean'] = []
    result['EpRewMean'] = []
    result['EpCostMean'] = []
    if cost_method == 'lagrangian':
        result_file = '/home/shaoqian/CMDP/baselines/baselines/models/trpo_mpi/' + env_id + '/' + str(
            seed) + '/cost_' + str(max_cost) + '_v2/lagrangian/'
    else:
        result_file = '/home/shaoqian/CMDP/baselines/baselines/models/trpo_mpi/'+env_id+ '/'+str(seed)+'/cost_' + str(max_cost) +'_v3/'
    mkdir(result_file)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            torch.save(result, os.path.join(result_file, 'results_dict_' + str(iters_so_far) + '.pt'))
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), result_file+env_id)
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        add_ctarg_and_cadv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret, ctarg,tdlamcostret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"],\
                                                      seg["cadv"], seg["tdlamcostret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        cpredbefore = seg["cpred"]  # predicted cost value function before udpate

        if cost_method=='bvf':
            violation_masks, vio_rate = check_violation(seg, max_cost)

            atarg = violation_masks * atarg - (np.ones(len(atarg), "float32") - violation_masks) * ctarg

        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        #if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        #if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        #args = seg["ob"], seg["ac"], atarg
        #fvpargs = [arr[::5] for arr in args]

        if cost_method == 'lagrangian':
            args = seg["ob"], seg["ac"], atarg, ctarg
            fvpargs = [arr[::5] for arr in args]
        else:
            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        with timed("cf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamcostret"]),
                include_final_partial_batch=False, batch_size=64):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)  # update running mean/std for policy
                    g = allmean(compute_cflossandgrad(mbob, mbret))
                    costadam.update(g, vf_stepsize)

        if cost_method == 'lagrangian':
                for i in range(len(seg["ep_costs"])):
                    U.get_session().run(train_penalty, feed_dict={cur_cost_ph: seg['ep_costs'][i]})


        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        logger.record_tabular("ev_tdlam_cost_before", explained_variance(cpredbefore, tdlamcostret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"],seg["ep_costs"] ) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews, tra_costs = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        costbuffer.extend(tra_costs)


        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpCostMean", np.mean(costbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        result['EpLenMean'].append(np.mean(lenbuffer))
        result['EpRewMean'].append(np.mean(rewbuffer))
        result['EpCostMean'].append(np.mean(costbuffer))

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)



        if rank==0:
            logger.dump_tabular()

        if iters_so_far % 1000 == 0:
            torch.save(result, os.path.join(result_file, 'results_dict_' + str(iters_so_far) + '.pt'))
    return pi




def learn_cost_r(*,
        network,
        env,
        total_timesteps,
        max_cost,  # add maximum cost
        env_id,
        timesteps_per_batch=2000,#1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy_cost_r(name=name, ob_space=ob_space, ac_space=ac_space,
                                         reuse=reuse, hid_size=100, num_hid_layers=2)

    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    # ADD for cost
    cost_atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Cost target advantage function
    cost_ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical cost

    cost_r_ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical reviewer cost

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))  # (v_prdict-return）^2

    # add for cost
    cost_vferr = tf.reduce_mean(tf.square(pi.cost_vpred - cost_ret))

    # add for cost reviewer
    cost_r_vferr = tf.reduce_mean(tf.square(pi.cost_r_vpred - cost_r_ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    cost_var_list = [v for v in all_var_list if v.name.startswith("pi/cost")]
    cost_r_var_list = [v for v in all_var_list if v.name.startswith("pi/cr")]
    assert len(var_list) == len(vf_var_list) + 1
    assert len(var_list) == len(cost_var_list) + 1
    assert len(var_list) == len(cost_r_var_list) + 1

    vfadam = MpiAdam(vf_var_list)
    costadam = MpiAdam(cost_var_list)
    costradam = MpiAdam(cost_r_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
    compute_cflossandgrad = U.function([ob, cost_ret], U.flatgrad(cost_vferr, cost_var_list))
    compute_cfrlossandgrad = U.function([ob, cost_r_ret], U.flatgrad(cost_r_vferr, cost_r_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    costadam.sync()
    costradam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_cost_r(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    costbuffer = deque(maxlen=40) # rolling buffer for episode costs

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'
    result = {}
    result['EpLenMean'] = []
    result['EpRewMean'] = []
    result['EpCostMean'] = []
    result_file = '/home/shaoqian/CMDP/baselines/baselines/models/trpo_mpi/'+env_id+ '/'+str(seed)+'/cost_r_' + str(max_cost) +'/'
    mkdir(result_file)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            torch.save(result, os.path.join(result_file, 'results_dict_' + str(iters_so_far) + '.pt'))
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), result_file+env_id)
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        add_ctarg_and_cadv(seg, gamma, lam)
        add_ctarg_and_cadv_r(seg, gamma, lam)  # calculate return and adv for cost reviewer

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret, ctarg,tdlamcostret,tdlamcostretr = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"],\
                                                      seg["cadv"], seg["tdlamcostret"],seg['tdlamcostretr']
        vpredbefore = seg["vpred"] # predicted value function before udpate
        cpredbefore = seg["cpred"]  # predicted cost value function before udpate
        cpredrbefore = seg["cpredr"]  # predicted cost reviewer value function before udpate

        #violation_masks, vio_rate = check_violation(seg, max_cost)
        violation_masks = np.less(tdlamcostret + tdlamcostretr, max_cost)
        atarg = violation_masks * atarg - (np.ones(len(atarg), "float32") - violation_masks) * ctarg

        #atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        with timed("cf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamcostret"]),
                include_final_partial_batch=False, batch_size=64):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)  # update running mean/std for policy
                    g = allmean(compute_cflossandgrad(mbob, mbret))
                    costadam.update(g, vf_stepsize)

        with timed("cfr"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamcostretr"]),
                                                         include_final_partial_batch=False, batch_size=64):
                    if hasattr(pi, "ob_rms"):
                        pi.ob_rms.update(mbob)  # update running mean/std for policy
                    g = allmean(compute_cfrlossandgrad(mbob, mbret))
                    costradam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        logger.record_tabular("ev_tdlam_cost_before", explained_variance(cpredbefore, tdlamcostret))
        logger.record_tabular("ev_tdlam_cost_reviwer_before", explained_variance(cpredrbefore, tdlamcostretr))

        lrlocal = (seg["ep_lens"], seg["ep_rets"],seg["ep_costs"] ) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews, tra_costs = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        costbuffer.extend(tra_costs)


        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpCostMean", np.mean(costbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        result['EpLenMean'].append(np.mean(lenbuffer))
        result['EpRewMean'].append(np.mean(rewbuffer))
        result['EpCostMean'].append(np.mean(costbuffer))

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

        if iters_so_far % 1000 == 0:
            torch.save(result, os.path.join(result_file, 'results_dict_' + str(iters_so_far) + '.pt'))
    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

