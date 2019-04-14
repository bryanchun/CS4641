###
### Imports
###

from mdptoolbox import example, mdp # import mdptoolbox then mdptoolbox.example does not work
# mdptoolbox?
# mdptoolbox.example
from mdplib import PolicyIteration, QLearning

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import random
from timeit import default_timer as timer

from functools import partial

###
### Constants
### - List of candidate hyperparameter values
###

GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9]  # controls degree of influence of previous reward on current reward
EPSILONS = np.logspace(-2, -10, 5)
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
EXPLORE_FRACTIONS = [0.01, 0.03, 0.05]
# p random action probability is an exploration strategy

###
### MDP / RL problems
### 1. Grid World (Berkeley CS188)
### 2. Taxi-v2 (openai gym)
###

def grid_world(r_terminal=1, r_nonterminal=0, p=0.1):
  p_adv = 1 - 2*p
  p_tot = 1
  
  P = [
    # a = a0
    [
      # s = (0,0) ... (2,3)
      [p_adv+p,p,0,0,0,0,0,0,0,0,0,0],
      [p,p_adv,p,0,0,0,0,0,0,0,0,0],
      [0,p,p_adv,p,0,0,0,0,0,0,0,0],
      [0,0,0,p_tot,0,0,0,0,0,0,0,0],  # row 1
      [p_adv,0,0,0,p+p,0,0,0,0,0,0,0],
      [p_tot,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,p_adv,0,0,0,p,p,0,0,0,0],
      [0,0,0,0,0,0,0,p_tot,0,0,0,0],  # row 2
      [0,0,0,0,p_adv,0,0,0,p,p,0,0],
      [0,0,0,0,0,0,0,0,p,p_adv,p,0],
      [0,0,0,0,0,0,p_adv,0,0,p,0,p],
      [0,0,0,0,0,0,0,p_adv,0,0,p,p]   # row 3
      
    ],
    # a = a1
    [
      [p,p,0,0,p_adv,0,0,0,0,0,0,0],
      [p,p_adv,p,0,0,0,0,0,0,0,0,0],
      [0,p,0,p,0,0,p_adv,0,0,0,0,0],
      [0,0,0,p_tot,0,0,0,0,0,0,0,0],  # row 1
      [0,0,0,0,p+p,0,0,0,p_adv,0,0,0],
      [p_tot,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,p,p,0,0,p_adv,0],
      [0,0,0,0,0,0,0,p_tot,0,0,0,0],  # row 2
      [0,0,0,0,0,0,0,0,p_adv+p,p,0,0],
      [0,0,0,0,0,0,0,0,p,p_adv,p,0],
      [0,0,0,0,0,0,0,0,0,p,p_adv,p],
      [0,0,0,0,0,0,0,0,0,0,p,p_adv+p]   # row 3
    ],
    # a = a2
    [
      [p_adv+p,0,0,0,p,0,0,0,0,0,0,0],
      [p_adv,p+p,0,0,0,0,0,0,0,0,0,0],
      [0,p_adv,p,0,0,0,p,0,0,0,0,0],
      [0,0,0,p_tot,0,0,0,0,0,0,0,0],  # row 1
      [p,0,0,0,p_adv,0,0,0,p,0,0,0],
      [p_tot,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,p,0,0,0,p_adv,0,0,0,p,0],
      [0,0,0,0,0,0,0,p_tot,0,0,0,0],  # row 2
      [0,0,0,0,p,0,0,0,p_adv+p,0,0,0],
      [0,0,0,0,0,0,0,0,p_adv,p+p,0,0],
      [0,0,0,0,0,0,p,0,0,p_adv,p,0],
      [0,0,0,0,0,0,0,p,0,0,p_adv,p]   # row 3
    ],
    # a = a3
    [
      [p,p_adv,0,0,p,0,0,0,0,0,0,0],
      [0,p+p,p_adv,0,0,0,0,0,0,0,0,0],
      [0,0,p,p_adv,0,0,p,0,0,0,0,0],
      [0,0,0,p_tot,0,0,0,0,0,0,0,0],  # row 1
      [p,0,0,0,p_adv,0,0,0,p,0,0,0],
      [p_tot,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,p,0,0,0,0,p_adv,0,0,p,0],
      [0,0,0,0,0,0,0,p_tot,0,0,0,0],  # row 2
      [0,0,0,0,p,0,0,0,p,p_adv,0,0],
      [0,0,0,0,0,0,0,0,0,p+p,p_adv,0],
      [0,0,0,0,0,0,p,0,0,0,p,p_adv],
      [0,0,0,0,0,0,0,p,0,0,0,p_adv+p]   # row 3
    ]
  ]
  # FIX: StochasticError: 'PyMDPToolbox - The transition probability matrix is not stochastic.'
  # All rows must sum to 1, even for unreachable states. Grid (1, 1)
  # Let it be any will do, since it anyway cannot be reached
  
  r0, r = r_terminal, r_nonterminal
  
  R = [
    [ r, r, r, 0, r, 0, r, 0, r, r, r, -r0 ],
    [ r, r, r, 0, r, 0, r, 0, r, r, r, r ],
    [ r, r, r, 0, r, 0, r, 0, r, r, r, r ],
    [ r, r, r0, 0, r, 0, -r0, 0, r, r, r, r ]
  ]
  
  return (np.array(P), np.array(R).T)


def forest(S=50, r1=60, r2=30, p=0.1):
  return example.forest(S=S, r1=r1, r2=r2, p=p)

def forest_reward(policy, r1=60, r2=30):
  if np.all(policy == 0):
    return r1
  if np.all(policy[:-1] == 0) and policy[-1] == 1:
    return (len(policy) - 2) + r2
  bitstr = ''.join(map(str, policy))
  waits = bitstr.split('1')
  effective_waits = list(filter(lambda w: len(w) > 0, waits[:-1]))
  oldest_reward = r1 if policy[-1] == 0 else r2
  return sum(len(effective_wait) for effective_wait in effective_waits) + oldest_reward

###
### Planning algorithms (given MDP) / RL algorithm (given samples)
###


def vi(P, R, mdp_name, discount, epsilon, verbose=False, max_iter=1000):
  # print(f'ValueIteration on {mdp_name}')
  # print(f'>> discount = {discount}, epsilon = {epsilon}')
  _vi = mdp.ValueIteration(P, R, discount, epsilon, max_iter=max_iter)
  if verbose:
    _vi.setVerbose()
  start = timer()
  _vi.run()
  elapsed = timer() - start
  # print('V', _vi.V)
  # print('policy', _vi.policy)
  # print('iter', _vi.iter)
  # print('time', elapsed)
  desc = f'Value Iteration\ndiscount: {discount}, epsilon: {epsilon}\niters: {_vi.iter}, elapsed: {"%.5f" % elapsed}s'
  return _vi.V, _vi.policy, _vi.iter, elapsed, desc
      


def pi(P, R, mdp_name, discount, verbose=False, max_iter=1000):
  # print(f'PolicyIteration on {mdp_name}')
  # print(f'>> discount = {discount}')
  _pi = mdp.PolicyIteration(P, R, discount, max_iter=max_iter)
  if verbose:
    _pi.setVerbose()
  start = timer()
  _pi.run()
  elapsed = timer() - start
  # print('V', _pi.V)
  # print('policy', _pi.policy)
  # print('iter', _pi.iter)
  # print('time', elapsed)
  desc = f'Policy Iteration\ndiscount: {discount}\niters: {_pi.iter}, elapsed: {"%.5f" % elapsed}s'
  return _pi.V, _pi.policy, _pi.iter, elapsed, desc
  
  
def ql(P, R, mdp_name, discount, lr, ex_frac, verbose=False, n_iter=10000, one=False):
  # print(f'QLearning on {mdp_name}')
  ex_int = int(ex_frac * n_iter)
  # print(f'>> discount = {discount}, lr = {lr}, ex_int = {ex_int}')
  _ql = QLearning(P, R, discount, learning_rate=lr, explore_interval=ex_int, n_iter=n_iter)
  if verbose:
    _ql.setVerbose()
  start = timer()
  _ql.run()
  elapsed = timer() - start
  # print('V', _ql.V)
  # print('policy', _ql.policy)
  # print('iter', n_iter)
  # print('time', elapsed)
  # print('total reward', _ql.totalR)
  if not one:
    desc = f'Q-Learning\ndiscount: {discount}, learning_rate: {lr}, exporation_interval: {ex_int}\niters: {n_iter}, elapsed: {"%.5f" % elapsed}s, total_reward: {_ql.totalR}'
  else:
    desc = f'({discount}, {lr}, {n_iter}, {"%.5f" % elapsed}s, {_ql.totalR})'
  return _ql.V, _ql.policy, n_iter, elapsed, _ql.totalR, desc
  
###
### Visualisation utilities
###

policy_to_arrow_map = {
  0: '^',
  1: '˅',
  2: '<',
  3: '>'
}

policy_to_arrow = np.vectorize(lambda i: policy_to_arrow_map[i])

def grid_world_policy(flat_policy, dims):
  policy = np.array(flat_policy).reshape(dims)
  policy = policy_to_arrow(policy)
  # grid_world_map
  policy[1][1] = 'X'
  policy[0][3] = '+'
  policy[1][3] = '-'
  return policy
 
def vis_map_one(ax, vs, policy, dims, policy_callback):
  vs = np.array(vs).reshape(dims)
  policy = policy_callback(policy, dims)
  im = ax.imshow(vs)
  ax.set_xticks([])
  ax.set_yticks([])
  
  for i in range(dims[0]):
    for j in range(dims[1]):
        text = ax.text(j, i, policy[i, j], ha="center", va="center", color="w", fontsize=20)
  return im
  
def vis_map(vs, policy, desc, dims, mdp_name, policy_callback):
  title = f'{mdp_name} Policy with {desc}'
  label = 'V-value'
  fig, ax = plt.subplots()

  im = vis_map_one(ax, vs, policy, dims)
  
  cbar = ax.figure.colorbar(im, ax=ax)
  cbar.ax.set_ylabel(label, rotation=-90, va="bottom")
  fig.tight_layout()
  plt.title(title)
  plt.show()
  
vis_grid_world = partial(vis_map, dims=(3, 4), mdp_name='Grid World', policy_callback=grid_world_policy)


def vis_grid_world_cross(ex_frac, discounts=(0.1, 0.99), lrs=(0.1, 0.9)):
  
  d1, d2 = discounts
  l1, l2 = lrs
  
  n_iter = 10000
  ex_int = int(ex_frac * n_iter)
  
  f, axarr = plt.subplots(2, 2, figsize=(9, 7))
  f.suptitle(f'Grid World Policy with Q-Learning\nexploration_interval: {ex_int}\n(discount, learning_rate, iters, elapsed, total_reward)')

  vs3, policy3, iter3, elapsed3, totalR3, desc3 = ql1(discount=d1, lr=l1, ex_frac=ex_frac, one=True)
  vis_map_one(axarr[0, 0], vs3, policy3, (3, 4), grid_world_policy)
  axarr[0, 0].set_title(desc3)

  vs3, policy3, iter3, elapsed3, totalR3, desc3 = ql1(discount=d1, lr=l2, ex_frac=ex_frac, one=True)
  vis_map_one(axarr[0, 1], vs3, policy3, (3, 4), grid_world_policy)
  axarr[0, 1].set_title(desc3)

  vs3, policy3, iter3, elapsed3, totalR3, desc3 = ql1(discount=d2, lr=l2, ex_frac=ex_frac, one=True)
  vis_map_one(axarr[1, 0], vs3, policy3, (3, 4), grid_world_policy)
  axarr[1, 0].set_title(desc3)

  vs3, policy3, iter3, elapsed3, totalR3, desc3 = ql1(discount=d2, lr=l1, ex_frac=ex_frac, one=True)
  im = vis_map_one(axarr[1, 1], vs3, policy3, (3, 4), grid_world_policy)
  axarr[1, 1].set_title(desc3)

  label = 'V-value'
  # cbar = axarr[1, 1].figure.colorbar(im, ax=axarr[1, 1])
  # cbar.ax.set_ylabel(label, rotation=-90, va="bottom")
  cbar = f.colorbar(im, ax=axarr.flat)
  cbar.ax.set_ylabel(label, rotation=-90, va="bottom", fontsize=14)

  plt.show()


def vis_forest(vs, policy, desc):
  S = len(vs)
  mdp_name = 'Forest'
  f, axarr = plt.subplots(2, sharex=True, figsize=(6, 6))
  reward = forest_reward(policy)
  title = f'{mdp_name} Policy and V-value with {desc}' + f', reward: {reward}'
  f.suptitle(title)
  axarr[0].scatter(np.arange(S), policy)
  axarr[0].set_ylabel("Action under Policy")
  axarr[1].plot(np.arange(S), vs)
  axarr[1].set_ylabel("V-value")
  axarr[1].set_xlabel("State (Year)")
  plt.show()
  
  
def vis_forest_cross(ex_frac, discounts=(0.1, 0.99), lrs=(0.1, 0.9)):

  n_iter = 10000
  ex_int = int(ex_frac * n_iter)
  configurations = [(d, l) for d in discounts for l in lrs]

  fig = plt.figure(figsize=(10, 9))
  outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.3)
  fig.suptitle(f'Forest Policy with Q-Learning\nexploration_interval: {ex_int}\n(discount, learning_rate, iters, elapsed, total_reward), reward')
  
  for i, (d, l) in enumerate(configurations):
      inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                      subplot_spec=outer[i], wspace=0.1, hspace=0.1)

      vs, policy, iter3, elapsed3, totalR3, desc = ql2(discount=d, lr=l, ex_frac=ex_frac, one=True)
      S = len(vs)

      ax_p = plt.Subplot(fig, inner[0])
      ax_p.scatter(np.arange(S), policy)
      reward = forest_reward(policy)
      ax_p.set_ylabel("Action under Policy")
      ax_p.set_xticks([])
      ax_p.set_title(desc + f", {reward}")
      fig.add_subplot(ax_p)

      ax_v = plt.Subplot(fig, inner[1])
      ax_v.plot(np.arange(S), vs)
      ax_v.set_ylabel("V-value")
      ax_v.set_xlabel("State (Year)")
      fig.add_subplot(ax_v)
  
  plt.show()
  
###
### Pre-main
###

P1, R1 = grid_world()

vi1 = partial(vi, P=P1, R=R1, mdp_name='Grid World')
pi1 = partial(pi, P=P1, R=R1, mdp_name='Grid World')

P2, R2 = forest()

vi2 = partial(vi, P=P2, R=R2, mdp_name='Forest')
pi2 = partial(pi, P=P2, R=R2, mdp_name='Forest')

ql1 = partial(ql, P=P1, R=R1, mdp_name='Grid World')
ql2 = partial(ql, P=P2, R=R2, mdp_name='Forest', n_iter=10000)