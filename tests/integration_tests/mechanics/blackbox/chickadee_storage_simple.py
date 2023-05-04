'''
Test that storage is properly handled in Chickadee
using the pyOpySparse dispatcher
'''
import time
import numpy as np
import chickadee

n = 40 # number of time points
time_horizon = np.linspace(0, n-1 , n)

steam = 'steam'

def smr_cost(dispatch: dict) -> float:
    '''Ecomonic cost function
    :param dispatch: dict, component dispatch
    :returns: float, the cost of running that dispatch

    This function will receive a dict representing a dispatch for a particular
    component and is required to return a float representing the economic cost
    of running that dispatch. A negative cost indicates an overall economic cost
    while a postive value indicates an economic benefit.

    This cost function is not required if the `external_obj_func` option is used
    in the dispatcher.
    '''
    # Impose a high ramp cost
    ramp_cost = 5*sum(abs(np.diff(dispatch[steam])))

    return sum(-0.1 * dispatch[steam] - ramp_cost)

def smr_transfer(inputs: list) -> list:
    return {}


smr_capacity = np.ones(n)*1200
smr_ramp = 600*np.ones(n)
smr_guess = np.ones(n) * 500
smr = chickadee.PyoptSparseComponent('smr', smr_capacity, smr_ramp, -smr_ramp,
                    steam, smr_transfer, smr_cost, produces=steam, guess=smr_guess)

def tes_transfer(inputs: list, init_store):
    tmp = np.insert(inputs, 0, init_store)
    return np.cumsum(tmp)[1:]

def tes_cost(dispatch):
    # Simulating high-capital and low-operating costs
    return -1000 - 0.01*np.sum(dispatch[steam])

tes_capacity = np.ones(n)*800
tes_ramp = np.ones(n)*50
tes_guess = np.zeros(n)
tes = chickadee.PyoptSparseComponent('tes', tes_capacity, -tes_ramp,
                                    tes_ramp, steam, tes_transfer,
                                    tes_cost, stores=steam, guess=tes_guess)


def load_transfer(inputs: list):
    return {}

def load_cost(dispatch):
    return sum(5.0 * dispatch[steam])

load_capacity = -(20*np.sin(time_horizon) + 500)
load_ramp = 1e10*np.ones(n)
load = chickadee.PyoptSparseComponent('load', load_capacity, load_ramp, load_ramp,
                                steam, load_transfer, load_cost,
                                consumes=steam, dispatch_type='fixed')

# dispatcher = chickadee.PyOptSparse(window_length=20)
dispatcher = chickadee.BlackboxDispatcher(window_length=20)

comps = [smr, tes, load]

start_time = time.time()
sol = dispatcher.dispatch(comps, time_horizon, [], verbose=False)
end_time = time.time()
# print('Full optimal dispatch:', optimal_dispatch)
print('Dispatch time:', end_time - start_time)
print('Obj Value: ', sol.objval)

import matplotlib.pyplot as plt
plt.subplot(3,1,1)
plt.plot(sol.time, sol.dispatch['tes'][steam], label='TES activity')
plt.plot(sol.time, sol.storage['tes'], label='TES storage level')
plt.plot(sol.time, tes_capacity*np.ones(len(time_horizon)), label='TES Max Capacity')
plt.plot(sol.time, tes_ramp, label='TES ramp')
ymax = max(sol.storage['tes'])
plt.vlines([w[0] for w in sol.time_windows], 0,
           ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0,
           ymax, colors='blue', linestyles='--')
plt.legend()

plt.subplot(3,1,2)
plt.plot(sol.time, sol.dispatch['smr'][steam], label='Heat generation')
plt.plot(sol.time, sol.dispatch['load'][steam], label='Heat load')
plt.plot(sol.time, sol.storage['tes'], label='Heat storage')
# plt.plot(sol.time[:-1], turbine_ramp, label='turbine ramp')
ymax = max(sol.dispatch['smr'][steam])
plt.vlines([w[0] for w in sol.time_windows], 0,
           ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0,
           ymax, colors='blue', linestyles='--')
plt.legend()

balance = sol.dispatch['load'][steam] - sol.dispatch['tes'][steam] + sol.dispatch['smr'][steam]

plt.subplot(3,1,3)
plt.plot(sol.time, balance, label='Steam Balance Error')
plt.legend()
# print(sol.dispatch)
# print(sol.storage)
plt.show()
