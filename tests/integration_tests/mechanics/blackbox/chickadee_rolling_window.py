'''
Test that rolling-window dispatch is working properly in Chickadee
using the Pyomo dispatcher
'''
import time
import matplotlib.pyplot as plt
import numpy as np
import chickadee

n = 20 # number of time points
time_horizon = np.linspace(0, n-1 , n)

steam = 'steam'
electricity = 'electricity'

def smr_cost(dispatch):
  return sum(-0.001 * dispatch[steam])

def smr_transfer(inputs):
  return {}

smr_capacity = np.ones(n)*1200
smr_ramp = 600*np.ones(n)
# Purposely off by a little bit
smr_guess = (100*np.sin(time_horizon) + 500)/0.65
smr = chickadee.PyomoBlackboxComponent('smr', smr_capacity, smr_ramp, -smr_ramp, steam,
                                smr_transfer, smr_cost, produces=steam, guess=smr_guess)

def turbine_transfer(inputs):
  effciency = 0.7  # Just a random guess at a turbine efficiency
  return {steam: -1/effciency * inputs}

def turbine_cost(dispatch):
  return sum(-1 * dispatch[steam])

turbine_capacity = np.ones(n)*1000
turbine_guess = 100*np.sin(time_horizon) + 500
turbine_ramp = 100*np.ones(n)
turbine = chickadee.PyomoBlackboxComponent('turbine', turbine_capacity,
                                turbine_ramp, -turbine_ramp, electricity,
                                turbine_transfer, turbine_cost,
                                produces=electricity, consumes=steam, guess=turbine_guess)

def el_market_transfer(inputs):
  return {}

def elm_cost(dispatch):
  return sum(5.0 * dispatch[electricity])

# elm_capacity = np.ones(n)*-800
elm_capacity = -(100*np.sin(time_horizon) + 500)
elm_ramp = 1e10*np.ones(n)
elm = chickadee.PyomoBlackboxComponent('el_market', elm_capacity, elm_ramp, -elm_ramp,
                                electricity, el_market_transfer, elm_cost,
                                consumes=electricity, dispatch_type='fixed')

dispatcher = chickadee.BlackboxDispatcher(window_length=10)

comps = [smr, turbine, elm]
# comps = [smr, tes, turbine, elm]

start_time = time.time()
sol = dispatcher.dispatch(comps, time_horizon, verbose=False)
end_time = time.time()
# print('Full optimal dispatch:', optimal_dispatch)
print('Dispatch time:', end_time - start_time)

# Check to make sure that the ramp rate is never too high
ramp = np.diff(sol.dispatch['turbine'][electricity])
# assert max(ramp) <= turbine.ramp_rate_up[0], 'Max ramp rate exceeded!'


balance_el = sol.dispatch['turbine'][electricity] + \
    sol.dispatch['el_market'][electricity]
balance_steam = sol.dispatch['turbine'][steam] + \
    sol.dispatch['smr'][steam]


plt.subplot(2, 1, 1)
plt.plot(sol.time, sol.dispatch['turbine'][electricity], label='El gen')
plt.plot(sol.time, sol.dispatch['el_market'][electricity], label='El cons')
plt.plot(sol.time, balance_el, label='Electricity balance')
plt.plot(sol.time[:-1], ramp, label='Ramp rate')
ymax = max(sol.dispatch['turbine'][electricity])
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sol.time, sol.dispatch['turbine'][steam], label='turbine')
plt.plot(sol.time, sol.dispatch['smr'][steam], label='SMR')
plt.plot(sol.time, balance_steam, label='Steam balance')
plt.plot(sol.time[:-1], ramp, label='Ramp rate')
ymax = max(sol.dispatch['smr'][steam])
plt.legend()

plt.savefig('output.png')
plt.show()

print(sol.dispatch['smr'][steam])
