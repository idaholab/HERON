# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Produces example plots for Activity reporting example in manual.
"""
import matplotlib.pyplot as plt

times = list(range(5))
activity = [-1, -2, 3, -1.5, -1]

# storage
fig, ax = plt.subplots()
ax.set_title('Storage Activity Example')
levels = []
for a, act in enumerate(activity):
  levels.append(-1*sum(activity[:a]))
  label = 'assumed' if a == 0 else None
  l_p, = ax.plot([a, a+1], [act, act], 'g--')
  if a > 0:
    ax.plot([a, a], [act, activity[a-1]], 'g--')
levels.append(-1*sum(activity))
ax2 = ax.twinx()
l_r, = ax2.plot(times + [5], levels, 'ko', label='reported')
l_a, = ax2.plot(times + [5], levels, 'r:', label='assumed')
ax.set_xlabel('time (h)')
ax.set_ylabel('production rate (u/s)', color='g')
ax.tick_params(axis='y', labelcolor='g')
ax.set_ylim([-4, 4])
ax2.set_ylabel('net level (u)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax.legend([l_r, l_p, l_a], ['reported (level)', 'assumed (production rate)', 'assumed (level)'], loc='upper left')
plt.savefig('storage_activity_explained.png')

# production
activity = [1, 4, 3, 1, 2]
fig, ax = plt.subplots()
ax.set_title('Producer Activity Example')
l_r, = ax.plot(times, activity, 'ko', label='reported (production rate)')
l_a, = ax.plot(times, activity, 'g--', label='assumed (production rate)')
ax.set_ylabel('production rate (u/s)')
ax.set_xlabel('time (h)')
ax.legend()
plt.savefig('producer_activity_explained.png')

plt.show()
