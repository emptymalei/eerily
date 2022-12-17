# %% [markdown]
# # Using Data Generator Utilities

# %% [markdown]
# In this tutorial, we explore some utilities for generating time series data. EERILY provides noise generators (`eerily.generators.utils.noises`), event generators (`eerily.generators.utils.events`), a basic stepper to help the users create iterators of time series (`eerily.generators.utils.stepper`), and finally factories to unify them.

# %%
import matplotlib.pyplot as plt

seed = 42
length = 20

# %% [markdown]
# ## Noise
# Noise is an important part of a real word dataset. To generate realistic time series data, we always add noise to the data points.

# %%
from eerily.generators.utils.noises import GaussianNoise

# %% [markdown]
# GaussianNoise requires a mean and a standard deviation.

# %%
gn_mu = 1
gn_std = 0.1
gn = GaussianNoise(mu=gn_mu, std=gn_std, seed=seed)

# %% [markdown]
# The Gaussian noise `gn` is an iterator.

# %%
gn_data = [next(gn) for _ in range(length)]
gn_data

# %%
fig, ax = plt.subplots(figsize=(10, 6.18))
ax.plot(range(length), gn_data, color="k", marker="o")
ax.set_xlabel("Time Step")
ax.set_ylabel("Noise")
ax.set_title("GaussianNoise Example")


# %% [markdown]
# ## Events
# Suppose we set a sensor on a road to record whether a car is passing. Assuming the traffic flow is sparse, we expect to see some spikes in this time series. A simple model is a Poisson process.
#
# In this section, we demo the `PoissonEvent` class.

# %%
from eerily.generators.utils.events import PoissonEvent

# %% [markdown]
# `PoissonEvent` requires one argument `rate`. We can also set the `seed` for reproducibility.
# %%
pe_rate = 0.5

# %%
pe = PoissonEvent(rate=pe_rate, seed=seed)

# %% [markdown]
# The event defined `pe` is an iterator. We iterate to get some data.

# %%
pe_data = [next(pe) for _ in range(length)]
pe_data

# %%
fig, ax = plt.subplots(figsize=(10, 6.18))
ax.plot(range(length), pe_data, color="k", marker="o")
ax.set_xlabel("Time Step")
ax.set_ylabel("Event")
ax.set_title("PoissonEvent Example")
