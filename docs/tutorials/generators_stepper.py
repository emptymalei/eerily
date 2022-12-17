# %% [markdown]
# # Using Data Generator Stepper

# %% [markdown]
# Stepper is the core of the data generators. In EERILY, we create iterators called steppers which we can iterate over to get the samples.
# In this tutorial we generate some neuronal spiking time series.


# %%
import matplotlib.pyplot as plt

from eerily.generators.spiking import SpikingEventParams, SpikingEventStepper
from eerily.generators.utils.events import PoissonEvent
from eerily.generators.utils.noises import LogNormalNoise

seed = 42

# %% [markdown]
# We first create the spiking event timing using a PoissonEvent.

# %%
spiking_rate = 0.1
spiking_level_mu = 0.1
spiking_level_std = 0.05

spiking = PoissonEvent(rate=spiking_rate, seed=seed)

# %% [markdown]
# `spiking` is only the indicator of the time where a spike happens as it can only take values 0 or 1. We have to specify the level of such spikes. Here we use a log normal distribution.

# %%
spiking_level = LogNormalNoise(mu=spiking_level_mu, std=spiking_level_std, seed=seed)

# %% [markdown]
# To create realistic data, we also need some background noise. We create such background values using log normal distributions.

# %%
background_mu = 1
background_std = 0.1
background = LogNormalNoise(mu=background_mu, std=background_std, seed=seed)

# %% [markdown]
# We assemble these configurations for the spiking stepper.

# %%
se_params = SpikingEventParams(
    spike=spiking,
    spike_level=spiking_level,
    background=background,
    variable_names=["event"],
    initial_state=0,
)

# %% [markdown]
# We create the spiking stepper using the above configuration.

# %%
se = SpikingEventStepper(se_params)

# %% [markdown]
# In this example, we create a time series with 100 steps.

# %%
length = 100
se_data = [next(se) for _ in range(length)]

# %%
fig, ax = plt.subplots(figsize=(10, 6.18))
ax.plot(range(length), se_data, color="k", marker="o")
ax.set_xlabel("Time Step")
ax.set_ylabel("Potential")
ax.set_title("Spiking Example")
