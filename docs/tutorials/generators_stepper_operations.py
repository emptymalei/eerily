# %% [markdown]
# # Combining Data Generators

# %% [markdown]
# Steppers can be combined into one single generator. In this example, we reuse the concepts in the tutorial "Using Data Generator Stepper".


# %%
import matplotlib.pyplot as plt

from eerily.generators.spiking import SpikingEventParams, SpikingEventStepper
from eerily.generators.utils.events import PoissonEvent
from eerily.generators.utils.noises import LogNormalNoise

seed = 42

# %% [markdown]
# We first create the spiking event timing using a PoissonEvent.

# %%
spiking_1 = PoissonEvent(rate=0.1, seed=seed)
spiking_2 = PoissonEvent(rate=0.9, seed=seed)

# %%
spiking_level_1 = LogNormalNoise(mu=0.1, std=0.05, seed=seed)
spiking_level_2 = LogNormalNoise(mu=0.1, std=0.05, seed=seed)

# %%
background_1 = LogNormalNoise(mu=1, std=0.1, seed=seed)
background_2 = LogNormalNoise(mu=1, std=0.1, seed=seed)

# %% [markdown]
# Create the spiking steppers:

# %%
se_params_1 = SpikingEventParams(
    spike=spiking_1,
    spike_level=spiking_level_1,
    background=background_1,
    variable_names=["event"],
    initial_state=0,
)

se_params_2 = SpikingEventParams(
    spike=spiking_2,
    spike_level=spiking_level_2,
    background=background_2,
    variable_names=["event"],
    initial_state=0,
)

# %%
se_1 = SpikingEventStepper(se_params_1, length=70)
se_2 = SpikingEventStepper(se_params_2, length=30)

# %% [markdown]
# We concat the two steppers.

# %%
generator = se_1 + se_2

# %%
fig, ax = plt.subplots(figsize=(10, 6.18))
ax.plot(range(100), list(generator), color="k", marker="o")
ax.set_xlabel("Time Step")
ax.set_ylabel("Potential")
ax.set_title("Spiking Example")

# %% [markdown]
# We can observe the potential is higher after step 70. This is because we have a higher spiking rate in the second stepper.
