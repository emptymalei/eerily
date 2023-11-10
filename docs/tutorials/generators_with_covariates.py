# %% [markdown]
# # Generator with Covariates

# %% [markdown]
# Many dynamical systems have observable covariates. For example,
# the sales of an article are related to the discounts.
# In this tutorial we generate some time series data with different discounts.


# %%
import matplotlib.pyplot as plt

from eerily.generators.elasticity import ElasticityStepper, LinearElasticityParams
from eerily.generators.naive import (
    ConstantStepper,
    ConstStepperParams,
    SequenceStepper,
    SequenceStepperParams,
)

# %%
length = 10
elasticity = iter([-3] * length)
log_prices = iter(range(length))

initial_condition = {"log_demand": 3, "log_price": 0.5, "elasticity": None}

lep = LinearElasticityParams(
    initial_state=initial_condition,
    log_prices=log_prices,
    elasticity=elasticity,
    variable_names=["log_demand", "log_price", "elasticity"],
)

es = ElasticityStepper(model_params=lep)

next(es)

# %% [markdown]
# ## Combining Multiple Elasticity Steppers

# %%
elasticity_1 = iter([-3] * length)
log_prices_1 = iter(range(length))

lep_1 = LinearElasticityParams(
    initial_state=initial_condition,
    log_prices=log_prices_1,
    elasticity=elasticity_1,
    variable_names=["log_demand", "log_price", "elasticity"],
)
es_1 = ElasticityStepper(model_params=lep_1, length=length)


# %%
ssp = SequenceStepperParams(initial_state=[0], variable_names=["steps"], step_sizes=[1])
ss = SequenceStepper(model_params=ssp, length=length)

# %%
csp = ConstStepperParams(initial_state=["brand_1"], variable_names=["name"])
cs = ConstantStepper(model_params=csp, length=length)

# %%
# We can combine the two steppers using `&`.

# %%
generator = es_1 & ss

list(generator)
