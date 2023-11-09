# %% [markdown]
# # Generator with Covariates

# %% [markdown]
# Many dynamical systems have observable covariates. For example,
# the sales of an article are related to the discounts.
# In this tutorial we generate some time series data with different discounts.


# %%
import matplotlib.pyplot as plt

from eerily.generators.elasticity import ElasticityStepper, LinearElasticityParams

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
# %%
