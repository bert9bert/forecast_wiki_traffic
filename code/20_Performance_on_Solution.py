
# setup and paths
## prebuilt libraries
import pandas as pd
import numpy as np

## project libraries
from projutils import smape

## paths
data_intermed_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/data_intermed"
outputs_nb_fldrpath = "/media/disk1/forecast_wiki_traffic/output"

all_models_store_path = data_intermed_nb_fldrpath + "/all_models_store.h5"


# combine candidate models with solutions
with pd.HDFStore(all_models_store_path, mode="r") as s:
    submission_forecast_all = s.select(key="/submission_forecast_all")
    solution = s.select(key="/solution")

submission_forecast_all.index = submission_forecast_all.index.droplevel("Id")
solution.index = solution.index.droplevel("Id")

submission_solution = pd.merge(
    submission_forecast_all,
    solution,
    left_index = True, right_index = True
)

assert(len(submission_solution) == len(submission_forecast_all))
assert(len(submission_solution) == len(solution))

del submission_forecast_all, solution


# calculate SMAPE on all candidate models using the solution to the forecast window

mod_names = [v for v in submission_solution.columns if v[:4]=="mod_" or v[:7]=="ensmod_" or v[:12]=="postValiMod_"]

submission_solution_smape = [smape(submission_solution[v], submission_solution["Visits"]) for v in mod_names]

submission_solution_smape = pd.DataFrame({"model":mod_names, "smape":submission_solution_smape})

# save
submission_solution_smape.to_csv(outputs_nb_fldrpath + "/losses_submission_solution.csv", index=False)
