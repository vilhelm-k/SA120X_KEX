import os
import pandas as pd


def save_solutions(model, x, t, solution_name):
    if not os.path.exists(f"saved_solutions/{solution_name}"):
        os.makedirs(f"saved_solutions/{solution_name}")
    x_df = pd.DataFrame([(k, i, j, x[k, i, j].X) for k, i, j in x], columns=["Caregiver", "From", "To", "Value"])
    t_df = pd.DataFrame([(k, i, t[k, i].X) for k, i in t], columns=["Caregiver", "Node", "Time"])
    x_df.to_csv(f"saved_solutions/{solution_name}/x.csv", index=False)
    t_df.to_csv(f"saved_solutions/{solution_name}/t.csv", index=False)
