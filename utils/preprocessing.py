import pandas as pd
import numpy as np


def create_time_matrix(filename):
    """
    Creates a time matrix from a csv file.
    The csv file is in seconds while the output is in minutes.
    """
    time_matrix = pd.read_csv(filename, index_col=0)
    time_matrix = time_matrix.astype("float64") / 60
    time_matrix.columns = time_matrix.columns.astype(int)
    time_matrix.index = time_matrix.index.astype(int)
    return time_matrix


def create_caregivers_df(filename):
    """
    Creates a caregivers dataframe from a csv file.
    Creates start_minutes and end_minutes
    """
    caregivers = pd.read_csv(filename, index_col=0)
    caregivers["Attributes"] = caregivers["Attributes"].apply(lambda x: np.array(eval(x)))
    caregivers["start_minutes"] = (
        pd.to_datetime(caregivers["EarliestStartTime"]).dt.hour * 60
        + pd.to_datetime(caregivers["EarliestStartTime"]).dt.minute
    )
    caregivers["end_minutes"] = (
        pd.to_datetime(caregivers["LatestEndTime"]).dt.hour * 60
        + pd.to_datetime(caregivers["LatestEndTime"]).dt.minute
    )
    return caregivers


def create_tasks_df(filename, only_client_tasks=True):
    """
    Creates a tasks dataframe from a csv file.
    Creates start_minutes, end_minutes and duration_minutes
    """
    tasks = pd.read_csv(filename, index_col=0)
    if only_client_tasks:
        tasks = tasks[tasks["TaskType"].isin(["Hemtjänst", "Dubbelbemanning"])]
        tasks["ClientID"] = tasks["ClientID"].astype(int)
    tasks["start_minutes"] = (
        pd.to_datetime(tasks["StartTime"]).dt.hour * 60 + pd.to_datetime(tasks["StartTime"]).dt.minute
    )
    tasks["end_minutes"] = pd.to_datetime(tasks["EndTime"]).dt.hour * 60 + pd.to_datetime(tasks["EndTime"]).dt.minute
    tasks["duration_minutes"] = tasks["end_minutes"] - tasks["start_minutes"]
    return tasks


def create_clients_df(filename):
    """
    Creates a clients dataframe from a csv file.
    """
    clients = pd.read_csv(filename, index_col=0)
    clients["Requirements"] = clients["Requirements"].apply(lambda x: np.array(eval(x)))
    return clients
