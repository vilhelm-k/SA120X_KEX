import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all optimization models"""

    # Initialization
    def __init__(
        self,
        caregivers,  # ID,ModeOfTransport,Attributes,EarliestStartTime,LatestEndTime,StartLocation,EndLocation
        tasks,  # ID,ClientID,StartTime,EndTime,TaskType,PlannedCaregiverID
        clients,  # ID,Requirements,x,y
        drive_time_matrix,  # ClientIDs. HQ is 0. Col is from, row is to.
        walk_time_matrix,
        bicycle_time_matrix,
        continuity,
    ):
        """
        Initialize the base model with common data

        Args:
            caregivers (pd.DataFrame): DataFrame containing caregiver information
            tasks (pd.DataFrame): DataFrame containing task information
            clients (pd.DataFrame): DataFrame containing client information
            drive_time_matrix (pd.DataFrame): Matrix of travel times between locations for driving
            walk_time_matrix (pd.DataFrame): Matrix of travel times for walking
            bicycle_time_matrix (pd.DataFrame): Matrix of travel times for bicycling
        """
        # Input data
        self.caregivers = caregivers
        self.tasks = tasks
        self.clients = clients
        self.drive_time_matrix = drive_time_matrix
        self.walk_time_matrix = walk_time_matrix
        self.bicycle_time_matrix = bicycle_time_matrix
        self.continuity = continuity

        # Preprocessed input data
        self.K = self.caregivers.index.tolist()
        self.V = self.tasks.index.tolist()
        self.C = self.clients.index.tolist()
        self.s = {i: self.tasks.loc[i, "duration_minutes"] for i in self.V}  # s[i] Service time for i
        self.e = {i: self.tasks.loc[i, "start_minutes"] for i in self.V}  # e[i] Earliest start time for i
        self.l = {i: self.tasks.loc[i, "end_minutes"] for i in self.V}  # l[i] Latest end time for
        self.Vk = self.__determine_qualified_tasks()  # Qualified tasks for each caregiver
        self.Vc = self.__determine_client_tasks()  # Tasks for each client
        self.A = self.__determine_pair_feasibility()
        self.H = self.__determine_continuity()  # H[k,c] represents if k has visited c historically

        # Model variables
        self.model = None
        self.x = None
        self.t = None

        # Postprocessed results
        self.routes = None
        self.arrivals = None
        self.breaks = None

    def __determine_continuity(self):
        """
        Determine the continuity of care for each caregiver and task.
        """
        continuity = {}
        for k in self.K:
            for c in self.C:
                most_visits = eval(self.continuity.loc[c, "MostVisits"])
                continuity[k, c] = 1 if k in most_visits else 0
        return continuity

    def c(self, k, i, j):
        """
        Get the travel times between locations for each caregiver and task.
        """
        mode_of_transport = self.caregivers.loc[k, "ModeOfTransport"]
        match mode_of_transport:
            case "car":
                time_matrix = self.drive_time_matrix
            case "pedestrian":
                time_matrix = self.walk_time_matrix
            case "bicycle":
                time_matrix = self.bicycle_time_matrix
            case _:
                raise ValueError(f"Unknown mode of transport: {mode_of_transport}")

        if i == "start":
            start_at_home = self.get_endpoint(k, "start") == "Home"
            return 0 if start_at_home else time_matrix.loc[0, self.get_location(j)]
        if j == "end":
            end_at_home = self.get_endpoint(k, "end") == "Home"
            return 0 if end_at_home else time_matrix.loc[self.get_location(i), 0]
        return time_matrix.loc[self.get_location(i), self.get_location(j)]

    def __determine_pair_feasibility(self):
        """
        Determine infeasible task pairs based on time constraints.
        """
        pair_feasibility = {}
        for k in self.K:
            for i in self.V:
                for j in self.V:
                    if j < i:
                        first, last = (i, j) if self.e[i] < self.e[j] else (j, i)
                        pair_feasibility[k, first, last] = self.e[last] >= self.l[first] + self.c(k, first, last)
        return pair_feasibility

    def __determine_client_tasks(self):
        """
        Determine which tasks are associated with each client.
        """
        client_tasks = {}
        for c in self.C:
            client_tasks[c] = self.tasks[self.tasks["ClientID"] == c].index.tolist()
        return client_tasks

    def __determine_qualified_tasks(self):
        """
        Determine which tasks each caregiver is qualified to perform.
        """
        caregiver_tasks = {}
        for k in self.K:
            caregiver_attributes = self.caregivers.loc[k, "Attributes"]
            qualified_clients = self.clients[
                self.clients["Requirements"].apply(lambda req: np.dot(req, caregiver_attributes) == 0)
            ].index.tolist()

            # Filter tasks to only include those with qualified clients
            caregiver_tasks[k] = self.tasks[self.tasks["ClientID"].isin(qualified_clients)].index.tolist()

        return caregiver_tasks

    def determine_feasible_breaks(self, break_length):
        """
        Determine feasible breaks for each caregiver based on their schedule.
        """
        breaks = {}
        for k, i, j in self.A:
            if self.A[k, i, j]:
                breaks[k, i, j] = self.e[j] - self.l[i] - self.c(k, i, j) >= break_length
        return breaks

    # Model building
    @abstractmethod
    def build(self):
        """Build the model. To be implemented by subclasses."""
        pass

    def optimize(self, **kwargs):
        """
        Optimize the model with given parameters

        Args:
            warm_start (bool): Whether to use warm start values from PlannedCaregiverID.
                              Setting this to True can significantly improve solve times
                              by starting from a feasible solution based on existing assignments.
            **kwargs: Optimization parameters like TimeLimit, MIPGap, etc.

        Returns:
            The optimized model
        """
        if self.model is None:
            raise ValueError("Model must be built before optimization")

        # Set any optimization parameters
        for param, value in kwargs.items():
            self.model.setParam(param, value)

        self.model.optimize()
        return self.model

    # Helper functions
    def get_location(self, task_id):
        """
        Get the location of a task given its ID.

        Args:
            task_id (int): The ID of the task

        Returns:
            The location of the task.
        """
        return self.tasks.loc[task_id, "ClientID"]

    def get_endpoint(self, caregiver_id, endpoint):
        """
        Get the endpoint of a caregiver given its ID.

        Args:
            caregiver_id (int): The ID of the caregiver
            endpoint (str): The endpoint to get ("start" or "end")

        Returns:
            The endpoint of the caregiver.
        """
        if endpoint == "start":
            return self.caregivers.loc[caregiver_id, "StartLocation"]
        elif endpoint == "end":
            return self.caregivers.loc[caregiver_id, "EndLocation"]
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

    # Postprocessing
    def _extract_routes(self):
        """
        Extracts the ordered route into a dictionary for each caregiver.
        """
        # First pass: build adjacency lists for each caregiver
        adjacency = {k: {} for k in self.K}

        for k, i, j in self.x:
            if self.x[k, i, j].X > 0.5:
                adjacency[k][i] = j

        # Second pass: traverse adjacency lists to build ordered routes
        routes = {k: [] for k in self.K}

        for k in self.K:
            current = "start"
            while current in adjacency[k] and current != "end":
                next_node = adjacency[k][current]
                routes[k].append((current, next_node))
                current = next_node

        self.routes = routes
        return routes

    def _extract_breaks(self):
        """
        Extracts the break assignments for each caregiver.
        """
        breaks = {k: [] for k in self.K}

        # Iterate over all caregivers and tasks
        for k in self.K:
            for i in self.V:
                if self.B[k, i].X > 0.5:
                    breaks[k].append(i)

        self.breaks = breaks
        return breaks

    def _extract_arrival_times(self):
        """
        Extract the arrival times at each task for each caregiver.
        """
        self.arrivals = {}
        for k in self.K:
            self.arrivals[k] = {}
            self.arrivals[k]["start"] = self.S[k].X
            self.arrivals[k]["end"] = self.E[k].X

            for _, j in self.routes[k]:
                if j == "end":
                    continue
                self.arrivals[k][j] = self.e[j]
        return self.arrivals

    def get_solution(self):
        """
        Get the solution of the optimized model.
        To be implemented by subclasses based on their specific structure.
        """
        if self.model is None:
            raise ValueError("Model must be optimized before extracting the solution")
        if self.model.Status != GRB.OPTIMAL:
            print(f"Model not optimally solved. Status: {self.model.Status}")
        self._extract_routes()
        self._extract_arrival_times()
        if hasattr(self, "B"):
            self._extract_breaks()
        return self.routes, self.arrivals, self.breaks
