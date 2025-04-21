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

        # Preprocessed input data - these are small and used frequently, so keep as dictionaries
        self.K = self.caregivers.index.tolist()
        self.V = self.tasks.index.tolist()
        self.C = self.clients.index.tolist()
        self.s = {i: self.tasks.loc[i, "duration_minutes"] for i in self.V}  # s[i] Service time for i
        self.e = {i: self.tasks.loc[i, "start_minutes"] for i in self.V}  # e[i] Earliest start time for i
        self.l = {i: self.tasks.loc[i, "end_minutes"] for i in self.V}  # l[i] Latest end time for

        # Model variables
        self.model = None
        self.x = None
        self.t = None

        # Postprocessed results
        self.routes = None
        self.arrivals = None
        self.breaks = None

        # Save actual routes and arrival times from PlannedCaregiverID
        self.real_routes, self.real_arrivals = self.__extract_real_routes_and_arrivals()

    def __extract_real_routes_and_arrivals(self):
        """
        Extract the real routes and arrival times based on the PlannedCaregiverID in tasks
        Returns:
            tuple: (real_routes, real_arrivals) dictionaries
        """
        # Initialize dictionaries for real routes and arrivals
        real_routes = {k: [] for k in self.K}
        real_arrivals = {k: {} for k in self.K}

        # Extract unique caregivers from tasks with PlannedCaregiverID
        if "PlannedCaregiverID" not in self.tasks.columns:
            return real_routes, real_arrivals

        # Get caregivers that have assigned tasks
        assigned_caregivers = set(self.tasks["PlannedCaregiverID"].dropna()).intersection(set(self.K))

        # Process each caregiver's route
        for k in assigned_caregivers:
            # Get all tasks for this caregiver sorted by start time
            caregiver_tasks = self.tasks[self.tasks["PlannedCaregiverID"] == k].sort_values("start_minutes")

            if caregiver_tasks.empty:
                continue

            # Set start and end times
            first_task = caregiver_tasks.iloc[0].name
            last_task = caregiver_tasks.iloc[-1].name

            # Calculate start and end times
            real_arrivals[k]["start"] = self.e[first_task] - self.c(k, "start", first_task)
            real_arrivals[k]["end"] = self.l[last_task] + self.c(k, last_task, "end")

            # Build the route
            prev_task = "start"
            for idx, task in caregiver_tasks.iterrows():
                # Add this leg of the route (prev_task -> current_task)
                real_routes[k].append((prev_task, idx))

                # Record arrival time (use the planned start time)
                real_arrivals[k][idx] = task["start_minutes"]

                # Update previous task
                prev_task = idx

            # Add the final leg back to end
            real_routes[k].append((prev_task, "end"))

        return real_routes, real_arrivals

    def is_historically_visited(self, k, c):
        """
        Check if caregiver k has historically visited client c.
        """
        most_visits = eval(self.continuity.loc[c, "MostVisits"])
        return 1 if k in most_visits else 0

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

    def is_pair_feasible(self, k, i, j):
        """
        Check if a pair of tasks is feasible based on time constraints.
        """
        if i == j:
            return False

        # Determine which task comes first in time
        if self.e[i] < self.e[j]:
            first, last = i, j
        else:
            first, last = j, i

        # Check if there's enough time between the tasks
        return self.e[last] >= self.l[first] + self.c(k, first, last)

    def get_client_tasks(self, c, index=False):
        """
        Get tasks associated with client c.

        Args:
            c: Client ID
            index: If True, return row positions in the dataframe. If False, return task indices.

        Returns:
            List of task IDs or positions in the dataframe
        """
        mask = self.tasks["ClientID"] == c
        if index:
            return [i for i, m in enumerate(mask) if m]
        else:
            return self.tasks[mask].index.tolist()

    def is_caregiver_qualified(self, k, task_id):
        """
        Check if caregiver k is qualified to perform task_id.
        """
        caregiver_attributes = self.caregivers.loc[k, "Attributes"]
        client_id = self.get_location(task_id)
        client_requirements = self.clients.loc[client_id, "Requirements"]

        # Check if the caregiver meets all requirements (dot product is 0)
        return np.dot(client_requirements, caregiver_attributes) == 0

    def is_break_feasible(self, k, i, j, break_length):
        """
        Check if a break is feasible between tasks i and j for caregiver k.
        """
        if i == j:
            return False

        # Determine which task comes first in time
        if self.e[i] < self.e[j]:
            first, last = i, j
        else:
            first, last = j, i

        # Check if there's enough time for a break between the tasks
        return self.e[last] - self.l[first] - self.c(k, first, last) >= break_length

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
        self._extract_routes()
        self._extract_arrival_times()
        if hasattr(self, "B"):
            self._extract_breaks()
        return self.routes, self.arrivals, self.breaks
