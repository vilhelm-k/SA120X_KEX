import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all optimization models"""

    # Initialization
    def __init__(self, caregivers, tasks, clients, drive_time_matrix, walk_time_matrix, bicycle_time_matrix):
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

        # Preprocessed input data
        self.K = self.caregivers.index.tolist()
        self.V = self.tasks.index.tolist()
        self.c = self.__calculate_travel_times()  # c[k,i,j] Travel time for k from i to j
        self.s = self.__calculate_service_times()  # s[i] Service time at i
        self.caregiver_tasks = self.__determine_qualified_tasks()  # Qualified tasks for each caregiver

        # Model variables
        self.model = None
        self.x = None
        self.t = None

        # Postprocessed results
        self.routes = None
        self.arrivals = None

    def __calculate_travel_times(self):
        """
        Calculate the travel times between locations for each caregiver and task.
        """
        c = {}
        for k in self.K:
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
            for i in self.V + ["start"]:
                for j in self.V + ["end"]:
                    if i != j and not (i == "start" and j == "end"):
                        # Calculate travel time based on locations
                        if i == "start":
                            # From start location to task
                            start_location = self.get_endpoint(k, "start")
                            travel_time = 0 if start_location == "Home" else time_matrix.loc[0, self.get_location(j)]
                        elif j == "end":
                            # From task to end location
                            end_location = self.get_endpoint(k, "end")
                            travel_time = 0 if end_location == "Home" else time_matrix.loc[self.get_location(i), 0]
                        else:
                            # From task to task
                            travel_time = time_matrix.loc[self.get_location(i), self.get_location(j)]
                        c[k, i, j] = travel_time
        return c

    def __calculate_service_times(self):
        """
        Calculate the service times for each task.
        """
        s = {}
        for i in self.V:
            s[i] = self.tasks.loc[i, "duration_minutes"]
        return s

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

    # Model building
    @abstractmethod
    def build(self):
        """Build the model. To be implemented by subclasses."""
        pass

    def optimize(self, **kwargs):
        """
        Optimize the model with given parameters

        Args:
            **kwargs: Optimization parameters

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
    def __extract_routes(self):
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

    @abstractmethod
    def _extract_arrival_times(self):
        """
        Extracts the arrival times at each task for each caregiver.
        """
        pass

    def get_solution(self):
        """
        Get the solution of the optimized model.
        To be implemented by subclasses based on their specific structure.
        """
        if self.model is None:
            raise ValueError("Model must be optimized before extracting the solution")
        if self.model.Status != GRB.OPTIMAL:
            print(f"Model not optimally solved. Status: {self.model.Status}")
        self.__extract_routes()
        self._extract_arrival_times()

    def get_solution_details(self):
        """
        Get detailed information about the solution.

        Returns:
            dict: Dictionary containing solution details including:
                - routes: Routes for each caregiver
                - arrivals: Arrival times for each caregiver at each task
                - total_travel_time: Total travel time for each caregiver
                - total_waiting_time: Total waiting time for each caregiver
                - utilization: Utilization of each caregiver
        """
        if self.routes is None or self.arrivals is None:
            self.get_solution()

        details = {
            "routes": self.routes,
            "arrivals": self.arrivals,
            "total_travel_time": {},
            "total_service_time": {},
            "total_waiting_time": {},
            "utilization": {},
        }

        # Calculate additional metrics
        for k in self.K:
            # Skip caregivers with no tasks
            if not self.routes[k]:
                continue

            travel_time = 0
            service_time = 0
            waiting_time = 0

            # Process each leg of the route
            for i, j in self.routes[k]:
                if j != "end":
                    # Add travel time
                    travel_time += self.c[k, i, j]

                    # Add service time for task j
                    service_time += self.s[j]

                    # Calculate waiting time if arriving before start window
                    start_minutes = self.tasks.loc[j, "start_minutes"]
                    arrival_time = self.arrivals[k][j]
                    if arrival_time < start_minutes:
                        waiting_time += start_minutes - arrival_time

            details["total_travel_time"][k] = travel_time
            details["total_service_time"][k] = service_time
            details["total_waiting_time"][k] = waiting_time

            # Calculate utilization (service time / total time)
            total_time = self.arrivals[k]["end"] - self.arrivals[k]["start"]
            if total_time > 0:
                details["utilization"][k] = service_time / total_time
            else:
                details["utilization"][k] = 0

        return details
