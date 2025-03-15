import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_routes(model, x, t):
    """
    Extract the routes from the solved Gurobi model and return as a dictionary.

    Parameters:
    - model: Solved Gurobi model
    - x: Dictionary of x variables (caregiver routes)

    Returns:
    - routes: Dictionary of routes for each caregiver
    """
    if model.Status != 2:  # Check if model is solved optimally
        print(f"Model not optimally solved. Status: {model.Status}")

    routes = {}

    for k, i, j in x:
        if x[k, i, j].X > 0.5:
            routes[k] = []
            routes[k].append((i, j))
    return routes


def calculate_arrival_times(model, x, t):
    """
    Extract the arrival times from the solved Gurobi model and return as a dictionary.

    Parameters:
    - model: Solved Gurobi model
    - x: Dictionary of x variables (caregiver routes)
    - t: Dictionary of t variables (arrival times)

    Returns:
    - arrival_times: Dictionary of arrival times for each caregiver
    """
    if model.Status != 2:  # Check if model is solved optimally
        print(f"Model not optimally solved. Status: {model.Status}")

    routes = calculate_routes(model, x, t)

    arrival_times = {}

    for k in routes:
        all_visited_nodes = set([i for i, j in routes[k]] + [j for i, j in routes[k]])
        for node in all_visited_nodes:
            arrival_times[k, node] = t[k, node].X
    return arrival_times


def visualize_home_care_schedule(
    model, x, t, caregivers, tasks, drive_time_matrix, walk_time_matrix=None, bike_time_matrix=None
):
    """
    Create a comprehensive home care schedule visualization using Matplotlib.
    Uses actual travel times from the drive_time_matrix.

    Parameters:
    - model: Solved Gurobi model
    - x: Dictionary of x variables (caregiver routes)
    - t: Dictionary of t variables (arrival times)
    - caregivers: DataFrame of caregiver data
    - tasks: DataFrame of task data
    - drive_time_matrix: Matrix of travel times between locations

    Returns:
    - fig: Matplotlib figure object
    """
    if model.Status != 2:  # Check if model is solved optimally
        print(f"Model not optimally solved. Status: {model.Status}")

    # 1. Extract solution data
    routes = {}
    arrival_times = {}

    for k in caregivers.index:
        routes[k] = []
        arrival_times[k] = {}

        # Get start time
        arrival_times[k]["start"] = t[k, "start"].X if (k, "start") in t else 0
        arrival_times[k]["end"] = t[k, "end"].X if (k, "end") in t else 0

        # Find all tasks visited by this caregiver
        for i in ["start"] + tasks.index.tolist():
            for j in tasks.index.tolist() + ["end"]:
                if i != j and (k, i, j) in x and x[k, i, j].X > 0.5:
                    if j != "end":  # Only add tasks, not the end node
                        routes[k].append(j)
                    if j != "end":
                        arrival_times[k][j] = t[k, j].X

    # Helper function to safely get travel time and ensure scalar return
    def get_travel_time(from_id, to_id):
        """Safely get travel time between locations, ensuring a scalar return value"""
        try:
            # First try direct lookup
            result = drive_time_matrix.loc[from_id, to_id]
            # If this is a Series or DataFrame, extract a scalar
            if isinstance(result, (pd.Series, pd.DataFrame)):
                # For simplicity, just use the first value or the mean
                return float(result.iloc[0] if hasattr(result, "iloc") else result.mean())
            return float(result)  # Convert to float if it's already a scalar
        except (KeyError, TypeError):
            try:
                # If HQ (0) is not in matrix, use first task's time as approximation
                if from_id == 0 and to_id in drive_time_matrix.columns:
                    # Use average time to location as an approximation for HQ to location
                    return float(drive_time_matrix[to_id].mean())
                elif to_id == 0 and from_id in drive_time_matrix.index:
                    # Use average time from location as an approximation for location to HQ
                    return float(drive_time_matrix.loc[from_id].mean())
                else:
                    # If both locations are valid in the matrix
                    if from_id in drive_time_matrix.index and to_id in drive_time_matrix.columns:
                        result = drive_time_matrix.loc[from_id, to_id]
                        if isinstance(result, (pd.Series, pd.DataFrame)):
                            return float(result.iloc[0] if hasattr(result, "iloc") else result.mean())
                        return float(result)
                    else:
                        # Default value if we can't determine
                        return 15.0  # Default 15 minutes
            except Exception as e:
                # Final fallback
                print(f"Error calculating travel time ({from_id} to {to_id}): {e}")
                return 15.0  # Default 15 minutes

    # 2. Create schedule data for visualization
    schedule_data = []

    for k in caregivers.index:
        start_time = arrival_times[k]["start"]
        current_time = start_time

        # Process each task
        prev_location = "Home"
        prev_client_id = None  # Will handle specially for first task

        for i, task_id in enumerate(routes[k]):
            task_start = arrival_times[k][task_id]
            task_duration = tasks.loc[task_id, "duration_minutes"]
            client_id = tasks.loc[task_id, "ClientID"]

            # Calculate actual travel time from previous location
            if prev_location == "Home" or prev_client_id is None:
                # First task, traveling from depot/home
                start_loc = caregivers.loc[k, "StartLocation"] if "StartLocation" in caregivers.columns else "Home"
                if start_loc == "Home":
                    travel_time = 0.0  # No travel needed if starting at home
                else:
                    # Approximate travel time from HQ to first client
                    travel_time = get_travel_time(0, client_id)
            else:
                # Travel between clients
                travel_time = get_travel_time(prev_client_id, client_id)

            # Add travel time from previous location if applicable
            travel_time_float = float(travel_time)  # Ensure scalar
            if travel_time_float > 0:
                from_text = "From Home" if prev_client_id is None else f"From Client {prev_client_id}"
                schedule_data.append(
                    {
                        "Caregiver": k,
                        "Activity": "Travel",
                        "Start": current_time,
                        "Duration": travel_time_float,
                        "Client": f"{from_text} to {client_id}",
                        "Color": "orange",
                    }
                )
                current_time += travel_time_float

            # Add waiting time if there's still a gap after travel
            if current_time < task_start:
                waiting_duration = task_start - current_time
                schedule_data.append(
                    {
                        "Caregiver": k,
                        "Activity": "Waiting",
                        "Start": current_time,
                        "Duration": waiting_duration,
                        "Client": None,
                        "Color": "lightgray",
                    }
                )
                current_time = task_start

            # Add the task
            schedule_data.append(
                {
                    "Caregiver": k,
                    "Activity": f"Task {task_id}",
                    "Start": current_time,
                    "Duration": task_duration,
                    "Client": f"Client {client_id}",
                    "Color": "skyblue",
                }
            )
            current_time += task_duration

            # Update previous location for next iteration
            prev_location = "Client"
            prev_client_id = client_id

        # Handle travel to end location if applicable
        if routes[k] and "end" in arrival_times[k]:
            end_location = caregivers.loc[k, "EndLocation"] if "EndLocation" in caregivers.columns else "Home"
            if end_location != "Home":
                # Get last client ID
                last_client_id = tasks.loc[routes[k][-1], "ClientID"]

                # Add travel to HQ/depot
                travel_time = get_travel_time(last_client_id, 0)  # Using safe travel time function
                travel_time_float = float(travel_time)  # Ensure scalar

                if travel_time_float > 0:
                    schedule_data.append(
                        {
                            "Caregiver": k,
                            "Activity": "Travel",
                            "Start": current_time,
                            "Duration": travel_time_float,
                            "Client": "To End (HQ)",
                            "Color": "orange",
                        }
                    )
                    current_time += travel_time_float

            # Add any remaining waiting time until official end
            if current_time < arrival_times[k]["end"]:
                waiting_duration = arrival_times[k]["end"] - current_time
                schedule_data.append(
                    {
                        "Caregiver": k,
                        "Activity": "Waiting",
                        "Start": current_time,
                        "Duration": waiting_duration,
                        "Client": None,
                        "Color": "lightgray",
                    }
                )

    # 3. Create the visualization
    df = pd.DataFrame(schedule_data)

    if df.empty:
        print("No schedule data to visualize.")
        return None

    # Calculate figure dimensions
    num_caregivers = len(caregivers)
    fig_height = max(8, num_caregivers * 1.5)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, fig_height))

    # Get y positions for each caregiver
    caregivers_list = sorted(df["Caregiver"].unique())
    y_positions = {cg: i for i, cg in enumerate(caregivers_list)}

    # Calculate time range
    min_time = df["Start"].min()
    max_time = (df["Start"] + df["Duration"]).max()

    # Plot each activity as a colored rectangle
    for _, row in df.iterrows():
        y_pos = y_positions[row["Caregiver"]]
        x_start = row["Start"]
        width = row["Duration"]

        rect = patches.Rectangle(
            (x_start, y_pos - 0.4), width, 0.8, linewidth=1, edgecolor="black", facecolor=row["Color"], alpha=0.7
        )
        ax.add_patch(rect)

        # Add text labels for tasks
        if "Task" in row["Activity"]:
            # Only add text if there's enough space
            if width > 30:
                ax.text(
                    x_start + width / 2,
                    y_pos,
                    f"{row['Activity']}\n{row['Client']}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        elif row["Activity"] == "Travel" and width > 15:
            # Add travel labels if there's enough space
            ax.text(x_start + width / 2, y_pos, "Travel", ha="center", va="center", fontsize=8, color="black")

    # Set up the axis
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"Caregiver {cg}" for cg in y_positions.keys()])
    ax.set_ylim(-0.5, len(y_positions) - 0.5)

    # Set x-axis to show hours
    max_hours = np.ceil(max_time / 60)
    hour_ticks = np.arange(0, max_hours * 60 + 1, 60)
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{int(h/60)}:00" for h in hour_ticks])
    ax.set_xlim(min_time - 15, max_time + 15)

    # Add grid, title and labels
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    ax.set_title("Home Care Scheduling Solution", fontsize=16)
    ax.set_xlabel("Time (hours)", fontsize=12)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor="skyblue", edgecolor="black", alpha=0.7, label="Task"),
        patches.Patch(facecolor="orange", edgecolor="black", alpha=0.7, label="Travel"),
        patches.Patch(facecolor="lightgray", edgecolor="black", alpha=0.7, label="Waiting"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # Highlight task time windows (safely handle any errors)
    try:
        for _, task in tasks.iterrows():
            # Add a very light background rectangle to show the time window
            window_rect = patches.Rectangle(
                (task["start_minutes"], -0.5),
                task["end_minutes"] - task["start_minutes"],
                len(caregivers_list),
                linewidth=0,
                facecolor="lightgray",
                alpha=0.1,
                zorder=0,  # Ensure it's behind everything
            )
            ax.add_patch(window_rect)
    except Exception as e:
        print(f"Warning: Could not draw time windows: {e}")

    # Add summary statistics
    total_task_time = df[df["Activity"].str.contains("Task")]["Duration"].sum()
    total_travel_time = df[df["Activity"] == "Travel"]["Duration"].sum()
    total_waiting_time = df[df["Activity"] == "Waiting"]["Duration"].sum()

    stats_text = (
        f"Total Statistics:\n"
        f"Task Time: {total_task_time:.0f} min\n"
        f"Travel Time: {total_travel_time:.0f} min\n"
        f"Waiting Time: {total_waiting_time:.0f} min"
    )

    # Add text box for statistics
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)
    return fig


# Example usage:
# model_subset.optimize()
# fig = visualize_home_care_schedule(model_subset, x_subset, t_subset, caregivers_subset, tasks_subset, drive_time_matrix)
# plt.savefig('home_care_schedule.png', dpi=300, bbox_inches='tight')
# plt.show()
