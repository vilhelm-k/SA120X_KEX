import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


### Need to completely redo the following...
def visualize_schedule(routes, arrivals, travel_times, service_times):
    """
    Create a comprehensive home care schedule visualization using Matplotlib.
    Uses actual travel times from the drive_time_matrix.

    Parameters:
    - routes: Dictionary for each k, with edges (i,j)
    - arrivals: Arrival times. Dictionary for each k, with a list of [task, time]
    - travel_times: c[k,i,j], time for k from i to j
    - service_times: time spent at i

    Returns:
    - fig: Matplotlib figure object
    """
    # 2. Create schedule data for visualization
    schedule_data = []

    for k in routes:
        if routes[k] == []:
            continue
        current_time = arrivals[k, "start"]
        for i, j in routes[k]:
            task_start = arrivals[k, i]
            if i != "start":
                # Waiting
                if current_time < task_start:
                    waiting_duration = task_start - current_time
                    schedule_data.append(
                        {
                            "Caregiver": k,
                            "Activity": "Waiting",
                            "Start": current_time,
                            "Duration": waiting_duration,
                            "Color": "lightgray",
                        }
                    )
                    current_time = task_start
                # Activity
                task_duration = service_times[i]
                schedule_data.append(
                    {
                        "Caregiver": k,
                        "Activity": f"Task {i}",
                        "Start": current_time,
                        "Duration": task_duration,
                        "Color": "skyblue",
                    }
                )
                current_time += task_duration
            # Travel
            travel_duration = travel_times[k, i, j]
            schedule_data.append(
                {
                    "Caregiver": k,
                    "Activity": "Travel",
                    "Start": current_time,
                    "Duration": travel_duration,
                    "Color": "orange",
                }
            )
            current_time += travel_duration
    # 3. Create the visualization
    df = pd.DataFrame(schedule_data)

    if df.empty:
        print("No schedule data to visualize.")
        return None

    # Calculate figure dimensions
    num_caregivers = len(routes)
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
                    f"{row['Activity']}",
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


# Example usage:
# model_subset.optimize()
# fig = visualize_home_care_schedule(model_subset, x_subset, t_subset, caregivers_subset, tasks_subset, drive_time_matrix)
# plt.savefig('home_care_schedule.png', dpi=300, bbox_inches='tight')
# plt.show()
