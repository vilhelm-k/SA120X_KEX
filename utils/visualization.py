import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from utils.metrics import calculate_metrics


def visualize_schedule(model):
    """
    Create a home care schedule visualization using Matplotlib.
    Shows each caregiver's route with indices for tasks and highlights breaks.

    Parameters:
    - model: A solved optimization model instance

    Returns:
    - fig: Matplotlib figure object
    """
    # Check if model has been solved
    if model.routes is None or model.arrivals is None:
        raise ValueError("Model must be solved before visualization.")

    # Extract data from the model
    routes = model.routes
    arrivals = model.arrivals
    service_times = model.s
    
    # Check if breaks exist in the model
    has_breaks = hasattr(model, 'breaks') and model.breaks is not None

    # Create schedule data for visualization
    schedule_data = []

    for k in model.K:
        if not routes[k]:  # Skip empty routes
            continue

        current_time = arrivals[k]["start"]
        for index, (i, j) in enumerate(routes[k]):
            if i != "start":
                # Get actual arrival time
                task_arrival = arrivals[k][i]

                # Add waiting if arriving before task start
                if current_time < task_arrival:
                    waiting_duration = task_arrival - current_time
                    schedule_data.append(
                        {
                            "Caregiver": k,
                            "Activity": "Waiting",
                            "Index": None,
                            "Start": current_time,
                            "Duration": waiting_duration,
                            "Color": "lightgray",
                        }
                    )
                    current_time = task_arrival

                # Activity
                task_duration = service_times[i]
                schedule_data.append(
                    {
                        "Caregiver": k,
                        "Activity": f"Task {i}",
                        "Index": index,
                        "Start": current_time,
                        "Duration": task_duration,
                        "Color": "skyblue",
                    }
                )
                current_time += task_duration
                
                # Add break if this task is followed by a break
                if has_breaks and k in model.breaks and i in model.breaks[k]:
                    # Assume break is 30 minutes (or adjust as needed)
                    break_duration = 30  # This could be parametrized or determined from model
                    schedule_data.append(
                        {
                            "Caregiver": k,
                            "Activity": "Break",
                            "Index": None,
                            "Start": current_time,
                            "Duration": break_duration,
                            "Color": "lightgreen",
                        }
                    )
                    current_time += break_duration

            # Travel
            travel_duration = model.c(k, i, j)
            schedule_data.append(
                {
                    "Caregiver": k,
                    "Activity": "Travel",
                    "Index": index,
                    "Start": current_time,
                    "Duration": travel_duration,
                    "Color": "orange",
                }
            )
            current_time += travel_duration

        # Handle waiting at the end
        if current_time < arrivals[k]["end"]:
            waiting_duration = arrivals[k]["end"] - current_time
            schedule_data.append(
                {
                    "Caregiver": k,
                    "Activity": "Waiting",
                    "Index": None,
                    "Start": current_time,
                    "Duration": waiting_duration,
                    "Color": "lightgray",
                }
            )

    # Create the visualization
    df = pd.DataFrame(schedule_data)

    if df.empty:
        print("No schedule data to visualize.")
        return None

    # Calculate figure dimensions
    num_caregivers = len([k for k in routes if routes[k]])
    fig_height = max(6, num_caregivers * 0.8)  # Half the height per caregiver

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, fig_height))

    # Get y positions for each caregiver in reverse order
    y_positions = {cg: len(model.K) - 1 - i for i, cg in enumerate(model.K)}

    # Calculate time range
    min_time = df["Start"].min()
    max_time = (df["Start"] + df["Duration"]).max()

    # Plot each activity as a colored rectangle
    for _, row in df.iterrows():
        y_pos = y_positions[row["Caregiver"]]
        x_start = row["Start"]
        width = row["Duration"]

        # Make the rectangles half as tall (0.4 -> 0.2) but keep same spacing
        rect = patches.Rectangle(
            (x_start, y_pos - 0.4), width, 0.8, linewidth=1, edgecolor="black", facecolor=row["Color"], alpha=0.7
        )
        ax.add_patch(rect)

        # Add text labels for tasks (not for breaks)
        if "Task" in row["Activity"]:
            # Determine what text to display based on width
            if width >= 40:
                # If box is wide enough, show task info
                task_id = row["Activity"].split(" ")[1]
                label = f"{task_id}\n({int(row['Index'])})"
            else:
                # For narrow boxes, just show index
                label = f"{int(row['Index'])}"

            ax.text(
                x_start + width / 2,
                y_pos,
                label,
                ha="center",
                va="center",
                fontsize=8,  # Slightly smaller font for smaller rectangles
                fontweight="bold",
            )
        # No longer adding text for breaks

    # Set up the axis
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(
        [f"Caregiver {cg}\n {model.caregivers.loc[cg, 'ModeOfTransport']}" for cg in y_positions.keys()]
    )
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
    
    # Add break to legend if breaks exist
    if has_breaks:
        legend_elements.append(
            patches.Patch(facecolor="lightgreen", edgecolor="black", alpha=0.7, label="Break")
        )
        
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


def display_metrics_summary(model):
    """
    Display a summary of the most relevant metrics as a formatted table.
    
    Parameters:
    - model: A solved optimization model instance
    
    Returns:
    - None (prints the summary table to the console)
    """
    # Calculate metrics
    metrics = calculate_metrics(model)
    aggregate_metrics = metrics["aggregate_metrics"]
    agg = aggregate_metrics
    continuity = aggregate_metrics["continuity"]["system_continuity"]
    
    # Create summary table with formatting
    print("\n" + "="*80)
    print(" "*30 + "HOME CARE SCHEDULE METRICS SUMMARY")
    print("="*80)
    
    # Basic metrics
    print("\nSCHEDULE OVERVIEW:")
    print(f"  Total tasks:              {agg['total']['number_of_tasks']}")
    print(f"  Active caregivers:        {agg['active_caregivers']}/{agg['total_caregivers']}")
    
    # Time allocation
    print("\nTIME ALLOCATION:")
    print(f"  Service time:             {agg['total']['service_time']:.0f} min ({agg['proportions']['service']:.1f}%)")
    print(f"  Travel time:              {agg['total']['travel_time']:.0f} min ({agg['proportions']['travel']:.1f}%)")
    print(f"  Waiting time:             {agg['total']['waiting_time']:.0f} min ({agg['proportions']['waiting']:.1f}%)")
    print(f"  Break time:               {agg['total']['break_time']:.0f} min ({agg['proportions']['break']:.1f}%)")
    print(f"  Total schedule time:      {agg['total']['schedule_time']:.0f} min")
    
    # Continuity metrics
    print("\nCONTINUITY METRICS:")
    print(f"  Historical visits:        {continuity['total_historical_tasks']}/{continuity['total_tasks']} ({continuity['historical_task_percentage']:.1f}%)")
    print(f"  Avg caregivers per client: {continuity['avg_caregivers_per_client']:.1f}")
    print(f"  Historical continuity:    {continuity['avg_historical_continuity']:.1f}%")
    print(f"  Perfect continuity:       {continuity['perfect_continuity_clients']} clients")
    print(f"  Perfect historical:       {continuity['perfect_historical_continuity_clients']} clients")
    
    # Average caregiver stats
    print("\nAVERAGE PER CAREGIVER:")
    print(f"  Tasks:                    {agg['average']['tasks_per_caregiver']:.1f}")
    print(f"  Service time:             {agg['average']['service_time']:.0f} min")
    print(f"  Travel time:              {agg['average']['travel_time']:.0f} min")
    print(f"  Waiting time:             {agg['average']['waiting_time']:.0f} min")
    print(f"  Break time:               {agg['average']['break_time']:.0f} min")
    print(f"  Utilization:              {agg['average']['utilization']:.1f}%")
    
    print("="*80 + "\n")
    
    return None


def visualize_metrics(model, display_mode="dashboard", dashboard_figsize=(20, 10)):
    """
    Create visualizations of the metrics from the model solution.

    Parameters:
    - model: A solved optimization model instance
    - display_mode: String, controls which figures to display
        - 'dashboard': Only display the dashboard (default)
        - 'individual': Only display the individual figures
        - 'all': Display both dashboard and individual figures
        - 'none': Don't display any figures, just return them
    - dashboard_figsize: Size of the dashboard figure

    Returns:
    - If display_mode is 'dashboard': The dashboard figure
    - If display_mode is 'individual': A tuple of individual figures (fig1, fig2, fig3, fig4, fig6, fig7)
    - If display_mode is 'all' or 'none': A tuple of all figures (fig1, fig2, fig3, fig4, fig6, fig7, dashboard_fig)
    """
    
    # Turn off interactive mode to prevent automatic display
    was_interactive = plt.isinteractive()
    if was_interactive:
        plt.ioff()
    metrics = calculate_metrics(model)
    caregiver_metrics = metrics["caregiver_metrics"]
    aggregate_metrics = metrics["aggregate_metrics"]
    continuity_metrics = aggregate_metrics["continuity"]

    # Only include caregivers with assigned tasks
    active_caregivers = {k: v for k, v in caregiver_metrics.items() if v["number_of_tasks"] > 0}

    # Create figures only if there are active caregivers
    if not active_caregivers:
        print("No active caregivers to visualize.")
        return None

    # Get list of caregivers and use consistent indices
    caregivers = list(active_caregivers.keys())
    index = np.arange(len(caregivers))  # Use numeric indices for x-axis positions
    caregiver_labels = [str(cg) for cg in caregivers]  # Convert all caregiver IDs to strings for labels
    
    # Safety check - ensure lengths match
    if len(index) != len(caregivers) or len(index) != len(caregiver_labels):
        print(f"Warning: Mismatch in array lengths: index={len(index)}, caregivers={len(caregivers)}, labels={len(caregiver_labels)}")
        # Try to correct by using the smallest length
        min_len = min(len(index), len(caregivers), len(caregiver_labels))
        index = index[:min_len]
        caregivers = caregivers[:min_len]
        caregiver_labels = caregiver_labels[:min_len]

    # 1. Time allocation bar chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Prepare data for stacked bar chart
    service_times = [active_caregivers[k]["service_time"] for k in caregivers]
    travel_times = [active_caregivers[k]["travel_time"] for k in caregivers]
    waiting_times = [active_caregivers[k]["waiting_time"] for k in caregivers]
    break_times = [active_caregivers[k]["break_time"] for k in caregivers]

    # Create stacked bar chart using index positions
    ax1.bar(index, service_times, label="Service Time", color="skyblue")

    # Calculate bottom positions for stacking
    service_bottom = service_times
    ax1.bar(index, travel_times, bottom=service_bottom, label="Travel Time", color="orange")

    # Calculate bottom for waiting times
    service_travel_bottom = [s + t for s, t in zip(service_times, travel_times)]
    ax1.bar(index, waiting_times, bottom=service_travel_bottom, label="Waiting Time", color="lightgray")
    
    # Add break times
    service_travel_waiting_bottom = [s + t + w for s, t, w in zip(service_times, travel_times, waiting_times)]
    ax1.bar(index, break_times, bottom=service_travel_waiting_bottom, label="Break Time", color="lightgreen")

    # Add total time as text on top of each bar
    for i, k in enumerate(caregivers):
        total_time = active_caregivers[k]["total_time"]
        ax1.text(
            i, sum([service_times[i], travel_times[i], waiting_times[i], break_times[i]]) + 5, f"{total_time:.0f} min", ha="center"
        )

    # Set x-ticks to show actual caregiver IDs
    ax1.set_xticks(index)
    ax1.set_xticklabels(caregiver_labels, rotation=45, ha='right')

    ax1.set_xlabel("Caregiver")
    ax1.set_ylabel("Time (minutes)")
    ax1.set_title("Time Allocation by Caregiver")
    ax1.legend()

    # 2. Utilization chart
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    utilizations = [active_caregivers[k]["utilization"] for k in caregivers]

    # Use index positions for x-axis
    ax2.bar(index, utilizations, color="skyblue")

    # Set x-ticks to show actual caregiver IDs
    ax2.set_xticks(index)
    ax2.set_xticklabels(caregiver_labels, rotation=45, ha='right')

    ax2.set_xlabel("Caregiver")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_title("Caregiver Utilization")
    ax2.set_ylim(0, 100)

    # Add average utilization line
    avg_util = aggregate_metrics["average"]["utilization"]
    ax2.axhline(y=avg_util, linestyle="--", color="red", label=f"Average: {avg_util:.1f}%")
    ax2.legend()

    # 3. Pie chart of overall time allocation
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    labels = ["Service Time", "Travel Time", "Waiting Time", "Break Time"]
    sizes = [
        aggregate_metrics["total"]["service_time"],
        aggregate_metrics["total"]["travel_time"],
        aggregate_metrics["total"]["waiting_time"],
        aggregate_metrics["total"]["break_time"],
    ]
    colors = ["skyblue", "orange", "lightgray", "lightgreen"]

    # Calculate percentages for labels
    total = sum(sizes)
    percentages = [100 * size / total if total > 0 else 0 for size in sizes]
    label_texts = [f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)]

    # Create the pie chart
    wedges, texts, autotexts = ax3.pie(
        sizes,
        labels=label_texts,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
        explode=(0.05, 0, 0, 0),  # Slightly explode the service time slice
    )

    # Improve text legibility
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax3.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    ax3.set_title("Overall Time Allocation")

    # 4. Tasks per caregiver
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    tasks_per_caregiver = [active_caregivers[k]["number_of_tasks"] for k in caregivers]

    # Use index positions for x-axis
    ax4.bar(index, tasks_per_caregiver, color="skyblue")

    # Set x-ticks to show actual caregiver IDs
    ax4.set_xticks(index)
    ax4.set_xticklabels(caregiver_labels, rotation=45, ha='right')

    ax4.set_xlabel("Caregiver")
    ax4.set_ylabel("Number of Tasks")
    ax4.set_title("Tasks Assigned per Caregiver")

    # Add average line
    avg_tasks = aggregate_metrics["average"]["tasks_per_caregiver"]
    ax4.axhline(y=avg_tasks, linestyle="--", color="red", label=f"Average: {avg_tasks:.1f}")
    ax4.legend()

    # NEW: 6. Historical visits pie chart
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    
    # Get all client data and system continuity metrics
    client_continuity = continuity_metrics["client_continuity"]
    system_continuity = continuity_metrics["system_continuity"]
    
    # Get historical vs non-historical visit counts
    historical_visits = system_continuity["total_historical_tasks"]
    non_historical_visits = system_continuity["total_tasks"] - historical_visits
    
    # Create pie chart
    hist_labels = ["Historical Caregiver Visits", "Non-Historical Caregiver Visits"]
    hist_sizes = [historical_visits, non_historical_visits]
    hist_colors = ["#4CAF50", "#F44336"]  # Green for historical, red for non-historical
    
    historical_percentage = system_continuity["historical_task_percentage"]
    hist_label_texts = [
        f"Historical ({historical_percentage:.1f}%)", 
        f"Non-Historical ({100-historical_percentage:.1f}%)"
    ]
    
    wedges, texts, autotexts = ax6.pie(
        hist_sizes,
        labels=hist_label_texts,
        colors=hist_colors,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
        explode=(0.05, 0),  # Explode the historical slice
    )
    
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
        
    ax6.axis("equal")
    ax6.set_title("Proportion of Visits by Historical vs. Non-Historical Caregivers")
    
    # NEW: 7. Client visits and caregivers chart
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    
    # Group clients by number of visits
    visit_groups = {}
    for client_id, data in client_continuity.items():
        visits = data["total_tasks"]
        if visits not in visit_groups:
            visit_groups[visits] = []
        visit_groups[visits].append(data["unique_caregivers"])
    
    # Prepare data for plot
    visit_counts = sorted(visit_groups.keys())
    caregiver_counts = [visit_groups[vc] for vc in visit_counts]
    
    # Safety check - need at least one data point
    if not visit_counts or len(visit_counts) == 0:
        ax7.text(0.5, 0.5, "No client visit data available", 
                ha='center', va='center', fontsize=14, transform=ax7.transAxes)
        ax7.set_title("Distribution of Caregivers per Client Grouped by Visit Count")
    else:
        # Calculate average caregivers per visit group
        avg_caregivers = [np.mean(caregivers) if caregivers else 0 for caregivers in caregiver_counts]
        
        # Create positions for x-axis
        positions = np.arange(len(visit_counts))
        
        # Safety check for violin plot - needs at least 2 data points per group
        valid_violin_data = []
        valid_positions = []
        
        for i, caregivers in enumerate(caregiver_counts):
            # Add individual data points with jitter
            x = np.ones_like(caregivers) * i
            jitter = np.random.normal(0, 0.04, size=len(caregivers))
            ax7.scatter(x + jitter, caregivers, color='blue', alpha=0.5, s=40)
            
            # Only include groups with enough data for violin plot
            if len(caregivers) >= 2:
                valid_violin_data.append(caregivers)
                valid_positions.append(positions[i])
            
            # Add average as text
            if caregivers:  # Only add if there's data
                max_val = max(caregivers)
                ax7.text(i, max_val + 0.2, f"Avg: {avg_caregivers[i]:.1f}", ha='center')
        
        # Create violin plots only if we have valid data
        if valid_violin_data:
            violin_parts = ax7.violinplot(valid_violin_data, positions=valid_positions, showmedians=True)
            
            # Change violin colors
            for pc in violin_parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)
        
        # Set axis labels and title
        ax7.set_xticks(positions)
        ax7.set_xticklabels([f"{vc} Visits" for vc in visit_counts])
        ax7.set_xlabel("Number of Visits per Client")
        ax7.set_ylabel("Number of Unique Caregivers")
        ax7.set_title("Distribution of Caregivers per Client Grouped by Visit Count")
        
        # Add grid lines
        ax7.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure y-axis starts at 0 and has integer ticks
        max_caregivers = max([max(cg) if cg else 0 for cg in caregiver_counts], default=1)
        ax7.set_ylim(0, max_caregivers + 1)
        ax7.set_yticks(np.arange(0, max_caregivers + 1))

    # Create the dashboard figure
    dashboard_fig = plt.figure(figsize=dashboard_figsize)
    gs = gridspec.GridSpec(2, 3, figure=dashboard_fig)

    # We'll create the plots directly on the dashboard figure
    # Handle the case where there's only one active caregiver
    x_positions = index  # Same positions as used for individual charts

    # 1. Time allocation bar chart (top left)
    dash_ax1 = dashboard_fig.add_subplot(gs[0, 0])
    dash_ax1.bar(x_positions, service_times, label="Service", color="skyblue")
    dash_ax1.bar(x_positions, travel_times, bottom=service_bottom, label="Travel", color="orange")
    dash_ax1.bar(x_positions, waiting_times, bottom=service_travel_bottom, label="Waiting", color="lightgray")
    dash_ax1.bar(x_positions, break_times, bottom=service_travel_waiting_bottom, label="Break", color="lightgreen")
    
    # Ensure tick positions and labels match exactly
    dash_ax1.set_xticks(x_positions)
    dash_ax1.set_xticklabels(caregiver_labels, rotation=45, ha='right')
    dash_ax1.set_xlabel("Caregiver")
    dash_ax1.set_ylabel("Time (minutes)")
    dash_ax1.set_title("Time Allocation by Caregiver")
    dash_ax1.legend()

    # 2. Overall time allocation pie chart (top middle)
    dash_ax2 = dashboard_fig.add_subplot(gs[0, 1])
    dash_ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    dash_ax2.axis("equal")
    dash_ax2.set_title("Overall Time Allocation")

    # 3. Historical visits pie chart (top right)
    dash_ax3 = dashboard_fig.add_subplot(gs[0, 2])
    dash_ax3.pie(hist_sizes, labels=hist_labels, colors=hist_colors, autopct="%1.1f%%", startangle=90)
    dash_ax3.axis("equal")
    dash_ax3.set_title("Historical vs. Non-Historical Visits")

    # 4. Utilization chart (bottom left)
    dash_ax4 = dashboard_fig.add_subplot(gs[1, 0])
    dash_ax4.bar(x_positions, utilizations, color="skyblue")
    
    # Ensure tick positions and labels match exactly
    dash_ax4.set_xticks(x_positions)
    dash_ax4.set_xticklabels(caregiver_labels, rotation=45, ha='right')
    dash_ax4.axhline(y=avg_util, linestyle="--", color="red", label=f"Avg: {avg_util:.1f}%")
    dash_ax4.set_xlabel("Caregiver")
    dash_ax4.set_ylabel("Utilization (%)")
    dash_ax4.set_title("Caregiver Utilization")
    dash_ax4.set_ylim(0, 100)
    dash_ax4.legend()

    # 5. Tasks per caregiver (bottom middle)
    dash_ax5 = dashboard_fig.add_subplot(gs[1, 1])
    dash_ax5.bar(x_positions, tasks_per_caregiver, color="skyblue")
    
    # Ensure tick positions and labels match exactly
    dash_ax5.set_xticks(x_positions)
    dash_ax5.set_xticklabels(caregiver_labels, rotation=45, ha='right')
    dash_ax5.axhline(y=avg_tasks, linestyle="--", color="red", label=f"Avg: {avg_tasks:.1f}")
    dash_ax5.set_xlabel("Caregiver")
    dash_ax5.set_ylabel("Number of Tasks")
    dash_ax5.set_title("Tasks Assigned per Caregiver")
    dash_ax5.legend()

    # 6. Client visits and caregivers chart (bottom right) - replaces the text summary
    dash_ax6 = dashboard_fig.add_subplot(gs[1, 2])
    
    # Get visit groups data for violin plot
    client_continuity = continuity_metrics["client_continuity"]
    visit_groups = {}
    for client_id, data in client_continuity.items():
        visits = data["total_tasks"]
        if visits not in visit_groups:
            visit_groups[visits] = []
        visit_groups[visits].append(data["unique_caregivers"])
    
    # Prepare data for plot
    visit_counts = sorted(visit_groups.keys())
    caregiver_counts = [visit_groups[vc] for vc in visit_counts]
    
    # Safety check - need at least one data point
    if not visit_counts or len(visit_counts) == 0:
        dash_ax6.text(0.5, 0.5, "No client visit data available", 
                ha='center', va='center', fontsize=14, transform=dash_ax6.transAxes)
        dash_ax6.set_title("Caregivers per Client by Visit Count")
    else:
        # Create positions for x-axis
        positions = np.arange(len(visit_counts))
        
        # Add individual data points with jitter
        for i, caregivers in enumerate(caregiver_counts):
            x = np.ones_like(caregivers) * i
            jitter = np.random.normal(0, 0.04, size=len(caregivers))
            dash_ax6.scatter(x + jitter, caregivers, color='blue', alpha=0.5, s=40)
        
        # Only create violin plots if we have enough data
        valid_violin_data = []
        valid_positions = []
        
        for i, caregivers in enumerate(caregiver_counts):
            if len(caregivers) >= 2:
                valid_violin_data.append(caregivers)
                valid_positions.append(positions[i])
        
        if valid_violin_data:
            violin_parts = dash_ax6.violinplot(valid_violin_data, positions=valid_positions, showmedians=True)
            
            # Change violin colors
            for pc in violin_parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)
        
        # Set axis labels and ticks
        dash_ax6.set_xticks(positions)
        dash_ax6.set_xticklabels([f"{vc}" for vc in visit_counts])
        dash_ax6.set_xlabel("Visits per Client")
        dash_ax6.set_ylabel("Caregivers")
        dash_ax6.set_title("Caregivers per Client by Visit Count")
        
        # Add grid and set y-axis limits
        dash_ax6.grid(True, linestyle='--', alpha=0.7)
        max_caregivers = max([max(cg) if cg else 0 for cg in caregiver_counts], default=1)
        dash_ax6.set_ylim(0, max_caregivers + 1)
        dash_ax6.set_yticks(np.arange(0, max_caregivers + 1))

    plt.tight_layout()

    # Handle display based on display_mode
    individual_figs = [fig1, fig2, fig3, fig4, fig6, fig7]
    all_figs = individual_figs + [dashboard_fig]

    # Close figures we don't want to display
    if display_mode == "dashboard":
        for fig in individual_figs:
            plt.close(fig)
        # Only display the dashboard
        if was_interactive:
            plt.ion()  # Restore interactive mode
            dashboard_fig.show()
        return dashboard_fig

    elif display_mode == "individual":
        plt.close(dashboard_fig)
        # Only display individual figures
        if was_interactive:
            plt.ion()  # Restore interactive mode
            for fig in individual_figs:
                fig.show()
        return tuple(individual_figs)

    elif display_mode == "all":
        # Display all figures
        if was_interactive:
            plt.ion()  # Restore interactive mode
            for fig in all_figs:
                fig.show()
        return tuple(all_figs)

    else:  # 'none' or any other value
        # Don't display any figures, just return them
        for fig in all_figs:
            plt.close(fig)
        if was_interactive:
            plt.ion()  # Restore interactive mode
        return tuple(all_figs)


def visualize_routes(model, caregiver_ids=None, subplot_mode=False, figsize=None, dpi=100):
    """
    Visualize routes taken by selected caregivers with a simple, clean layout.

    Args:
        model: The optimized BaseModel or FlexibleModel instance with solution
        caregiver_ids: List of caregiver IDs to display (if None, all caregivers are shown)
        figsize: Size of the figure to create
        dpi: Resolution of the figure
        subplot_mode: If True, create a separate subplot for each caregiver;
                     if False, plot all routes on the same axes (default)

    Returns:
        If subplot_mode is False:
            matplotlib figure and single axis object
        If subplot_mode is True:
            matplotlib figure and array of axis objects (one per caregiver)
    """

    # Use model.clients
    clients_df = model.clients

    # Extract solution if not already done
    if model.routes is None or model.arrivals is None:
        raise ValueError("Model must be solved before visualization.")

    # If caregiver_ids is None, use all caregivers
    if caregiver_ids is None:
        caregiver_ids = model.K

    if figsize is None:
        height = 4 / 3 * len(caregiver_ids) if subplot_mode else 8
        figsize = (10, max(8, height))

    # Set up colors for caregivers - use distinct colors
    colors = list(mcolors.TABLEAU_COLORS)

    # Get active clients
    active_clients = model.tasks["ClientID"].unique()
    active_client_data = clients_df.loc[clients_df.index.isin(active_clients)]

    # Helper function to plot a caregiver's route
    def plot_caregiver_route(ax, k, k_idx):
        color = colors[k_idx % len(colors)]

        # Skip caregivers with no routes
        if not model.routes[k]:
            return False

        # Process route data
        route_x = []
        route_y = []
        route_tasks = []

        # Handle start point
        start_loc = model.get_endpoint(k, "start")
        if start_loc == "HQ":
            start_x, start_y = 0, 0
            route_x.append(start_x)
            route_y.append(start_y)
            route_tasks.append(None)

        # Process each task in the route
        for i, j in model.routes[k]:
            if j != "end":
                client_id = model.get_location(j)
                client_x = clients_df.loc[client_id, "x"]
                client_y = clients_df.loc[client_id, "y"]
                route_x.append(client_x)
                route_y.append(client_y)
                route_tasks.append(j)

        # Handle end point
        end_loc = model.get_endpoint(k, "end")
        if end_loc == "HQ":
            end_x, end_y = 0, 0
            route_x.append(end_x)
            route_y.append(end_y)
            route_tasks.append(None)

        # Skip if no route was created
        if len(route_x) < 1:
            return False

        # Plot the route
        ax.plot(
            route_x,
            route_y,
            color=color,
            linewidth=2,
            alpha=0.7,
            label=f"Caregiver {k} ({model.caregivers.loc[k, 'ModeOfTransport']})",
        )

        # Add order numbers for each visit
        visit_order = 1
        for i in range(len(route_x)):
            if route_tasks[i] is not None:
                ax.text(
                    route_x[i],
                    route_y[i],
                    str(visit_order),
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="circle", facecolor="white", edgecolor=color, linewidth=1.5, alpha=0.9),
                    zorder=3,
                )
                visit_order += 1

        return True

    if subplot_mode:
        # Create a figure with subplots for each caregiver
        n_caregivers = len(caregiver_ids)
        n_cols = min(3, n_caregivers)  # Use max 3 columns (changed from 2)
        n_rows = int(np.ceil(n_caregivers / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

        # Handle the case of a single caregiver (axes won't be an array)
        if n_caregivers == 1:
            axes = np.array([axes])

        # Flatten axes array for easier iteration
        axes = np.array(axes).flatten()

        # For each caregiver, create a separate subplot
        for k_idx, k in enumerate(caregiver_ids):
            ax = axes[k_idx]

            # Plot all client nodes in the background
            ax.scatter(active_client_data["x"], active_client_data["y"], color="lightgray", s=40, alpha=0.5)

            # Plot the caregiver's route
            has_route = plot_caregiver_route(ax, k, k_idx)

            # Plot HQ
            ax.scatter(0, 0, color="black", s=80, marker="s")

            # Set title and labels
            ax.set_title(
                f"Caregiver {k} ({model.caregivers.loc[k, "ModeOfTransport"]}) {" (No Route)" if not has_route else ""}"
            )
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.3)

            # Adjust axes to have some margin
            ax.margins(0.1)

        # Hide any unused subplots
        for idx in range(len(caregiver_ids), len(axes)):
            axes[idx].set_visible(False)

    else:
        # Original single plot implementation
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot all client nodes in the background
        ax.scatter(active_client_data["x"], active_client_data["y"], color="lightgray", s=40, alpha=0.5)

        # For each caregiver, plot their route
        for k_idx, k in enumerate(caregiver_ids):
            plot_caregiver_route(ax, k, k_idx)

        # Plot HQ
        ax.scatter(0, 0, color="black", s=80, marker="s")

        # Add legend for caregivers
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Set axis labels
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.3)

        # Adjust axes to have some margin
        ax.margins(0.1)

    plt.tight_layout()
    return fig, ax if not subplot_mode else (fig, axes)
