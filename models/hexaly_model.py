import hexaly.optimizer
from .base_model import BaseModel


class HexalyModel(BaseModel):
    def build(
        self,
        overtime_penalty=2,
        min_tour_duration=60 * 3,
        worktime_per_break=5 * 60,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=60,
        day_continuity_penalty=10,
        distance_cost=2,
        time_limit=60 * 8,
        wiggle_room=30,
        between_break_time=60 * 3,
        break_boundary_time=60 * 2,
        evening_shift_start=15 * 60 + 30,
    ):
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            # ---- Base Model Construction ----
            model = optimizer.model

            # Zero indexing our sets
            V_num = len(self.V)
            K_num = len(self.K)
            C_num = len(self.C)

            max_travel = self.get_max_travel_time()
            earliest_start = int(min(self.e.values()) - wiggle_room - max_travel)
            latest_start = int(max(self.e.values()) + wiggle_room)
            latest_end = int(max(self.l.values()) + wiggle_room + max_travel)

            # ---- Model Variables ----
            task_sequences = [model.list(V_num) for _ in range(K_num)]
            model.constraint(model.partition(task_sequences))
            start_times = [model.int(earliest_start, latest_start) for _ in range(K_num)]
            caregiver_breaks = [model.list(V_num) for _ in range(K_num)]

            # ---- Model Parameters ----
            # Visit parameters
            service_time = model.array([self.s[i] for i in self.V])
            earliest = model.array([max(self.e[i] - wiggle_room, 450) for i in self.V])
            latest = model.array([self.l[i] + wiggle_room for i in self.V])

            # Distances
            dist_matrix = model.array([[[self.c(k, i, j) for j in self.V] for i in self.V] for k in self.K])
            dist_start = model.array([[self.c(k, "start", i) for i in self.V] for k in self.K])
            dist_end = model.array([[self.c(k, i, "end") for i in self.V] for k in self.K])

            # Continuity parameters
            def calculate_continuity_penalty(k, c):
                return day_continuity_penalty + continuity_penalty * (1 - self.is_historically_visited(k, c))

            client_tasks = model.array([self.get_client_tasks(c, True) for c in self.C])
            client_visit_cost = model.array([[calculate_continuity_penalty(k, c) for c in self.C] for k in self.K])

            # Base
            end_time = [None] * K_num
            lateness = [None] * K_num
            dist_routes = [None] * K_num
            tour_duration = [None] * K_num
            # Overtime
            overtime = [None] * K_num
            # Continuity
            continuity_penalty = [None] * K_num
            # Break
            required_breaks = [None] * K_num
            missed_breaks = [None] * K_num
            break_end_times = [None] * K_num
            break_violation = [None] * K_num

            caregivers_used = [(model.count(task_sequences[k]) > 0) for k in range(K_num)]

            for k, caregiver in enumerate(self.K):
                sequence = task_sequences[k]
                c = model.count(sequence)
                start = start_times[k]
                breaks = caregiver_breaks[k]
                b = model.count(breaks)

                forbidden = []
                # Forbidding unallowed stops
                for i, task in enumerate(self.V):
                    if not self.is_caregiver_qualified(caregiver, task):
                        forbidden.append(task)
                if forbidden:
                    model.constraint(model.count(model.intersection(sequence, model.array(forbidden))) == 0)

                # End time of each visit
                end_time_lambda = model.lambda_function(
                    lambda i, prev: model.max(
                        earliest[sequence[i]],
                        model.iif(
                            i == 0,
                            start + dist_start[k][sequence[0]],
                            prev
                            + model.at(dist_matrix, k, sequence[i - 1], sequence[i])
                            + model.iif(model.contains(breaks, sequence[i - 1]), break_length, 0),
                        ),
                    )
                    + service_time[sequence[i]],
                )
                end_time[k] = model.array(model.range(0, c), end_time_lambda, 0)

                # Lateness
                late_lambda = model.lambda_function(lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
                lateness[k] = model.sum(model.range(0, c), late_lambda)

                # Distance driven
                dist_lambda = model.lambda_function(lambda i: model.at(dist_matrix, k, sequence[i - 1], sequence[i]))

                dist_routes[k] = model.sum(model.range(1, c), dist_lambda) + model.iif(
                    c > 0, dist_start[k][sequence[0]] + dist_end[k][sequence[c - 1]], 0
                )

                # Tour duration. First term is the home arrival
                tour_duration[k] = model.iif(
                    c > 0,
                    model.max(
                        min_tour_duration,
                        end_time[k][c - 1] + dist_end[k][sequence[c - 1]] - start,
                    ),
                    0,
                )

                # Breaks
                required_breaks[k] = model.iif(
                    start < evening_shift_start, model.floor(tour_duration[k] / worktime_per_break), 0
                )
                model.constraint(
                    model.and_(model.range(0, b), model.lambda_function(lambda i: model.contains(sequence, breaks[i])))
                )
                missed_breaks[k] = model.max(0, required_breaks[k] - b)

                break_end_time_lambda = model.lambda_function(
                    lambda i: end_time[k][
                        model.create_expression(hexaly.optimizer.HxOperator.INDEXOF, sequence, breaks[i])
                    ]
                    + break_length
                )
                break_end_times[k] = model.array(model.range(0, b), break_end_time_lambda)
                break_violation_lambda = model.lambda_function(
                    lambda i: model.iif(
                        i < b - 1,
                        model.max(0, break_end_times[k][i - 1] - break_end_times[k][i] - between_break_time),
                        model.max(
                            0,
                            break_end_times[k][i]
                            - (end_time[k][c - 1] + dist_end[k][sequence[c - 1]])
                            + break_boundary_time,
                        ),
                    )
                )
                break_violation[k] = model.iif(
                    b > 0,
                    model.max(0, start - break_end_times[k][0] + between_break_time)
                    + model.sum(model.range(0, b), break_violation_lambda),
                    0,
                )

                # Overtime
                overtime[k] = model.iif(
                    c > 0,
                    model.max(0, tour_duration[k] - regular_hours),
                    0,
                )

                # Continuity
                continuity_lambda = model.lambda_function(
                    lambda i: model.iif(
                        model.count(model.intersection(sequence, client_tasks[i])) > 0,
                        model.at(client_visit_cost, k, i),
                        0,
                    )
                )
                continuity_penalty[k] = model.sum(model.range(0, C_num), continuity_lambda)

            total_lateness = model.sum(lateness)
            total_tour_duration = model.sum(tour_duration)
            total_overtime = model.sum(overtime)
            total_continuity_penalty = model.sum(continuity_penalty)
            total_distance = model.sum(dist_routes)
            total_missed_breaks = model.sum(missed_breaks)
            total_break_violation = model.sum(break_violation)

            model.minimize(total_lateness)
            model.minimize(total_missed_breaks)
            model.minimize(total_break_violation)
            model.minimize(
                total_tour_duration
                + overtime_penalty * total_overtime
                + total_continuity_penalty
                + distance_cost * total_distance
            )

            model.close()
            optimizer.param.time_limit = time_limit
            optimizer.solve()

            # ---- Solution Extraction ----
            self.routes = {k: [] for k in self.K}
            self.arrivals = {k: {} for k in self.K}
            self.breaks = {k: [] for k in self.K}
            for k, caregiver in enumerate(self.K):
                if not caregivers_used[k].value:
                    continue

                sequence_value = task_sequences[k].value
                end_time_value = end_time[k].value

                c = len(sequence_value)

                self.routes[caregiver].append(("start", self.V[sequence_value[0]]))
                self.arrivals[caregiver]["start"] = start_times[k].value

                for i in range(c):
                    source = self.V[sequence_value[i]]
                    self.arrivals[caregiver][source] = end_time_value[i] - self.s[source]
                    if i < c - 1:
                        destination = self.V[sequence_value[i + 1]]
                        self.routes[caregiver].append((source, destination))

                end_idx = sequence_value[c - 1]
                self.routes[caregiver].append((self.V[end_idx], "end"))
                self.arrivals[caregiver]["end"] = end_time_value[c - 1] + self.c(caregiver, self.V[end_idx], "end")

                # Breaks
                real_breaks = [b for b in caregiver_breaks[k].value if b in sequence_value]
                self.breaks[caregiver] = [self.V[b] for b in real_breaks]

    def get_solution(self):
        return self.routes, self.arrivals, self.breaks
