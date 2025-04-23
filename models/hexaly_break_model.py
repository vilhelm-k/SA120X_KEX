import hexaly.optimizer
from .base_model import BaseModel
import math


class HexalyBreakModel(BaseModel):
    def build(
        self,
        overtime_penalty=2,
        min_tour_duration=60 * 3,
        regular_hours=8 * 60,
        break_length=40,
        continuity_penalty=60,
        worktime_per_break=60 * 6,
        day_continuity_penalty=10,
        distance_cost=2,
        time_limit=60 * 8,
        wiggle_room=30,
        evening_shift=17 * 60,
    ):
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            # ---- Base Model Construction ----
            m = optimizer.model

            # Zero indexing our sets
            V_num = len(self.V)
            K_num = len(self.K)
            C_num = len(self.C)

            max_travel = self.get_max_travel_time()
            earliest_start = math.floor(min(self.e.values()) - wiggle_room - max_travel)
            latest_start = math.ceil(max(self.e.values()) + wiggle_room)
            total_service_time = sum(self.s.values())
            max_breaks = math.ceil(1.5 * total_service_time / worktime_per_break)  # Should be enough breaks

            # ---- Model Variables ----
            task_sequences = [m.list(V_num + max_breaks) for _ in range(K_num + 1)]
            m.constraint(m.partition(task_sequences))
            start_times = [m.int(earliest_start, latest_start) for _ in range(K_num)]

            # Fictive caregiver can't visit any real clients
            for i in range(V_num):
                m.constraint(m.contains(task_sequences[K_num], i) == False)

            # ---- Model Parameters ----
            # Visit parameters
            service_time = m.array([self.s[i] for i in self.V])
            earliest = m.array([max(self.e[i] - wiggle_room, 450) for i in self.V])
            latest = m.array([self.l[i] + wiggle_room for i in self.V])

            # Distances
            dist_matrix = m.array([[[self.c(k, i, j) for j in self.V] for i in self.V] for k in self.K])
            dist_start = m.array([[self.c(k, "start", i) for i in self.V] for k in self.K])
            dist_end = m.array([[self.c(k, i, "end") for i in self.V] for k in self.K])

            # Continuity parameters
            def calculate_continuity_penalty(k, c):
                return day_continuity_penalty + continuity_penalty * (1 - self.is_historically_visited(k, c))

            client_tasks = m.array([self.get_client_tasks(c, True) for c in self.C])
            client_visit_cost = m.array([[calculate_continuity_penalty(k, c) for c in self.C] for k in self.K])

            end_time = [None] * K_num
            home_time = [None] * K_num
            lateness = [None] * K_num
            dist_routes = [None] * K_num
            tour_duration = [None] * K_num
            overtime = [None] * K_num
            continuity_penalty = [None] * K_num
            locations = [None] * K_num
            distances = [None] * K_num
            route_exhaustion = [None] * K_num

            caregivers_used = [(m.count(task_sequences[k]) > 0) for k in range(K_num)]

            for k, caregiver in enumerate(self.K):
                sequence = task_sequences[k]
                c = m.count(sequence)
                start = start_times[k]

                forbidden = []
                # Forbidding unallowed stops
                for i, task in enumerate(self.V):
                    if not self.is_caregiver_qualified(caregiver, task):
                        forbidden.append(task)
                if forbidden:
                    m.constraint(m.count(m.intersection(sequence, m.array(forbidden))) == 0)

                distance_lambda = m.lambda_function(
                    lambda i: m.iif(
                        sequence[i] < V_num,
                        m.iif(
                            i == 0,
                            dist_start[k][sequence[i]],
                            m.iif(
                                sequence[i - 1] < V_num,
                                m.at(dist_matrix, k, sequence[i - 1], sequence[i]),
                                m.at(dist_matrix, k, sequence[i - 2], sequence[i]),
                            ),
                        ),
                        0,
                    )
                )
                distances[k] = m.array(m.range(0, c), distance_lambda)

                end_time_lambda = m.lambda_function(
                    lambda i, prev: m.iif(
                        sequence[i] < V_num,
                        m.max(earliest[sequence[i]], prev + distances[k][i]) + service_time[sequence[i]],
                        prev + break_length,
                    )
                )

                end_time[k] = m.array(m.range(0, c), end_time_lambda, start)

                # Lateness
                late_lambda = m.lambda_function(
                    lambda i: m.iif(
                        sequence[i] < V_num,
                        m.max(0, end_time[k][i] - latest[sequence[i]]),
                        0,  # Break node, no lateness
                    )
                )
                lateness[k] = m.sum(m.range(0, c), late_lambda)

                # Distance driven
                dist_lambda = m.lambda_function(lambda i: distances[k][i])
                dist_routes[k] = m.sum(m.range(0, c), dist_lambda) + m.iif(c > 0, dist_end[k][sequence[c - 1]], 0)

                # Tour duration. First term is the home arrival
                home_time[k] = m.iif(c > 0, end_time[k][c - 1] + dist_end[k][sequence[c - 1]], start)
                tour_duration[k] = m.iif(
                    c > 0,
                    m.max(
                        min_tour_duration,
                        home_time[k] - start,
                    ),
                    0,
                )

                # Breaks
                route_exhaustion_lambda = m.lambda_function(
                    lambda i, prev: m.iif(
                        sequence[i] < V_num,
                        m.iif(i == 0, end_time[k][0] - start, prev + end_time[k][i] - end_time[k][i - 1]),
                        0,
                    )
                )
                route_exhaustion[k] = m.array(m.range(0, c), route_exhaustion_lambda, 0)

                exhaustion_lambda = m.lambda_function(lambda i: route_exhaustion[k][i] <= worktime_per_break)
                m.constraint(m.and_(m.range(0, c), exhaustion_lambda))

                # Overtime
                overtime[k] = m.iif(
                    c > 0,
                    m.max(0, tour_duration[k] - regular_hours),
                    0,
                )

                # Continuity
                continuity_lambda = m.lambda_function(
                    lambda i: m.iif(
                        m.count(m.intersection(sequence, client_tasks[i])) > 0,
                        m.at(client_visit_cost, k, i),
                        0,
                    )
                )
                continuity_penalty[k] = m.sum(m.range(0, C_num), continuity_lambda)

            total_lateness = m.sum(lateness)
            total_tour_duration = m.sum(tour_duration)
            total_overtime = m.sum(overtime)
            total_continuity_penalty = m.sum(continuity_penalty)
            total_distance = m.sum(dist_routes)

            m.minimize(total_lateness)
            m.minimize(
                total_tour_duration
                + overtime_penalty * total_overtime
                + total_continuity_penalty
                + distance_cost * total_distance
            )

            m.close()
            optimizer.param.time_limit = time_limit
            optimizer.solve()

            self.routes = {k: [] for k in self.K}
            self.arrivals = {k: {} for k in self.K}
            self.breaks = {k: [] for k in self.K}
            for k, caregiver in enumerate(self.K):
                if not caregivers_used[k].value:
                    continue

                sequence_value = task_sequences[k].value
                end_time_value = end_time[k].value
                real_sequence = []
                for idx, i in enumerate(sequence_value):
                    if i < V_num:
                        real_sequence.append(i)
                    else:
                        self.breaks[caregiver].append(self.V[real_sequence[-1]])

                c = len(real_sequence)

                self.routes[caregiver].append(("start", self.V[real_sequence[0]]))
                self.arrivals[caregiver]["start"] = start_times[k].value

                for i in range(c):
                    source = self.V[real_sequence[i]]
                    for idx, j in enumerate(sequence_value):
                        if j == real_sequence[i]:
                            break
                    self.arrivals[caregiver][source] = end_time_value[idx] - self.s[source]
                    if i < c - 1:
                        destination = self.V[real_sequence[i + 1]]
                        self.routes[caregiver].append((source, destination))

                end_idx = real_sequence[c - 1]
                self.routes[caregiver].append((self.V[end_idx], "end"))
                for idx, j in enumerate(sequence_value):
                    if j == end_idx:
                        break
                self.arrivals[caregiver]["end"] = end_time_value[idx] + self.c(caregiver, self.V[end_idx], "end")

    def get_solution(self):
        return self.routes, self.arrivals, self.breaks
