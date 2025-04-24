import hexaly.optimizer
from .base_model import BaseModel
import math


class HexalyFixedModel(BaseModel):
    def build(
        self,
        overtime_penalty=0.8,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=60,
        worktime_per_break=5 * 60,
        day_continuity_penalty=10,
        time_limit=60 * 8,
        lateness_penalty=5,
        caregiver_penalty=60,
        evening_shift_time=15 * 60 + 30,
        break_penalty=None,
    ):
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            # ---- Base Model Construction ----
            m = optimizer.model

            # Zero indexing our sets
            V_num = len(self.V)
            K_num = len(self.K)
            C_num = len(self.C)

            # ---- Model Variables ----
            task_sequences = [m.list(V_num) for _ in range(K_num)]
            m.constraint(m.partition(task_sequences))

            # Weights
            def calculate_weight(k, i, j):
                return self.l[j] - self.l[i] - lateness_penalty * (min(0, self.e[j] - self.l[i] - self.c(k, i, j)))

            dist_matrix = m.array([[[calculate_weight(k, i, j) for j in self.V] for i in self.V] for k in self.K])
            dist_start = m.array([[self.c(k, "start", i) + self.s[i] for i in self.V] for k in self.K])
            dist_end = m.array([[self.c(k, i, "end") for i in self.V] for k in self.K])

            # Possible breaks
            breaks_array = m.array(
                [
                    [[int(self.is_break_feasible(k, i, j, break_length)) for j in self.V] for i in self.V]
                    for k in self.K
                ]
            )
            earliest = m.array([self.e[i] for i in self.V])

            # Continuity parameters
            def calculate_continuity_penalty(k, c):
                return day_continuity_penalty + continuity_penalty * (1 - self.is_historically_visited(k, c))

            client_tasks = m.array([self.get_client_tasks(c, True) for c in self.C])
            client_visit_cost = m.array([[calculate_continuity_penalty(k, c) for c in self.C] for k in self.K])

            tour_duration = [None] * K_num
            overtime = [None] * K_num
            continuity_penalty = [None] * K_num
            required_breaks = [None] * K_num
            breaks = [None] * K_num
            exhaustion = [None] * K_num

            caregivers_used = [(m.count(task_sequences[k]) > 0) for k in range(K_num)]

            for k, caregiver in enumerate(self.K):
                sequence = task_sequences[k]
                c = m.count(sequence)

                forbidden = []
                # Forbidding unallowed stops
                for i, task in enumerate(self.V):
                    if not self.is_caregiver_qualified(caregiver, task):
                        forbidden.append(task)
                if forbidden:
                    m.constraint(m.count(m.intersection(sequence, m.array(forbidden))) == 0)

                # Distance driven
                duration_lambda = m.lambda_function(lambda i: m.at(dist_matrix, k, sequence[i - 1], sequence[i]))

                tour_duration[k] = m.sum(m.range(1, c), duration_lambda) + m.iif(
                    c > 0, dist_start[k][sequence[0]] + dist_end[k][sequence[c - 1]], 0
                )

                # Breaks
                start = m.iif(c > 0, earliest[sequence[0]] - dist_start[k][sequence[0]], 0)
                required_breaks[k] = m.iif(
                    start < evening_shift_time, m.floor(tour_duration[k] / worktime_per_break), 0
                )
                breaks_lambda = m.lambda_function(lambda i: m.at(breaks_array, k, sequence[i], sequence[i + 1]))
                breaks[k] = m.array(m.range(0, c - 1), breaks_lambda)
                # exhaustion_lambda = m.lambda_function(
                #     lambda i, prev: m.iif(
                #         breaks[k][i] == 1,
                #         0,
                #         prev
                #         + m.iif(
                #             i == 0, dist_start[k][sequence[0]], m.at(dist_matrix, k, sequence[i], sequence[i + 1])
                #         ),
                #     )
                # )
                # exhaustion[k] = m.array(m.range(0, c - 1), exhaustion_lambda, 0)
                # m.constraint(
                #     m.and_(m.range(0, c - 1), m.lambda_function(lambda i: exhaustion[k][i] <= worktime_per_break))
                # )
                m.constraint(m.sum(breaks[k]) >= required_breaks[k])

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

            total_tour_duration = m.sum(tour_duration)
            total_overtime = m.sum(overtime)
            total_continuity_penalty = m.sum(continuity_penalty)
            total_caregivers_used = m.sum(caregivers_used)

            m.minimize(
                total_tour_duration
                + overtime_penalty * total_overtime
                + total_continuity_penalty
                + caregiver_penalty * total_caregivers_used
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
                breaks_value = breaks[k].value

                c = len(sequence_value)

                start_task = self.V[sequence_value[0]]
                self.routes[caregiver].append(("start", start_task))
                self.arrivals[caregiver]["start"] = self.e[start_task] - self.c(caregiver, "start", start_task)

                for i in range(c):
                    source = self.V[sequence_value[i]]
                    self.arrivals[caregiver][source] = self.e[source]
                    if i < c - 1:
                        destination = self.V[sequence_value[i + 1]]
                        self.routes[caregiver].append((source, destination))

                end_task = self.V[sequence_value[c - 1]]
                self.routes[caregiver].append((end_task, "end"))
                self.arrivals[caregiver]["end"] = self.l[end_task] + self.c(caregiver, end_task, "end")

                # Breaks
                for idx, is_break in enumerate(breaks_value):
                    if not is_break:
                        continue
                    break_task = self.V[sequence_value[idx]]
                    self.breaks[caregiver].append(break_task)

    def get_solution(self):
        return self.routes, self.arrivals, self.breaks
