import hexaly.optimizer
from .base_model import BaseModel


class HexalyModel(BaseModel):
    def build(
        self,
        overtime_penalty=1.5,
        caregiver_penalty=60,
        worktime_per_break=5 * 60,
        regular_hours=8 * 60,
        break_length=30,
        continuity_penalty=5,
        lateness_penalty=5,
    ):
        with hexaly.optimizer.HexalyOptimizer() as optimizer:
            # ---- Base Model Construction ----
            self.model = optimizer.model

            # Zero indexing our sets
            V_num = len(self.V)
            K_num = len(self.K)

            # time_matrix = self.model.array([[self.l[j] - self.l[i] for j in self.V] for i in self.V])
            # start_weight = self.model.array([[self.c(k, "start", i) + self.s[i] for i in self.V] for k in self.K])
            # end_weight = self.model.array([[self.c(k, i, "end") for i in self.V] for k in self.K])
            task_sequences = [self.model.list(V_num) for _ in range(K_num)]
            self.model.constraint(self.model.partition(task_sequences))

            service_time = self.model.array([self.s[i] for i in self.V])
            earliest = self.model.array([self.e[i] for i in self.V])
            latest = self.model.array([self.l[i] for i in self.V])
            dist_matrix = self.model.array([[[self.c(k, i, j) for j in self.V] for i in self.V] for k in self.K])
            dist_start = self.model.array([[self.c(k, "start", i) for i in self.V] for k in self.K])
            dist_end = self.model.array([[self.c(k, i, "end") for i in self.V] for k in self.K])

            end_time = [None] * K_num
            lateness = [None] * K_num
            tour_duration = [None] * K_num

            caregivers_used = [(self.model.count(task_sequences[k]) > 0) for k in range(K_num)]

            for k, caregiver in enumerate(self.K):
                sequence = task_sequences[k]
                c = self.model.count(sequence)

                # Forbidding unallowed stops
                for i, task in enumerate(self.V):
                    if not self.is_caregiver_qualified(caregiver, task):
                        self.model.constraint(self.model.not_(self.model.contains(sequence, i)))

                # End time of each visit
                end_time_lambda = self.model.lambda_function(
                    lambda i, prev: self.model.max(
                        earliest[sequence[i]],
                        self.model.iif(
                            i == 0,
                            earliest[sequence[i]],
                            prev + self.model.at(dist_matrix, k, sequence[i - 1], sequence[i]),
                        ),
                    )
                    + service_time[sequence[i]],
                )
                end_time[k] = self.model.array(self.model.range(0, c), end_time_lambda, 0)

                # Lateness
                late_lambda = self.model.lambda_function(
                    lambda i: self.model.max(0, end_time[k][i] - latest[sequence[i]])
                )
                lateness[k] = self.model.sum(self.model.range(0, c), late_lambda)

                # Tour duration. First term is the home arrival
                tour_duration[k] = self.model.iif(
                    c > 0,
                    end_time[k][c - 1] + dist_end[k][sequence[c - 1]] - dist_start[k][sequence[0]],
                    0,
                )

            total_lateness = self.model.sum(lateness)
            total_tour_duration = self.model.sum(tour_duration)

            self.model.minimize(lateness_penalty * total_lateness + total_tour_duration)

            self.model.close()
            optimizer.param.time_limit = 10
            optimizer.solve()

            ### Solution extraction

            self.routes = {k: [] for k in self.K}
            self.arrivals = {k: [] for k in self.K}
            for k, caregiver in enumerate(self.K):
                if not caregivers_used[k].value:
                    continue

                sequence_value = task_sequences[k].value
                c = len(sequence_value)

                # Routes
                self.routes[caregiver].append(("start", self.V[sequence_value[0]]))
                for i in range(len(sequence_value) - 1):
                    self.routes[caregiver].append((self.V[sequence_value[i]], self.V[sequence_value[i + 1]]))
                self.routes[caregiver].append((self.V[c - 1], "end"))

                # Arrivals
                end_time_value = end_time[k].value
                self.arrivals[caregiver] = {}
                self.arrivals[caregiver]["start"] = end_time_value[0] - self.s[self.V[sequence_value[0]]]
                self.arrivals[caregiver]["end"] = self.arrivals[caregiver]["start"] + tour_duration[k].value
                for idx, i in enumerate(sequence_value):
                    self.arrivals[caregiver][self.V[i]] = end_time_value[idx] - self.s[self.V[i]]
