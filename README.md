# Home Health Care Model

### Sets and Parameters

- $V$: Set of all patient tasks (indexed by $i$).
- $K$: Set of caregivers (indexed by $k$).
- For each caregiver $k$, we define two auxiliary nodes:
$$
    \sigma_k := \text{``start''}, 
    \quad
    \tau_k := \text{``end''}.
$$
- To handle these properly, let
$$
    V^k_\sigma := V \cup \{\sigma_k\},
    \quad
    V^k_\tau := V \cup \{\tau_k\}.
$$
In the constraints below, we only use arcs 
$
    (i,j)\; \text{with}\; i \in V^k_\sigma,\; j \in V^k_\tau,\; i \neq j
$
and exclude any arcs leading \emph{into} $\sigma_k$ or \emph{out of} $\tau_k$.
- $c_{ij} \in \mathbb{R}_{\geq 0}$: Travel time from node $i$ to node $j$. 
        In practice, $c_{\sigma_k,j}$ and $c_{i,\tau_k}$ depend on each caregiver’s actual start/end location 
        (and may be 0 for caregivers starting/ending at home).
- $[e_i, l_i] \subset \mathbb{R}$: Time window for patient task $i$, 
        with earliest possible start $e_i$ and latest possible start $l_i$.
- $s_i \in \mathbb{R}_{\geq 0}$: Service (visit) duration for task $i$.
- $p_i^k \in \{0,1\}$: Binary parameter indicating whether caregiver $k$ is qualified to perform task $i$.
- $M \in \mathbb{R}_{\geq 0}$: A sufficiently large constant, used for big-$M$ linearization.

## Flexible Model

### Decision Variables

  - $x_{ij}^k \in \{0,1\}$: Binary variable indicating if caregiver $k$ travels directly from node $i$ to node $j$. 
        Defined only for \mbox{$i \in V^k_\sigma$, $j \in V^k_\tau$, $i \neq j$}.
        This automatically disallows arcs from $\tau_k$ or into $\sigma_k$.
  - $t_i^k \in \mathbb{R}_{\geq 0}$: Arrival (start-of-service) time of caregiver $k$ at node $i$.  
        In particular, $t_{\sigma_k}^k$ denotes the time caregiver $k$ begins work, 
        and $t_{\tau_k}^k$ is the time they reach $\tau_k$ (i.e., finish their last task).

### Optimization Problem

$$
\begin{align}
\text{minimize} \quad 
& \sum_{k \in K} \bigl(t_{\tau_k}^k - t_{\sigma_k}^k\bigr)
&& \tag{V1} \label{eq:MinimizeTotalTime}\\[6pt]
\text{subject to}\quad
& \sum_{k \in K} \sum_{\substack{j \in V^k_\sigma \\ j \neq i}} x_{ji}^k = 1 
  && \forall\, i \in V 
  && \tag{V2} \label{eq:UniqueVisit}\\[3pt]
& \sum_{\substack{j \in V^k_\tau \\ j \neq i}} x_{ij}^k
  \;-\;
  \sum_{\substack{j \in V^k_\sigma \\ j \neq i}} x_{ji}^k 
  \;=\; 0
  && \forall\, k \in K,\;\forall\, i \in V
  && \tag{V3} \label{eq:FlowConservation}\\[3pt]
& \sum_{\substack{j \in V^k_\tau \\ j \neq \sigma_k}} x_{\sigma_k j}^k \;\le\; 1
  && \forall\, k \in K 
  && \tag{V4} \label{eq:RouteCompletion}\\[3pt]
& x_{ij}^k \;\le\; p_j^k
  && \forall\, k \in K,\;\forall\, i \in V^k_\sigma,\;\forall\, j \in V,\; i \neq j
  && \tag{V5} \label{eq:Qualification}\\[3pt]
& t_i^k \;\ge\; e_i 
  && \forall\, k \in K,\;\forall\, i \in V 
  && \tag{V6} \label{eq:EarliestArrival}\\[3pt]
& t_i^k \;\le\; l_i - s_i 
  && \forall\, k \in K,\;\forall\, i \in V 
  && \tag{V7} \label{eq:LatestArrival}\\[3pt]
& t_j^k \;\ge\; t_i^k + c_{ij} + s_i \;-\; M\bigl(1 - x_{ij}^k\bigr)
  && \forall\, k \in K,\;\forall\, i \in V^k_\sigma,\;\forall\, j \in V^k_\tau,\; i \neq j
  && \tag{V8} \label{eq:TemporalFeasibility}
\end{align}
$$

### Constraint explanations

- **(V2) Unique Visit:** Ensures each patient task $i$ is visited exactly once by exactly one caregiver.
- **(V3) Flow Conservation:** For each caregiver $k$, flow into a task $i$ from the set $V^k_\sigma$ 
      equals flow out to the set $V^k_\tau$, maintaining a continuous path.
- **(V4) Route Completion:** Caregivers start at $\sigma_k$ at most once (i.e.\ either they have a route or remain unused).
- **(V5) Qualification:** Disallows visiting tasks $j$ that caregiver $k$ is not qualified to handle.
- **(V6, V7) Earliest/Latest Arrival:** Ensures each visit starts within the allowable time window.
- **(V8) Temporal Feasibility:** Maintains proper ordering of travel and service times, with a big-$M$ 
      to deactivate the time link if $x_{ij}^k = 0$.

## Fixed Model
### Decision Variables

  - $x_{ij}^k \in \{0,1\}$: Binary variable indicating if caregiver $k$ travels directly from node $i$ to node $j$. 
        Defined only for \mbox{$i \in V^k_\sigma$, $j \in V^k_\tau$, $i \neq j$}.
        This automatically disallows arcs from $\tau_k$ or into $\sigma_k$.
  - $T_\sigma^k \in \mathbb{R}_{\geq 0}$: Starting time of caregiver $k$.
  - $T_\tau^k \in \mathbb{R}_{\geq 0}$: Ending time of caregiver $k$.

### Optimization Problem

$$
\begin{align}
\text{minimize} \quad 
& \sum_{k \in K} \bigl( T_\tau^k - T_\sigma^k \bigr)
&& \tag{B1} \label{eq:B_MinimizeTotalTime}\\[6pt]
\text{subject to}\quad
& \sum_{k \in K} \sum_{\substack{j \in V^k_\sigma \\ j \neq i}} x_{ji}^k = 1 
  && \forall\, i \in V 
  && \tag{B2} \label{eq:B_UniqueVisit}\\[3pt]
& \sum_{\substack{j \in V^k_\tau \\ j \neq i}} x_{ij}^k
  \;-\;
  \sum_{\substack{j \in V^k_\sigma \\ j \neq i}} x_{ji}^k 
  \;=\; 0
  && \forall\, k \in K,\;\forall\, i \in V
  && \tag{B3} \label{eq:B_FlowConservation}\\[3pt]
& \sum_{\substack{j \in V^k_\tau \\ j \neq \sigma_k}} x_{\sigma_k j}^k \;\le\; 1
  && \forall\, k \in K 
  && \tag{B4} \label{eq:B_RouteCompletion}\\[3pt]
& x_{ij}^k \;\le\; p_j^k
  && \forall\, k \in K,\;\forall\, i \in V^k_\sigma,\;\forall\, j \in V,\; i \neq j
  && \tag{B5} \label{eq:B_Qualification}\\[3pt]
& (e_i - l_j - c_{ij}) x_{ij}^k \geq 0
  && \forall\, k \in K,\;\forall\, i, j \in V,\; i \neq j
  && \tag{B6} \label{eq:B_TemporalFeasibility}\\[3pt]
& T_{\sigma_k}^k \leq \sum_{i \in V} x_{\sigma_k i} \cdot (e_i - c_{\sigma_k i})
  && \forall\, k \in K
  && \tag{B7} \label{eq:B_StartTime}\\[3pt]
& T_{\tau_k}^k \geq \sum_{i \in V} x_{i \tau_k} \cdot (l_i + c_{i\tau_k})
  && \forall\, k \in K
  && \tag{B8} \label{eq:B_EndTime}
\end{align}
$$

### Constraint explanations

- **(B2) Unique Visit:** Ensures each patient task $i$ is visited exactly once by exactly one caregiver.
- **(B3) Flow Conservation:** For each caregiver $k$, flow into a task $i$ from the set $V^k_\sigma$ 
      equals flow out to the set $V^k_\tau$, maintaining a continuous path.
- **(B4) Route Completion:** Caregivers start at $\sigma_k$ at most once (i.e.\ either they have a route or remain unused).
- **(B5) Qualification:** Disallows visiting tasks $j$ that caregiver $k$ is not qualified to handle.
- **(B6) Temporal Feasibility:** Only permits routes to be active if they are temporally feasible.
