from ortools.linear_solver import pywraplp
import networkx as nx


def build_l21_ortools(G: nx.Graph,
                      M: int | None = None,
                      L: int | None = None,
                      solver_name: str = "CPLEX_MIXED_INTEGER_PROGRAMMING"):
    """
    G: grafo simples não-direcionado (networkx.Graph)
    M, L: constantes 'big-M'. Se None, usam um bound simples baseado em |V|.
    solver_name:
        - "CBC_MIXED_INTEGER_PROGRAMMING" (default, open source)
        - "CPLEX_MIXED_INTEGER_PROGRAMMING" (se você tiver CPLEX instalado e integrado)
    """

    # --- Pré-processamento ---
    nodes = list(G.nodes())
    n = len(nodes)

    if M is None:
        # bound simples: pior caso span <= n (rótulos 0..n-1)
        M = n
    if L is None:
        L = n

    # arestas distância 1
    E1 = [(u, v) if u < v else (v, u) for u, v in G.edges()]
    E1 = list(set(E1))  # garantir unicidade

    # pares distância 2
    dist2_pairs = set()
    for u in nodes:
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=2)
        for v, d in lengths.items():
            if d == 2 and u < v:
                dist2_pairs.add((u, v))
    E2 = list(dist2_pairs)

    # --- Criar solver ---
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Não foi possível criar solver '{solver_name}'")

    # --- Variáveis ---
    # xi ∈ N≥0
    x = {
        i: solver.IntVar(0.0, solver.infinity(), f"x_{i}")
        for i in nodes
    }

    # z ∈ N≥0
    z = solver.IntVar(0.0, solver.infinity(), "z")

    # bij ∈ {0,1} para d=1
    b = {}
    for (i, j) in E1:
        b[(i, j)] = solver.BoolVar(f"b_{i}_{j}")

    # dij ∈ {0,1} para d=2
    dvar = {}
    for (i, j) in E2:
        dvar[(i, j)] = solver.BoolVar(f"d_{i}_{j}")

    # --- Restrições ---

    # xi ≤ z para todo i
    for i in nodes:
        solver.Add(x[i] <= z)

    # d(vi, vj) = 1:
    # xi − xj ≥ 2 − M·(1 − bij)
    # xj − xi ≥ 2 − M·bij
    for (i, j) in E1:
        bij = b[(i, j)]
        solver.Add(x[i] - x[j] >= 2 - M * (1 - bij))
        solver.Add(x[j] - x[i] >= 2 - M * bij)

    # d(vi, vj) = 2:
    # xi − xj ≥ 1 − L·(1 − dij)
    # xj − xi ≥ 1 − L·dij
    for (i, j) in E2:
        dij = dvar[(i, j)]
        solver.Add(x[i] - x[j] >= 1 - L * (1 - dij))
        solver.Add(x[j] - x[i] >= 1 - L * dij)

    # --- Função objetivo ---
    solver.Minimize(z)

    return solver, x, z, b, dvar


def solve_l21_ortools(G: nx.Graph,
                      M: int | None = None,
                      L: int | None = None,
                      solver_name: str = "CBC_MIXED_INTEGER_PROGRAMMING"):
    solver, x, z, b, dvar = build_l21_ortools(G, M, L, solver_name)

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print("Nenhuma solução encontrada.")
        return None

    span = z.solution_value()
    labeling = {i: int(x[i].solution_value()) for i in G.nodes()}

    return {
        "status": status,
        "span": span,
        "labeling": labeling,
        "x": x,
        "z": z,
        "b": b,
        "d": dvar,
        "solver": solver,
    }


# ----------------- EXEMPLO DE USO -----------------
if __name__ == "__main__":
    # Exemplo: ciclo C_5
    G = nx.cycle_graph(5)

    result = solve_l21_ortools(G)

    if result is not None:
        print("Span ótimo:", result["span"])
        print("Rótulos:")
        for i, val in result["labeling"].items():
            print(f"x_{i} = {val}")