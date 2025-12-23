from docplex.mp.model import Model
import networkx as nx
from scipy.io import mmread
from pathlib import Path
import csv
import time

input_directory_name = "data"
pasta_entrada = Path(input_directory_name)

pasta_saida = Path("results_" + input_directory_name)
pasta_saida.mkdir(exist_ok=True)

pasta_saida_rotulacao = Path("labelings_" + input_directory_name)
pasta_saida_rotulacao.mkdir(exist_ok=True)

TIME_LIMIT_MINUTES = 15


def _neighbors_at_distance_1(graph: nx.Graph):
    return {u: set(graph.neighbors(u)) for u in graph.nodes()}


def _neighbors_at_distance_2(graph: nx.Graph, dist1):
    dist2 = dict()
    for v in graph:
        dist2[v] = set()
        for u in graph[v]:
            for w in graph[u]:
                if w != v and w not in dist1[v]:
                    dist2[v].add(w)
    return dist2

def greedy_labeling(graph, order, ub=None):
    f = {i: -1 for i in order}
    Pro = {}

    if ub is None:
        delta = max(dict(graph.degree()).values(), default=0)
        ub = delta * delta + delta 

    possible_labeling = list(range(0, ub + 1))

    k = 0
    for i in order:
        Pro[i] = []
        for neighbor in graph.neighbors(i):
            if f[neighbor] != -1:
                Pro[i].append(f[neighbor])
                Pro[i].append(f[neighbor] - 1)
                Pro[i].append(f[neighbor] + 1)
            for nn in graph.neighbors(neighbor):
                if f[nn] != -1:
                    Pro[i].append(f[nn])

        candidates = [x for x in possible_labeling if x not in Pro[i]]
        if not candidates:
            # fallback bem raro: expande o universo de rótulos
            new_max = possible_labeling[-1] + 100
            possible_labeling.extend(range(possible_labeling[-1] + 1, new_max + 1))
            candidates = [x for x in possible_labeling if x not in Pro[i]]

        f[i] = min(candidates)
        k = max(k, f[i])

    return k, f

def processar_arquivo(arquivo_path: Path):
    arquivo_path = Path(arquivo_path)

    A = mmread(arquivo_path)
    G = nx.from_scipy_sparse_array(A)

    # remover laços
    G.remove_edges_from([(u, v) for u, v in G.edges() if u == v])

    edge_count = G.number_of_edges()
    vertex_count = G.number_of_nodes()
    density = (2.0 * edge_count) / (vertex_count * (vertex_count - 1)) if vertex_count > 1 else 0.0
    max_degree = max(dict(G.degree()).values(), default=0)
    min_degree = min(dict(G.degree()).values(), default=0)

    # --- Greedy para solução inicial (MIP start) ---
    order = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)

    ub_greedy = max_degree**2 + max_degree  # coerente com seu bound
    k0, f0 = greedy_labeling(G, order, ub=ub_greedy)

    dist1 = _neighbors_at_distance_1(G)
    dist2 = _neighbors_at_distance_2(G, dist1)

    # --- Modelo DOcplex ---
    mdl = Model(name=f"L21_{arquivo_path.stem}")

    # variáveis
    x = mdl.integer_var_list(vertex_count, lb=0, name="x")
    z = mdl.integer_var(lb=0, name="z")

    # --- MIP start: setar valores iniciais ---
    for i in range(vertex_count):
        x[i].start = int(f0[i])

    z.start = int(k0)

    # Dá um upper bound inicial pro modelo
    mdl.add_constraint(z <= int(k0), ctname="ub_from_greedy")

    # x[i] <= z
    for i in range(vertex_count):
        mdl.add_constraint(x[i] <= z, ctname=f"ub_{i}")

    # big-M
    M = max_degree**2 + max_degree + 2
    L = max_degree**2 + max_degree + 1

    # distância 1 (dif >= 2)
    b = {}
    for i in range(vertex_count):
        for j in dist1[i]:
            if i < j:
                b[(i, j)] = mdl.binary_var(name=f"b_{i}_{j}")
                mdl.add_constraint(x[i] - x[j] >= 2 - M * (1 - b[(i, j)]), ctname=f"d1a_{i}_{j}")
                mdl.add_constraint(x[j] - x[i] >= 2 - M * b[(i, j)], ctname=f"d1b_{i}_{j}")

    # distância 2 (dif >= 1)
    d = {}
    for i in range(vertex_count):
        for j in dist2[i]:
            if i < j:
                d[(i, j)] = mdl.binary_var(name=f"d_{i}_{j}")
                mdl.add_constraint(x[i] - x[j] >= 1 - L * (1 - d[(i, j)]), ctname=f"d2a_{i}_{j}")
                mdl.add_constraint(x[j] - x[i] >= 1 - L * d[(i, j)], ctname=f"d2b_{i}_{j}")

    # objetivo
    mdl.minimize(z)

    # time limit (segundos)
    mdl.parameters.timelimit = TIME_LIMIT_MINUTES * 60

    # (opcional) log do CPLEX no terminal
    mdl.parameters.mip.display = 2

    start = time.time()
    sol = mdl.solve(log_output=False)
    elapsed_ms = (time.time() - start) * 1000.0

    if sol is None:
        # sem solução (infeasible/unknown/time limit sem incumbente)
        return (vertex_count, edge_count, density, max_degree, min_degree, elapsed_ms, -1, "NO_SOLUTION")

    # status
    # (DOcplex normalmente usa solve_status + details)
    status_str = str(mdl.solve_details.status)
    # marcar "OPTIMAL" vs "BEST FOUND"
    if mdl.solve_details.status == "optimal":
        label_status = "OPTIMAL"
    else:
        # pode ser feasible, time_limit, etc.
        label_status = "BEST FOUND"

    # salvar rotulação
    nome_csv = pasta_saida_rotulacao / f"{arquivo_path.stem}.csv"
    with open(nome_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex", "label"])
        for i in G.nodes():
            writer.writerow([i, int(sol.get_value(x[i]))])

    lambda_val = sol.get_value(z)
    return (vertex_count, edge_count, density, max_degree, min_degree, elapsed_ms, lambda_val, label_status)


# loop dos arquivos
for arquivo in pasta_entrada.iterdir():
    if arquivo.suffix == ".mtx":
        try:
            print(f"Processando {arquivo.name}")
            resultado = processar_arquivo(arquivo)

            nome_csv = pasta_saida / f"{arquivo.stem}.csv"
            with open(nome_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["graph", "#vertices", "#edges", "density", "max_degree", "min_degree", "time(ms)", "lambda", "status"])
                writer.writerow([
                    arquivo.stem,
                    resultado[0],
                    resultado[1],
                    f"{resultado[2]:.5f}",
                    resultado[3],
                    resultado[4],
                    f"{resultado[5]:.2f}",
                    resultado[6],
                    resultado[7]
                ])

            print(f"[OK] {arquivo.name} → {nome_csv.name}")
        except Exception as e:
            print(f"Erro ao processar {arquivo.name}: {e}")