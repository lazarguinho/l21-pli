from docplex.mp.model import Model
import networkx as nx
from scipy.io import mmread
from pathlib import Path
import csv
import time
from tqdm import tqdm

input_directory_name = "data"
pasta_entrada = Path(input_directory_name)

pasta_saida = Path("results_" + input_directory_name)
pasta_saida.mkdir(exist_ok=True)

pasta_saida_rotulacao = Path("labelings_" + input_directory_name)
pasta_saida_rotulacao.mkdir(exist_ok=True)

TIME_LIMIT_MINUTES = 15


def _neighbors_at_distance_1(graph: nx.Graph):
    return {u: set(graph.neighbors(u)) for u in graph.nodes()}


def _neighbors_at_distance_2_fast(graph: nx.Graph, dist1):
    # dist2[v] = (união dos N(u) para u em N(v)) - N(v) - {v}
    dist2 = {}
    for v in graph.nodes():
        Nv = dist1[v]
        s = set()
        for u in Nv:
            s |= dist1[u]
        s.discard(v)
        s -= Nv
        dist2[v] = s
    return dist2


def check_L21(G, f):
    # dist1
    for u, v in G.edges():
        if abs(f[u] - f[v]) < 2:
            return False
    # dist2
    for u in G.nodes():
        Nu = set(G.neighbors(u))
        for w in Nu:
            for v in G.neighbors(w):
                if v != u and v not in Nu:
                    if abs(f[u] - f[v]) < 1:
                        return False
    return True


def greedy_labeling(graph, order, ub=None):
    f = {i: -1 for i in order}

    if ub is None:
        delta = max(dict(graph.degree()).values(), default=0)
        ub = delta * delta + delta

    possible = list(range(0, ub + 1))

    k = 0
    for i in order:
        forbidden = set()

        for neighbor in graph.neighbors(i):
            ln = f[neighbor]
            if ln != -1:
                forbidden.add(ln)
                forbidden.add(ln - 1)
                forbidden.add(ln + 1)
            for nn in graph.neighbors(neighbor):
                lnn = f[nn]
                if lnn != -1:
                    forbidden.add(lnn)

        chosen = None
        for x in possible:
            if x not in forbidden:
                chosen = x
                break

        if chosen is None:
            start = possible[-1] + 1
            end = start + 200
            possible.extend(range(start, end + 1))
            for x in range(start, end + 1):
                if x not in forbidden:
                    chosen = x
                    break

        if chosen is None:
            raise RuntimeError("Greedy não encontrou rótulo mesmo após expandir.")

        f[i] = chosen
        k = max(k, chosen)

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

    # --- Greedy ---
    order = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    ub_greedy = max_degree**2 + max_degree

    t0 = time.time()
    k0, f0 = greedy_labeling(G, order, ub=ub_greedy)
    greedy_time = time.time() - t0

    ok = check_L21(G, f0)
    print(f"greedy: {greedy_time:.2f}s | k0={k0} | ok={ok}")

    # --- dist1/dist2 ---
    t0 = time.time()
    dist1 = _neighbors_at_distance_1(G)
    dist2 = _neighbors_at_distance_2_fast(G, dist1)
    dist_time = time.time() - t0
    print(f"dist2: {dist_time:.2f}s")

    # --- Modelo ---
    mdl = Model(name=f"L21_{arquivo_path.stem}")

    x = mdl.integer_var_list(vertex_count, lb=0, name="x")
    z = mdl.integer_var(lb=0, name="z")

    # x[i] <= z
    for i in range(vertex_count):
        mdl.add_constraint(x[i] <= z, ctname=f"ub_{i}")

    # big-M
    M = max_degree**2 + max_degree + 2
    L = max_degree**2 + max_degree + 1

    # distância 1
    b = {}
    for i in range(vertex_count):
        for j in dist1[i]:
            if i < j:
                var = mdl.binary_var(name=f"b_{i}_{j}")
                b[(i, j)] = var
                mdl.add_constraint(x[i] - x[j] >= 2 - M * (1 - var), ctname=f"d1a_{i}_{j}")
                mdl.add_constraint(x[j] - x[i] >= 2 - M * var, ctname=f"d1b_{i}_{j}")

    # distância 2
    d = {}
    for i in range(vertex_count):
        for j in dist2[i]:
            if i < j:
                var = mdl.binary_var(name=f"d_{i}_{j}")
                d[(i, j)] = var
                mdl.add_constraint(x[i] - x[j] >= 1 - L * (1 - var), ctname=f"d2a_{i}_{j}")
                mdl.add_constraint(x[j] - x[i] >= 1 - L * var, ctname=f"d2b_{i}_{j}")

    mdl.minimize(z)

    # time limit e log
    mdl.parameters.timelimit = TIME_LIMIT_MINUTES * 60
    mdl.parameters.mip.display = 2

    # --- Ajuste para grafos densos (achar incumbente) ---
    mdl.parameters.emphasis.mip = 1
    mdl.parameters.mip.strategy.heuristicfreq = 10

    # --- MIP start "de verdade" ---
    if ok:
        s = mdl.new_solution()

        # x e z
        for i in range(vertex_count):
            s.add_var_value(x[i], int(f0[i]))
        s.add_var_value(z, int(k0))

        # b e d coerentes com o ordering das labels
        for (i, j), var in b.items():
            s.add_var_value(var, 1 if f0[i] >= f0[j] else 0)
        for (i, j), var in d.items():
            s.add_var_value(var, 1 if f0[i] >= f0[j] else 0)

        # IMPORTANTÍSSIMO: registra o MIP start no modelo
        mdl.add_mip_start(s)
        print("[INFO] MIP start adicionado via add_mip_start().")

        # Só agora faz sentido colocar z <= k0 (porque o greedy é viável)
        mdl.add_constraint(z <= int(k0), ctname="ub_from_greedy")
    else:
        print("[INFO] Greedy inválido -> sem MIP start e sem z<=k0.")

    # --- solve ---
    t0 = time.time()
    sol = mdl.solve(log_output=True)
    solve_time = time.time() - t0
    print(f"solve: {solve_time:.2f}s")
    print("status:", mdl.solve_details.status)

    elapsed_ms = solve_time * 1000.0

    if sol is None:
        return (vertex_count, edge_count, density, max_degree, min_degree, elapsed_ms, -1,
                f"NO_SOLUTION ({mdl.solve_details.status})")

    label_status = "OPTIMAL" if mdl.solve_details.status == "optimal" else f"BEST FOUND ({mdl.solve_details.status})"

    # salvar rotulação
    nome_csv = pasta_saida_rotulacao / f"{arquivo_path.stem}.csv"
    with open(nome_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex", "label"])
        for i in G.nodes():
            writer.writerow([i, int(sol.get_value(x[i]))])

    lambda_val = sol.get_value(z)
    return (vertex_count, edge_count, density, max_degree, min_degree, elapsed_ms, lambda_val, label_status)


# loop com progresso
arquivos_mtx = sorted([a for a in pasta_entrada.iterdir() if a.suffix == ".mtx"])
for arquivo in tqdm(arquivos_mtx, desc="Processando grafos", unit="grafo"):
    try:
        print(f"\nProcessando {arquivo.name}")
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
        tqdm.write(f"[ERRO] {arquivo.name}: {e}")