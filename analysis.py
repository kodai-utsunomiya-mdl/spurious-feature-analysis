# sp/analysis.py

import numpy as np
import time
from itertools import combinations, product, combinations_with_replacement
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F

# 'pot'ライブラリのインポート
try:
    import ot
except ImportError:
    print("Warning: 'pot' library not found. Wasserstein distance analysis will not be available.")
    print("Please install it using: pip install pot")
    ot = None

# ==============================================================================
# ヤコビアン計算のヘルパー関数
# ==============================================================================
def get_model_jacobian(model, X_subset, device):
    """モデルのヤコビアンの期待値を計算"""
    model.eval()
    jacobians = []
    X_subset = X_subset.to(device)
    
    for i in range(len(X_subset)):
        x_i = X_subset[i:i+1]
        x_i.requires_grad_(True)
        
        y_pred, _ = model(x_i)
        
        # モデルの全パラメータに対する勾配を計算
        grad_params = torch.autograd.grad(y_pred, model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_params])
        jacobians.append(flat_grad.cpu().detach().numpy())
        
    return np.mean(jacobians, axis=0)

# ==============================================================================
# 汎化ギャップ計算のためのヘルパー関数
# ==============================================================================
def get_eta_vector(model, param_groups):
    """ optimizer.param_groups から学習率etaのベクトルを再構築する """
    eta_tensors = []
    # optimizer作成時に model.parameters() が渡されているため，順序は一致する
    for group in param_groups:
        lr = group['lr']
        for p in group['params']:
            eta_tensors.append(torch.full_like(p.data, lr))
            
    return torch.cat([t.view(-1) for t in eta_tensors]).cpu().numpy()

# ==============================================================================
# 汎化ギャップの分析
# ==============================================================================
def analyze_generalization_gap(model, X_data, y_data, a_data, device, config, epoch, history, param_groups, loss_function, dataset_type):
    """ グループごとの汎化ギャップ Γ_g を推定する """
    print(f"\nAnalyzing GENERALIZATION GAP on {dataset_type} data...")
    
    # 1. 定数と変数の準備
    base_lr = config['learning_rate']
    t = base_lr * epoch  # 経過時間 t の近似
    
    # 経験損失の差 R_N(theta_0) - R_N(theta_t)
    initial_loss = history['train_avg_loss'][0]
    current_loss = history['train_avg_loss'][-1]
    loss_diff = initial_loss - current_loss
    if loss_diff < 0:
        print(f"Warning: Negative loss difference ({loss_diff:.4f}). Setting generalization gap to 0.")
        loss_diff = 0.0
        
    # eta ベクトルの取得
    eta_vec = get_eta_vector(model, param_groups)
    
    # 2. グループごとに計算
    gen_gap_results = {}
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        N_g = mask.sum().item()
        
        if N_g == 0:
            gen_gap_results[f"G({y_val},{a_val})"] = np.nan
            continue
            
        X_group, y_group = X_data[mask], y_data[mask]
        
        # jacobian_num_samples に基づいてサブサンプリング
        num_samples_to_use = config.get('jacobian_num_samples', 100)
        if N_g > num_samples_to_use:
            indices = np.random.choice(N_g, num_samples_to_use, replace=False)
            X_subset, y_subset = X_group[indices], y_group[indices]
        else:
            X_subset, y_subset = X_group, y_group
            num_samples_to_use = N_g

        # LPKトレース項の和 (sum_i K_t(z_i, z_i)) を計算
        sum_of_integrals = 0.0
        for i in range(num_samples_to_use):
            x_i, y_i = X_subset[i:i+1], y_subset[i:i+1]
            
            model.zero_grad()
            scores, _ = model(x_i.to(device))
            
            if loss_function == 'logistic':
                loss = F.softplus(-y_i.to(device) * scores).mean()
            else:  # mse
                loss = F.mse_loss(scores, y_i.to(device))
                
            loss.backward()
            
            grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).cpu().numpy()
            
            # eta-norm の2乗を計算: ||∇l||^2_η
            norm_sq = np.sum(eta_vec * (grad_vec ** 2))
            
            # 積分を近似: ∫||∇l||^2 ds ≈ t * ||∇l||^2
            integral_approx = t * norm_sq
            sum_of_integrals += integral_approx
            
        # サブサンプリングした場合は，グループ全体の和にスケールアップ
        scaled_sum_of_integrals = sum_of_integrals * (N_g / num_samples_to_use)
        
        # Γ_g を計算
        if scaled_sum_of_integrals < 0:
            gamma_g = 0.0
        else:
            gamma_g = (np.sqrt(loss_diff) / N_g) * np.sqrt(scaled_sum_of_integrals)

        gen_gap_results[f"G({y_val},{a_val})"] = gamma_g
        
    return gen_gap_results

# ==============================================================================
# ベクトル全体の相互情報量を計算する関数
# ==============================================================================
def vector_mutual_information(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> float:
    """ Kozachenko-Leonenko推定法を用いてベクトルXとラベルyの相互情報量を計算 """
    n_samples = X.shape[0]
    unique_classes, y_indices, class_counts = np.unique(y, return_inverse=True, return_counts=True)

    radii = np.zeros(n_samples)
    # 各クラス内でk最近傍までの距離（半径）を計算
    for i, c in enumerate(unique_classes):
        class_indices = np.where(y_indices == i)[0]
        X_class = X[class_indices]
        if len(X_class) <= n_neighbors: continue

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='chebyshev', algorithm='ball_tree')
        nn.fit(X_class)
        class_radii, _ = nn.kneighbors(X_class, n_neighbors=n_neighbors + 1)
        radii[class_indices] = class_radii[:, n_neighbors]

    radii[radii == 0] = 1e-15  # 半径が0にならないようにする

    # 全データで，各点の半径内にいくつの他の点が含まれるかを数える
    nn_full = NearestNeighbors(metric='chebyshev', algorithm='ball_tree')
    nn_full.fit(X)
    m_i_counts = nn_full.radius_neighbors(X, radius=radii, return_distance=False)
    m_i = np.array([len(neighbors) for neighbors in m_i_counts])

    # 相互情報量を計算
    mi_val = digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(class_counts[y_indices])) - np.mean(digamma(m_i))

    return max(0, mi_val)

# ==============================================================================
# 条件付き相互情報量を計算するヘルパー関数
# ==============================================================================
def conditional_mutual_information(Z: np.ndarray, target: np.ndarray, condition: np.ndarray,
                                     n_neighbors: int = 5) -> float:
    """ 条件変数でデータを分割し，各部分集合の相互情報量の加重平均を計算 """
    total_cmi = 0.0
    unique_conditions = np.unique(condition)
    for c_val in unique_conditions:
        p_condition = np.mean(condition == c_val)
        indices = (condition == c_val)
        Z_sub, target_sub = Z[indices], target[indices]

        if len(Z_sub) < n_neighbors * 2: continue # サンプルが少なすぎる場合はスキップ

        mi_sub = vector_mutual_information(Z_sub, target_sub, n_neighbors=n_neighbors)
        if not np.isnan(mi_sub):
            total_cmi += p_condition * mi_sub

    return total_cmi

# ==============================================================================
# 条件付きWasserstein距離を計算するヘルパー関数
# ==============================================================================
def conditional_wasserstein_distance(Z: np.ndarray, primary_var: np.ndarray, condition_var: np.ndarray) -> float:
    """
    条件変数でデータを分割し，各部分集合における主変数間のWasserstein距離の加重平均を計算．
    具体的には， E_c [ W_2^2( P(Z | primary=v1, condition=c), P(Z | primary=v2, condition=c) ) ] を計算．
    """
    if ot is None:
        return np.nan

    total_wd = 0.0
    unique_conditions = np.unique(condition_var)

    for c_val in unique_conditions:
        p_condition = np.mean(condition_var == c_val)
        indices_c = (condition_var == c_val)

        Z_sub = Z[indices_c]
        primary_sub = primary_var[indices_c]

        unique_primary_vals = np.unique(primary_sub)
        if len(unique_primary_vals) < 2:
            continue

        # 主変数が2値 (+1, -1) であることを仮定
        val1, val2 = unique_primary_vals[0], unique_primary_vals[1]

        indices_p1 = (primary_sub == val1)
        indices_p2 = (primary_sub == val2)

        Z1 = Z_sub[indices_p1]
        Z2 = Z_sub[indices_p2]

        if Z1.shape[0] == 0 or Z2.shape[0] == 0:
            continue

        # コスト行列を計算 (2乗ユークリッド距離)
        cost_matrix = ot.dist(Z1, Z2, metric='sqeuclidean')

        # 各サンプルの重み (一様分布)
        n1, n2 = Z1.shape[0], Z2.shape[0]
        a_weights = np.ones((n1,)) / n1
        b_weights = np.ones((n2,)) / n2

        # 2-Wasserstein距離の2乗 (EMD) を計算
        try:
            wasserstein_dist_sq = ot.emd2(a_weights, b_weights, cost_matrix)
            if not np.isnan(wasserstein_dist_sq):
                total_wd += p_condition * wasserstein_dist_sq
        except Exception as e:
            print(f"Warning: Wasserstein distance calculation failed for a subgroup. Error: {e}")

    return total_wd

# ==============================================================================
# グループ間のWasserstein距離を計算する関数
# ==============================================================================
def analyze_intergroup_distances(Z, y_np, a_np, dataset_type):
    """ 4つの(y,a)グループ間の全6ペアのWasserstein距離を計算 """
    print(f"\nAnalyzing INTER-GROUP Wasserstein distances on {dataset_type} data...")
    if ot is None:
        print("Skipping inter-group WD analysis because 'pot' library is not found.")
        return {}

    distances = {}
    groups_data = {
        (y, a): Z[(y_np == y) & (a_np == a)]
        for y in [-1, 1] for a in [-1, 1]
    }

    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for (y1, a1), (y2, a2) in combinations(group_keys, 2):
        key_name = f"G({y1},{a1})_vs_G({y2},{a2})"

        Z1 = groups_data[(y1, a1)]
        Z2 = groups_data[(y2, a2)]

        if Z1.shape[0] == 0 or Z2.shape[0] == 0:
            distances[key_name] = np.nan
            print(f"  - Skipping {key_name} due to empty group.")
            continue

        try:
            cost_matrix = ot.dist(Z1, Z2, metric='sqeuclidean')
            weights1 = ot.unif(Z1.shape[0])
            weights2 = ot.unif(Z2.shape[0])
            wd_sq = ot.emd2(weights1, weights2, cost_matrix)
            distances[key_name] = wd_sq
        except Exception as e:
            distances[key_name] = np.nan
            print(f"Warning: Wasserstein distance calculation failed for {key_name}. Error: {e}")

    # 結果の表示
    print("-" * 40)
    for name, dist in distances.items():
        print(f"  {name:<25}: {dist:.4f}")
    print("-" * 40)

    return distances

# ==============================================================================
# 【重心法】重心ベースのWasserstein距離を計算する関数
# ==============================================================================
def barycentric_wasserstein_distance(Z: np.ndarray, y_np: np.ndarray, a_np: np.ndarray,
                                     dist_type: str, barycenter_support_size: int = 100) -> float:
    """
    重心を計算し，重心分布間のWasserstein距離を計算．
    dist_type: 'core' または 'spurious'
    """
    if ot is None:
        return np.nan

    groups = {(y, a): Z[(y_np == y) & (a_np == a)] for y in [-1, 1] for a in [-1, 1]}
    barycenters = {}

    if dist_type == 'core':
        primary_vars, condition_vars = [-1, 1], [-1, 1] # y, a
        condition_prob_fn = lambda c: np.mean(a_np == c)
        print("Calculating barycentric CORE distance...")
    elif dist_type == 'spurious':
        primary_vars, condition_vars = [-1, 1], [-1, 1] # a, y
        condition_prob_fn = lambda c: np.mean(y_np == c)
        print("Calculating barycentric SPURIOUS distance...")
    else:
        raise ValueError("dist_type must be 'core' or 'spurious'")

    for p_val in primary_vars:
        measures_locations, measures_weights, barycenter_weights = [], [], []

        for c_val in condition_vars:
            key = (p_val, c_val) if dist_type == 'core' else (c_val, p_val)
            group_data = groups.get(key)

            if group_data is not None and group_data.shape[0] > 0:
                measures_locations.append(group_data)
                measures_weights.append(ot.unif(group_data.shape[0]))
                barycenter_weights.append(condition_prob_fn(c_val))

        if not measures_locations:
            barycenters[p_val] = None
            continue

        barycenter_weights = np.array(barycenter_weights)
        barycenter_weights /= barycenter_weights.sum()

        combined_dist = np.vstack(measures_locations)
        support_size = min(barycenter_support_size, combined_dist.shape[0])
        if support_size == 0:
            barycenters[p_val] = None
            continue
        init_support_indices = np.random.choice(combined_dist.shape[0], support_size, replace=False)
        X_init = combined_dist[init_support_indices, :]

        try:
            barycenter = ot.lp.free_support_barycenter(
                measures_locations=measures_locations,
                measures_weights=measures_weights,
                X_init=X_init,
                weights=barycenter_weights
            )
            barycenters[p_val] = barycenter
        except Exception as e:
            print(f"Warning: Barycenter calculation failed for primary_val={p_val}: {e}")
            barycenters[p_val] = None

    b1 = barycenters.get(primary_vars[0])
    b2 = barycenters.get(primary_vars[1])

    if b1 is not None and b2 is not None:
        try:
            cost_matrix = ot.dist(b1, b2, metric='sqeuclidean')
            n1, n2 = b1.shape[0], b2.shape[0]
            w1, w2 = np.ones((n1,)) / n1, np.ones((n2,)) / n2
            return ot.emd2(w1, w2, cost_matrix)
        except Exception as e:
            print(f"Warning: Final distance calculation between barycenters failed. Error: {e}")
            return np.nan
    else:
        return np.nan

# ==============================================================================
# 平均変位ベクトルを計算する関数
# ==============================================================================
def get_transport_direction(Z1, Z2):
    """
    2つの分布間の平均変位ベクトルをメモリ効率良く計算
    E[(z2 - z1)] where (z1, z2) ~ pi*
    """
    if ot is None or Z1.shape[0] == 0 or Z2.shape[0] == 0:
        d = Z1.shape[1] if Z1.shape[0] > 0 and Z1.ndim > 1 else (Z2.shape[1] if Z2.shape[0] > 0 and Z2.ndim > 1 else 1)
        return np.zeros(d)

    try:
        n1, d = Z1.shape
        n2 = Z2.shape[0]

        # 1. コスト行列と輸送計画を計算
        cost_matrix = ot.dist(Z1, Z2, metric='euclidean')
        a_weights = ot.unif(n1)
        b_weights = ot.unif(n2)
        transport_plan = ot.emd(a_weights, b_weights, cost_matrix) # shape (n1, n2)

        # 2. 巨大な中間配列を作らず，ループで計算
        average_displacement_vector = np.zeros(d, dtype=np.float64)

        # transport_planの非ゼロ要素だけをループすることでさらに効率化
        non_zero_indices = np.argwhere(transport_plan > 1e-15) # 計算誤差を考慮

        for i, j in non_zero_indices:
            weight = transport_plan[i, j]
            displacement = Z2[j] - Z1[i]
            average_displacement_vector += weight * displacement

        return average_displacement_vector
    except Exception as e:
        print(f"Warning: Could not compute true transport direction: {e}")
        d = Z1.shape[1] if Z1.ndim > 1 else 1
        return np.zeros(d)

# ==============================================================================
# 【ベクトル平均化】アライメント分析
# ==============================================================================
def analyze_vector_averaging_alignment(Z, y_np, a_np, w_classifier, dataset_type):
    """
    分類器の重みベクトルと，コア/スプリアス特徴の方向とのアライメントを計算（ベクトル平均化）
    """
    print(f"\nAnalyzing VECTOR-AVERAGING classifier alignment on {dataset_type} data...")
    if ot is None:
        print("Skipping alignment analysis because 'pot' library is not found.")
        return np.nan, np.nan

    groups = {(y, a): Z[(y_np == y) & (a_np == a)] for y in [-1, 1] for a in [-1, 1]}

    # --- コア特徴の方向ベクトル (v_core) を計算 ---
    v_core_sum = np.zeros_like(w_classifier)
    total_weight_core = 0
    for a_val in [-1, 1]:
        p_a = np.mean(a_np == a_val)
        if p_a > 0:
            Z_neg_y = groups.get((-1, a_val), np.array([]))
            Z_pos_y = groups.get((1, a_val), np.array([]))
            v_cond_a = get_transport_direction(Z_neg_y, Z_pos_y)
            v_core_sum += p_a * v_cond_a
            total_weight_core += p_a
    v_core = v_core_sum / total_weight_core if total_weight_core > 0 else v_core_sum

    # --- スプリアス特徴の方向ベクトル (v_spurious) を計算 ---
    v_spurious_sum = np.zeros_like(w_classifier)
    total_weight_spurious = 0
    for y_val in [-1, 1]:
        p_y = np.mean(y_np == y_val)
        if p_y > 0:
            Z_neg_a = groups.get((y_val, -1), np.array([]))
            Z_pos_a = groups.get((y_val, 1), np.array([]))
            v_cond_y = get_transport_direction(Z_neg_a, Z_pos_a)
            v_spurious_sum += p_y * v_cond_y
            total_weight_spurious += p_y
    v_spurious = v_spurious_sum / total_weight_spurious if total_weight_spurious > 0 else v_spurious_sum

    # --- コサイン類似度でアライメントを計算 ---
    def cosine_similarity(v1, v2):
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    core_alignment = cosine_similarity(w_classifier, v_core)
    spurious_alignment = cosine_similarity(w_classifier, v_spurious)

    print(f"[{dataset_type}] Vec-Avg Core Alignment: {core_alignment:.4f}, Vec-Avg Spurious Alignment: {spurious_alignment:.4f}")
    return core_alignment, spurious_alignment

# ==============================================================================
# 【重心法】アライメント分析
# ==============================================================================
def analyze_barycentric_alignment(Z, y_np, a_np, w_classifier, dataset_type, barycenter_support_size=100):
    """
    分類器のアライメントを計算する（重心法）
    """
    print(f"\nAnalyzing BARYCENTRIC classifier alignment on {dataset_type} data...")
    if ot is None:
        print("Skipping alignment analysis because 'pot' library is not found.")
        return np.nan, np.nan

    groups = {(y, a): Z[(y_np == y) & (a_np == a)] for y in [-1, 1] for a in [-1, 1]}

    # --- Core Direction (v_core) の計算 ---
    barycenters = {}
    for y_val in [-1, 1]:
        print(f"  Calculating barycenter for Y={y_val}...")
        measures_locations, measures_weights, barycenter_weights = [], [], []
        for a_val in [-1, 1]:
            group_data = groups.get((y_val, a_val))
            if group_data is not None and group_data.shape[0] > 0:
                measures_locations.append(group_data)
                measures_weights.append(ot.unif(group_data.shape[0]))
                barycenter_weights.append(np.mean(a_np == a_val))

        if not measures_locations:
            barycenters[y_val] = None
            continue

        barycenter_weights = np.array(barycenter_weights)
        barycenter_weights /= barycenter_weights.sum()

        combined_dist = np.vstack(measures_locations)
        support_size = min(barycenter_support_size, combined_dist.shape[0])
        if support_size == 0:
            barycenters[y_val] = None
            continue
        init_support_indices = np.random.choice(combined_dist.shape[0], support_size, replace=False)
        X_init = combined_dist[init_support_indices, :]

        try:
            barycenter = ot.lp.free_support_barycenter(
                measures_locations=measures_locations,
                measures_weights=measures_weights,
                X_init=X_init,
                weights=barycenter_weights
            )
            barycenters[y_val] = barycenter
        except Exception as e:
            print(f"Warning: Barycenter calculation failed for Y={y_val}: {e}")
            barycenters[y_val] = None

    if barycenters.get(-1) is not None and barycenters.get(1) is not None:
        v_core = get_transport_direction(barycenters[-1], barycenters[1])
    else:
        v_core = np.zeros_like(w_classifier)

    # --- Spurious Direction (v_spurious) の計算 ---
    barycenters = {}
    for a_val in [-1, 1]:
        print(f"  Calculating barycenter for A={a_val}...")
        measures_locations, measures_weights, barycenter_weights = [], [], []
        for y_val in [-1, 1]:
            group_data = groups.get((y_val, a_val))
            if group_data is not None and group_data.shape[0] > 0:
                measures_locations.append(group_data)
                measures_weights.append(ot.unif(group_data.shape[0]))
                barycenter_weights.append(np.mean(y_np == y_val))

        if not measures_locations:
            barycenters[a_val] = None
            continue

        barycenter_weights = np.array(barycenter_weights)
        barycenter_weights /= barycenter_weights.sum()

        combined_dist = np.vstack(measures_locations)
        support_size = min(barycenter_support_size, combined_dist.shape[0])
        if support_size == 0:
            barycenters[a_val] = None
            continue
        init_support_indices = np.random.choice(combined_dist.shape[0], support_size, replace=False)
        X_init = combined_dist[init_support_indices, :]

        try:
            barycenter = ot.lp.free_support_barycenter(
                measures_locations=measures_locations,
                measures_weights=measures_weights,
                X_init=X_init,
                weights=barycenter_weights
            )
            barycenters[a_val] = barycenter
        except Exception as e:
            print(f"Warning: Barycenter calculation failed for A={a_val}: {e}")
            barycenters[a_val] = None

    if barycenters.get(-1) is not None and barycenters.get(1) is not None:
        v_spurious = get_transport_direction(barycenters[-1], barycenters[1])
    else:
        v_spurious = np.zeros_like(w_classifier)

    def cosine_similarity(v1, v2):
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    core_alignment = cosine_similarity(w_classifier, v_core)
    spurious_alignment = cosine_similarity(w_classifier, v_spurious)

    print(f"[{dataset_type}] Barycentric Core Alignment: {core_alignment:.4f}, Barycentric Spurious Alignment: {spurious_alignment:.4f}")
    return core_alignment, spurious_alignment


# ==============================================================================
# 【重心法】輸送アライメント分析
# ==============================================================================
def analyze_transport_alignment(Z, y_np, a_np, w_classifier, dataset_type, barycenter_support_size=100):
    """
    分類器wが何に基づいて形成されたかを，輸送コストの方向性分解によって分析
    """
    print(f"\nAnalyzing TRANSPORT-BASED classifier alignment on {dataset_type} data...")
    if ot is None:
        print("Skipping alignment analysis because 'pot' library is not found.")
        return np.nan, np.nan

    # --------------------------------------------------------------------------
    # ヘルパー関数: 2つの重心分布から輸送アライメントを計算
    # --------------------------------------------------------------------------
    def _calculate_transport_alignment(barycenter1, barycenter2, w_classifier):
        if barycenter1 is None or barycenter2 is None or barycenter1.shape[0] < 2 or barycenter2.shape[0] < 2:
            return np.nan

        try:
            n1, n2 = barycenter1.shape[0], barycenter2.shape[0]
            weights1, weights2 = ot.unif(n1), ot.unif(n2)

            # 1. 最適輸送計画 T を計算
            cost_matrix = ot.dist(barycenter1, barycenter2, metric='sqeuclidean')
            transport_plan = ot.emd(weights1, weights2, cost_matrix)

            # 2. 総輸送コストとw方向の輸送コストを計算
            w_norm = w_classifier / (np.linalg.norm(w_classifier) + 1e-9)

            total_transport_cost = 0.0
            w_direction_transport_cost = 0.0

            non_zero_indices = np.argwhere(transport_plan > 1e-15)

            for i, j in non_zero_indices:
                mass = transport_plan[i, j]
                p1 = barycenter1[i]
                p2 = barycenter2[j]

                displacement_vector = p2 - p1

                # 総輸送コストへの寄与
                total_transport_cost += mass * np.sum(displacement_vector**2)

                # w方向の輸送コストへの寄与 (変位ベクトルのw方向への射影長の2乗)
                projected_length_sq = (displacement_vector @ w_norm)**2
                w_direction_transport_cost += mass * projected_length_sq

            # 3. 輸送アライメントを計算
            if total_transport_cost < 1e-9:
                return 1.0 if w_direction_transport_cost < 1e-9 else 0.0

            alignment_ratio = w_direction_transport_cost / total_transport_cost
            return np.clip(alignment_ratio, 0.0, 1.0)

        except Exception as e:
            print(f"Warning: Transport alignment calculation failed: {e}")
            return np.nan

    # --------------------------------------------------------------------------
    # 重心の計算
    # --------------------------------------------------------------------------
    groups = {(y, a): Z[(y_np == y) & (a_np == a)] for y in [-1, 1] for a in [-1, 1]}

    def _get_barycenter(keys, weight_fn):
        measures_locations, measures_weights, barycenter_weights = [], [], []
        for key in keys:
            group_data = groups.get(key)
            if group_data is not None and group_data.shape[0] > 1:
                measures_locations.append(group_data)
                measures_weights.append(ot.unif(group_data.shape[0]))
                barycenter_weights.append(weight_fn(key))

        if not measures_locations: return None

        barycenter_weights = np.array(barycenter_weights)
        if barycenter_weights.sum() == 0: return None
        barycenter_weights /= barycenter_weights.sum()

        combined_dist = np.vstack(measures_locations)
        support_size = min(barycenter_support_size, combined_dist.shape[0])
        if support_size < 2: return None
        init_support_indices = np.random.choice(combined_dist.shape[0], support_size, replace=False)
        X_init = combined_dist[init_support_indices, :]

        try:
            return ot.lp.free_support_barycenter(
                measures_locations, measures_weights, X_init, weights=barycenter_weights
            )
        except Exception as e:
            print(f"Warning: Barycenter calculation failed for keys {keys}: {e}")
            return None

    # --- Core Alignment の計算 ---
    print("  Calculating core barycenters (Y=-1 vs Y=+1)...")
    barycenter_y_neg = _get_barycenter([(-1, -1), (-1, 1)], lambda key: np.mean(a_np == key[1]))
    barycenter_y_pos = _get_barycenter([(1, -1), (1, 1)], lambda key: np.mean(a_np == key[1]))
    core_transport_alignment = _calculate_transport_alignment(barycenter_y_neg, barycenter_y_pos, w_classifier)

    # --- Spurious Alignment の計算 ---
    print("  Calculating spurious barycenters (A=-1 vs A=+1)...")
    barycenter_a_neg = _get_barycenter([(-1, -1), (1, -1)], lambda key: np.mean(y_np == key[0]))
    barycenter_a_pos = _get_barycenter([(-1, 1), (1, 1)], lambda key: np.mean(y_np == key[0]))
    spurious_transport_alignment = _calculate_transport_alignment(barycenter_a_neg, barycenter_a_pos, w_classifier)

    print(f"[{dataset_type}] Transport Core Alignment: {core_transport_alignment:.4f}, Transport Spurious Alignment: {spurious_transport_alignment:.4f}")
    return core_transport_alignment, spurious_transport_alignment

# ==============================================================================
# ★勾配グラム行列の分析
# ==============================================================================
def analyze_gradient_gram_matrix(model, X_data, y_data, a_data, device, loss_function, dataset_type, optimizer_params):
    """グループ間の勾配の内積からなるグラム行列を計算（学習率を考慮）"""
    print(f"\nAnalyzing GRADIENT GRAM MATRIX on {dataset_type} data...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_grads = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_grads[(y_val, a_val)] = None
            continue
        
        X_group, y_group = X_data[mask], y_data[mask]
        
        model.zero_grad()
        scores, _ = model(X_group.to(device))
        
        if loss_function == 'logistic':
            loss = torch.nn.functional.softplus(-y_group.to(device) * scores).mean()
        else: # mse
            loss = torch.nn.functional.mse_loss(scores, y_group.to(device))
            
        loss.backward()
        
        # フラット化せず，パラメータごとの勾配テンソルのリストとして保持
        grads_list = [p.grad.cpu().numpy() for p in model.parameters() if p.grad is not None]
        group_grads[(y_val, a_val)] = grads_list

    gram_matrix_results = {}
    # 対角成分も計算するため combinations_with_replacement を使用
    for (y1, a1), (y2, a2) in combinations_with_replacement(group_keys, 2):
        grads1, grads2 = group_grads.get((y1, a1)), group_grads.get((y2, a2))
        
        key_name = f"G({y1},{a1})_vs_G({y2},{a2})"
        if grads1 is not None and grads2 is not None:
            # 層ごとの学習率を考慮した重み付き内積を計算
            weighted_dot_product = 0.0
            param_idx = 0
            # optimizer_params の順序は model.parameters() と一致していることを前提とする
            for group in optimizer_params:
                lr = group['lr']
                for _ in group['params']:
                    # パラメータが勾配を持つことを確認
                    if param_idx < len(grads1) and param_idx < len(grads2):
                        grad1_p = grads1[param_idx].flatten()
                        grad2_p = grads2[param_idx].flatten()
                        weighted_dot_product += lr * np.dot(grad1_p, grad2_p)
                    param_idx += 1
            gram_matrix_results[key_name] = weighted_dot_product
        else:
            gram_matrix_results[key_name] = np.nan
            
    return gram_matrix_results

# ==============================================================================
# ヤコビアンノルムの分析
# ==============================================================================
def analyze_jacobian_norms(model, X_data, y_data, a_data, device, num_samples, dataset_type):
    """グループごとのヤコビアンノルムと内積を計算"""
    print(f"\nAnalyzing JACOBIAN NORMS on {dataset_type} data (using {num_samples} samples per group)...")
    group_keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    group_jacobians = {}

    for y_val, a_val in group_keys:
        mask = (y_data == y_val) & (a_data == a_val)
        if mask.sum() == 0:
            group_jacobians[(y_val, a_val)] = None
            continue
        
        X_group = X_data[mask]
        # サンプル数が指定数より多い場合はランダムサンプリング
        if len(X_group) > num_samples:
            indices = np.random.choice(len(X_group), num_samples, replace=False)
            X_subset = X_group[indices]
        else:
            X_subset = X_group
            
        group_jacobians[(y_val, a_val)] = get_model_jacobian(model, X_subset, device)

    jacobian_results = {}
    # ノルムの計算
    for (y, a), jacobian in group_jacobians.items():
        key_name = f"norm_G({y},{a})"
        if jacobian is not None:
            jacobian_results[key_name] = np.linalg.norm(jacobian)**2
        else:
            jacobian_results[key_name] = np.nan

    # 内積の計算
    for (y1, a1), (y2, a2) in combinations(group_keys, 2):
        jac1, jac2 = group_jacobians.get((y1, a1)), group_jacobians.get((y2, a2))
        key_name = f"dot_G({y1},{a1})_vs_G({y2},{a2})"
        if jac1 is not None and jac2 is not None:
            jacobian_results[key_name] = np.dot(jac1, jac2)
        else:
            jacobian_results[key_name] = np.nan
            
    return jacobian_results

# ==============================================================================
# 勾配グラム行列のスペクトル分析
# ==============================================================================
def analyze_gradient_gram_spectrum(gram_matrix_results, dataset_type):
    """勾配グラム行列の固有値と主固有ベクトルを計算"""
    print(f"\nAnalyzing GRADIENT GRAM SPECTRUM on {dataset_type} data...")
    group_order = [(-1,-1), (-1,1), (1,-1), (1,1)]
    G = np.zeros((4, 4))
    
    for i, g1 in enumerate(group_order):
        for j, g2 in enumerate(group_order):
            # 対称性を利用
            if i <= j:
                key = f"G({g1[0]},{g1[1]})_vs_G({g2[0]},{g2[1]})"
                val = gram_matrix_results.get(key, np.nan)
                G[i, j] = G[j, i] = val
            
    if np.isnan(G).any():
        print("Gram matrix contains NaN values. Skipping spectrum analysis.")
        return {'eigenvalues': [np.nan]*4, 'eigenvector1': [np.nan]*4, 'eigenvector2': [np.nan]*4}
        
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        # 固有値を降順にソート
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        # 上位2つの固有ベクトル
        main_eigenvector = eigenvectors[:, sorted_indices[0]]
        second_eigenvector = eigenvectors[:, sorted_indices[1]]
        
        return {
            'eigenvalues': sorted_eigenvalues.tolist(),
            'eigenvector1': main_eigenvector.tolist(),
            'eigenvector2': second_eigenvector.tolist()
        }
    except np.linalg.LinAlgError as e:
        print(f"Eigendecomposition failed: {e}")
        return {'eigenvalues': [np.nan]*4, 'eigenvector1': [np.nan]*4, 'eigenvector2': [np.nan]*4}

# ==============================================================================
# 多数派/少数派の勾配ノルム比の分析
# ==============================================================================
def analyze_gradient_norm_ratio(gram_matrix_results, dataset_type):
    """多数派グループと少数派グループの勾配ノルムの比を計算"""
    print(f"\nAnalyzing GRADIENT NORM RATIO on {dataset_type} data...")
    
    # 勾配ノルムの2乗はグラム行列の対角成分
    norm_sq_maj1 = gram_matrix_results.get('G(-1,-1)_vs_G(-1,-1)', np.nan)
    norm_sq_maj2 = gram_matrix_results.get('G(1,1)_vs_G(1,1)', np.nan)
    norm_sq_min1 = gram_matrix_results.get('G(-1,1)_vs_G(-1,1)', np.nan)
    norm_sq_min2 = gram_matrix_results.get('G(1,-1)_vs_G(1,-1)', np.nan)

    # NaNチェック
    if any(np.isnan([norm_sq_maj1, norm_sq_maj2, norm_sq_min1, norm_sq_min2])):
        print("Cannot calculate norm ratio due to missing norm values.")
        return {'ratio': np.nan}

    # 各ノルムを計算し，合計
    norm_maj = np.sqrt(norm_sq_maj1) + np.sqrt(norm_sq_maj2)
    norm_min = np.sqrt(norm_sq_min1) + np.sqrt(norm_sq_min2)

    # ゼロ除算を回避
    if norm_min < 1e-9:
        ratio = np.inf if norm_maj > 1e-9 else np.nan
    else:
        ratio = norm_maj / norm_min

    return {'ratio': ratio}


# ==============================================================================
# 各層の分析関数
# ==============================================================================
def analyze_layer(layer_name, Z, y_np, a_np, n_neighbors, dataset_type, analyze_mi=True):
    """ 指定された層の特徴量について情報量と条件付きWDを計算し表示する """
    print("\n" + "#"*70 + f"\n### Analyzing Layer: {layer_name} on {dataset_type} Data ###\n" + "#"*70)

    if Z.ndim > 2: Z = Z.reshape(Z.shape[0], -1)
    if Z.ndim == 1: Z = Z.reshape(-1, 1)

    # --- 相互情報量の計算 ---
    core_info, spurious_info, spurious_info_ratio = np.nan, np.nan, np.nan
    if analyze_mi:
        print(f"Calculating mutual information (n_neighbors={n_neighbors})...")
        start_time_mi = time.time()
        core_info = conditional_mutual_information(Z, y_np, a_np, n_neighbors=n_neighbors)
        spurious_info = conditional_mutual_information(Z, a_np, y_np, n_neighbors=n_neighbors)
        print(f"MI calculation took: {time.time() - start_time_mi:.2f} seconds")
        total_unique_info = core_info + spurious_info
        spurious_info_ratio = spurious_info / total_unique_info if total_unique_info > 1e-9 else 0.0

    # --- 条件付きWasserstein距離の計算 ---
    wd_core, wd_spurious, wd_ratio, wd_simple_ratio = np.nan, np.nan, np.nan, np.nan
    if ot is not None:
        print(f"Calculating conditional Wasserstein distances...")
        start_time_wd = time.time()
        wd_core = conditional_wasserstein_distance(Z, y_np, a_np)
        wd_spurious = conditional_wasserstein_distance(Z, a_np, y_np)
        total_wd = wd_core + wd_spurious
        wd_ratio = wd_spurious / total_wd if not np.isnan(total_wd) and total_wd > 1e-9 else 0.0
        wd_simple_ratio = wd_spurious / wd_core if not np.isnan(wd_core) and wd_core > 1e-9 else 0.0
        print(f"WD calculation took: {time.time() - start_time_wd:.2f} seconds")
    else:
        print("Skipping Wasserstein distance calculation ('pot' library not found).")

    # --- 結果の表示 ---
    print("\n" + "="*65 + f"\nAnalysis Results for Layer: {layer_name} ({dataset_type})\n" + "="*65)
    nats_to_bits = np.log2(np.e)
    if analyze_mi:
        print(f"Core Information         I(Z; Y | A) = {core_info:.4f} nats ({core_info * nats_to_bits:.4f} bits)")
        print(f"Spurious Information     I(Z; A | Y) = {spurious_info:.4f} nats ({spurious_info * nats_to_bits:.4f} bits)")
        print(f"Spurious Info Ratio                = {spurious_info_ratio:.2%}")
        print("-" * 65)
    print(f"Cond. Core Dist (W-sq)         = {wd_core:.4f}")
    print(f"Cond. Spurious Dist (W-sq)     = {wd_spurious:.4f}")
    print(f"Cond. Spurious Dist Ratio      = {wd_ratio:.2%}")
    print(f"Cond. Spurious Simple Ratio    = {wd_simple_ratio:.4f}")

    return core_info, spurious_info, spurious_info_ratio, wd_core, wd_spurious, wd_ratio, wd_simple_ratio


# ==============================================================================
# 全ての分析を統括するラッパー関数
# ==============================================================================
def run_all_analyses(config, epoch, layers, model, train_outputs, test_outputs,
                     X_train, y_train, a_train, X_test, y_test, a_test, histories,
                     optimizer_params, history): 
    """設定に基づいてすべての分析を実行し，結果をhistory辞書に保存する"""
    y_train_np, a_train_np = y_train.numpy(), a_train.numpy()
    y_test_np, a_test_np = y_test.numpy(), a_test.numpy()
    
    analysis_target = config['analysis_target']

    # --- 勾配グラム行列関連の分析 ---
    grad_gram_train_results, grad_gram_test_results = None, None
    run_grad_gram_related_analysis = config.get('analyze_gradient_gram', False) or \
                                     config.get('analyze_gradient_gram_spectrum', False) or \
                                     config.get('analyze_gradient_norm_ratio', False)

    if run_grad_gram_related_analysis:
        if analysis_target in ['train', 'both']:
            grad_gram_train_results = analyze_gradient_gram_matrix(
                model, X_train, y_train, a_train, config['device'], config['loss_function'], "Train", optimizer_params)
            if config.get('analyze_gradient_gram', False):
                histories['grad_gram_train'][epoch] = grad_gram_train_results
        if analysis_target in ['test', 'both']:
            grad_gram_test_results = analyze_gradient_gram_matrix(
                model, X_test, y_test, a_test, config['device'], config['loss_function'], "Test", optimizer_params)
            if config.get('analyze_gradient_gram', False):
                histories['grad_gram_test'][epoch] = grad_gram_test_results

    if config.get('analyze_gradient_gram_spectrum', False):
        if grad_gram_train_results:
            histories['grad_gram_spectrum_train'][epoch] = analyze_gradient_gram_spectrum(grad_gram_train_results, "Train")
        if grad_gram_test_results:
            histories['grad_gram_spectrum_test'][epoch] = analyze_gradient_gram_spectrum(grad_gram_test_results, "Test")

    if config.get('analyze_gradient_norm_ratio', False):
        if grad_gram_train_results:
            histories['grad_norm_ratio_train'][epoch] = analyze_gradient_norm_ratio(grad_gram_train_results, "Train")
        if grad_gram_test_results:
            histories['grad_norm_ratio_test'][epoch] = analyze_gradient_norm_ratio(grad_gram_test_results, "Test")


    # --- ヤコビアンノルムの分析 ---
    if config.get('analyze_jacobian_norm', False):
        if analysis_target in ['train', 'both']:
            histories['jacobian_norm_train'][epoch] = analyze_jacobian_norms(
                model, X_train, y_train, a_train, config['device'], config['jacobian_num_samples'], "Train")
        if analysis_target in ['test', 'both']:
            histories['jacobian_norm_test'][epoch] = analyze_jacobian_norms(
                model, X_test, y_test, a_test, config['device'], config['jacobian_num_samples'], "Test")

    # --- 汎化ギャップの推定 (訓練セットのみ) ---
    if config.get('analyze_generalization_gap', False):
        if analysis_target in ['train', 'both']:
            histories['gen_gap_train'][epoch] = analyze_generalization_gap(
                model, X_train, y_train, a_train, config['device'], config, epoch,
                history, optimizer_params, config['loss_function'], "Train")

    # --- 重み行列の特異値解析 ---
    if config['analyze_weight_singular_values']:
        sv_results_epoch = {}
        for i, layer in enumerate(model.layers):
            layer_name = f'layer_{i+1}'
            if layer_name in layers and hasattr(layer, 'weight'):
                W = layer.weight.data.cpu().numpy()
                S = np.linalg.svd(W, compute_uv=False)
                sv_results_epoch[layer_name] = S[:10]
        if 'logit' in layers and hasattr(model.classifier, 'weight'):
            W_classifier = model.classifier.weight.data.cpu().numpy()
            S_classifier = np.linalg.svd(W_classifier, compute_uv=False)
            sv_results_epoch['logit'] = S_classifier[:10]
        histories['weight_sv'][epoch] = sv_results_epoch
        print(f"Computed top 10 singular values for weight matrices of {len(sv_results_epoch)} layers.")

    # --- アクティベーションの特異値解析 ---
    if config['analyze_activation_singular_values']:
        if analysis_target in ['train', 'both']:
            act_sv_train = {layer: np.linalg.svd(train_outputs[layer].numpy().reshape(len(y_train), -1), compute_uv=False)[:10] for layer in layers}
            histories['activation_sv_train'][epoch] = act_sv_train
            print(f"Computed top 10 singular values for TRAIN activations of {len(act_sv_train)} layers.")
        if analysis_target in ['test', 'both']:
            act_sv_test = {layer: np.linalg.svd(test_outputs[layer].numpy().reshape(len(y_test), -1), compute_uv=False)[:10] for layer in layers}
            histories['activation_sv_test'][epoch] = act_sv_test
            print(f"Computed top 10 singular values for TEST activations of {len(act_sv_test)} layers.")

    # --- 各層のループ分析 ---
    mi_train, mi_test = {}, {}
    cond_wd_train, cond_wd_test = {}, {}
    inter_wd_train, inter_wd_test = {}, {}
    bary_wd_train, bary_wd_test = {}, {}

    for layer in layers:
        # Train
        if analysis_target in ['train', 'both'] and train_outputs is not None:
            Z_train = train_outputs[layer].numpy()
            if config['analyze_mutual_information'] or config['analyze_conditional_distance']:
                c_mi, s_mi, r_mi, c_wd, s_wd, r_wd, sr_wd = analyze_layer(layer, Z_train, y_train_np, a_train_np, config['n_neighbors_mi'], "Train", config['analyze_mutual_information'])
                if config['analyze_mutual_information']: mi_train[layer] = (c_mi, s_mi, r_mi)
                if config['analyze_conditional_distance']: cond_wd_train[layer] = (c_wd, s_wd, r_wd, sr_wd)
            if config['analyze_intergroup_distance']:
                inter_wd_train[layer] = analyze_intergroup_distances(Z_train, y_train_np, a_train_np, "Train")
            if config['analyze_barycentric_distance']:
                c_bwd = barycentric_wasserstein_distance(Z_train, y_train_np, a_train_np, 'core', config['barycenter_support_size'])
                s_bwd = barycentric_wasserstein_distance(Z_train, y_train_np, a_train_np, 'spurious', config['barycenter_support_size'])
                r_bwd = s_bwd / (c_bwd + s_bwd) if not np.isnan(c_bwd + s_bwd) and (c_bwd + s_bwd) > 1e-9 else 0.0
                sr_bwd = s_bwd / c_bwd if not np.isnan(c_bwd) and c_bwd > 1e-9 else np.nan
                bary_wd_train[layer] = (c_bwd, s_bwd, r_bwd, sr_bwd)
        # Test
        if analysis_target in ['test', 'both'] and test_outputs is not None:
            Z_test = test_outputs[layer].numpy()
            if config['analyze_mutual_information'] or config['analyze_conditional_distance']:
                c_mi, s_mi, r_mi, c_wd, s_wd, r_wd, sr_wd = analyze_layer(layer, Z_test, y_test_np, a_test_np, config['n_neighbors_mi'], "Test", config['analyze_mutual_information'])
                if config['analyze_mutual_information']: mi_test[layer] = (c_mi, s_mi, r_mi)
                if config['analyze_conditional_distance']: cond_wd_test[layer] = (c_wd, s_wd, r_wd, sr_wd)
            if config['analyze_intergroup_distance']:
                inter_wd_test[layer] = analyze_intergroup_distances(Z_test, y_test_np, a_test_np, "Test")
            if config['analyze_barycentric_distance']:
                c_bwd = barycentric_wasserstein_distance(Z_test, y_test_np, a_test_np, 'core', config['barycenter_support_size'])
                s_bwd = barycentric_wasserstein_distance(Z_test, y_test_np, a_test_np, 'spurious', config['barycenter_support_size'])
                r_bwd = s_bwd / (c_bwd + s_bwd) if not np.isnan(c_bwd + s_bwd) and (c_bwd + s_bwd) > 1e-9 else 0.0
                sr_bwd = s_bwd / c_bwd if not np.isnan(c_bwd) and c_bwd > 1e-9 else np.nan
                bary_wd_test[layer] = (c_bwd, s_bwd, r_bwd, sr_bwd)

    if mi_train: histories['mi_train'][epoch] = mi_train
    if mi_test: histories['mi_test'][epoch] = mi_test
    if cond_wd_train: histories['cond_wd_train'][epoch] = cond_wd_train
    if cond_wd_test: histories['cond_wd_test'][epoch] = cond_wd_test
    if inter_wd_train: histories['intergroup_wd_train'][epoch] = inter_wd_train
    if inter_wd_test: histories['intergroup_wd_test'][epoch] = inter_wd_test
    if bary_wd_train: histories['bary_wd_train'][epoch] = bary_wd_train
    if bary_wd_test: histories['bary_wd_test'][epoch] = bary_wd_test
    
    # --- Bregman Divergence ---
    if config['analyze_bregman_divergence']:
        if config['analyze_conditional_distance'] and config['analyze_barycentric_distance']:
            breg_train, breg_test = {}, {}
            for layer in layers:
                if layer in cond_wd_train and layer in bary_wd_train:
                    d_c = cond_wd_train[layer][0] - bary_wd_train[layer][0]
                    d_s = cond_wd_train[layer][1] - bary_wd_train[layer][1]
                    breg_train[layer] = (d_c, d_s)
                if layer in cond_wd_test and layer in bary_wd_test:
                    d_c = cond_wd_test[layer][0] - bary_wd_test[layer][0]
                    d_s = cond_wd_test[layer][1] - bary_wd_test[layer][1]
                    breg_test[layer] = (d_c, d_s)
            if breg_train: histories['bregman_wd_train'][epoch] = breg_train
            if breg_test: histories['bregman_wd_test'][epoch] = breg_test
            print("Computed Wasserstein Bregman Divergence.")
        else:
            print("Warning: Bregman Divergence requires Conditional and Barycentric analyses. Skipping.")

    # --- アライメント分析 ---
    if train_outputs is not None and test_outputs is not None:
        final_layer = f'layer_{config["num_hidden_layers"]}'
        w_classifier = model.classifier.weight.data.cpu().numpy().flatten()
        
        # Vector Averaging
        if config['analyze_vector_averaging_alignment']:
            if analysis_target in ['train', 'both']:
                Z_final = train_outputs[final_layer].numpy()
                c_a, s_a = analyze_vector_averaging_alignment(Z_final, y_train_np, a_train_np, w_classifier, "Train")
                r_a = abs(s_a) / (abs(c_a) + abs(s_a)) if (abs(c_a) + abs(s_a)) > 1e-9 else 0.0
                sr_a = abs(s_a) / abs(c_a) if abs(c_a) > 1e-9 else np.nan
                histories['vec_avg_align_train'][epoch] = (c_a, s_a, r_a, sr_a)
            if analysis_target in ['test', 'both']:
                Z_final = test_outputs[final_layer].numpy()
                c_a, s_a = analyze_vector_averaging_alignment(Z_final, y_test_np, a_test_np, w_classifier, "Test")
                r_a = abs(s_a) / (abs(c_a) + abs(s_a)) if (abs(c_a) + abs(s_a)) > 1e-9 else 0.0
                sr_a = abs(s_a) / abs(c_a) if abs(c_a) > 1e-9 else np.nan
                histories['vec_avg_align_test'][epoch] = (c_a, s_a, r_a, sr_a)
                
        # Barycentric (Vector-based)
        if config['analyze_barycentric_alignment']:
            if analysis_target in ['train', 'both']:
                Z_final = train_outputs[final_layer].numpy()
                c_a, s_a = analyze_barycentric_alignment(Z_final, y_train_np, a_train_np, w_classifier, "Train", config['barycenter_support_size'])
                r_a = abs(s_a) / (abs(c_a) + abs(s_a)) if (abs(c_a) + abs(s_a)) > 1e-9 else 0.0
                sr_a = abs(s_a) / abs(c_a) if abs(c_a) > 1e-9 else np.nan
                histories['bary_align_train'][epoch] = (c_a, s_a, r_a, sr_a)
            if analysis_target in ['test', 'both']:
                Z_final = test_outputs[final_layer].numpy()
                c_a, s_a = analyze_barycentric_alignment(Z_final, y_test_np, a_test_np, w_classifier, "Test", config['barycenter_support_size'])
                r_a = abs(s_a) / (abs(c_a) + abs(s_a)) if (abs(c_a) + abs(s_a)) > 1e-9 else 0.0
                sr_a = abs(s_a) / abs(c_a) if abs(c_a) > 1e-9 else np.nan
                histories['bary_align_test'][epoch] = (c_a, s_a, r_a, sr_a)
                
        # Barycentric (Transport-based)
        if config['analyze_transport_alignment']:
            if analysis_target in ['train', 'both']:
                Z_final = train_outputs[final_layer].numpy()
                c_a, s_a = analyze_transport_alignment(Z_final, y_train_np, a_train_np, w_classifier, "Train", config['barycenter_support_size'])
                r_a = s_a / (c_a + s_a) if not np.isnan(c_a + s_a) and (c_a + s_a) > 1e-9 else 0.0
                sr_a = s_a / c_a if not np.isnan(c_a) and c_a > 1e-9 else np.nan
                histories['transport_align_train'][epoch] = (c_a, s_a, r_a, sr_a)
            if analysis_target in ['test', 'both']:
                Z_final = test_outputs[final_layer].numpy()
                c_a, s_a = analyze_transport_alignment(Z_final, y_test_np, a_test_np, w_classifier, "Test", config['barycenter_support_size'])
                r_a = s_a / (c_a + s_a) if not np.isnan(c_a + s_a) and (c_a + s_a) > 1e-9 else 0.0
                sr_a = s_a / c_a if not np.isnan(c_a) and c_a > 1e-9 else np.nan
                histories['transport_align_test'][epoch] = (c_a, s_a, r_a, sr_a)
