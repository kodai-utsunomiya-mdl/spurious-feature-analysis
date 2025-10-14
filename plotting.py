# sp/plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from itertools import combinations, product
import pandas as pd

# 'pot'ライブラリのインポート
try:
    import ot
except ImportError:
    print("Warning: 'pot' library not found. Some plots will not be generated.")
    ot = None

plt.rc("figure", dpi=100, facecolor=(1, 1, 1))
plt.rc("font", family='stixgeneral', size=13)
plt.rc("axes", facecolor='white', titlesize=16)
plt.rc("mathtext", fontset='cm')
plt.rc('text', usetex=False)

# ==============================================================================
# 共通ヘルパー関数
# ==============================================================================

def _save_and_close(fig, save_dir, filename):
    """
    MatplotlibのFigureを指定されたディレクトリにファイルとして保存し，メモリを解放
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")

# ==============================================================================
# t-SNE可視化関数
# ==============================================================================

def visualize_tsne_layers(train_outputs, y_train, a_train, test_outputs, y_test, a_test, 
                          target_layers, epoch_num, save_dir):
    """
    各層の特徴表現をt-SNEで可視化し，ファイルに保存
    """
    title_suffix = f" (Epoch {epoch_num})"
    print(f"\nVisualizing and saving t-SNE representations{title_suffix}...")
    
    num_plots = len(target_layers)
    fig, axes = plt.subplots(2, num_plots, figsize=(6 * num_plots, 10), squeeze=False)
    fig.suptitle(f't-SNE Visualization of Layer Representations{title_suffix}', fontsize=16)

    groups = {
        '$y = +1$, $a = +1$ (Maj)': {'y': 1, 'a': 1, 'color': 'red', 'marker': 'o'},
        '$y = +1$, $a = -1$ (Min)': {'y': 1, 'a': -1, 'color': 'orange', 'marker': 'x'},
        '$y = -1$, $a = +1$ (Min)': {'y': -1, 'a': 1, 'color': 'blue', 'marker': 'x'},
        '$y = -1$, $a = -1$ (Maj)': {'y': -1, 'a': -1, 'color': 'cyan', 'marker': 'o'}
    }
    
    y_train_np, a_train_np = y_train.numpy(), a_train.numpy()
    y_test_np, a_test_np = y_test.numpy(), a_test.numpy()

    for i, layer_name in enumerate(target_layers):
        # 訓練データとテストデータのループ
        for j, (split_name, Z_dict, y_np, a_np, ax) in enumerate([
            ("Train", train_outputs, y_train_np, a_train_np, axes[0, i]),
            ("Test", test_outputs, y_test_np, a_test_np, axes[1, i])
        ]):
            if layer_name not in Z_dict:
                print(f"Warning: {layer_name} not found in {split_name} outputs for t-SNE. Skipping.")
                ax.axis('off')
                continue

            Z = Z_dict[layer_name].numpy()
            if Z.ndim > 2: Z = Z.reshape(Z.shape[0], -1)

            perplexity_val = min(30, Z.shape[0] - 1)
            if perplexity_val <= 0:
                print(f"Warning: Not enough samples in {split_name} data for t-SNE ({layer_name}). Skipping.")
                ax.axis('off')
                continue
                
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity_val, learning_rate='auto', init='pca', random_state=42)
                print(f"Running t-SNE for {split_name} Data ({layer_name})...")
                Z_2d = tsne.fit_transform(Z)
            except Exception as e:
                print(f"t-SNE with PCA init failed for {split_name} Data ({layer_name}): {e}. Retrying with random init.")
                tsne = TSNE(n_components=2, perplexity=perplexity_val, learning_rate='auto', init='random', random_state=42)
                Z_2d = tsne.fit_transform(Z)

            for label, props in groups.items():
                mask = (y_np == props['y']) & (a_np == props['a'])
                ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=props['color'], marker=props['marker'], label=label, alpha=0.7, s=10)
            ax.set_title(f'{split_name} Data - {layer_name}')
            ax.set_xticks([])
            ax.set_yticks([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_and_close(fig, save_dir, f"tsne_epoch_{epoch_num}.png")

# ==============================================================================
# 学習履歴プロット関数
# ==============================================================================

def plot_training_history(history_df, save_dir):
    """学習過程の各種メトリクスをプロットし，2つの画像ファイルとして保存"""
    epochs = history_df.index + 1
    
    # --- Figure 1: 平均とワーストグループのメトリクス ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Overall and Worst-Group Metrics Over Epochs', fontsize=16)

    axes1[0, 0].plot(epochs, history_df['train_avg_loss'], 'b-', label='Train Avg Loss')
    axes1[0, 0].plot(epochs, history_df['test_avg_loss'], 'b--', label='Test Avg Loss')
    axes1[0, 0].set(title='Average Loss', xlabel='Epochs', ylabel='Loss')
    axes1[0, 0].legend(); axes1[0, 0].grid(True)
    axes1[0, 1].plot(epochs, history_df['train_avg_acc'], 'r-', label='Train Avg Accuracy')
    axes1[0, 1].plot(epochs, history_df['test_avg_acc'], 'r--', label='Test Avg Accuracy')
    axes1[0, 1].set(title='Average Accuracy', xlabel='Epochs', ylabel='Accuracy', ylim=(0, 1.05))
    axes1[0, 1].legend(); axes1[0, 1].grid(True)
    axes1[1, 0].plot(epochs, history_df['train_worst_loss'], 'g-', label='Train Worst-Group Loss')
    axes1[1, 0].plot(epochs, history_df['test_worst_loss'], 'g--', label='Test Worst-Group Loss')
    axes1[1, 0].set(title='Worst-Group Loss', xlabel='Epochs', ylabel='Loss')
    axes1[1, 0].legend(); axes1[1, 0].grid(True)
    axes1[1, 1].plot(epochs, history_df['train_worst_acc'], 'm-', label='Train Worst-Group Accuracy')
    axes1[1, 1].plot(epochs, history_df['test_worst_acc'], 'm--', label='Test Worst-Group Accuracy')
    axes1[1, 1].set(title='Worst-Group Accuracy', xlabel='Epochs', ylabel='Accuracy', ylim=(0, 1.05))
    axes1[1, 1].legend(); axes1[1, 1].grid(True)
    
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig1, save_dir, "training_history_main.png")

    # --- Figure 2: グループごとのメトリクス ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
    fig2.suptitle('Per-Group Metrics Over Epochs', fontsize=16)
    group_labels = {0: '$y=-1, a=-1$', 1: '$y=-1, a=+1$', 2: '$y=+1, a=-1$', 3: '$y=+1, a=+1$'}
    
    train_losses = np.array(history_df['train_group_losses'].tolist())
    test_losses = np.array(history_df['test_group_losses'].tolist())
    train_accs = np.array(history_df['train_group_accs'].tolist())
    test_accs = np.array(history_df['test_group_accs'].tolist())
    
    for i in range(4):
        r, c = i // 2, i % 2
        ax_loss = axes2[r, c]
        ax_acc = ax_loss.twinx()
        p1, = ax_loss.plot(epochs, train_losses[:, i], 'c-', label=f'Train Loss ({group_labels[i]})')
        p2, = ax_loss.plot(epochs, test_losses[:, i], 'c--', label=f'Test Loss ({group_labels[i]})')
        ax_loss.set_ylabel('Loss', color='c'); ax_loss.tick_params(axis='y', labelcolor='c')
        ax_loss.grid(True, axis='y', linestyle=':')
        p3, = ax_acc.plot(epochs, train_accs[:, i], 'y-', label=f'Train Acc ({group_labels[i]})')
        p4, = ax_acc.plot(epochs, test_accs[:, i], 'y--', label=f'Test Acc ({group_labels[i]})')
        ax_acc.set_ylabel('Accuracy', color='y'); ax_acc.tick_params(axis='y', labelcolor='y'); ax_acc.set_ylim(0, 1.05)
        ax_loss.set(title=f'Group: {group_labels[i]}', xlabel='Epochs')
        ax_loss.legend(handles=[p1, p2, p3, p4], loc='best')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig2, save_dir, "training_history_groups.png")

def plot_misclassification_rates(final_metrics_series, dataset_name, save_dir):
    """最終テストセットのグループ別誤分類率をプロット"""
    print("\nPlotting misclassification rates...")
    if 'WaterBirds' in dataset_name:
        group_labels = ['Landbird\non Land\n($y=-1, a=-1$)', 'Landbird\non Water\n($y=-1, a=+1$)', 
                        'Waterbird\non Land\n($y=+1, a=-1$)', 'Waterbird\non Water\n($y=+1, a=+1$)']
    else: # ColoredMNIST
        group_labels = ['Digit<5, Green\n($y=-1, a=-1$)', 'Digit<5, Red\n($y=-1, a=+1$)',
                        'Digit>=5, Green\n($y=+1, a=-1$)', 'Digit>=5, Red\n($y=+1, a=+1$)']
    
    group_accs = final_metrics_series['test_group_accs']
    rates = [1 - (acc if not np.isnan(acc) else 0) for acc in group_accs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(group_labels, rates, color=['cyan', 'blue', 'orange', 'red'])
    ax.set(ylabel='Misclassification Rate', title='Final Test Set Misclassification Rate by Group')
    ax.set_ylim(0, max(1.05, max(rates) * 1.2 if rates else 1.05))
    ax.bar_label(bars, fmt='%.3f', fontsize=10, padding=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    fig.tight_layout()
    _save_and_close(fig, save_dir, "misclassification_rates.png")

# ==============================================================================
# 分析結果の時系列プロット関数
# ==============================================================================
def _plot_evolution_layered(history_train, history_test, target_layers, plot_configs, suptitle, filename, save_dir):
    """【レイヤー別】時系列プロットの共通ロジック"""
    if not history_train and not history_test:
        print(f"No layered history to plot for {filename}.")
        return

    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    num_metrics = len(next(iter(next(iter(history_train.values())).values()))) if history_train else 0
    plot_data_train = {layer: [[] for _ in range(num_metrics)] for layer in target_layers}
    plot_data_test = {layer: [[] for _ in range(num_metrics)] for layer in target_layers}

    for history, plot_data in [(history_train, plot_data_train), (history_test, plot_data_test)]:
        if not history: continue
        for epoch in epochs:
            for layer in target_layers:
                metrics = history.get(epoch, {}).get(layer)
                if metrics:
                    for i in range(num_metrics): plot_data[layer][i].append(metrics[i])
                else:
                    for i in range(num_metrics): plot_data[layer][i].append(np.nan)

    colors = plt.cm.viridis(np.linspace(0, 1, len(target_layers)))

    for i, config in enumerate(plot_configs):
        ax = axes.flatten()[i]
        metric_idx = config['metric_idx']
        for j, layer in enumerate(target_layers):
            if history_train: ax.plot(epochs, plot_data_train[layer][metric_idx], marker='o', linestyle='-', color=colors[j], label=f'{layer} (Train)')
            if history_test: ax.plot(epochs, plot_data_test[layer][metric_idx], marker='x', linestyle='--', color=colors[j], label=f'{layer} (Test)')

        ax.set(title=config['title'], xlabel='Epoch', ylabel=config['ylabel'])
        if config.get('log'): ax.set_yscale('log')
        if 'ylim' in config: ax.set_ylim(config['ylim'])
        ax.legend()
        ax.grid(True, which="both", ls="--")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_dir, filename)

def _plot_evolution_single(history_train, history_test, plot_configs, suptitle, filename, save_dir):
    """【単一系列】時系列プロットの共通ロジック（アライメント用）"""
    if not history_train and not history_test:
        print(f"No single-series history to plot for {filename}.")
        return

    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    num_metrics = len(next(iter(history_train.values()))) if history_train else 0
    plot_data_train = [[] for _ in range(num_metrics)]
    plot_data_test = [[] for _ in range(num_metrics)]

    for history, plot_data in [(history_train, plot_data_train), (history_test, plot_data_test)]:
        if not history: continue
        for epoch in epochs:
            metrics = history.get(epoch)
            if metrics:
                for i in range(num_metrics): plot_data[i].append(metrics[i])
            else:
                for i in range(num_metrics): plot_data[i].append(np.nan)
    
    for i, config in enumerate(plot_configs):
        ax = axes.flatten()[i]
        metric_idx = config['metric_idx']
        if history_train: ax.plot(epochs, plot_data_train[metric_idx], marker='o', linestyle='-', color='g', label='Train')
        if history_test: ax.plot(epochs, plot_data_test[metric_idx], marker='x', linestyle='--', color='r', label='Test')
        
        ax.set(title=config['title'], xlabel='Epoch', ylabel=config['ylabel'])
        if config.get('log'): ax.set_yscale('log')
        if 'ylim' in config: ax.set_ylim(config['ylim'])
        ax.legend()
        ax.grid(True, which="both", ls="--")
        
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_dir, filename)

# --- ラッパー関数 ---

def plot_mi_evolution(mi_history_train, mi_history_test, all_target_layers, save_dir):
    configs = [
        {'metric_idx': 0, 'title': 'Core Information: $I(Z; Y | A)$', 'ylabel': 'nats'},
        {'metric_idx': 1, 'title': 'Spurious Information: $I(Z; A | Y)$', 'ylabel': 'nats'},
        {'metric_idx': 2, 'title': 'Spurious Information Ratio', 'ylabel': 'Ratio', 'ylim': (0, 1.05)},
    ]
    _plot_evolution_layered(mi_history_train, mi_history_test, all_target_layers, configs, 
                    'Evolution of Mutual Information (Solid=Train, Dashed=Test)', 
                    'mi_evolution.png', save_dir)

def plot_conditional_wd_evolution(wd_history_train, wd_history_test, all_target_layers, save_dir):
    if ot is None: return
    configs = [
        {'metric_idx': 0, 'title': 'Cond. Core Distance (sq)', 'ylabel': 'Squared W-Dist (log)', 'log': True},
        {'metric_idx': 1, 'title': 'Cond. Spurious Distance (sq)', 'ylabel': 'Squared W-Dist (log)', 'log': True},
        {'metric_idx': 2, 'title': 'Spurious Ratio', 'ylabel': 'Ratio', 'ylim': (0, 1.05)},
        {'metric_idx': 3, 'title': 'Simple Ratio (Spurious/Core)', 'ylabel': 'Ratio (log)', 'log': True},
    ]
    _plot_evolution_layered(wd_history_train, wd_history_test, all_target_layers, configs,
                    'Evolution of Conditional Wasserstein Distances',
                    'conditional_wd_evolution.png', save_dir)

def plot_barycentric_wd_evolution(wd_history_train, wd_history_test, all_target_layers, save_dir):
    if ot is None: return
    configs = [
        {'metric_idx': 0, 'title': 'Bary. Core Distance (sq)', 'ylabel': 'Squared W-Dist (log)', 'log': True},
        {'metric_idx': 1, 'title': 'Bary. Spurious Distance (sq)', 'ylabel': 'Squared W-Dist (log)', 'log': True},
        {'metric_idx': 2, 'title': 'Spurious Ratio', 'ylabel': 'Ratio', 'ylim': (0, 1.05)},
        {'metric_idx': 3, 'title': 'Simple Ratio (Spurious/Core)', 'ylabel': 'Ratio (log)', 'log': True},
    ]
    _plot_evolution_layered(wd_history_train, wd_history_test, all_target_layers, configs,
                    'Evolution of Barycentric Wasserstein Distances',
                    'barycentric_wd_evolution.png', save_dir)

def plot_bregman_wd_evolution(wd_history_train, wd_history_test, all_target_layers, save_dir):
    if ot is None: return
    configs = [
        {'metric_idx': 0, 'title': 'Core Bregman Divergence', 'ylabel': 'Divergence (log)', 'log': True},
        {'metric_idx': 1, 'title': 'Spurious Bregman Divergence', 'ylabel': 'Divergence (log)', 'log': True},
    ]
    _plot_evolution_layered(wd_history_train, wd_history_test, all_target_layers, configs,
                    'Evolution of Wasserstein Bregman Divergence',
                    'bregman_divergence_evolution.png', save_dir)

def plot_vector_averaging_alignment_evolution(history_train, history_test, save_dir):
    configs = [
        {'metric_idx': 0, 'title': 'Core Alignment', 'ylabel': 'Cosine Similarity'},
        {'metric_idx': 1, 'title': 'Spurious Alignment', 'ylabel': 'Cosine Similarity'},
        {'metric_idx': 2, 'title': 'Spurious Ratio', 'ylabel': 'Ratio', 'ylim': (0, 1.05)},
        {'metric_idx': 3, 'title': 'Simple Ratio (Spurious/Core)', 'ylabel': 'Ratio (log)', 'log': True},
    ]
    _plot_evolution_single(history_train, history_test, configs, 
                           'Evolution of Vector-Averaging Classifier Alignment', 
                           'vec_avg_alignment_evolution.png', save_dir)

def plot_barycentric_alignment_evolution(history_train, history_test, save_dir):
    configs = [
        {'metric_idx': 0, 'title': 'Core Alignment', 'ylabel': 'Cosine Similarity'},
        {'metric_idx': 1, 'title': 'Spurious Alignment', 'ylabel': 'Cosine Similarity'},
        {'metric_idx': 2, 'title': 'Spurious Ratio', 'ylabel': 'Ratio', 'ylim': (0, 1.05)},
        {'metric_idx': 3, 'title': 'Simple Ratio (Spurious/Core)', 'ylabel': 'Ratio (log)', 'log': True},
    ]
    _plot_evolution_single(history_train, history_test, configs,
                           'Evolution of Barycentric Classifier Alignment',
                           'barycentric_alignment_evolution.png', save_dir)

def plot_transport_alignment_evolution(history_train, history_test, save_dir):
    configs = [
        {'metric_idx': 0, 'title': 'Transport Core Alignment', 'ylabel': 'Alignment Ratio'},
        {'metric_idx': 1, 'title': 'Transport Spurious Alignment', 'ylabel': 'Alignment Ratio'},
        {'metric_idx': 2, 'title': 'Spurious Ratio', 'ylabel': 'Ratio', 'ylim': (-0.05, 1.05)},
        {'metric_idx': 3, 'title': 'Simple Ratio (Spurious/Core)', 'ylabel': 'Ratio (log)', 'log': True},
    ]
    _plot_evolution_single(history_train, history_test, configs, 
                    'Evolution of Transport-Based Classifier Alignment', 
                    'transport_alignment_evolution.png', save_dir)
                    
def plot_intergroup_wd_evolution(wd_history_train, wd_history_test, all_target_layers, save_dir):
    if ot is None or (not wd_history_train and not wd_history_test): return
    epochs = sorted(wd_history_train.keys() if wd_history_train else wd_history_test.keys())
    layers = [l for l in all_target_layers if any(l in wd_history_train.get(e, {}) or l in wd_history_test.get(e, {}) for e in epochs)]
    if not layers: return
    
    pair_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations([(-1,-1), (-1,1), (1,-1), (1,1)], 2)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pair_keys)))
    
    fig, axes = plt.subplots(len(layers), 1, figsize=(15, 6 * len(layers)), squeeze=False)
    fig.suptitle('Evolution of Inter-Group Wasserstein Distances (Squared)', fontsize=16)

    for i, layer_name in enumerate(layers):
        ax = axes[i, 0]
        for j, p_key in enumerate(pair_keys):
            if wd_history_train:
                vals = [wd_history_train.get(e, {}).get(layer_name, {}).get(p_key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='o', markersize=4, linestyle='-', color=colors[j], label=f'{p_key} (Train)')
            if wd_history_test:
                vals = [wd_history_test.get(e, {}).get(layer_name, {}).get(p_key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='x', markersize=4, linestyle='--', color=colors[j])
        ax.set(title=f'Layer: {layer_name}', xlabel='Epoch', ylabel='Squared W-Dist (log scale)', yscale='log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, which="both", ls="--")

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    _save_and_close(fig, save_dir, 'intergroup_wd_evolution.png')

def plot_singular_value_evolution(sv_history, layers, title_prefix, save_dir, top_k=10):
    if not sv_history: return
    epochs = sorted(sv_history.keys())
    layers_with_hist = [l for l in layers if any(l in sv_history.get(e, {}) for e in epochs)]
    if not layers_with_hist: return
    
    fig, axes = plt.subplots(1, len(layers_with_hist), figsize=(7 * len(layers_with_hist), 5), squeeze=False)
    fig.suptitle(f'Evolution of Top {top_k} Singular Values of {title_prefix}', fontsize=16)

    for i, layer in enumerate(layers_with_hist):
        ax = axes.flatten()[i]
        sv_vals = np.array([sv_history[e].get(layer, [np.nan]*top_k) for e in epochs])
        for k in range(min(top_k, sv_vals.shape[1])):
            ax.plot(epochs, sv_vals[:, k], label=f'SV {k+1}')
        ax.set(title=f'Layer: {layer}', xlabel='Epoch', ylabel='Singular Value (log)', yscale='log')
        ax.legend()
        ax.grid(True, which="both", ls="--")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_dir, f'singular_values_{title_prefix.replace(" ", "_").lower()}.png')


# 勾配グラム行列のプロット関数
def plot_gradient_gram_evolution(history_train, history_test, save_dir):
    if not history_train and not history_test: return
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    group_keys = [(-1,-1), (-1,1), (1,-1), (1,1)]
    maj_maj_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations(group_keys, 2) if y1==a1 and y2==a2]
    min_min_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations(group_keys, 2) if y1!=a1 and y2!=a2]
    maj_min_keys = [f"G({y1},{a1})_vs_G({y2},{a2})" for (y1, a1), (y2, a2) in combinations(group_keys, 2) if (y1==a1 and y2!=a2) or (y1!=a1 and y2==a2)]
    
    plot_configs = [
        ('Majority-Majority', maj_maj_keys),
        ('Minority-Minority', min_min_keys),
        ('Majority-Minority', maj_min_keys)
    ]

    for title, keys in plot_configs:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Evolution of Gradient Gram Matrix ({title})', fontsize=16)
        colors = plt.cm.jet(np.linspace(0, 1, len(keys)))

        for i, key in enumerate(keys):
            if history_train:
                vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{key} (Train)')
            if history_test:
                vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
                ax.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i])

        ax.set(xlabel='Epoch', ylabel='Inner Product (log)', yscale='symlog')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, which="both", ls="--")
        fig.tight_layout(rect=[0, 0, 0.85, 0.95])
        _save_and_close(fig, save_dir, f'gradient_gram_{title.lower().replace("-", "_")}.png')

# ヤコビアンノルムのプロット関数
def plot_jacobian_norm_evolution(history_train, history_test, save_dir):
    if not history_train and not history_test: return
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    # Norms
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig1.suptitle('Evolution of Jacobian Norms', fontsize=16)
    norm_keys = [k for k in next(iter(history_train.values())).keys() if k.startswith('norm')]
    colors = plt.cm.jet(np.linspace(0, 1, len(norm_keys)))
    for i, key in enumerate(norm_keys):
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax1.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax1.plot(epochs, vals, marker='x', linestyle='--', color=colors[i])
    ax1.set(xlabel='Epoch', ylabel='Squared Norm (log)', yscale='log')
    ax1.legend(); ax1.grid(True, which="both", ls="--")
    _save_and_close(fig1, save_dir, 'jacobian_norms.png')

    # Inner Products
    dot_keys = [k for k in next(iter(history_train.values())).keys() if k.startswith('dot')]
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig2.suptitle('Evolution of Jacobian Inner Products', fontsize=16)
    colors = plt.cm.jet(np.linspace(0, 1, len(dot_keys)))
    for i, key in enumerate(dot_keys):
        if history_train:
            vals = [history_train.get(e, {}).get(key, np.nan) for e in epochs]
            ax2.plot(epochs, vals, marker='o', linestyle='-', color=colors[i], label=f'{key} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get(key, np.nan) for e in epochs]
            ax2.plot(epochs, vals, marker='x', linestyle='--', color=colors[i])
    ax2.set(xlabel='Epoch', ylabel='Inner Product (log)', yscale='symlog')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax2.grid(True, which="both", ls="--")
    fig2.tight_layout(rect=[0, 0, 0.8, 0.95])
    _save_and_close(fig2, save_dir, 'jacobian_inner_products.png')

# 汎化ギャップのプロット関数
def plot_generalization_gap_evolution(history_df, gen_gap_history, save_dir):
    """汎化ギャップの推定値と実際のギャップの変遷をプロット"""
    if not gen_gap_history:
        print("No generalization gap history to plot.")
        return

    epochs = sorted(gen_gap_history.keys())
    if not epochs: return
    
    # 実際の汎化ギャップを計算
    # history_dfのインデックスは0から始まるので，epochsリスト（1から始まる）に合わせて調整
    df_indices = [e - 1 for e in epochs if e - 1 < len(history_df)]
    actual_epochs = [e for e in epochs if e - 1 < len(history_df)]
    if not actual_epochs: return
    
    actual_gap_avg = (history_df.loc[df_indices, 'test_avg_loss'] - history_df.loc[df_indices, 'train_avg_loss']).values
    actual_gap_worst = (history_df.loc[df_indices, 'test_worst_loss'] - history_df.loc[df_indices, 'train_worst_loss']).values
    
    # グループごとの実際の汎化ギャップ
    test_group_losses = np.array(history_df.loc[df_indices, 'test_group_losses'].tolist())
    train_group_losses = np.array(history_df.loc[df_indices, 'train_group_losses'].tolist())
    actual_group_gaps = test_group_losses - train_group_losses
    
    group_keys = [f"G({y},{a})" for y in [-1, 1] for a in [-1, 1]]
    colors = plt.cm.jet(np.linspace(0, 1, len(group_keys)))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Evolution of Generalization Gap (Estimated vs. Actual)', fontsize=16)

    # 推定ギャップをプロット
    for i, key in enumerate(group_keys):
        estimated_vals = [gen_gap_history.get(e, {}).get(key, np.nan) for e in epochs]
        ax.plot(epochs, estimated_vals, marker='o', markersize=4, linestyle='--', color=colors[i], label=f'Estimated Gap ({key})')

    # 実際のギャップをプロット
    ax.plot(actual_epochs, actual_gap_avg, marker='', linestyle='-', color='black', linewidth=2, label='Actual Gap (Average)')
    # グループごとの実際のギャップもプロット
    for i, key in enumerate(group_keys):
         ax.plot(actual_epochs, actual_group_gaps[:, i], marker='', linestyle='-', color=colors[i], linewidth=1.5, label=f'Actual Gap ({key})')


    ax.set(xlabel='Epoch', ylabel='Generalization Gap (Loss)', yscale='log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which="both", ls="--")
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    _save_and_close(fig, save_dir, 'generalization_gap_evolution.png')

# 勾配グラム行列のスペクトルプロット関数
def plot_gradient_gram_spectrum_evolution(history_train, history_test, save_dir):
    """勾配グラム行列の固有値と主固有ベクトルの成分の変遷をプロット"""
    if not history_train and not history_test:
        print("No gradient gram spectrum history to plot.")
        return
    
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)) # グラフエリアを3つに変更
    fig.suptitle('Evolution of Gradient Gram Matrix Spectrum', fontsize=16)

    # --- 1. 固有値のプロット ---
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for i in range(4):
        if history_train:
            vals = [history_train.get(e, {}).get('eigenvalues', [np.nan]*4)[i] for e in epochs]
            ax1.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'λ_{i+1} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get('eigenvalues', [np.nan]*4)[i] for e in epochs]
            ax1.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'λ_{i+1} (Test)')
    ax1.set(xlabel='Epoch', ylabel='Eigenvalue', title='Eigenvalues (λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄)', yscale='symlog', ylim_bottom=0)
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    # --- 2. 主固有ベクトルの成分のプロット ---
    ax2 = axes[1]
    group_labels = ['$G_{(-1,-1)}$', '$G_{(-1,1)}$', '$G_{(1,-1)}$', '$G_{(1,1)}$']
    for i in range(4):
        if history_train:
            vals = [history_train.get(e, {}).get('eigenvector1', [np.nan]*4)[i] for e in epochs]
            ax2.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{group_labels[i]} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get('eigenvector1', [np.nan]*4)[i] for e in epochs]
            ax2.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'{group_labels[i]} (Test)')
    ax2.set(xlabel='Epoch', ylabel='Component Value', title='Components of 1st Eigenvector (u₁)', ylim=(-1.05, 1.05))
    ax2.legend()
    ax2.grid(True, which="both", ls="--")

    # --- 3. 2番目の固有ベクトルの成分のプロット ---
    ax3 = axes[2]
    for i in range(4):
        if history_train:
            vals = [history_train.get(e, {}).get('eigenvector2', [np.nan]*4)[i] for e in epochs]
            ax3.plot(epochs, vals, marker='o', markersize=3, linestyle='-', color=colors[i], label=f'{group_labels[i]} (Train)')
        if history_test:
            vals = [history_test.get(e, {}).get('eigenvector2', [np.nan]*4)[i] for e in epochs]
            ax3.plot(epochs, vals, marker='x', markersize=3, linestyle='--', color=colors[i], label=f'{group_labels[i]} (Test)')
    ax3.set(xlabel='Epoch', ylabel='Component Value', title='Components of 2nd Eigenvector (u₂)', ylim=(-1.05, 1.05))
    ax3.legend()
    ax3.grid(True, which="both", ls="--")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_dir, 'gradient_gram_spectrum_evolution.png')

# 勾配ノルム比のプロット関数
def plot_gradient_norm_ratio_evolution(history_train, history_test, save_dir):
    """多数派/少数派の勾配ノルム比の変遷をプロット"""
    if not history_train and not history_test:
        print("No gradient norm ratio history to plot.")
        return
        
    epochs = sorted(history_train.keys() if history_train else history_test.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Evolution of Gradient Norm Ratio (Majority / Minority)', fontsize=16)
    
    if history_train:
        vals = [history_train.get(e, {}).get('ratio', np.nan) for e in epochs]
        ax.plot(epochs, vals, marker='o', markersize=4, linestyle='-', color='blue', label='Train')
    if history_test:
        vals = [history_test.get(e, {}).get('ratio', np.nan) for e in epochs]
        ax.plot(epochs, vals, marker='x', markersize=4, linestyle='--', color='cyan', label='Test')

    ax.set(xlabel='Epoch', ylabel='Norm Ratio', yscale='log')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_dir, 'gradient_norm_ratio_evolution.png')


# ==============================================================================
# 全てのプロットを統括するラッパー関数
# ==============================================================================

def plot_all_results(history_df, analysis_histories, layers, save_dir, config):
    """全てのプロット関数を呼び出し，結果を保存"""
    print("\n--- Generating and saving all plots ---")
    
    plot_training_history(history_df, save_dir)
    plot_misclassification_rates(history_df.iloc[-1], config['dataset_name'], save_dir)

    if config['analyze_mutual_information']:
        plot_mi_evolution(analysis_histories['mi_train'], analysis_histories['mi_test'], layers, save_dir)
    if config['analyze_intergroup_distance']:
        plot_intergroup_wd_evolution(analysis_histories['intergroup_wd_train'], analysis_histories['intergroup_wd_test'], layers, save_dir)
    if config['analyze_conditional_distance']:
        plot_conditional_wd_evolution(analysis_histories['cond_wd_train'], analysis_histories['cond_wd_test'], layers, save_dir)
    if config['analyze_barycentric_distance']:
        plot_barycentric_wd_evolution(analysis_histories['bary_wd_train'], analysis_histories['bary_wd_test'], layers, save_dir)
    if config['analyze_bregman_divergence']:
        plot_bregman_wd_evolution(analysis_histories['bregman_wd_train'], analysis_histories['bregman_wd_test'], layers, save_dir)
    
    if config['analyze_vector_averaging_alignment']:
        plot_vector_averaging_alignment_evolution(analysis_histories['vec_avg_align_train'], analysis_histories['vec_avg_align_test'], save_dir)
    if config['analyze_barycentric_alignment']:
        plot_barycentric_alignment_evolution(analysis_histories['bary_align_train'], analysis_histories['bary_align_test'], save_dir)
    if config['analyze_transport_alignment']:
        plot_transport_alignment_evolution(analysis_histories['transport_align_train'], analysis_histories['transport_align_test'], save_dir)

    if config['analyze_weight_singular_values']:
        plot_singular_value_evolution(analysis_histories['weight_sv'], layers, "Weight_Matrices", save_dir)
    if config['analyze_activation_singular_values']:
        plot_singular_value_evolution(analysis_histories['activation_sv_train'], layers, "Train_Activations", save_dir)
        plot_singular_value_evolution(analysis_histories['activation_sv_test'], layers, "Test_Activations", save_dir)

    if config.get('analyze_gradient_gram', False):
        plot_gradient_gram_evolution(analysis_histories['grad_gram_train'], analysis_histories['grad_gram_test'], save_dir)
    if config.get('analyze_jacobian_norm', False):
        plot_jacobian_norm_evolution(analysis_histories['jacobian_norm_train'], analysis_histories['jacobian_norm_test'], save_dir)
    if config.get('analyze_generalization_gap', False):
        plot_generalization_gap_evolution(history_df, analysis_histories['gen_gap_train'], save_dir)
        
    if config.get('analyze_gradient_gram_spectrum', False):
        plot_gradient_gram_spectrum_evolution(
            analysis_histories['grad_gram_spectrum_train'],
            analysis_histories['grad_gram_spectrum_test'],
            save_dir
        )
    if config.get('analyze_gradient_norm_ratio', False):
        plot_gradient_norm_ratio_evolution(
            analysis_histories['grad_norm_ratio_train'],
            analysis_histories['grad_norm_ratio_test'],
            save_dir
        )
