import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

# フォント設定
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("3要素の群別分析")
print("=" * 80)

# データ読み込み
df = pd.read_csv('element_judgment_29participants_complete.csv')

# 要素リスト
elements = [
    ('Element1_Obligation', '要素1: コミュニケーション続行義務意識'),
    ('Element2_Burden', '要素2: 対人配慮負担'),
    ('Element3_Rejection', '要素3: 拒絶・評価懸念')
]

# 出力フォルダ作成
import os
os.makedirs('element_analysis', exist_ok=True)

results_summary = []

for col, name in elements:
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    # データ不足を除外
    df_analysis = df[df[col] != 'データ不足'].copy()
    
    # 群別の統計
    print("\n【記述統計】")
    for group in ['有効', '不変', '逆効果']:
        subset = df_analysis[df_analysis[col] == group]
        if len(subset) > 0:
            mean_delta = subset['Delta_STAI'].mean()
            sd_delta = subset['Delta_STAI'].std()
            median_delta = subset['Delta_STAI'].median()
            min_delta = subset['Delta_STAI'].min()
            max_delta = subset['Delta_STAI'].max()
            
            print(f"\n{group}群 (n={len(subset)}):")
            print(f"  平均ΔSTAI-S: {mean_delta:+.2f}")
            print(f"  SD: {sd_delta:.2f}")
            print(f"  中央値: {median_delta:+.2f}")
            print(f"  範囲: {min_delta:+.0f} ~ {max_delta:+.0f}")
    
    # 群間比較(データ不足を除外)
    groups_data = []
    group_names = []
    
    for group in ['有効', '不変', '逆効果']:
        subset = df_analysis[df_analysis[col] == group]
        if len(subset) > 0:
            groups_data.append(subset['Delta_STAI'].values)
            group_names.append(group)
    
    if len(groups_data) >= 2:
        print("\n【群間比較】")
        
        # 正規性検定
        print("\n正規性検定 (Shapiro-Wilk):")
        all_normal = True
        for i, (data, gname) in enumerate(zip(groups_data, group_names)):
            if len(data) >= 3:
                w, p = stats.shapiro(data)
                print(f"  {gname}群: W={w:.4f}, p={p:.4f}", end="")
                if p < 0.05:
                    print(" (非正規)")
                    all_normal = False
                else:
                    print(" (正規)")
            else:
                print(f"  {gname}群: n<3のためスキップ")
                all_normal = False
        
        # 等分散性検定
        if len(groups_data) >= 2 and all(len(g) >= 2 for g in groups_data):
            stat, p_levene = stats.levene(*groups_data)
            print(f"\n等分散性検定 (Levene): F={stat:.4f}, p={p_levene:.4f}", end="")
            if p_levene < 0.05:
                print(" (等分散性なし)")
            else:
                print(" (等分散性あり)")
        
        # 群間検定
        if len(groups_data) == 2:
            # 2群の場合: t検定 or Mann-Whitney U検定
            if all_normal:
                t_stat, p_value = stats.ttest_ind(groups_data[0], groups_data[1])
                print(f"\n対応なしt検定: t={t_stat:.3f}, p={p_value:.4f}", end="")
            else:
                u_stat, p_value = stats.mannwhitneyu(groups_data[0], groups_data[1])
                print(f"\nMann-Whitney U検定: U={u_stat:.3f}, p={p_value:.4f}", end="")
        
        elif len(groups_data) >= 3:
            # 3群の場合: ANOVA or Kruskal-Wallis検定
            if all_normal:
                f_stat, p_value = stats.f_oneway(*groups_data)
                print(f"\n一元配置分散分析 (ANOVA): F={f_stat:.3f}, p={p_value:.4f}", end="")
            else:
                h_stat, p_value = stats.kruskal(*groups_data)
                print(f"\nKruskal-Wallis検定: H={h_stat:.3f}, p={p_value:.4f}", end="")
        
        # 有意性判定
        if p_value < 0.001:
            print(" ***")
            sig = "***"
        elif p_value < 0.01:
            print(" **")
            sig = "**"
        elif p_value < 0.05:
            print(" *")
            sig = "*"
        else:
            print(" (n.s.)")
            sig = "n.s."
        
        # 効果量 (η² or ε²)
        if len(groups_data) >= 2:
            # 全データを1つの配列に
            all_data = np.concatenate(groups_data)
            # 群ラベルを作成
            group_labels = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups_data)])
            
            # η² (eta squared) を計算
            ss_between = sum([len(g) * (np.mean(g) - np.mean(all_data))**2 for g in groups_data])
            ss_total = np.sum((all_data - np.mean(all_data))**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            print(f"効果量 (η²): {eta_squared:.3f}", end="")
            if eta_squared >= 0.14:
                print(" (大)")
            elif eta_squared >= 0.06:
                print(" (中)")
            elif eta_squared >= 0.01:
                print(" (小)")
            else:
                print(" (極小)")
        
        # 結果を保存
        results_summary.append({
            'Element': name,
            'n_groups': len(groups_data),
            'group_names': ', '.join(group_names),
            'p_value': p_value,
            'significance': sig,
            'effect_size': eta_squared if len(groups_data) >= 2 else np.nan
        })
    
    # ===================================
    # 可視化: 箱ひげ図
    # ===================================
    if len(groups_data) >= 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        positions = list(range(1, len(groups_data) + 1))
        
        bp = ax.boxplot(groups_data, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                      markeredgecolor='red', markersize=10),
                        medianprops=dict(color='black', linewidth=2.5),
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2))
        
        # 色設定
        colors = {'有効': '#2E86AB', '不変': '#A8DADC', '逆効果': '#F18F01'}
        for patch, gname in zip(bp['boxes'], group_names):
            patch.set_facecolor(colors.get(gname, 'gray'))
            patch.set_alpha(0.7)
        
        # 個別データポイント
        for i, (data, gname) in enumerate(zip(groups_data, group_names)):
            y = data
            x = np.random.normal(positions[i], 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.5, s=80, c=colors.get(gname, 'gray'), 
                      edgecolors='black', linewidth=1)
        
        # 0のライン
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                  label='変化なし (ΔSTAI-S=0)')
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{g}\n(n={len(d)})' for g, d in zip(group_names, groups_data)], 
                          fontsize=12, fontweight='bold')
        ax.set_ylabel('ΔSTAI-S (B条件 - A条件)', fontsize=14, fontweight='bold')
        ax.set_title(f'{name}\n群別ΔSTAI-S比較', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # 統計情報を追加
        if len(groups_data) == 2:
            test_name = "t検定" if all_normal else "Mann-Whitney"
        else:
            test_name = "ANOVA" if all_normal else "Kruskal-Wallis"
        
        ax.text(0.02, 0.98, 
               f'{test_name}: p={p_value:.4f} {sig}\nη²={eta_squared:.3f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # ファイル名を作成
        element_num = col.replace('Element', '').replace('_Obligation', '').replace('_Burden', '').replace('_Rejection', '')
        filename = f'element_analysis/element{element_num}_group_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ 図を保存: {filename}")

# ===================================
# サマリーテーブル作成
# ===================================
print("\n" + "=" * 80)
print("分析結果サマリー")
print("=" * 80)

summary_df = pd.DataFrame(results_summary)
print("\n", summary_df.to_string(index=False))

# CSV保存
summary_df.to_csv('element_analysis/analysis_summary.csv', index=False, encoding='utf-8-sig')
print("\n✓ サマリーをCSVで保存: element_analysis/analysis_summary.csv")

print("\n" + "=" * 80)
print("✅ 分析完了!")
print("=" * 80)
print("\n保存先: element_analysis/ フォルダ")
print("  - element1_group_comparison.png")
print("  - element2_group_comparison.png")
print("  - element3_group_comparison.png")
print("  - analysis_summary.csv")
print("=" * 80)