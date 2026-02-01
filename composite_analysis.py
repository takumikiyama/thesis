"""
composite_analysis.py
複合分析: 質的判定（3要素）とΔSTAI-Sの関連分析

実行方法:
    python composite_analysis.py

必要ファイル:
    element_judgment_29participants_complete.csv (同一ディレクトリ)

出力:
    各要素について以下を算出・表示:
    - 群別の記述統計 (n, 平均, SD, 中央値, 範囲)
    - 正規性検定の結果と検定手法の選択理由
    - 検定結果と有意性
    - 効果量 (η²)
    - LaTeX記載用のまとめ行
"""

import pandas as pd
import numpy as np
from scipy import stats


# ============================================================
# 設定
# ============================================================
CSV_FILE = 'element_judgment_29participants_complete.csv'

ELEMENTS = [
    ('Element1_Obligation', '要素1: コミュニケーション続行義務意識'),
    ('Element2_Burden',     '要素2: 対人配慮負担'),
    ('Element3_Rejection',  '要素3: 拒絶・評価懸念'),
]

JUDGMENT_LABELS = ['有効', '不変', '逆効果']  # データ不足は除外対象


# ============================================================
# 補助関数
# ============================================================
def classify_eta_sq(eta_sq):
    """η²の値をCohen (1988) の基準で分類する
       小: η² ≥ .01 / 中: η² ≥ .06 / 大: η² ≥ .14
    """
    if   eta_sq >= 0.14: return '大'
    elif eta_sq >= 0.06: return '中'
    elif eta_sq >= 0.01: return '小'
    else:                return '極小'


def format_p_latex(p_value):
    """p値をLaTeX記載用にフォーマットする (小数第3位)"""
    if p_value < 0.001:
        return 'p < .001'
    else:
        rounded = round(p_value, 3)
        return f'p = .{int(rounded * 1000):03d}'


def calc_eta_squared(group_arrays):
    """η²（イータ二乗）を計算する

    η² = SS_between / SS_total
      SS_between: 群間平方和 = Σ n_i × (群平均_i − 全体平均)²
      SS_total:   全体平方和 = Σ (個別値 − 全体平均)²

    「全体の変動のうち、群の違いによって説明できる割合」を表す。
    """
    all_data   = np.concatenate(group_arrays)
    grand_mean = np.mean(all_data)

    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in group_arrays
    )
    ss_total = np.sum((all_data - grand_mean) ** 2)

    return ss_between / ss_total if ss_total > 0 else 0.0


# ============================================================
# データ読み込み
# ============================================================
df = pd.read_csv(CSV_FILE)
print(f"読み込みデータ: {len(df)} 人")


# ============================================================
# ステップ1: ΔSTAI-Sの算出
#
# 定義: ΔSTAI-S = B条件スコア − A条件スコア
#   正の値 → A条件（メッセージマッチング有）で不安が低かった
#   負の値 → B条件（メッセージマッチング無）で不安が低かった
# ============================================================
df['Delta_STAI'] = df['B_Score'] - df['A_Score']

print("\n" + "=" * 70)
print(" ステップ1: ΔSTAI-S の算出 (B条件 − A条件)")
print("=" * 70)
print(f"  全体平均: {df['Delta_STAI'].mean():+.2f} (SD = {df['Delta_STAI'].std():.2f})\n")


# ============================================================
# ステップ2〜4: 各要素について群間比較を実行
# ============================================================
for col, name in ELEMENTS:
    print("=" * 70)
    print(f" {name}")
    print("=" * 70)

    # ---------------------------------------------------------
    # ステップ2: データ不足を除外し、判定群に分類
    # ---------------------------------------------------------
    df_elem   = df[df[col] != 'データ不足'].copy()
    n_excluded = len(df) - len(df_elem)

    print(f"\n【ステップ2: 群分類】")
    print(f"  データ不足で除外: {n_excluded} 人  →  分析対象: {len(df_elem)} 人\n")

    groups = {}  # {判定ラベル: ΔSTAI-Sの配列}
    for label in JUDGMENT_LABELS:
        subset = df_elem[df_elem[col] == label]
        if len(subset) > 0:
            groups[label] = subset['Delta_STAI'].values
            print(f"  {label}群: n={len(subset):2d} | "
                  f"平均={subset['Delta_STAI'].mean():+6.2f} | "
                  f"SD={subset['Delta_STAI'].std():5.2f} | "
                  f"中央値={subset['Delta_STAI'].median():+5.1f} | "
                  f"範囲=[{subset['Delta_STAI'].min():+3.0f}, "
                  f"{subset['Delta_STAI'].max():+3.0f}]")

    group_names  = list(groups.keys())
    group_arrays = list(groups.values())
    n_groups     = len(groups)

    # ---------------------------------------------------------
    # ステップ3: 正規性検定 (Shapiro-Wilk)
    #
    # 各群のΔSTAI-Sが正規分布に従うか確認する。
    # n < 3 の群は検定実行不可なため、非正規として扱う。
    # この結果がステップ4の検定手法選択に使われる。
    # ---------------------------------------------------------
    print(f"\n【ステップ3: 正規性検定 (Shapiro-Wilk)】")

    all_normal = True
    for gname, gdata in groups.items():
        if len(gdata) >= 3:
            w, p   = stats.shapiro(gdata)
            is_normal = (p >= 0.05)
            if not is_normal:
                all_normal = False
            mark = '正規  ✓' if is_normal else '非正規 ✗'
            print(f"  {gname}群 (n={len(gdata):2d}): W={w:.4f}, p={p:.4f}  →  {mark}")
        else:
            all_normal = False
            print(f"  {gname}群 (n={len(gdata):2d}): n < 3 のため検定スキップ  →  非正規として扱う ✗")

    # ---------------------------------------------------------
    # ステップ4: 検定手法の選択と実行
    #
    # 選択基準:
    #   3群の場合:
    #     全群が正規  → ANOVA（一元配置分散分析）
    #     一部が非正規 → Kruskal-Wallis検定（順位に基づくノンパラメトリック）
    #   2群の場合:
    #     全群が正規  → 対応なしt検定
    #     一部が非正規 → Mann-Whitney U検定
    # ---------------------------------------------------------
    print(f"\n【ステップ4: 群間比較】")
    print(f"  正規性の判定: {'全群が正規' if all_normal else '一部の群が非正規'}")

    if n_groups == 3:
        if all_normal:
            f_stat, p_value = stats.f_oneway(*group_arrays)
            df_between = n_groups - 1
            df_within  = sum(len(g) for g in group_arrays) - n_groups
            test_name  = 'ANOVA'
            stat_str   = f'F({df_between}, {df_within}) = {f_stat:.3f}'
        else:
            h_stat, p_value = stats.kruskal(*group_arrays)
            test_name = 'Kruskal-Wallis'
            stat_str  = f'H = {h_stat:.3f}'

    elif n_groups == 2:
        if all_normal:
            t_stat, p_value = stats.ttest_ind(*group_arrays)
            df_t       = sum(len(g) for g in group_arrays) - 2
            test_name  = 't検定'
            stat_str   = f't({df_t}) = {t_stat:.3f}'
        else:
            u_stat, p_value = stats.mannwhitneyu(
                *group_arrays, alternative='two-sided'
            )
            test_name = 'Mann-Whitney U'
            stat_str  = f'U = {u_stat:.3f}'

    # 有意性判定
    if   p_value < 0.001: sig = '***'
    elif p_value < 0.01:  sig = '**'
    elif p_value < 0.05:  sig = '*'
    else:                 sig = 'n.s.'

    print(f"  選択手法: {test_name} "
          f"({'パラメトリック' if all_normal else 'ノンパラメトリック'})")
    print(f"  検定結果: {stat_str}, {format_p_latex(p_value)}  {sig}")

    # --- 効果量 η² ---
    eta_sq    = calc_eta_squared(group_arrays)
    eta_label = classify_eta_sq(eta_sq)
    print(f"  効果量:   η² = {eta_sq:.3f} ({eta_label})")

    # --- LaTeX記載用まとめ ---
    print(f"\n  ┌─── LaTeX記載用 ──────────────────────────────────────┐")
    print(f"  │  {test_name}: {stat_str}, {format_p_latex(p_value)}, η² = {eta_sq:.2f}  {sig}")
    print(f"  └──────────────────────────────────────────────────────┘\n")


print("=" * 70)
print(" 分析完了")
print("=" * 70)