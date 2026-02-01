import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

# フォント設定(日本語の警告を抑制)
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("STAI-S 統計分析レポート")
print("=" * 80)

# データ読み込み
df = pd.read_csv('element_judgment_29participants_complete.csv')

# 欠損値を除外
df_clean = df[df['A_Score'].notna() & df['B_Score'].notna()].copy()

print(f"\n【データ概要】")
print(f"総参加者数: {len(df_clean)}人")
print(f"A条件平均: {df_clean['A_Score'].mean():.2f} (SD={df_clean['A_Score'].std():.2f})")
print(f"B条件平均: {df_clean['B_Score'].mean():.2f} (SD={df_clean['B_Score'].std():.2f})")

# ΔSTAI-S計算
df_clean['Delta_STAI'] = df_clean['B_Score'] - df_clean['A_Score']
print(f"平均ΔSTAI-S: {df_clean['Delta_STAI'].mean():.2f} (SD={df_clean['Delta_STAI'].std():.2f})")

print("\n" + "=" * 80)
print("1. 対応ありt検定 (A条件 vs B条件)")
print("=" * 80)

# 正規性の検定
shapiro_a = stats.shapiro(df_clean['A_Score'])
shapiro_b = stats.shapiro(df_clean['B_Score'])

print(f"\n【正規性の検定 (Shapiro-Wilk test)】")
print(f"A条件: W = {shapiro_a.statistic:.4f}, p = {shapiro_a.pvalue:.4f}")
print(f"B条件: W = {shapiro_b.statistic:.4f}, p = {shapiro_b.pvalue:.4f}")

if shapiro_a.pvalue > 0.05 and shapiro_b.pvalue > 0.05:
    print("→ 正規性が確認されたため、対応ありt検定を使用")
    test_name = "Paired t-test"
else:
    print("→ 正規性が確認されないため、Wilcoxon検定を推奨")
    test_name = "Wilcoxon signed-rank test (recommended)"

# 対応ありt検定
t_stat, t_pvalue = stats.ttest_rel(df_clean['A_Score'], df_clean['B_Score'])

print(f"\n【対応ありt検定の結果】")
print(f"t({len(df_clean)-1}) = {t_stat:.3f}")
print(f"p = {t_pvalue:.4f}")

if t_pvalue < 0.001:
    sig = "***"
elif t_pvalue < 0.01:
    sig = "**"
elif t_pvalue < 0.05:
    sig = "*"
else:
    sig = "n.s."

print(f"有意性: {sig}")

# 効果量(Cohen's d for paired design)
# 対応あり設計では、差分スコアのSDを使用する（pooled SDは独立2群用）
diff_scores = df_clean['A_Score'] - df_clean['B_Score']
cohens_d = diff_scores.mean() / diff_scores.std()

print(f"Cohen's d = {cohens_d:.3f}")
print(f"  (差分スコア法: mean_diff = {diff_scores.mean():.3f}, SD_diff = {diff_scores.std():.3f})")

print("\n" + "=" * 80)
print("2. Wilcoxon符号順位検定 (ノンパラメトリック)")
print("=" * 80)

# Wilcoxon検定
w_stat, w_pvalue = stats.wilcoxon(df_clean['A_Score'], df_clean['B_Score'])

print(f"\n【Wilcoxon検定の結果】")
print(f"W = {w_stat:.1f}")
print(f"p = {w_pvalue:.4f}")

if w_pvalue < 0.001:
    w_sig = "***"
elif w_pvalue < 0.01:
    w_sig = "**"
elif w_pvalue < 0.05:
    w_sig = "*"
else:
    w_sig = "n.s."

print(f"有意性: {w_sig}")

print("\n" + "=" * 80)
print("3. ΔSTAI-S と 3要素判定の相関分析")
print("=" * 80)

# 要素の数値化
element_mapping = {'有効': 1, '不変': 0, '逆効果': -1, 'データ不足': np.nan}

for elem_col in ['Element1_Obligation', 'Element2_Burden', 'Element3_Rejection']:
    df_clean[f'{elem_col}_Score'] = df_clean[elem_col].map(element_mapping)

# 各要素との相関
elements = [
    ('Element1_Obligation_Score', '要素1: コミュニケーション続行義務意識'),
    ('Element2_Burden_Score', '要素2: 対人配慮負担'),
    ('Element3_Rejection_Score', '要素3: 拒絶・評価懸念')
]

print("\n【Spearman順位相関係数】")
print("(有効=1, 不変=0, 逆効果=-1 として数値化)")

for elem_score, elem_name in elements:
    # 欠損値を除外
    valid_data = df_clean[[elem_score, 'Delta_STAI']].dropna()
    
    if len(valid_data) > 2:
        rho, p_val = stats.spearmanr(valid_data[elem_score], valid_data['Delta_STAI'])
        
        if p_val < 0.001:
            sig_mark = "***"
        elif p_val < 0.01:
            sig_mark = "**"
        elif p_val < 0.05:
            sig_mark = "*"
        else:
            sig_mark = "n.s."
        
        print(f"\n{elem_name}")
        print(f"  n = {len(valid_data)}")
        print(f"  ρ = {rho:.3f}, p = {p_val:.4f} ({sig_mark})")
    else:
        print(f"\n{elem_name}")
        print(f"  データ不足のため計算不可")

print("\n" + "=" * 80)
print("4. 要素別の記述統計")
print("=" * 80)

for elem_col, elem_name in [
    ('Element1_Obligation', '要素1'),
    ('Element2_Burden', '要素2'),
    ('Element3_Rejection', '要素3')
]:
    print(f"\n【{elem_name}】")
    value_counts = df_clean[elem_col].value_counts()
    for judgment in ['有効', '不変', '逆効果', 'データ不足']:
        count = value_counts.get(judgment, 0)
        percentage = (count / len(df_clean)) * 100 if len(df_clean) > 0 else 0
        print(f"  {judgment}: {count}人 ({percentage:.1f}%)")
        
        # 各判定群のΔSTAI-S平均
        if count > 0:
            subset = df_clean[df_clean[elem_col] == judgment]
            if len(subset) > 0:
                mean_delta = subset['Delta_STAI'].mean()
                print(f"    → 平均ΔSTAI-S: {mean_delta:+.2f}")

print("\n" + "=" * 80)
print("分析完了!")
print("=" * 80)
print("\n【論文での記載例】")
print("""
Methods セクション:
  統計解析にはPython 3.9 (SciPy 1.13, pandas 2.3, Matplotlib 3.9)を使用した。
  A条件とB条件のSTAI-Sスコアの比較には、正規性検定(Shapiro-Wilk test)の結果に基づき、
  対応ありt検定またはWilcoxon符号順位検定を用いた。
  ΔSTAI-Sと各要素の判定結果の相関分析には、Spearmanの順位相関係数を用いた。
  有意水準は5%とした。

Results セクション:
  A条件(M=XX.XX, SD=XX.XX)とB条件(M=XX.XX, SD=XX.XX)の間に
  統計的に有意な差は認められなかった(t(XX)=XX.XX, p=X.XXX, n.s.)。
""")

print("\n統計結果をコピーして論文に使用してください。")