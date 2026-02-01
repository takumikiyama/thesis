# 修論分析コード

慶應義塾大学大学院メディアデザイン研究科 修士論文  
「Happy Ice Cream: 大学生のオンライン会話開始における心理的障壁を軽減するメッセージマッチングアプリケーション」  
の統計分析コード

## ファイル一覧

- `stai_analysis.py` - STAI-S（状態不安尺度）の分析スクリプト
- `composite_analysis.py` - 質的判定と量的データの複合分析スクリプト
- `Analyze_elements_groups.py` - 3要素の群別比較分析スクリプト
- `element_judgment_29participants_complete.csv` - 参加者データ（29名）

## 実行方法
```bash
# 必要なライブラリのインストール
pip install pandas numpy scipy matplotlib

# スクリプトの実行
python stai_analysis.py
python composite_analysis.py
python Analyze_elements_groups.py
```

## データ形式

CSVファイルには以下のカラムが含まれています：
- `Participant_ID`: 参加者ID
- `A_Score`: 仕様A（メッセージマッチング有）のSTAI-Sスコア
- `B_Score`: 仕様B（メッセージマッチング無）のSTAI-Sスコア
- `Element1_Obligation`: 要素1（コミュニケーション続行義務）の判定
- `Element2_Burden`: 要素2（対人配慮負担）の判定
- `Element3_Rejection`: 要素3（拒絶・評価懸念）の判定

## 著者

城山 拓海（Takumi Kiyama）  
慶應義塾大学大学院 メディアデザイン研究科  
2025年度修了予定
```

---

## **論文に記載するURL**

あなたのリポジトリのURLはこちらです：
```
https://github.com/takumikiyama/thesis
