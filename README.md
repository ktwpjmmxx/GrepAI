# Generative AI Handbook

生成AI・LLMに関する学習内容を体系的に整理したナレッジベースです。
基礎知識から実践的なライブラリ活用、設計パターン、リスク管理まで、カテゴリー別にドキュメントを蓄積していきます。

本リポジトリは学習の進捗に合わせて随時更新されます。各チャプターのステータスは下記の目次を参照してください。

---

## 目次（Table of Contents）

| No. | チャプター | 概要 | ステータス |
|---|---|---|---|
| 01 | [AI・LLMの基礎知識](./docs/01_fundamentals/README.md) | Transformer、トークン、コンテキストウィンドウ、主要モデル比較 | 完了 |
| 02 | [アプリケーション導入時の実践ライブラリ](./docs/02_libraries_and_tools/README.md) | OpenAI API、LangChain、LlamaIndexの比較と実装 | 完了 |
| 03 | [設計思想・先進的なアーキテクチャ](./docs/03_design_patterns/README.md) | プロンプトエンジニアリング、RAG、エージェント型AI | 完了 |
| 04 | [AI利用時の注意点とリスク管理](./docs/04_precautions_and_security/README.md) | ハルシネーション対策、プロンプトインジェクション、プライバシー | 完了 |
| 05 | [評価・モニタリング](./docs/05_evaluation_and_monitoring/README.md) | RAG評価指標（Precision@k/Recall@k/RAGAS）とAI-Recipe-Bookでの実践例 | 完了 |
| 06 | ファインチューニング実践 | LoRA/PEFTを用いたモデルのカスタマイズ | 予定 |
| 07 | [MLOps / LLMOps](./docs/07_mlops_llmops/README.md) | 実験管理（MLflow）、デプロイ・運用・継続的改善 | 作成中（実験管理パートのみ執筆済み） |
| 08 | [LLM-as-a-judgeによる出力品質評価](./docs/08_llm_as_judge_evaluation/README.md) | プロンプト設計の定量比較、評価軸・バイアス排除の設計、AI-Recipe-Bookでの実践例 | 完了 |

新しいチャプターを追加する際は、この表に行を追加し、`docs/` 配下に同名フォルダを作成してください（詳細は下記「リポジトリ構成」を参照）。

---

## リポジトリ構成

```
generative-ai-handbook/
├── README.md                          # 本ファイル（全体の目次）
├── docs/
│   ├── 01_fundamentals/
│   │   ├── README.md                  # チャプター本体
│   │   └── assets/                    # このチャプター専用の図・画像
│   ├── 02_libraries_and_tools/
│   │   ├── README.md
│   │   └── assets/
│   ├── 03_design_patterns/
│   │   ├── README.md
│   │   └── assets/
│   ├── 04_precautions_and_security/
│   │   ├── README.md
│   │   └── assets/
│   ├── 05_evaluation_and_monitoring/
│   │   ├── README.md
│   │   └── assets/
│   ├── 07_mlops_llmops/
│   │   ├── README.md
│   │   └── assets/
│   ├── 08_llm_as_judge_evaluation/
│   │   ├── README.md
│   │   └── assets/
│   └── 0X_.../                        # 新規チャプターはこの命名規則で追加
└── assets/                            # リポジトリ全体で共有する画像・バナー
```

### 新しいチャプターを追加する手順

1. `docs/` 配下に `0X_チャプター名/` フォルダを作成する
2. フォルダ内に `README.md` を作成し、本文を記述する（図が必要な場合は `assets/` に格納するか、Mermaid記法を使用する）
3. 本ファイル冒頭の目次テーブルに行を追加する
4. 内容が他チャプターと関連する場合は、該当箇所に相互リンクを貼る（例：「RAGの詳細は [03_design_patterns](./docs/03_design_patterns/README.md) を参照」）

---

## 執筆方針

- **1チャプター1テーマ**を原則とし、肥大化した場合はサブファイル（`01a_xxx.md`など）に分割する
- 図解が有効な箇所は、可能な限り**Mermaid記法**を使用し、GitHub上でそのままレンダリングできるようにする
- コードスニペットは**Python**を基本言語とし、実行可能な最小構成で記載する
- 専門用語は初出時に**太字**で強調し、簡潔な定義を添える

---

## 関連リポジトリ

- 実装ポートフォリオ：[AI-Recipe-Book](https://github.com/ktwpjmmxx/AI-Recipe-Book) — RAGを用いたレシピ管理アプリ。05章で扱うRAG評価指標、08章で扱うLLM-as-a-judgeによるプロンプト比較実験は、いずれも本アプリの`search-assist`機能を対象に実測した内容を実例として掲載
- 実装ポートフォリオ：[mlflow-experiment-tracking](https://github.com/ktwpjmmxx/mlflow-experiment-tracking) — Fashion-MNIST分類を題材にした実験管理の実践プロジェクト。07章の内容と対応

---

## ライセンス

個人の学習記録として公開しています。内容の正確性については継続的に見直しを行っていますが、
利用は自己責任でお願いします。
