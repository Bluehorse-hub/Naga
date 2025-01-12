# Naga

> [!NOTE]
> [Hydra](https://hydra.cc/docs/intro/)のようなハイパーパラメータ管理のためのライブラリ

### 特徴
- 一般的なPythonの文法による実装のため可読性が高い

- naga.pyに機能を追加することでカスタマイズ可能

### 機能
- 指定したハイパーパラメータでグリッドサーチを行う

- 実験時のハイパーパラメータと損失の推移を出力

## お試し

このリポジトリをクローン

```
git clone https://github.com/Bluehorse-hub/Naga.git
```

リポジトリ内に入る

```
cd Naga
```

設定ファイルから環境を複製する

```
conda env create -n naga -f env.yaml
```

環境に入る

```
conda activate naga
```

設定ファイルを読み込むためのライブラリを入れる

```
conda install conda-forge::omegaconf
```

`dummy_train.py`を実行して結果を見る

```
python dummy_train.py
```

## 事前準備

### ライブラリのインストール

設定ファイルを用いるので**omegaconf**をインストールする

pip

```
pip install omegaconf
```

conda

```
conda install conda-forge::omegaconf
```

### フォルダ構成

```
.
└── project
    ├── config
    │   └── paramas.yaml
    ├── train.py
    └── naga.py
```

### 設定ファイルの書き方
探索したいパラメータを設定ファイル(拡張子は**yaml**)に書く

```yaml
params:
    lr: [0.01, 0.02, 0.03]
    batch_size: [16, 32, 64]
```
> [!IMPORTANT]
>  `params`というセクションキーを付け、その階層に項目を**リスト型**で定義する  
>  例では　`params.lr`、`params.batch_size`となっている

## 実装方法

```python
# ライブラリのインポート
import naga

# 保存先のパスとまとめ用のリストを定義
time_path, best_judge_list = naga.preparation()

# 設定ファイルから組み合わせと名前を取得
params_value_conbinations, params_name_list = naga.loadyaml('./config/params.yaml')

# 実験管理用のループ
for i, (lr, batch_size) in enumerate(params_value_conbinations):

    # 学習進捗の調査用と実験管理用のパスの初期化
    loss_list, dirs_path = naga.init()

    # 実験管理用のパスを更新
    dirs_path = naga.update_dirs_path(i, time_path, dirs_path)

    # 実験で使用したパラメータの値を保存
    params_value_list = [lr, batch_size]

    # フォルダを作成
    naga.makedirs(dirs_path)

    # 設定ファイルの作成
    naga.makeyaml(dirs_path, params_name_list, params_value_list)

    '''
    train()の処理
    '''

    # 実験結果を管理するためのリストに結果を追加
    best_judge_list.append(sum(loss_list))

    # 実験毎の結果をプロット
    naga.plot_loss_history(loss_list, dirs_path)

# 全ての実験結果をテキストファイルに出力
naga.best_study(time_path, best_judge_list)
```
