# sample-marimo-titanic

Marimo で Titanic Tutorial をやってみる。

## 環境構築

### Kaggle API の設定

[Kaggle API の設定](https://www.kaggle.com/docs/api) を参考に、API を設定してください。

### データのダウンロード

```bash
kaggle competitions download -c titanic -p data/
```

### インストール

```bash
uv sync
```

### インタラクティブ編集モードの起動

```bash
marimo edit titanic.py
```

### Jupyter Notebook ファイルへのエクスポート

```bash
marimo export ipynb ./titanic.py > titanic.ipynb
```

## フォーマット・静的検査

```bash
ruff check .
ruff check --fix .
ruff format .
```
