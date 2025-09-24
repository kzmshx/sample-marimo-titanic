import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Titanic Tutorial with Marimo""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _(pl):
    train_data = pl.read_csv("./data/train.csv")
    test_data = pl.read_csv("./data/test.csv")
    return test_data, train_data


@app.cell
def _(mo):
    head_cnt = mo.ui.slider(3, 10, label="head count")
    head_cnt
    return (head_cnt,)


@app.cell
def _(head_cnt, train_data):
    train_data.head(n=head_cnt.value)
    return


@app.cell
def _(head_cnt, test_data):
    test_data.head(n=head_cnt.value)
    return


@app.cell
def _(mo, pl, train_data):
    women = (
        train_data.filter(pl.col("Sex") == "female")
        .select(pl.col("Survived"))
        .to_series()
    )
    rate_women = sum(women) / len(women)

    mo.md(
        f"""
        ## % of women who survived

        {rate_women}
        """
    )
    return


@app.cell
def _(mo, pl, train_data):
    men = (
        train_data.filter(pl.col("Sex") == "male")
        .select(pl.col("Survived"))
        .to_series()
    )
    rate_men = sum(men) / len(men)

    mo.md(
        f"""
        ## % of men who survived

        {rate_men}
        """
    )
    return


@app.cell
def _(mo, train_data):
    features = mo.ui.multiselect(
        options=train_data.columns,
        label="features",
        value=["Pclass", "Sex", "SibSp", "Parch"],
    )
    return (features,)


@app.cell
def _(features, mo):
    mo.hstack([features, mo.md(f"Selected features: {', '.join(features.value)}")])
    return


@app.cell
def _(features, pl, test_data, train_data):
    from sklearn.ensemble import RandomForestClassifier

    categorical_features = [f for f in features.value if train_data[f].dtype == pl.Utf8]
    categories = {}
    for col in categorical_features:
        # trainとtestの両方からユニーク値を取得
        unique_vals = (
            pl.concat([train_data[col], test_data[col]]).unique().drop_nulls().to_list()
        )
        categories[col] = unique_vals

    y = train_data.select(pl.col("Survived")).to_numpy().ravel()
    X = (
        train_data.select(features.value)
        .to_dummies(columns=categorical_features)
        .to_numpy()
    )
    X_test = (
        test_data.select(features.value)
        .to_dummies(columns=categorical_features)
        .to_numpy()
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    return X_test, model


@app.cell
def _(X_test, model):
    predictions = model.predict(X_test)
    return (predictions,)


@app.cell
def _(pl, predictions, test_data):
    output = pl.DataFrame(
        {
            "PassengerId": test_data.select(pl.col("PassengerId")).to_series(),
            "Survived": predictions.tolist(),
        }
    )
    output.head(10)
    return (output,)


@app.cell
def _(output):
    print(output.head(10))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
