import typing as t
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def create_preprocesser(
    categorical_features: t.List[str], numerical_features: t.List[str]
) -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
