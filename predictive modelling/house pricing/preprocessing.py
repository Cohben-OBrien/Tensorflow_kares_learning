import numpy as np
from jupyter_lsp.types import SimpleSpecMaker
from mistune.markdown import preprocess
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector

numeric_attributes = ['longitude', 'latitude', 'housing_median_name', 'total_rooms', 'total_bedrooms',
                      'population', 'household', 'median_income', 'median_house_value']

category_attributes = ['ocean_proximity']

numeric_pipeline = Pipeline([
    SimpleImputer(strategy='median'),
    StandardScaler(),
])

cat_pipe = Pipeline([
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'),
])


processing = ColumnTransformer([

    ("numeric", numeric_attributes, numeric_pipeline),
    ("categorical", category_attributes, cat_pipe),
])

preprocessing = make_column_transformer(
    (numeric_attributes, make_column_selector(dtype_include=np.number)),
    (category_attributes, make_column_selector(dtype_include=object))
)



def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(X):
    return ['ratio']

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )