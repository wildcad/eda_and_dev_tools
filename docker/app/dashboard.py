import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# Загрузка данных
df = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/eda_and_dev_tools/main/datasets/abalone.csv')
df['Sex'] = df['Sex'].str.upper()
Q1 = df['Height'].quantile(0.25)
Q3 = df['Height'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Height'] >= Q1 - 1.5 * IQR) & (df['Height'] <= Q3 + 1.5 * IQR)]

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
numeric_columns.remove('Rings')
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

X = df.drop(columns=['Rings'])
y = df['Rings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', LGBMRegressor(random_state=42))])
pipe.fit(X_train, y_train)

X_test_processed = pipe.named_steps['preprocessor'].transform(X_test)
X_test_df = pd.DataFrame(X_test_processed,
                         columns=pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_columns).tolist() + numeric_columns)

# Создаём объяснителя
explainer = RegressionExplainer(pipe.named_steps['model'], X_test_df, y_test)

# Дашборд
db = ExplainerDashboard(explainer, title="Abalone Rings Regressor")

# Сохраняем
db.to_yaml("abalone_regression_dashboard.yaml", explainerfile="regression_explainer.joblib", dump_explainer=True)

# Объясняем модель
explainer_shap = shap.TreeExplainer(pipe.named_steps['model'])
shap_values = explainer_shap.shap_values(pipe.named_steps['preprocessor'].transform(X_test.iloc[[0]]))

# Выводим waterfall plot
print("Пример прогноза:")
shap.initjs()
shap.force_plot(explainer_shap.expected_value, shap_values[0], 
                pipe.named_steps['preprocessor'].transform(X_test.iloc[[0]]),
                feature_names=X_test_df.columns,
                matplotlib=True, show=False)
plt.show()