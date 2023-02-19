# Кейс №12

1. Загрузим набор данных Boston house prices из библиотеки scikit-learn:

```python
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target
```

2. Для анализа данных создадим таблицу pandas и проверим ее на наличие отсутствующих значений:

```python
import pandas as pd

df = pd.DataFrame(X, columns=boston.feature_names)
df['target'] = y
print(df.isnull().sum())
print(df.describe())
```

####Отсутствующих значений не обнаружено. Метод describe() позволяет получить основные выборочные характеристики признаков. Для нормализации данных воспользуемся классом StandardScaler из библиотеки scikit-learn:####

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
```

3. Разделим данные на обучающую и тестовую выборки в соотношении 80% к 20%:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
```

4. Обучим модели LinearRegression, LinearSVR, SVR с kernel='rbf' и degree=3, и SVR с kernel='poly' и degree=5:

```python
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR

models = {
    'LinearRegression': LinearRegression(),
    'LinearSVR': LinearSVR(),
    'SVR_rbf': SVR(kernel='rbf', degree=3, gamma=1),
    'SVR_poly': SVR(kernel='poly', degree=5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f'{name}: Train MAE={train_mae:.3f}, Test MAE={test_mae:.3f}')
 ```
    
5. Оценим качество моделей, используя функцию потерь MAE на обучающей и тестовой выборках. Для этого воспользуемся функцией mean_absolute_error из библиотеки scikit-learn:

```python
from sklearn.metrics import mean_absolute_error

for name, model in models.items():
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f'{name}: Train MAE={train_mae:.3f}, Test MAE={test_mae:.3f}')
 ```
    
6. Найдем модель с наименьшей ошибкой обучения и тестирования:

```python
LinearRegression: Train MAE=3.270, Test MAE=3.196
LinearSVR: Train MAE=4.595, Test MAE=4.273
SVR_rbf: Train MAE=2.658
```
