# Кейс №12

Условие:
1. Загрузите набор данных **Boston house prises**  (`load_boston.data`, `load_boston.target`).


2. Постройте таблицу `pandas` для данных и проанализируйте их структуру: отсутствующие значения, основные выборочные характеристики признаков. Нормализуйте данные.


3. Разделите данные на подвыборки для обучения и тестирования в соотношении **80%** к **20%**.


4. Обучите две регрессионные модели **LinearRegression** и **[Li](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)[nearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)** (используйте параметры по умолчанию) и две регрессионные модели, построенные с помощью метода опорных векторов, со следующими параметрами: `SVR(kernel=’rbf’, degree=3, gamma=1)` и `SVR(kernel=’poly’, degree=5)`.


5. Оцените качество моделей, используя функцию потерь MAE (mean_squared_error) на обучающей и тестовой выборках.


6. Какая модель дает меньшую ошибку обучения (ошибку тестирования)?

---

Решение:

1. Загрузка набора данных. В этом шаге мы используем функцию `load_boston()` из библиотеки `scikit-learn`, чтобы загрузить набор данных **Boston House Prices**. Этот набор данных содержит информацию о различных характеристиках жилья в городе Бостон и соответствующие им цены на недвижимость.

```python
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target
```

2. Анализ структуры данных и нормализация. В этом шаге мы создаем объект `DataFrame` с помощью библиотеки `pandas` и анализируем его структуру, используя методы ``isnull()`` и ``describe()``. Затем мы используем класс **StandardScaler** из библиотеки `scikit-learn`, чтобы нормализовать данные.

```python
import pandas as pd

df = pd.DataFrame(X, columns=boston.feature_names)
df['target'] = y
print(df.isnull().sum())
print(df.describe())
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
```

3. Разделение данных на обучающую и тестовую выборки. В этом шаге мы используем функцию `train_test_split()` из библиотеки `scikit-learn`, чтобы разделить данные на *обучающую* и *тестовую* выборки в соотношении **80%** к **20%**. Это позволяет проверить, насколько хорошо модель работает на новых данных.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
```

4. Обучение моделей. В этом шаге мы обучаем 4 различных модели регрессии на обучающих данных. Мы используем классы **LinearRegression**, **LinearSVR**, **SVR** с `kernel='rbf' и degree=3`, и **SVR** с `kernel='poly' и degree=5`. После обучения каждой модели мы получаем прогнозы на обучающих и тестовых данных.

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
    
5. Оценка качества моделей. В этом шаге мы оценим качество моделей, используя функцию потерь **MAE** на обучающей и тестовой выборках. Для этого используем функцию `mean_absolute_error()` из библиотеки `scikit-learn`, чтобы оценить качество каждой модели на обучающих и тестовых данных. Мы сравниваем ошибку абсолютного значения средней арифметической между фактическими значениями и прогнозами.

```python
from sklearn.metrics import mean_absolute_error

for name, model in models.items():
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f'{name}: Train MAE={train_mae:.3f}, Test MAE={test_mae:.3f}')
 ```
    
6. Определение модели с наименьшей ошибкой. В этом шаге мы находим модель с наименьшей ошибкой на обучающих и тестовых данных. Для этого мы используем результаты оценки моделей, полученные в предыдущем шаге, и выбираем модель с наименьшей ошибкой **MAE**. В нашем случае, лучшей моделью является **LinearRegression**, потому что она имеет наименьшую ошибку на тестовой выборке.

```python
LinearRegression: Train MAE=3.270, Test MAE=3.196
LinearSVR: Train MAE=4.595, Test MAE=4.273
SVR_rbf: Train MAE=2.658
```
