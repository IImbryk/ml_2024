## Домашняя работа

> Примечание: иногда ваш ответ не совпадает точно с одним из вариантов.
> Выберите вариант, который наиболее близок к вашему решению.

В этой домашней работе мы будем использовать набор данных "Bank Marketing". Скачайте его [здесь](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).

Вы можете сделать это с помощью команды `wget`:

```bash
wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
unzip bank+marketing.zip 
unzip bank.zip
```


В этом наборе данных целевой переменной является переменная `y` — подписался ли клиент на срочный депозит или нет.

### Подготовка набора данных

Для остальной части домашней работы вам нужно будет использовать только следующие столбцы:

* `'age'`,
* `'job'`,
* `'marital'`,
* `'education'`,
* `'balance'`,
* `'housing'`,
* `'contact'`,
* `'day'`,
* `'month'`,
* `'duration'`,
* `'campaign'`,
* `'pdays'`,
* `'previous'`,
* `'poutcome'`,
* `'y'`

Разделите данные на 3 части: обучение/валидация/тест с распределением 60%/20%/20%. Используйте функцию `train_test_split` для этого с параметром `random_state=1`.

### Вопрос 1: Важность признаков по ROC AUC

ROC AUC также можно использовать для оценки важности признаков числовых переменных.

Сделаем это:

* Для каждой числовой переменной используйте её как предсказание и вычислите AUC с переменной `y` в качестве фактического значения.
* Используйте для этого обучающую выборку.

Если ваш AUC < 0.5, инвертируйте эту переменную, добавив перед ней знак "-".

(например, `-df_train['engine_hp']`)

AUC может быть ниже 0.5, если переменная отрицательно коррелирует с целевой переменной. Вы можете изменить направление корреляции, взяв отрицательное значение этой переменной — тогда отрицательная корреляция станет положительной.

Какая числовая переменная (из следующих 4) имеет наивысший AUC?

- `balance`
- `day`
- `duration`
- `previous`

### Вопрос 2: Обучение модели

Примените one-hot-encoding с использованием `DictVectorizer` и обучите логистическую регрессию с этими параметрами:

```python
LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
```

Какой AUC у этой модели на валидационном наборе данных? (округлите до 3 знаков)

- 0.69
- 0.79
- 0.89
- 0.99


### Вопрос 3: Точность и полнота

Теперь давайте вычислим precision и recall для нашей модели.

* Оцените модель на всех порогах от 0.0 до 1.0 с шагом 0.01.
* Для каждого порога вычислите precision и recall.
* Постройте их графики.

На каком пороге precision и recall пересекаются?

* 0.265
* 0.465
* 0.665
* 0.865


### Вопрос 4: F1 score

Точность и полнота конфликтуют — когда одна увеличивается, другая уменьшается. Поэтому их часто объединяют в метрику F1 score, которая учитывает обе.

Это формула для вычисления F1:

$$F_1 = 2 \cdot \cfrac{P \cdot R}{P + R}$$

Где $P$ — это точность, а $R$ — полнота.

Давайте вычислим F1 для всех порогов от 0.0 до 1.0 с шагом 0.01.

При каком пороге F1 достигает максимума?

- 0.02
- 0.22
- 0.42
- 0.62


### Вопрос 5: 5-кратная кросс-валидация

Используйте класс `KFold` из библиотеки Scikit-Learn, чтобы оценить нашу модель на 5 разных фолдах:

```python
KFold(n_splits=5, shuffle=True, random_state=1)
```

* Итеративно пройдите по различным фолдам из `df_full_train`.
* Разделите данные на обучающую и валидационную выборки.
* Обучите модель на обучающей выборке с этими параметрами: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`.
* Используйте AUC для оценки модели на валидационной выборке.

Какова стандартная ошибка оценок на разных фолдах?

- 0.0001
- 0.006
- 0.06
- 0.26


### Вопрос 6: Тюнинг гиперпараметров

Теперь давайте используем 5-кратную кросс-валидацию, чтобы найти лучший параметр `C`.

* Переберите следующие значения `C`: `[0.000001, 0.001, 1]`.
* Инициализируйте `KFold` с теми же параметрами, что и ранее.
* Используйте следующие параметры для модели: `LogisticRegression(solver='liblinear', C=C, max_iter=1000)`.
* Вычислите среднюю оценку и стандартное отклонение (округлите среднее значение и std до 3 знаков).

Какое значение `C` приводит к лучшему среднему значению?

- 0.000001
- 0.001
- 1

Если есть несколько одинаковых значений, выберите наименьшее стандартное отклонение. Если и тогда есть совпадения, выберите наименьшее значение `C`.

