
## Цели нашего курса:

- Познакомиться с открытыми библиотеками Python для работы с табличными данными;
- Разработать простые модели на основе данных или реализовать компьютерную модель в ДУ;

## Требования нашего курса:

- Реализовать 1 индивидуальный проект: код опубликовать в github, продемонстрировать работу кода. Презентация на 5-8 минут о проекте и результатах работы. Проверить 1 работу сокурсника;
- Выполнение домашних работ и загрузка их в github или, если Вы готовы освоить больше материала и более сложного, напишите мне, подумаем, как поступить
- Небольшое индивидуальное собеседование (по необходимости).

# Формат занятий:
- Созвоны раз в 2 недели на 1-1,5 часа -- посещение не обязательно

## Требования к проекту:

- Источник данных — физическое устройство (датчики + arduino), генерация данных на основе модели, источника данных, парсинг данных с веб-страничек, форма ввода данных;
- Дополнительный источник данных — телеграм бот
- Сбор данных в БД;
- Подготовка данных, анализ данных, модель объекта;
- Визуализация данных и доп. анализ данных (дашборды — Grafana, Qlick Sense, Python Dash и другие);

Коротко, какие могут быть темы:

[Пример 1. Мониторинг электроэнергии на производстве. ](https://www.notion.so/1-c702e38dda4b4b8d921ce3b5cc45c944?pvs=21)

[Пример 2. Подбор слуховых аппаратов для пациентов](https://www.notion.so/2-800f16a63eb54b109bbee21fb6445c29?pvs=21)

[Пример 3. Модель газоперекачивающего агрегата](https://www.notion.so/3-b2bfdddd1a47492d93d1b146cb635a49?pvs=21)

[Пример 4. Динамическая модель опухоли](https://www.notion.so/4-f50445b2530345cc9f689ce64f1077c8?pvs=21)



## Где взять данные/модель для проекта
1. Взять общеизвестный датасет из открытого доступа, например, найти здесь (из других источников)	

https://www.kaggle.com/datasets 

https://github.com/awesomedata/awesome-public-datasets 
[](https://www.kaggle.com/datasets)https://archive-beta.ics.uci.edu/

https://data.worldbank.org/


2. Запросить данные на работе
3. Модель в ДУ можно подсмотреть тут или находить в научных статьях:
    
    [prbook.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/967bc864-fa93-4ebb-a65a-1738a06dd035/807630ea-9602-41b9-a301-d10520a98cc2/prbook.pdf)

Репозиторий для вдохновения: https://github.com/YKatser/Industrial-ML/blob/main/cases.md

## План-капкан (будет заполняться по ходу)
| Дата       | Тема                | Домашка |
| ------------- |:------------------:| :-----|
| 12.09.2024     |  задачи ML, классификация, моделирование систем в ДУ | Выполнять в notebook, можно в colab/kaggle, можно локально. <br/> Финальный блокнот загрузить в свой гит. Набор данных по доходам (https://archive.ics.uci.edu/dataset/2/adult) в последней ячейке блокнота ответить на вопросы : <br/> 1) Число столбцов в наборе данных <br/> 2) Есть ли пропуски в данных? Если есть, то в каких столбцах <br/> 3) Кол-во уникальных значений в столбце race <br/> 4) Медиана hours-per-week <br/> 5) Кого больше - женщин или мужчин с ЗП >50K?<br/> 6) Заполните пропущенные данные в отдельных столбцах наиболее встречаемыми значениями. Как еще можно было бы заполнить пропущенные данные? |
|26.09.2024|линейная регрессия| Решим задачу предсказания оценки за экзамен по ТОЭ во 2 семестре (сообщите пожалуйста какой предмет был самым сложным во втором семестре 2 (лектор один и тот же у обеих групп). <br/> Построим модель предсказания оценки за этот предмет для будущих студентов:  <br/> 1) Необходимо всем вместе собрать данные. Для этого, напишите каждый в таблице гугла какой признак вам кажется значимым при определении оценки ([таблица](https://docs.google.com/spreadsheets/d/1het-urZJKtHMKdE84htxegRR8WXS9C5WEdajfBXRDrI/edit?usp=sharing) ) <br/> 2) Cоставлю опросник по признакам, которые вы предложите и пришлю  его вам, нужно будет заполнить до 1 октября, присылаю вам результаты <br/> 3) далее каждый составляет модель регрессии для определения оценки за предмет (можно использовать не только линейную регрессию), посчитать ошибку предсказания (MSE, R^2), сохранить модель (дедлайн 10 октября)


### Будем брать из курса ODS:

https://habr.com/ru/company/ods/blog/322626/

https://academy.yandex.ru/handbook/ml


