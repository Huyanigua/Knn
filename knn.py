import pandas as pd
import numpy as np

def normalize(coffntea):
    # нормализуем бинарные значения
    for index in coffntea.columns:
        if index in ["Выпиваете алкоголь",
                     "Характер",
                     "Как часто вы пропускаете завтраки?",
                     "Что вы предпочитаете?",
                     "Возраст",
                     "Как часто вы берете инициативу в свои руки?",
                     "Сколько спите ночью в среднем",
                     "Время подъема",
                     "Любимое время года?",
                     "Что пьют родители"]:
            continue

        for i in range(len(coffntea)):
            if coffntea.loc[i, index] in ["Да", "Восток", "Холодные", "Женский"]:
                coffntea.loc[i, index] = 1
            else:
                coffntea.loc[i, index] = 0

    # нормализуем значения с ЧИСЛОВЫМИ вариантами ответов
    for index in ["Возраст", "Время подъема", "Сколько спите ночью в среднем"]:
        coffntea[index] = coffntea[index].astype(float)
        min_value = coffntea[index].min()
        max_value = coffntea[index].max()
        for i in range(len(coffntea)):
            coffntea.loc[i, index] = (coffntea.loc[i, index] - min_value) / (max_value - min_value)

    # нормализуем значения с 4 различными СЛОВЕСНЫМИ вариантами ответов
    for index in ["Как часто вы берете инициативу в свои руки?", "Как часто вы пропускаете завтраки?"]:
        for i in range(len(coffntea)):
            match coffntea.loc[i, index]:
                case "Очень редко":
                    coffntea.loc[i, index] = 0
                case "Редко":
                    coffntea.loc[i, index] = 0.25
                case "Периодически":
                    coffntea.loc[i, index] = 0.5
                case "Часто":
                    coffntea.loc[i, index] = 0.75
                case "Очень часто":
                    coffntea.loc[i, index] = 1

    # нормализуем значения для родителей
    for i in range(len(coffntea)):
        match coffntea.loc[i, "Что пьют родители"]:
            case "Чай":
                coffntea.loc[i, "Что пьют родители"] = 1
            case "Чай и кофе":
                coffntea.loc[i, "Что пьют родители"] = 0.5
            case "Кофе":
                coffntea.loc[i, "Что пьют родители"] = 0

    # нормализуем значения для алкоголя
    for i in range(len(coffntea)):
        match coffntea.loc[i, "Выпиваете алкоголь"]:
            case "Да":
                coffntea.loc[i, "Выпиваете алкоголь"] = 1
            case "Редко":
                coffntea.loc[i, "Выпиваете алкоголь"] = 0.5
            case "Нет":
                coffntea.loc[i, "Выпиваете алкоголь"] = 0

    # нормализуем типы темперамента, заменяя один столбик на 4 отдельных
    coffntea["Холерик"] = 0
    coffntea["Меланхолик"] = 0
    coffntea["Флегматик"] = 0
    coffntea["Сангвиник"] = 0
    for i in range(len(coffntea)):
        match coffntea.loc[i, "Характер"]:
            case "Холерик":
                coffntea.loc[i, "Холерик"] = 1
            case "Меланхолик":
                coffntea.loc[i, "Меланхолик"] = 1
            case "Флегматик":
                coffntea.loc[i, "Флегматик"] = 1
            case "Сангвиник":
                coffntea.loc[i, "Сангвиник"] = 1
    coffntea.drop("Характер", axis=1, inplace=True)

    # Ну и конечно же ЛЮБИМОЕ ВРЕМЯ ГОДА КУДА ЖЕ БЕЗ НЕГО. Ну этот пункт я и придумал, так что сам виноват)))
    coffntea["Весна"] = 0
    coffntea["Лето"] = 0
    coffntea["Осень"] = 0
    coffntea["Зима"] = 0
    for i in range(len(coffntea)):
        match coffntea.loc[i, "Любимое время года?"]:
            case "Весна":
                coffntea.loc[i, "Весна"] = 1
            case "Лето":
                coffntea.loc[i, "Лето"] = 1
            case "Осень":
                coffntea.loc[i, "Осень"] = 1
            case "Зима":
                coffntea.loc[i, "Зима"] = 1
    coffntea.drop("Любимое время года?", axis=1, inplace=True)
    return coffntea


# Открываю файлы с обучающими и тестовыми данными и провожу нормализацию
coffntea_train = normalize(pd.read_csv("new.csv"))
coffntea_test = normalize(pd.read_csv("new_test.csv"))

# Создаю список со списками расстояний для каждого тестового элемента и сразу же их сортирую по возрастанию для каждого тестового элемента
all_distances = []
for i in range(len(coffntea_test)):
    distance = []
    for j in range(len(coffntea_train)):
        dist = 0
        for column in coffntea_test.columns:
            if column == "Что вы предпочитаете?":
                continue
            dist += (coffntea_test.loc[i, column] - coffntea_train.loc[j, column]) ** 2
        distance.append((dist ** 0.5, coffntea_train.loc[j, "Что вы предпочитаете?"]))
    all_distances.append(sorted(distance))
# print(all_distances)

# Прохожусь по всем к от 1 до 39
for k in range(1, 40, 2):
    guess = 0
    # Прохожусь по тестовым элементам по первым к значениям
    for i in range(len(all_distances)-1):
        coff = 0
        tea = 0
        for j in range(k-1):
            if all_distances[i][j][1] == "Кофе":
                coff += 1
            else:
                tea += 1
        if tea > coff:
            preference = "Чай"
        else:
            preference = "Кофе"
        if preference == coffntea_test.loc[i, "Что вы предпочитаете?"]:
            guess += 1
    print(f"accuracy with k = {k}: {guess/len(all_distances)}")
