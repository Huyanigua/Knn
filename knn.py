import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Создаем образец данных
data = {
    'Пол': ['Мужской', 'Женский', 'Мужской', 'Женский', 'Мужской', 'Женский'],
    'Возраст': [25, 30, 35, 20, 40, 28],
    'Характер': ['Экстраверт', 'Интроверт', 'Экстраверт', 'Интроверт', 'Экстраверт', 'Интроверт'],
    'Кофе/Чай': ['Кофе', 'Чай', 'Кофе', 'Чай', 'Кофе', 'Чай'],
    'Рост': [175, 160, 180, 165, 190, 170],
    'Вес': [70, 55, 80, 60, 90, 75],
    'Сколько часов спите': [7, 8, 6, 9, 5, 7],
    'Время года': ['Зима', 'Лето', 'Весна', 'Осень', 'Зима', 'Весна'],
    'Цвет глаз': ['Голубой', 'Зеленый', 'Карий', 'Серый', 'Голубой', 'Зеленый'],
    'Настроение': ['Хорошее', 'Плохое', 'Хорошее', 'Плохое', 'Хорошее', 'Плохое']
}

df = pd.DataFrame(data)

# Преобразуем переменные
le = LabelEncoder()
df['Пол'] = le.fit_transform(df['Пол'])
df['Характер'] = le.fit_transform(df['Характер'])
df['Время года'] = le.fit_transform(df['Время года'])
df['Цвет глаз'] = le.fit_transform(df['Цвет глаз'])
df['Кофе/Чай'] = le.fit_transform(df['Кофе/Чай'])
df['Настроение'] = le.fit_transform(df['Настроение'])

# Разделим данные на обучающую и тестовую выборки
X = df.drop('Кофе/Чай', axis=1)
y = df['Кофе/Чай']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализуем данные
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создаем классификатор k
knn = KNeighborsClassifier(n_neighbors=5)

# Обучаем модель
knn.fit(X_train, y_train)

# Прогнозируем утренний напиток
predicted_drink = knn.predict(X_test)

# Преобразуем числовые значения в строковые
drink_map = {0: 'Кофе', 1: 'Чай'}
predicted_drink_str = [drink_map[x] for x in predicted_drink]

# Выводим результаты
for i, drink in enumerate(predicted_drink_str):
    print(f"Утренний напиток для образца {i+1}: {drink}")
