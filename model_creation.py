# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:50:30 2024

@author: Metinov Adilet
"""

import rasterio
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def calculate_mean_value(src, crop_window):
    left, top, width, height = crop_window
    image_data = src.read(1, window=rasterio.windows.Window(left, top, width, height))
    return np.mean(image_data)

# Инициализируем список для сохранения результатов
results = []

# Путь к директории с изображениями
image_directory = 'C:\\Users\\Administrator\\Desktop\\Data\\'
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
crop_window = (0, 0, 1200, 1200)

# Цикл по годам
for year in range(2014, 2022):  # 2022, потому что range не включает конечное значение
    # Цикл для NDVI и NDWI
    for prefix in ['NDVI', 'NDWI']:
        for month in months:
            image_path = os.path.join(image_directory, f'{prefix}_Image_{year}_{month}.tif')
            if os.path.exists(image_path):
                with rasterio.open(image_path) as src:
                    mean_value = calculate_mean_value(src, crop_window)
                    results.append({'Year': year, 'Month': month, 'Type': prefix, 'Mean_Value': mean_value})
            else:
                print(f'Image for {year}_{month} not found.')

# Создаем DataFrame из списка результатов
results_df = pd.DataFrame(results)

# Сохраняем результаты в CSV файл
output_path = os.path.join(image_directory, 'mean_values_summary.csv')
results_df.to_csv(output_path, index=False)

print(f'Results saved to {output_path}')

# Загружаем данные
data_df = pd.read_csv(output_path)

# Проверяем на наличие nan в столбце 'Mean_Value' и заменяем на соответствующие значения
for i, row in data_df.iterrows():
    if pd.isnull(row['Mean_Value']):
        if row['Type'] == 'NDVI':
            data_df.at[i, 'Mean_Value'] = 0.2
        elif row['Type'] == 'NDWI':
            data_df.at[i, 'Mean_Value'] = -0.2

# Преобразуем 'Type' с помощью one-hot encoding
type_dummies = pd.get_dummies(data_df['Type'], prefix='Type')
data_df = pd.concat([data_df.drop('Type', axis=1), type_dummies], axis=1)

# Преобразование всех данных в float32
data_df['Year'] = data_df['Year'].astype('float32')
data_df['Month'] = data_df['Month'].astype('float32')
for column in type_dummies.columns:
    data_df[column] = data_df[column].astype('float32')

# Создаем признаки (X) и цель (y), убедимся, что все данные в нужном формате
X = data_df.drop('Mean_Value', axis=1).astype('float32')
y = data_df['Mean_Value'].astype('float32')

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Определение модели
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)

# Оценка модели на тестовых данных
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
# Сохранение обученной модели
model.save('C:\\Users\\Administrator\\Desktop\\Data\\Model\\Meer.h5')
