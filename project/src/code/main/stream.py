import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib

def add_feature(data):
    def calc_mean_room_area(data):
        return (data['area'] - data['kitchen_area']) / (abs(data['rooms']))

    data['mean_room_area'] = calc_mean_room_area(data)
    data['percent_of_kitchen_area'] = data['kitchen_area'] / data['area']
    data['percent_of_level'] = data['level'] / data['levels']
    return data

class RobustScaler():
    def __init__(self):
        self.scaler = joblib.load('scaler.pkl')

    def get_scaled_data(self, nums):
        return self.scaler.transform(nums)


class CatBoostRegressor():
    def __init__(self):
        self.model = joblib.load('cat_model.pkl')

    def predict_price(self, data):
        return self.model.predict(data)


class LightGBM():
    def __init__(self):
        self.model = joblib.load('lgbm.pkl')

    def predict_price(self, data):
        return self.model.predict(data)


coordinates = pd.read_csv('coordinates.csv')

st.title('Демо-версия сервиса по оценке квартир')
st.markdown('**Это демонстрационный вариант**')
st.markdown('Укажите параметры квартиры и узнайте ее стоимость')

show_data = st.sidebar.checkbox('Все параметры')
st.sidebar.info('Чем больше параметров вы укажете, тем точнее будет прогноз цены.')
st.sidebar.info('Заполните форму **Дополнительные параметры**')

# with st.form('form'):
a = st.text_area('Субъект: ', 'Москва')
b = st.text_area('Общая площадь: ', '50')
c = st.text_area('Количество комнат: ', '2')
if show_data == True:
    st.markdown('## Дополнительные параметры')
    st.subheader('Укажите другие параметры квартиры:')
    d = st.text_area('Этаж: ', '5')
    e = st.text_area('Этажность дома: ', '10')
    f = st.text_area('Площадь кухни: ', '10')
else:
    d = 6
    e = 12
    f = 10

st.sidebar.markdown('Выберите вид жилья')
select_object_type = st.sidebar.radio('', ('Вторичное жилье', 'Новостройка'))
if select_object_type == 'Вторичное жилье':
    g = 1
if select_object_type == 'Новостройка':
    g = 2

select_building_type = st.sidebar.selectbox('Выберите тип дома',
                                            ('Панельный', 'Монолитый', 'Кирпичный', 'Бетонный', 'Деревянный', 'Другое'))
if select_building_type == 'Панельный':
    h = 1
if select_building_type == 'Монолитый':
    h = 2
if select_building_type == 'Кирпичный':
    h = 3
if select_building_type == 'Бетонный':
    h = 4
if select_building_type == 'Деревянный':
    h = 5
if select_building_type == 'Другое':
    h = 0

data = {'state': str(a), 'area': int(b), 'rooms': int(c), 'level': int(d), 'levels': int(e), 'kitchen_area': int(f),
        'object_type': int(g), 'building_type': int(h)}
df = pd.DataFrame(data,
                  columns=['state', 'area', 'rooms', 'level', 'levels', 'kitchen_area', 'object_type', 'building_type'],
                  index=[0])

# Добавляем координаты по субъекту
df_with_coordinates = pd.merge(df, coordinates.loc[coordinates.state == a][['geo_lat', 'geo_lon', 'state']],
                               on='state').drop('state', axis=1)

# Добавляем временной признак
now = datetime.datetime.now()
first_date = datetime.datetime(2018, 2, 19)
df_with_coordinates['day_delta'] = (now - first_date).days
df_with_coordinates['hour'] = now.hour
df_with_coordinates['year'] = now.year
df_with_coordinates = add_feature(df_with_coordinates)
# Нормализуем числовые признаки
#nums = df_with_coordinates.drop(['object_type', 'building_type'], axis=1)
#scaler = RobustScaler()
#scaled_nums = pd.DataFrame(scaler.get_scaled_data(nums))

#ready_df = pd.concat([scaled_nums, df_with_coordinates['object_type'], df_with_coordinates['building_type']], axis=1)

model = LightGBM()
prediction = model.predict_price(df_with_coordinates)

if st.button('Узнать рекомендованную стоимость'):
    # st.markdown('**Рекомендованная цена квартиры**')
    #st.subheader(np.round(np.exp(prediction[0])))
    st.subheader(prediction[0])
else:
    st.write('Нажмите на кнопку, чтобы рассчитать стоимость!')
