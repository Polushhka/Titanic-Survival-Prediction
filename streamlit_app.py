import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction_logreg
from plotly.io import templates
templates.default = "plotly"

st.set_page_config(
    page_title="Анализ выживаемости пассажиров Титаника",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Мы заполняем нулевые значения поля «Возраст» средним возрастом людей, принадлежащих к тому же классу
    df["Age"] = df.groupby(["Pclass"])["Age"].transform(lambda x: x.fillna(x.mean()))
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Замена пропущенных значений в 'Embarked' наиболее частым значением
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df["Embarked"] = df["Embarked"].replace({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})

    # Удаление колонки 'Cabin' из-за большого количества пропусков
    df = df.drop(columns=['Cabin', 'PassengerId'])

    categorical_columns = ['Sex', 'Embarked', 'Pclass', 'Survived']
    for column in categorical_columns:
        df[column] = df[column].astype('category')
    return df


class Menu:
    apps = [
        {
            "func": home.app,
            "title": "Главная",
            "icon": "house-fill"
        },
        {
            "func": prediction_logreg.app,
            "title": "Прогнозирование",
            "icon": "person-check-fill"
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons = [app["icon"] for app in self.apps]

            selected = option_menu(
                "Меню",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )
            st.info("""
                ## Анализ выживаемости пассажиров Титаника
                Эта система анализа данных предназначена для изучения различных аспектов выживаемости пассажиров Титаника, включая социально-экономические факторы, возраст, пол и другие характеристики.
            """)
        return selected


if __name__ == '__main__':
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

    df = preprocess_data(current_dir / 'merged_data.csv')
    menu = Menu()
    st.sidebar.image(str(current_dir / 'images' / 'logo.png'))
    selected = menu.run()
    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break
