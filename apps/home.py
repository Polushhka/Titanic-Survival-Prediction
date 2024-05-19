import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


@st.cache_data
def get_melt_categorical(df, categorical_features):
    """Функция для отображения категориальных признаков"""
    # melt (расплавление)
    cat_df = pd.DataFrame(
        df[categorical_features].melt(
            var_name='column',
            value_name='value'
        ).value_counts()
    ).rename(columns={0: 'count'}).sort_values(by=['column', 'count'])
    return cat_df


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def get_profile_report(df):
    from pandas_profiling import ProfileReport
    pr = ProfileReport(df)
    return pr


@st.cache_data
def create_histogram(df, column_name):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",
        color='Survived',
        title=f"Распределение {column_name}",
        template="plotly"
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, numerical_features):
    corr = df[numerical_features].corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.values,
        colorscale='plotly3'
    )
    fig.update_layout(height=800)
    return fig


@st.cache_data
def get_simple_histograms(df, selected_category):
    fig = px.histogram(
        df,
        x=selected_category,
        color=selected_category,
        title=f'Распределение по {selected_category}'
    )
    return fig


@st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        palette='viridis',
        plot_kws={'alpha': 0.5, 's': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


@st.cache_data
def display_metrics(df):
    st.markdown("""
        В данном разделе представлены ключевые метрики, предоставляющие общий обзор данных о пассажирах Титаника.
    """)

    total_passengers = len(df)
    total_survivors = df['Survived'].astype(int).sum()
    survival_rate = (total_survivors / total_passengers) * 100

    average_age = df['Age'].mean()
    median_fare = df['Fare'].median()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего пассажиров", total_passengers)
    col2.metric("Выживших пассажиров", total_survivors)
    col3.metric("Процент выживания", f"{survival_rate:.2f}%")
    col4.metric("Средний возраст", f"{average_age:.1f} лет")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Медианная стоимость билета", f"${median_fare:.2f}")
    col2.metric("Пассажиров 1-го класса", len(df[df['Pclass'] == 1]))
    col3.metric("Пассажиров 2-го класса", len(df[df['Pclass'] == 2]))
    col4.metric("Пассажиров 3-го класса", len(df[df['Pclass'] == 3]))


def display_box_plot(df, numerical_features, categorical_features):
    c1, c2, c3 = st.columns(3)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='box_feature1')
    feature2 = c2.selectbox('Второй признак', categorical_features, key='box_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, *categorical_features], key='box_filter_by', index=2)

    if feature2 == filter_by:
        filter_by = None

    fig = px.box(
        df,
        x=feature1, y=feature2,
        color=filter_by,
        title=f"Распределение {feature1} по разным {feature2}",
    )
    fig.update_layout(height=900)
    st.plotly_chart(fig, use_container_width=True)


def app(df, current_dir: Path):
    st.title("Анализ выживаемости пассажиров Титаника")

    st.markdown("""
        ### Краткая история «Титаника»
        Закройте глаза и представьте себя в 1912 году. 
        Вы приближаетесь к берегам Ньюфаундленда, слушая шум волн, бьющихся о корпус корабля. Вдыхаете ледяной воздух атлантической ночи. Отлично! Вы на борту «Титаника».
        
        Но на этот раз мы отправляемся к данным, которые составляют его историю. Этот проект будет посвящен анализу данных, полученных во время трагического плавания. Мы будем исследовать набор данных, чтобы найти необходимую информацию о пассажирах, а также о факторах, повлиявших на их выживание.
        
        «Титаник» - океанский лайнер, построенный компанией White Star Line и спущенный на воду в 1912 году. Он считался «самым безопасным кораблем из когда-либо построенных» благодаря своим инновационным системам безопасности. 14 апреля 1912 года во время своего первого рейса из Саутгемптона (Великобритания) в Нью-Йорк «Титаник» столкнулся с айсбергом и затонул.
        
        Многие люди погибли из-за отсутствия на борту достаточного количества спасательных плотов. Катастрофа «Титаника» широко считается одной из самых страшных морских катастроф в истории и стала предметом бесчисленного количества книг, фильмов и исследований.
    """)
    st.image(str(current_dir / "images" / "main.bmp"), use_column_width='auto')
    st.info("""
        **Основная цель данного анализа** - ответить на несколько вопросов, связанных с выживаемостью пассажиров, таких как: какие факторы повлияли на выживаемость пассажиров? Сколько пассажиров было в зависимости от их класса? Были ли существенные различия между выживаемостью мужчин и женщин?

        На основе полученных результатов мы сделаем прогноз выживаемости пассажиров, основываясь на имеющихся у нас данных. Мы будем использовать модель логистической регрессии для оценки вероятности выживания и разделим пассажиров на две группы: выжившие и невыжившие.

        Мы надеемся, что результаты этого анализа помогут лучше понять факторы, повлиявшие на выживаемость пассажиров «Титаника», и внесут вклад в сохранение его истории.
    """)
    st.markdown("""<hr style="height:1px;background-color: #ff4b4b; align:left" /> """, unsafe_allow_html=True)

    st.markdown("""
        
        ## Область применения
        
        В качестве области применения для системы анализа данных выбрана история крушения Титаника. Это событие является одним из самых известных кораблекрушений в истории, произошедшим в апреле 1912 года. Анализ данных о пассажирах Титаника позволяет исследовать различные аспекты выживаемости при кораблекрушении, включая социально-экономические факторы, возраст, пол и другие характеристики.
        
        ## Ключевые параметры и характеристики данных
    """)
    tab1, tab2 = st.tabs(["Показать описание данных", "Показать пример данных"])
    with tab1:
        st.markdown(r"""
            | Поле         | Описание                                             | Тип переменной |
            |--------------|------------------------------------------------------|----------------|
            | PassengerId  | Уникальный идентификатор пассажира                  | Целочисленный  |
            | Survived     | Выжил ли пассажир (1 - да, 0 - нет)                 | Целочисленный  |
            | Pclass       | Класс билета (1, 2, 3)                               | Целочисленный  |
            | Name         | Имя пассажира                                        | Строковый       |
            | Sex          | Пол пассажира                                        | Строковый       |
            | Age          | Возраст пассажира                                    | Вещественный    |
            | SibSp        | Количество братьев и сестер / супругов на борту      | Целочисленный  |
            | Parch        | Количество родителей / детей на борту               | Целочисленный  |
            | Ticket       | Номер билета                                         | Строковый       |
            | Fare         | Стоимость билета                                     | Вещественный    |
            | Cabin        | Номер каюты                                          | Строковый       |
            | Embarked     | Порт посадки (C = Cherbourg, Q = Queenstown, S = Southampton) | Строковый       |
        """)
    with tab2:
        st.header("Пример данных")
        st.dataframe(df.head(50), height=400)

    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.header("Предварительный анализ данных")
    st.dataframe(get_data_info(df), use_container_width=True)

    st.header("Основные статистики для признаков")
    display_metrics(df)

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.header("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe(), use_container_width=True)
    with tab2:
        st.header("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include='category'), use_container_width=True)

    st.header("Визуализация данных")

    st.subheader("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox1"
    )
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("""
        ## Гистограммы и ящики с усами
    
        Гистограмма — это вид диаграммы, представляющий распределение числовых данных. Она помогает оценить плотность вероятности распределения данных. Гистограммы идеально подходят для иллюстрации распределений признаков, таких как возраст клиентов или продолжительность контакта в секундах.
        
        Ящик с усами — это еще один тип графика для визуализации распределения числовых данных. Он показывает медиану, первый и третий квартили, а также "усы", которые простираются до крайних точек данных, не считая выбросов. Ящики с усами особенно полезны для сравнения распределений между несколькими группами и выявления выбросов.
    """)
    display_box_plot(
        df,
        numerical_features,
        categorical_features
    )

    tab1, tab2 = st.tabs(["Простые графики", "Показать отчет о данных"])
    with tab1:
        st.subheader("Распределение сотрудников")
        st.subheader("Столбчатые диаграммы для категориальных признаков")
        selected_category_simple_histograms = st.selectbox(
            'Категория для анализа',
            categorical_features,
            key='category_get_simple_histograms'
        )
        st.plotly_chart(get_simple_histograms(df, selected_category_simple_histograms), use_container_width=True)
    with tab2:
        if st.button("Сформировать отчёт", use_container_width=True, type='primary'):
            st_profile_report(get_profile_report(df))

    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    def create_header_with_image(title, img_src):
        return f"""
        <div style='display:flex; align-items:left;'>
            <h1 style='color:#ff4b4b; font-size:24px; webkit-text-stroke: 0px white'>
                {title}
            </h1>
            <img src='{img_src}' alt={title} style='width:300px; margin-left:40px; margin: 10px 0 20px 40px; border-radius: 10px;'>
        </div>
        """

    st.markdown(create_header_with_image(
        'Распределение пассажиров по полу',
        'https://quo.eldiario.es/wp-content/uploads/2019/10/espanoles-en-el-titanic.jpg'),
        unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 1**", expanded=False):
        fig_sex = px.pie(df, names="Sex", color="Sex", hole=.2, color_discrete_map={"male": 'red', "female": 'cyan'})
        fig_sex.update_traces(text=df["Sex"].value_counts(), textinfo="label+percent+text")
        fig_sex.update_layout(legend_title="Пол")

        st.plotly_chart(fig_sex)
        st.info("""
            **Комментарий:**\n
            На борту Титаника было больше мужчин (64.8%), чем женщин.
        """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 2: Распределение пассажиров по классам
    st.markdown(create_header_with_image(
        'Распределение пассажиров по классам',
        'https://banderaroja.com.ve/wp-content/uploads/2017/03/titanic.jpg'),
        unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 2**", expanded=False):
        ax = sns.countplot(x="Pclass", data=df, palette="pastel")
        plt.style.use("dark_background")
        ax.bar_label(ax.containers[0])
        ax.set_xlabel("Класс")
        ax.set_ylabel("Количество пассажиров")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.info("""
             **Комментарий:**\n
             Большинство пассажиров Титаника были из третьего класса (491 человек), что вдвое больше, чем пассажиры первого и второго классов вместе взятые.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 3: Распределение пассажиров по классам и полу
    st.markdown(create_header_with_image(
        'Распределение пассажиров по классам и полу',
        'https://1.bp.blogspot.com/-myC9Qq_0bDQ/Xz41ONAgS9I/AAAAAAAAJPo/iqTaMb2Yt7wx-GyOUe80e-6S25A6tlLegCLcBGAsYHQ/w1200-h630-p-k-no-nu/f3177-titanic-kate-beer.jpg'),
        unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 3**", expanded=False):
        ax = sns.countplot(x="Pclass", data=df, palette="pastel", hue="Sex")
        plt.style.use("dark_background")
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.set_xlabel("Класс")
        ax.set_ylabel("Количество пассажиров")
        st.pyplot()

        st.info("""
            **Комментарий:**\n
            Здесь показано распределение пассажиров по классам и полу. Видно, что большинство пассажиров третьего класса - мужчины.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 4: Возраст пассажиров
    st.markdown(create_header_with_image(
        'Информация о возрасте пассажиров',
        'https://avatars.mds.yandex.net/get-kinopoisk-image/1900788/0a95f304-e8ce-4d41-8511-96d256169bd4/1920x'),
        unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 4**", expanded=False):
        fig_age = px.histogram(
            df, x="Age", nbins=15, text_auto=True,
            opacity=0.7, color_discrete_sequence=['indianred'], template="plotly_dark")
        fig_age.update_layout(
            xaxis_title="Возрастные группы",
            yaxis_title="Частота"
        )
        st.plotly_chart(fig_age)

        st.info("""
            **Комментарий:**\n
            Большинство пассажиров были в возрасте от 20 до 39 лет, особенно от 20 до 29 лет. Самому старшему пассажиру было 80 лет, самому младшему - несколько месяцев.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 6: Распределение пассажиров по месту посадки
    st.markdown(create_header_with_image(
        'Распределение пассажиров по месту посадки',
        'https://s1.eestatic.com/2018/04/16/social/titanic-twitter-barcos_300233521_74164994_1706x960.jpg'
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 6**", expanded=False):
        fig_embarked = px.pie(df, names='Embarked', hole=0.2)
        fig_embarked.update_layout(
            title="Распределение пассажиров по месту посадки",
            annotations=[dict(text='S - Southampton\nC - Cherbourg\nQ - Queenstown', x=0.5, y=-0.1, showarrow=False)]
        )
        st.plotly_chart(fig_embarked)

        st.info("""
             **Комментарий:**\n
             Большинство пассажиров (72.5%) сели на корабль в Саутгемптоне.
          """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 7: Распределение пассажиров по цене билета
    st.markdown(create_header_with_image(
        'Распределение пассажиров по цене билета',
        'https://boomway.ru/wp-content/uploads/2023/06/passazhirskiy-bilet-na-titanik.jpg'
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 7**", expanded=False):
        fig_fare2 = px.histogram(
            df, x="Fare", range_x=(0, 550), nbins=15,
            opacity=0.7, color_discrete_sequence=['indianred'], template="plotly_dark")
        fig_fare2.update_layout(
            xaxis_title="Цена билета",
            yaxis_title="Количество пассажиров"
        )
        st.plotly_chart(fig_fare2)

        st.info("""
            **Комментарий:**\n
            * На данном графике видно, что большинство билетов стоили менее 35 долларов, а самые дорогие билеты стоили 512.33 долларов.
            * Большинство пассажиров заплатили за билеты от 0 до 100 долларов, с пиком в диапазоне от 20 до 40 долларов.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 9: Цена билета в зависимости от класса
    st.markdown(create_header_with_image(
        'Цена билета в зависимости от класса',
        'https://avatars.dzeninfra.ru/get-zen_doc/5233619/pub_6215647992dc855fe5218182_629696add38d907fbe62e89a/scale_1200'
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 8**", expanded=False):
        fig_class_fare = px.scatter(
            df, x="Fare", y="Pclass", color="Sex", size="Fare", size_max=12,
            color_discrete_map={'male': 'green', 'female': 'blue'})
        fig_class_fare.update_yaxes(ticktext=["Первая класс", "Второй класс", "Третий класс"], tickvals=[1, 2, 3])
        st.plotly_chart(fig_class_fare)

        st.info("""
             **Комментарий:**\n
             Этот график показывает связь между ценой билета и классом. Большинство пассажиров первого класса заплатили от 0 до 300 долларов, в то время как пассажиры третьего класса заплатили от 0 до 100 долларов, с пиком в диапазоне от 0 до 10 долларов.
        """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 10: Откуда садились пассажиры в зависимости от класса?
    st.markdown(create_header_with_image(
        'Откуда садились пассажиры в зависимости от класса?',
        'https://cdn2.rsvponline.mx/files/rsvp/styles/wide/public/images/main/2020/titanic_first_class.jpg'
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 9**", expanded=False):
        ax = sns.countplot(x="Embarked", hue="Pclass", data=df, palette="pastel")
        plt.style.use("dark_background")
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.bar_label(ax.containers[2])
        ax.set_xticklabels(["Саутгемптон", "Шербур", "Квинстаун"])
        st.pyplot()
        st.info("""
             **Комментарий:**\n
             Большинство пассажиров сели на корабль в Саутгемптоне, затем в Шербуре и Квинстауне.
       """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 13: Цена билета в зависимости от возраста и пола
    st.markdown(create_header_with_image(
        'Цена билета в зависимости от возраста и пола',
        'https://ichef.bbci.co.uk/news/640/cpsprodpb/17F4E/production/_124162189_gettyimages-1371405694.jpg'
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 13**", expanded=False):
        fig_scatter_sex = px.scatter(df, x="Age", y="Fare", color="Sex")
        st.plotly_chart(fig_scatter_sex)

        st.info("""
               **Комментарий:**\n
               Этот график показывает связь между возрастом и ценой билета пассажиров Титаника в зависимости от пола.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 14: Цена билета в зависимости от возраста и выживаемости
    st.markdown(create_header_with_image(
        'Цена билета в зависимости от возраста и выживаемости',
        'https://ep00.epimg.net/cultura/imagenes/2012/04/13/album/1334327815_650313_1334328225_album_normal.jpg'
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 14**", expanded=False):
        fig_scatter_sur = px.scatter(df, x="Age", y="Fare", color="Survived")
        st.plotly_chart(fig_scatter_sur)

        st.info("""
              **Комментарий:**\n
              Этот график показывает связь между возрастом и ценой билета пассажиров Титаника в зависимости от выживаемости. Видно, что пассажиры, которые выжили, обычно платили более высокую цену за билеты, особенно в возрасте от 20 до 40 лет.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 15: Сравнения между классом, ценой билета и полом
    st.markdown(create_header_with_image(
        'Сравнения между классом, ценой билета и полом',
        'https://sabiasquehistoria.files.wordpress.com/2020/04/titanic3.jpg',
    ), unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 15**", expanded=False):
        fig_lmplot = sns.lmplot(x="Age", y="Fare", row="Sex", col="Pclass", data=df)
        st.pyplot(fig_lmplot)

        st.info("""
               **Комментарий:**\n
               Этот график показывает связь между возрастом и ценой билета пассажиров Титаника в зависимости от пола и класса. Видно, что в трех классах мужчины и женщины тратили на билеты примерно одинаково.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 16: 3D сравнение возраста, цены билета и места посадки
    st.markdown(create_header_with_image(
        '3D сравнение возраста, цены билета и места посадки',
        'https://www.eluniverso.com/resizer/0sV9GyChdiBXuLYCO924r_2RUNM=/456x336/smart/filters:quality(70)/cloudfront-us-east-1.images.arcpublishing.com/eluniverso/BAQ3IZYUPBASTBKCV57J4FOP2Q.jpg',
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 16**", expanded=False):
        fig_3d_titanic = px.scatter_3d(
            df, x="Age", y="Pclass", z="Fare", color="Survived",
            title="3D сравнение возраста, цены билета и места посадки",
            labels={"Age": "Возраст", "Pclass": "Класс", "Fare": "Цена билета"})
        st.plotly_chart(fig_3d_titanic)
        st.info("""
            **Комментарий:**\n
            Этот график показывает в 3D форматe связь между возрастом, ценой билета и местом посадки пассажиров Титаника. Видно, что более дорогие билеты обычно покупали пассажиры, которые выжили.
        """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 17: Солнцеобразный график с различными условиями
    st.markdown(create_header_with_image(
        'Солнцеобразный график с различными условиями',
        'https://img.microsiervos.com/images2021/titanic-southampton.jpg',
    ), unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 17**", expanded=False):
        fig_sunburst = px.sunburst(
            df, path=['Sex', 'Survived', 'Pclass', 'Embarked'], values=df.index,
            color_discrete_sequence=px.colors.qualitative.Dark24)
        fig_sunburst.update_traces(textinfo="label+value")
        st.plotly_chart(fig_sunburst)
        st.info("""
            **Комментарий:**\n
            График показывает распределение по полу, выживаемости, классу и месту посадки пассажиров Титаника. Это интерактивный способ лучше понять, как эти факторы влияли на выживаемость.
       """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 18: Количество выживших пассажиров
    st.markdown(create_header_with_image(
        'Сколько пассажиров выжили?',
        'https://static.dw.com/image/15881891_303.jpg',
    ), unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 18**", expanded=False):
        supervivientes = df.groupby(['Survived']).size().reset_index(name='counts')
        fig_supervivientes = px.bar(
            supervivientes, x='Survived', y='counts', color='Survived',
            labels={'Survived': 'Выжили', 'counts': 'Количество пассажиров'},
            title="Количество выживших пассажиров Титаника",
            template="plotly_dark")
        st.plotly_chart(fig_supervivientes)
        st.info("""
             **Комментарий:**\n
             Этот простой график показывает количество пассажиров, которые выжили или не выжили в катастрофе Титаника. Большинство не выжило.
        """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 19: Выживаемость пассажиров в зависимости от класса
    st.markdown(create_header_with_image(
        'Выживаемость пассажиров в зависимости от класса',
        'https://imagenes.lainformacion.com/files/twitter_thumbnail/uploads/imagenes/2019/09/04/5d6f97e6449e4.jpeg',
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 19**", expanded=False):
        ax = sns.countplot(x="Pclass", hue="Survived", data=df, palette="pastel")
        plt.style.use("dark_background")
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.set_xticklabels(["Первая", "Вторая", "Третья"])
        st.pyplot()
        st.info("""
             **Комментарий:**\n
             Этот график показывает количество выживших пассажиров в зависимости от класса. Видно, что большинство пассажиров третьего класса не выжило.
        """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 20: Вероятность выживания в зависимости от класса
    st.markdown(create_header_with_image(
        'Вероятность выживания в зависимости от класса',
        'https://historia.nationalgeographic.com.es/medio/2022/07/15/cpmcdtita-fe022_a9748bb4_1280x750.jpeg',
    ), unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 20**", expanded=False):
        grupo_clase = df.groupby(["Pclass", "Survived"]).size().reset_index(name="counts")
        grupo_clase["probability"] = grupo_clase.apply(
            lambda row: row.counts / grupo_clase[grupo_clase.Pclass == row.Pclass]["counts"].sum() * 100, axis=1).round(
            2)
        fig_supervivientes_clase2 = px.bar(
            grupo_clase, x="Pclass", y="probability",
            color="Survived",
            labels={"Pclass": "Класс", "probability": "Вероятность выживания"},
            text_auto=True,
            title="Вероятность выживания в зависимости от класса")
        st.plotly_chart(fig_supervivientes_clase2)
        st.info("""
              **Комментарий:**\n
              Этот график показывает вероятность выживания в зависимости от класса. Видно, что пассажиры третьего класса имели значительно меньшие шансы на выживание.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 21: Вероятность выживания в зависимости от места посадки
    st.markdown(create_header_with_image(
        'Вероятность выживания в зависимости от места посадки',
        'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/embarque-titanic-1523699983.jpg',
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 21**", expanded=False):
        grupo_embarque = df.groupby(["Embarked", "Survived"]).size().reset_index(name="counts")
        grupo_embarque["probability"] = grupo_embarque.apply(
            lambda row: row.counts / grupo_embarque[grupo_embarque.Embarked == row.Embarked]["counts"].sum() * 100,
            axis=1).round(2)
        fig_supervivientes_embarque2 = px.bar(
            grupo_embarque, x="Embarked", y="probability",
            color="Survived", labels={"Embarked": "Место посадки",
                                      "probability": "Вероятность выживания"},
            text_auto=True,
            title="Вероятность выживания в зависимости от места посадки")
        st.plotly_chart(fig_supervivientes_embarque2)
        st.info("""
              **Комментарий:**\n
              Этот график показывает вероятность выживания пассажиров в зависимости от места их посадки. Видно, что пассажиры, севшие в Шербуре, имели наибольшие шансы на выживание.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 22: Вероятность выживания в зависимости от пола
    st.markdown(create_header_with_image(
        'Вероятность выживания в зависимости от пола',
        'https://img.ecartelera.com/noticias/fotos/37000/37072/1.jpg',
    ), unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 22**", expanded=False):
        grupo_genero = df.groupby(["Sex", "Survived"]).size().reset_index(name="counts")
        grupo_genero["probability"] = grupo_genero.apply(
            lambda row: row.counts / grupo_genero[grupo_genero.Sex == row.Sex]["counts"].sum() * 100, axis=1).round(2)
        fig_supervivientes_genero = px.bar(
            grupo_genero, x="Sex", y="probability",
            color="Survived",
            labels={"Sex": "Пол", 'probability': "Вероятность выживания"},
            text_auto=True,
            title="Вероятность выживания в зависимости от пола")
        st.plotly_chart(fig_supervivientes_genero)
        st.info("""
              **Комментарий:**\n
              Этот график показывает вероятность выживания в зависимости от пола. Видно, что женщины имели значительно больше шансов на выживание, чем мужчины.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 23: Вероятность выживания в зависимости от возраста
    st.markdown(create_header_with_image(
        'Вероятность выживания в зависимости от возраста',
        'https://historia.nationalgeographic.com.es/medio/2017/04/07/los-huerfanos-del-titanic_a3402840.jpg',
    ), unsafe_allow_html=True)

    with st.expander("**Показать анализ графика 23**", expanded=False):
        edades = df.groupby(["Age", "Survived"]).size().reset_index(name="counts")
        edades["probability"] = edades.apply(
            lambda row: row.counts / edades[edades.Age == row.Age]["counts"].sum() * 100, axis=1).round(2)
        fig_supervivientes_edad = px.histogram(
            edades, x="Age", y="probability", color="Survived",
            histfunc="avg", title="Вероятность выживания в зависимости от возраста",
            text_auto=True,
            labels={"Age": "Возраст", "probability": "Вероятность выживания"})
        st.plotly_chart(fig_supervivientes_edad)
        st.info("""
              **Комментарий:**\n
              Этот график показывает вероятность выживания в зависимости от возраста. Видно, что дети до 10 лет имели наибольшие шансы на выживание, в то время как пассажиры старше 60 лет имели наименьшие шансы.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    # График 24: Вероятность выживания при путешествии в одиночку
    st.markdown(create_header_with_image(
        'Вероятность выживания при путешествии в одиночку',
        'https://pymstatic.com/60560/conversions/frases-titanic-social.jpg',
    ), unsafe_allow_html=True)
    with st.expander("**Показать анализ графика 24**", expanded=False):
        df['TravelAlone'] = df['SibSp'] + df['Parch']
        solitarios = df.groupby(["TravelAlone", "Survived"]).size().reset_index(name='counts')
        solitarios["probability"] = solitarios.apply(
            lambda row: row.counts / solitarios[solitarios.TravelAlone == row.TravelAlone]['counts'].sum() * 100,
            axis=1).round(2)
        fig_supervivientes_solos = px.bar(
            solitarios, x="TravelAlone", y="probability",
            text_auto=True,
            color="Survived",
            labels={"TravelAlone": "Путешествуют в одиночку (0 = Нет, 1 = Да)",
                    "probability": "Вероятность выживания"},
            title="Вероятность выживания при путешествии в одиночку"
        )
        st.plotly_chart(fig_supervivientes_solos)
        st.info("""
              **Комментарий:**\n
              Этот график показывает вероятность выживания в зависимости от того, путешествовал ли пассажир в одиночку или с семьей. Видно, что пассажиры, путешествующие в одиночку, имели меньшие шансы на выживание.
         """, icon='💡')

    # Разделитель
    st.markdown("""<hr style="height:2px;background-color: gray;" /> """, unsafe_allow_html=True)

    @st.cache_data
    def plot_metrics(df, columns):
        plt.figure(figsize=(16, 20))
        for i, col in enumerate(columns, start=1):
            plt.subplot(len(columns), 2, 2 * i - 1)
            sns.histplot(data=df, x=col, hue="Survived", multiple="stack", kde=True)
            plt.title(f'Гистограмма для {col}')

            plt.subplot(len(columns), 2, 2 * i)
            sns.boxplot(data=df, x="Survived", y=col)
            plt.title(f'Ящик с усами для {col}')

        plt.tight_layout()
        st.pyplot(plt.gcf())  # Отображение созданного графика в Streamlit
        plt.clf()

    st.markdown("### Визуализация основных данных о пассажирах ")
    plot_metrics(df, numerical_features[:5])

    st.header("Корреляционный анализ")
    st.plotly_chart(create_correlation_matrix(df, numerical_features), use_container_width=True)
    st.markdown("""
        Матрица корреляции позволяет определить связи между признаками. Значения в матрице колеблются от -1 до 1, где:
        
        - 1 означает положительную линейную корреляцию,
        - -1 означает отрицательную линейную корреляцию,
        - 0 означает отсутствие линейной корреляции.
        
        Корреляционная матрица представляет связь между различными числовыми параметрами. В данном случае:
        
        Интерпретация корреляционной матрицы:
        * Survived и Pclass: Корреляция -0.34 указывает на умеренную отрицательную связь; более высокий класс (нижний номер класса) связан с большей вероятностью выживания.
        * Survived и Fare: Корреляция 0.26 говорит о наличии положительной связи; более высокая цена билета связана с большей вероятностью выживания.
        * Pclass и Age: Корреляция -0.34 показывает, что в более высоких классах (1 и 2) средний возраст пассажиров выше.
        * Pclass и Fare: Сильная отрицательная корреляция (-0.55) указывает, что билеты высших классов (1 и 2) стоят дороже.
    """)

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features,
        default=numerical_features,
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=1,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
