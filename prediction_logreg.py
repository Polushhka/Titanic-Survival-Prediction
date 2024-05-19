import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from streamlit_option_menu import option_menu


@st.cache_data
def create_plot_roc_curve(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC кривая (AUC = %0.2f)' % roc_auc_score(y_true, y_prob)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash'),
        name='Диагональ'
    ))

    fig.update_layout(
        xaxis_title='Доля ложно-положительных результатов',
        yaxis_title='Доля истинно-положительных результатов',
        title='Кривая ROC',
        width=900
    )
    return fig


@st.cache_data
def create_plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(3)

    fig = px.imshow(
        cm,
        labels=dict(x="Предсказанный класс", y="Истинный класс"),
        x=['Нет', 'Да'], y=['Нет', 'Да'],
        title='Нормализованная матрица ошибок' if normalize else 'Матрица ошибок',
        color_continuous_scale='Blues'
    )
    fig.update_xaxes(tickangle=45, tickmode='array', tickvals=np.arange(2), ticktext=['Нет', 'Да'])
    fig.update_yaxes(tickangle=45, tickmode='array', tickvals=np.arange(2), ticktext=['Нет', 'Да'])
    # Добавление надписей
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(
                x=i, y=j,
                text=str(cm[j, i]),
                showarrow=False,
                font=dict(color="white" if cm[j, i] > thresh else "black"),
                align="center"
            )
    return fig


def score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        prob = clf.predict_proba(X_train)[:, 1]
        clf_report = classification_report(y_train, pred, output_dict=True)
        st.subheader("Результат обучения:")
        st.write(f"Точность модели: {accuracy_score(y_train, pred) * 100:.2f}%")
        st.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:", pd.DataFrame(clf_report).transpose())
        st.plotly_chart(create_plot_roc_curve(y_train, prob), use_container_width=True)
        st.plotly_chart(create_plot_confusion_matrix(y_train, pred, normalize=True), use_container_width=True)
    else:
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]
        clf_report = classification_report(y_test, pred, output_dict=True)
        st.subheader("Результат тестирования:")
        st.write(f"Точность модели: {accuracy_score(y_test, pred) * 100:.2f}%")
        st.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ:", pd.DataFrame(clf_report).transpose())
        st.plotly_chart(create_plot_roc_curve(y_test, prob), use_container_width=True)
        st.plotly_chart(create_plot_confusion_matrix(y_test, pred, normalize=True), use_container_width=True)


def print_model_adequacy_section(current_dir: Path):
    st.markdown(
        """
        ## Оценка адекватности модели
        При оценки адекватности модели важно использовать несколько метрик, которые помогают оценить различные аспекты производительности модели.
 
        ### Матрица ошибок (Confusion Matrix)
 
        Матрица ошибок позволяет визуально оценить, как модель справляется с каждым из классов задачи. Она показывает, сколько примеров, предсказанных в каждом классе, действительно принадлежат этому классу.
        """
    )
    st.image(str(current_dir / 'images' / 'matrix.jpg'))
    st.markdown(
        """
        ### Отчет о классификации (Precision, Recall, F1-Score)
        * Precision (Точность) описывает, какая доля положительных идентификаций была верной (TP / (TP + FP)).
        * Recall (Полнота) показывает, какая доля фактических положительных классов была идентифицирована (TP / (TP + FN)).
        * F1-Score является гармоническим средним Precision и Recall и помогает учесть обе эти метрики в одной.

        ### Кривая ROC и площадь под кривой AUC
        * ROC кривая (Receiver Operating Characteristic curve) помогает визуально оценить качество классификатора. Ось X показывает долю ложноположительных результатов (False Positive Rate), а ось Y — долю истинноположительных результатов (True Positive Rate).
        * AUC (Area Under Curve) — площадь под ROC кривой, которая дает количественную оценку производительности модели.
        """
    )


def app(df, current_dir: Path):
    st.title("Анализ выживаемости пассажиров Титаника")

    st.image(str(current_dir / "images" / "main-pred.jpg"), width=150, use_column_width='auto')

    categorical_features = df.select_dtypes(include='category').columns.tolist()

    X = df.drop(['Survived', 'Name', 'Ticket'], axis='columns')
    X = pd.get_dummies(X).drop('Sex_female', axis='columns')
    X.fillna({'Age': X.Age.median()}, inplace=True)
    Y = df[['Survived']]

    st.subheader('Разделение данных')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    st.write("Размер тренировочных данных:", X_train.shape, y_train.shape)
    st.write("Размер тестовых данных:", X_test.shape, y_test.shape)

    tab1, tab2 = st.tabs(["Тренировочные данные", "Тестовые данные"])

    with tab1:
        st.subheader("Тренировочные данные")
        st.markdown("""
          **Описание:** Тренировочные данные используются для подгонки модели и оценки её параметров.
          Эти данные получены путем исключения из исходного датасета столбцов с целевой переменной 'y'.

          **Данные тренировочного набора (X_train)**.
          Обучающий набор данных содержит информацию о признаках, используемых для обучения модели.
          """)
        st.dataframe(X_train.head(15), use_container_width=True)
        st.markdown("""
          **Целевая переменная (y_train)**.
          Целевая переменная содержит значения цены, которые модель должна научиться прогнозировать.
          В качестве целевой переменной для тренировочного набора используются исключительно значения столбца 'y'.
          """)
        st.dataframe(pd.DataFrame(y_train.head(15)).T)

    with tab2:
        st.subheader("Тестовые данные")
        st.markdown("""
          **Описание:** Тестовые данные используются для проверки точности модели на данных, которые не участвовали в тренировке.
          Это позволяет оценить, как модель будет работать с новыми, ранее не виденными данными.
          """)
        st.markdown("""
          **Данные тестового набора (X_test)**.
          Тестовый набор данных содержит информацию о признаках, используемых для оценки модели.
          """)
        st.dataframe(X_test.head(15), use_container_width=True)
        st.markdown("""
          **Целевая переменная (y_test)**.
          Целевая переменная представляет собой значения, которые модель пытается предсказать.
          """)
        st.dataframe(pd.DataFrame(y_test.head(15)).T)

    st.markdown(
        """
        # Моделирование
        ## Работа с несбалансированными данными
        Обратите внимание, что у нас есть несбалансированный набор данных, в котором большинство наблюдений относятся к одному типу ('NO'). В нашем случае, например, примерно 84% наблюдений имеют метку 'No', а только 16% - 'Yes', что делает этот набор данных несбалансированным.
        
        Для работы с такими данными необходимо принять определенные меры, иначе производительность нашей модели может существенно пострадать. В этом разделе я рассмотрю два подхода к решению этой проблемы.
        
        ### Увеличение числа примеров меньшинства или уменьшение числа примеров большинства
        В несбалансированных наборах данных основная проблема заключается в том, что данные сильно искажены, т.е. количество наблюдений одного класса значительно превышает количество наблюдений другого. Поэтому в этом подходе мы либо увеличиваем количество наблюдений для класса-меньшинства (oversampling), либо уменьшаем количество наблюдений для класса-большинства (undersampling).
        
        Стоит отметить, что в нашем случае количество наблюдений и так довольно мало, поэтому более подходящим будет метод увеличения числа примеров.
        
        Ниже я использовал технику увеличения числа примеров, известную как SMOTE (Synthetic Minority Oversampling Technique), которая случайным образом создает некоторые "синтетические" инстансы для класса-меньшинства, чтобы данные по обоим классам стали более сбалансированными.
        
        Важно использовать SMOTE до шага кросс-валидации, чтобы избежать переобучения модели, как это бывает при выборе признаков.
        
        ###  Выбор правильной метрики оценки
        Еще один важный аспект при работе с несбалансированными классами - это выбор правильных оценочных метрик.
        
        Следует помнить, что точность (accuracy) не является хорошим выбором. Это связано с тем, что из-за искажения данных даже алгоритм, всегда предсказывающий класс-большинство, может показать высокую точность. Например, если у нас есть 20 наблюдений одного типа и 980 другого, классификатор, предсказывающий класс-большинство, также достигнет точности 98%, но это не будет полезной информацией.
        
        В таких случаях мы можем использовать другие метрики, такие как:
        
        - **Точность (Precision)** — (истинно положительные)/(истинно положительные + ложно положительные)
        - **Полнота (Recall)** — (истинно положительные)/(истинно положительные + ложно отрицательные)
        - **F1-Score** — гармоническое среднее точности и полноты
        - **AUC ROC** — ROC-кривая, график между чувствительностью (Recall) и (1-specificity) (Специфичность=Точность)
        - **Матрица ошибок** — отображение полной матрицы ошибок
        """
    )

    st.markdown(
        r"""
        ### Логистическая регрессия
        Логистическая регрессия — это статистический метод анализа, используемый для моделирования зависимости дихотомической переменной (целевой переменной с двумя возможными исходами) от одного или нескольких предикторов (независимых переменных). Основное отличие логистической регрессии от линейной заключается в том, что первая предсказывает вероятность наступления события, используя логистическую функцию, что делает её идеальной для классификационных задач.
        
        #### Математическая модель
        
        ##### Логистическая функция (Сигмоид)
        Основой логистической регрессии является логистическая функция, также известная как сигмоид. Она описывается следующей формулой:
        
        $$
        P(y=1|X) = \frac{1}{1 + e^{-z}} 
        $$
        
        где $ z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n $ — линейная комбинация входных переменных $ X $ (независимых переменных) и коэффициентов модели $ \beta $ (включая свободный член $ \beta_0 $ и коэффициенты при переменных $ \beta_1, \beta_2, ..., \beta_n $).
        
        ##### Интерпретация коэффициентов
        Коэффициенты в логистической регрессии интерпретируются через шансы (odds) и логарифм шансов:
        
        - **Шансы**: Вероятность того, что событие произойдет, деленная на вероятность того, что событие не произойдет.
        
        $$ \text{odds} = \frac{P(y=1|X)}{1 - P(y=1|X)} = e^z $$
        
        - **Логарифм шансов**:
        
        $$\log(\text{odds}) = z = \beta_0 + \beta_1x_1 + ... + \beta_nx_n $$
        
        Каждый коэффициент $\beta_i $ показывает, как изменится логарифм шансов, если соответствующая переменная увеличится на одну единицу, при условии, что все остальные переменные остаются неизменными.
        
        #### Регуляризация
        
        Регуляризация используется для предотвращения переобучения путем добавления штрафа за слишком большие значения коэффициентов к функции потерь:
        
        - **L1-регуляризация (Lasso)**:
        
          Штрафует сумму абсолютных значений коэффициентов. Это может привести к обнулению некоторых коэффициентов, что делает модель разреженной и может помочь в отборе признаков.
        
        - **L2-регуляризация (Ridge)**:
        
          Штрафует сумму квадратов коэффициентов, что предотвращает их слишком большое увеличение и помогает уменьшить переобучение, но не делает модель разреженной.
        
        #### Оценка модели
        
        Для оценки качества модели логистической регрессии часто используют ROC AUC (Area Under the Receiver Operating Characteristics Curve). Эта метрика
        
         помогает оценить, насколько хорошо модель может различать два класса (например, положительный и отрицательный).
        
        - **ROC AUC**:
          - Чем ближе значение ROC AUC к 1, тем лучше модель различает два класса.
          - Значение 0.5 говорит о том, что модель работает не лучше случайного гадания.
        """
    )
    try:
        # def test1():
        # return joblib.load(str(current_dir / "models" / 'logistic_regression_model.joblib'))
        lin_reg = test1()
    except Exception as e:
        lin_reg = LogisticRegression(C=0.01, penalty='l2', solver='liblinear')
        lin_reg.fit(X_train, y_train)
        joblib.dump(lin_reg, str(current_dir / "models" / 'logistic_regression_model.joblib'))

    print_model_adequacy_section(current_dir)
    tab1, tab2 = st.tabs(["Результаты модели на зависимых данных", "Результаты модели на независимых данных", ])
    with tab1:
        score(lin_reg, X_train, y_train, X_test, y_test, train=True)
    with tab2:
        score(lin_reg, X_train, y_train, X_test, y_test, train=False)

    st.markdown(
        """
        ### Результаты
        
        На основе представленных результатов обучения и тестирования модели логистической регрессии, а также визуализаций ROC кривой и нормализованной матрицы ошибок, можно сделать следующие выводы:
        
        1. **Кривая ROC**:
        - Значения AUC (Area Under Curve) для обеих кривых ROC составляют 0.88 для обучающей выборки и 0.85 для тестовой выборки, что указывает на высокую способность модели различать классы. Это хороший результат, поскольку AUC значительно выше 0.5, что означало бы отсутствие дискриминационной способности у модели. Кривые ROC значительно выше диагонали, что также указывает на эффективность модели.
        
        2. **Матрица ошибок**:
        - Нормализованные матрицы ошибок для обучающего и тестового набора данных показывают, что модель имеет высокий процент истинно положительных результатов (True Positive Rate) по отношению к негативному классу, что соответствует высокому показателю Recall для класса 0.
        - Однако Recall для класса 1 (положительного класса) остается низким, что указывает на то, что значительное количество положительных случаев ошибочно классифицированы как негативные (False Negative).
        
        3. **Отчет по классификации**:
        - Точность модели (Accuracy) составляет 77.65% для обучающей выборки и 73.66% для тестовой выборки.
        - Precision для положительного класса (1) ниже, чем для негативного класса (0), что может указывать на осторожность модели в предсказании положительного класса.
        - Recall для положительного класса значительно ниже, чем для негативного, что подтверждает наблюдения из матрицы ошибок.
        
        4. **Интерпретация результатов**:
        - Модель демонстрирует хорошее общее качество предсказаний, но имеет тенденцию к лучшему выявлению негативных исходов (класс 0) в ущерб положительным (класс 1).
        - В контексте, например, кредитного скоринга это означало бы, что модель более консервативна и склонна к минимизации риска, за счет увеличения количества ложно-отрицательных результатов, тем самым возможно упуская потенциально надежных клиентов.
        - Возможно, потребуется пересмотреть порог классификации или внести корректировки в процесс подготовки данных и отбора признаков для улучшения предсказательной способности модели по отношению к положительному классу.
        """
    )

    with st.form("Ввод данных клиента"):
        st.subheader('Введите параметры для прогноза')
        st.markdown("Введите параметры пассажира для прогнозирования выживаемости:")

        # Используем option_menu для категориальных переменных
        Pclass = option_menu('Класс билета', sorted(df['Pclass'].unique()), icons=['star', 'star-half', 'star-fill'], menu_icon="person-badge", default_index=0, orientation='horizontal')
        Sex = option_menu('Пол', ['male', 'female'], icons=['gender-male', 'gender-female'], menu_icon="people", default_index=1, orientation='horizontal')
        Embarked = option_menu('Порт посадки', ['Cherbourg', 'Queenstown', 'Southampton'], icons=['globe-asia-australia', 'globe-americas', 'globe-central-south-asia'], menu_icon="house", default_index=2, orientation='horizontal')
        Age = st.slider('Возраст', 0, 80, 25)
        SibSp = st.number_input('Количество братьев и сестер / супругов на борту', min_value=0, max_value=20, value=3)
        Parch = st.number_input('Количество родителей / детей на борту', min_value=0, max_value=20, value=2)
        Fare = st.number_input('Стоимость билета', min_value=0.0, max_value=600.0, value=263.1, step=0.1)

        input_data = {
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Pclass_1': bool(Pclass == 1),
            'Pclass_2': bool(Pclass == 2),
            'Pclass_3': bool(Pclass == 3),
            'Sex_male': bool(Sex == 'male'),
            'Embarked_Cherbourg': bool(Embarked == 'Cherbourg'),
            'Embarked_Queenstown': bool(Embarked == 'Queenstown'),
            'Embarked_Southampton': bool(Embarked == 'Southampton')
        }
        if st.form_submit_button("Прогнозировать", type='primary', use_container_width=True):
            input_df = pd.DataFrame([input_data])
            # Преобразование категориальных переменных в числовые с использованием pd.get_dummies
            input_df = pd.get_dummies(input_df, drop_first=True)
            # Прогнозирование
            prediction = lin_reg.predict_proba(input_df).round(2) * 100
            prediction = prediction[0][1]
            if prediction < 40:
                st.warning(f"Прогноз успешно выполнен! Вероятность выживания: {prediction:.2f}%")
                color = 'darkorange'
            else:
                st.success(f"Прогноз успешно выполнен! Вероятность выживания: {prediction:.2f}%")
                color = 'darkgreen'

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}},
                title={"text": "Прогнозируемая вероятность выживания"}
            ))
            fig.update_layout(paper_bgcolor="#f0f2f6", font={'color': color, 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()
