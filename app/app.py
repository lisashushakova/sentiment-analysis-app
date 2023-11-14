import sys
from ast import literal_eval

import pandas as pd
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QPlainTextEdit, \
    QLabel, QComboBox
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from misc import mean_vectorizer, tfidf_vectorizer, preprocess_text, select_vocabulary, nn_vectorizer, DatasetType, \
    ClassifierType, WordVectorizationType, TextVectorizationType, Word2VecModelType, precision, recall, f1

from keras.models import load_model

STORE_TEXT_COUNT = 1000


# Подкласс QMainWindow для настройки главного окна приложения
class MainWindow(QMainWindow):

    def on_dataset_changed(self, value):
        self.result_widget.setText("Загрузка набора данных...\n")
        app.processEvents()
        if value == DatasetType.BANKS.value:
            self.dataset_type = DatasetType.BANKS
        elif value == DatasetType.MEDIA.value:
            self.dataset_type = DatasetType.MEDIA
        elif value == DatasetType.TWITTER.value:
            self.dataset_type = DatasetType.TWITTER
        self.load_words()
        print(f"Changed dataset type to {self.dataset_type}")
        self.vectorizer = None
        self.classifier = None
        self.result_widget.setText("Набор данных загружен!\n")
        app.processEvents()

    def on_classifier_changed(self, value):
        self.word_vectorization_selection_combo_box.clear()
        if value == ClassifierType.LOGISTIC_REGRESSION.value:
            self.classifier_type = ClassifierType.LOGISTIC_REGRESSION
            self.word_vectorization_selection_combo_box.addItems([e.value for e in WordVectorizationType])
            self.word2vec_model_type = None
            self.text_vectorization_selection_widget.show()
        elif value == ClassifierType.NAIVE_BAYESIAN.value:
            self.classifier_type = ClassifierType.NAIVE_BAYESIAN
            self.word_vectorization_selection_combo_box.addItems([e.value for e in WordVectorizationType])
            self.word2vec_model_type = None
            self.text_vectorization_selection_widget.show()
        elif value == ClassifierType.CNN.value:
            self.classifier_type = ClassifierType.CNN
            self.word_vectorization_selection_combo_box.addItems([WordVectorizationType.W2V.value])
            self.word_vectorization_type = WordVectorizationType.W2V
            self.word2vec_model_type = Word2VecModelType.NAVEC
            self.text_vectorization_selection_widget.hide()
        elif value == ClassifierType.RNN.value:
            self.classifier_type = ClassifierType.RNN
            self.word_vectorization_selection_combo_box.addItems([WordVectorizationType.W2V.value])
            self.word_vectorization_type = WordVectorizationType.W2V
            self.word2vec_model_type = Word2VecModelType.NAVEC
            self.text_vectorization_selection_widget.hide()
        print(f"Changed classifier type to {self.classifier_type}")
        self.vectorizer = None
        self.classifier = None
        self.result_widget.setText("\n")
        app.processEvents()

    def on_word_vectorization_changed(self, value):
        if value == WordVectorizationType.OHE.value:
            self.text_vectorization_type = WordVectorizationType.OHE
        elif value == WordVectorizationType.W2V.value:
            self.word_vectorization_type = WordVectorizationType.W2V
        print(f"Changed word vectorization type to {self.dataset_type}")
        self.vectorizer = None
        self.classifier = None
        self.result_widget.setText("\n")
        app.processEvents()

    def on_text_vectorization_changed(self, value):
        if value == TextVectorizationType.BOW.value:
            self.text_vectorization_type = TextVectorizationType.BOW
        elif value == TextVectorizationType.TF_IDF.value:
            self.text_vectorization_type = TextVectorizationType.TF_IDF
        print(f"Changed text vectorization type to {self.dataset_type}")
        self.vectorizer = None
        self.classifier = None
        self.result_widget.setText("\n")
        app.processEvents()

    def on_word2vec_model_changed(self, value):
        if value == Word2VecModelType.NAVEC.value:
            self.word2vec_model_type = Word2VecModelType.NAVEC
        elif value == Word2VecModelType.TRAINED.value:
            self.word2vec_model_type = Word2VecModelType.TRAINED
        print(f"Changed word2vec model type to {self.word2vec_model_type}")
        self.vectorizer = None
        self.classifier = None
        self.result_widget.setText("\n")
        app.processEvents()

    def on_text_changed(self):
        self.answer = None
        self.result_widget.setText("\n")
        app.processEvents()


    def on_start(self):
        answer_str = f"\n(Правильный ответ: {self.answer})" if self.answer else ""

        MAX_WORDS = 5000

        input_text = self.text_edit_widget.toPlainText()
        preprocessed_input_text = preprocess_text(input_text)

        if self.vectorizer is None:
            if self.classifier_type in [ClassifierType.LOGISTIC_REGRESSION, ClassifierType.NAIVE_BAYESIAN]:
                if self.word_vectorization_type == WordVectorizationType.OHE:
                    if self.text_vectorization_type == TextVectorizationType.BOW:
                        self.vectorizer = CountVectorizer(max_features=MAX_WORDS)
                    else:
                        self.vectorizer = TfidfVectorizer(max_features=MAX_WORDS)
                else:
                    self.result_widget.setText("Загрузка словаря...\n")
                    app.processEvents()
                    vocabulary, DIM = select_vocabulary(self.word2vec_model_type, self.dataset_type)
                    if self.text_vectorization_type == TextVectorizationType.BOW:
                        self.vectorizer = mean_vectorizer(vocabulary, DIM)
                    else:
                        self.vectorizer = tfidf_vectorizer(vocabulary, DIM)
            else:
                if self.dataset_type == DatasetType.TWITTER:
                    text_length = 20
                elif self.dataset_type == DatasetType.BANKS:
                    text_length = 400
                else:
                    text_length = 180
                self.vectorizer = nn_vectorizer(max_features=MAX_WORDS, length=text_length)
            self.result_widget.setText("Обновление векторизатора...\n")
            app.processEvents()
            if self.word_vectorization_type == WordVectorizationType.OHE:
                vectors = self.vectorizer.fit(self.data['Text']).transform(self.data['Text'])
            else:
                vectors = self.vectorizer.fit(self.data['PreprocessedText']).transform(self.data['PreprocessedText'])

        if self.classifier is None:
            x_train, x_test, y_train, y_test = train_test_split(vectors, self.data['Score'], test_size=0.2, random_state=42, shuffle=True)

            self.result_widget.setText("Загрузка классификатора...\n")
            app.processEvents()
            if self.classifier_type == ClassifierType.LOGISTIC_REGRESSION:
                self.classifier = LogisticRegression(random_state=42, max_iter=2000)
                self.classifier.fit(x_train, y_train)
            elif self.classifier_type == ClassifierType.NAIVE_BAYESIAN:
                if self.word_vectorization_type == WordVectorizationType.OHE:
                    self.classifier = MultinomialNB().fit(x_train, y_train)
                else:
                    self.classifier = GaussianNB().fit(x_train, y_train)
            else:
                dependencies = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                if self.classifier_type == ClassifierType.CNN:
                    model_part = 'cnn'
                else:
                    model_part = 'rnn'

                if self.word2vec_model_type == Word2VecModelType.NAVEC:
                    word2vec_model_part = 'navec'
                else:
                    word2vec_model_part = 'trained'

                if self.dataset_type == DatasetType.BANKS:
                    dataset_part = 'banks'
                elif self.dataset_type == DatasetType.MEDIA:
                    dataset_part = 'mass_media'
                else:
                    dataset_part = 'twitter'

                path = f'../models/saved/{model_part}_{word2vec_model_part}_{dataset_part}.keras'
                self.classifier = load_model(path, custom_objects=dependencies)


        if self.word_vectorization_type == WordVectorizationType.OHE:
            vectorized_input_text = self.vectorizer.transform([input_text])
        else:
            vectorized_input_text = self.vectorizer.transform([preprocessed_input_text])

        if self.classifier_type in [ClassifierType.LOGISTIC_REGRESSION, ClassifierType.NAIVE_BAYESIAN]:
            [[_, res]] = self.classifier.predict_proba(vectorized_input_text)
        else:
            [[res]] = self.classifier.predict(vectorized_input_text)

        self.result_widget.setText(
            f"Ответ классификатора: {['Отрицательный', 'Положительный'][round(res)]} ({res:3.2f}){answer_str}"
        )


    def load_words(self):
        if self.dataset_type == DatasetType.BANKS:
            load_preprocessed_dataset = "banks_preprocessed"
        elif self.dataset_type == DatasetType.MEDIA:
            load_preprocessed_dataset = "mass_media_balanced_preprocessed"
        elif self.dataset_type == DatasetType.TWITTER:
            load_preprocessed_dataset = "twitter_preprocessed"

        load_path = f"../data/preprocessed/{load_preprocessed_dataset}.csv"
        df = pd.read_csv(load_path)
        df['PreprocessedText'] = df['PreprocessedText'].apply(literal_eval)
        df = df.astype({'Score': object})
        mapping = {'Negative': 0, 'Positive': 1}
        df.replace({'Score': mapping}, inplace=True)
        self.data = df

    def on_randomize(self):
        r = self.data.sample()
        self.text_edit_widget.setPlainText(r['Text'].values[0])
        self.answer = ['Отрицательный', 'Положительный'][r['Score'].values[0]]

    def build_text_input_widget(self):

        text_input_layout = QHBoxLayout()
        text_input_widget = QWidget()
        text_input_widget.setLayout(text_input_layout)

        self.text_edit_widget = QPlainTextEdit()
        self.text_edit_widget.textChanged.connect(self.on_text_changed)
        text_input_layout.addWidget(self.text_edit_widget)

        buttons_block_layout = QVBoxLayout()
        buttons_block_widget = QWidget()
        buttons_block_widget.setLayout(buttons_block_layout)
        text_input_layout.addWidget(buttons_block_widget)

        randomize_button = QPushButton("Случайно")
        randomize_button.clicked.connect(self.on_randomize)
        buttons_block_layout.addWidget(randomize_button)

        start_button = QPushButton("Определить")
        start_button.clicked.connect(self.on_start)
        buttons_block_layout.addWidget(start_button)

        return text_input_widget

    def build_output_widget(self):

        output_layout = QVBoxLayout()
        output_widget = QWidget()
        output_widget.setLayout(output_layout)

        self.result_widget = QLabel(" \n ")
        self.result_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_layout.addWidget(self.result_widget)

        return output_widget

    def build_config_widget(self):

        config_layout = QVBoxLayout()
        config_widget = QWidget()
        config_widget.setLayout(config_layout)

        # DATASET SELECTION WIDGET

        dataset_selection_layout = QHBoxLayout()
        dataset_selection_widget = QWidget()
        dataset_selection_widget.setLayout(dataset_selection_layout)

        dataset_selection_label = QLabel("Набор данных")
        dataset_selection_layout.addWidget(dataset_selection_label)

        dataset_selection_combo_box = QComboBox()
        dataset_selection_combo_box.addItems([e.value for e in DatasetType])
        dataset_selection_combo_box.currentTextChanged.connect(self.on_dataset_changed)
        dataset_selection_layout.addWidget(dataset_selection_combo_box)

        config_layout.addWidget(dataset_selection_widget)

        # -------------------------



        # CLASSIFIER SELECTION WIDGET

        classifier_selection_layout = QHBoxLayout()
        classifier_selection_widget = QWidget()
        classifier_selection_widget.setLayout(classifier_selection_layout)

        classifier_selection_label = QLabel("Классификатор")
        classifier_selection_layout.addWidget(classifier_selection_label)

        classifier_selection_combo_box = QComboBox()
        classifier_selection_combo_box.addItems([e.value for e in ClassifierType])
        classifier_selection_combo_box.currentTextChanged.connect(self.on_classifier_changed)
        classifier_selection_layout.addWidget(classifier_selection_combo_box)

        config_layout.addWidget(classifier_selection_widget)

        # -------------------------



        # WORD VECTORIZATION SELECTION WIDGET

        word_vectorization_selection_layout = QHBoxLayout()
        word_vectorization_selection_widget = QWidget()
        word_vectorization_selection_widget.setLayout(word_vectorization_selection_layout)

        word_vectorization_selection_label = QLabel("Метод векторизации слов")
        word_vectorization_selection_layout.addWidget(word_vectorization_selection_label)

        self.word_vectorization_selection_combo_box = QComboBox()
        self.word_vectorization_selection_combo_box.addItems([e.value for e in WordVectorizationType])
        self.word_vectorization_selection_combo_box.currentTextChanged.connect(self.on_word_vectorization_changed)
        word_vectorization_selection_layout.addWidget(self.word_vectorization_selection_combo_box)

        config_layout.addWidget(word_vectorization_selection_widget)

        # -------------------------



        # TEXT VECTORIZATION SELECTION WIDGET

        text_vectorization_selection_layout = QHBoxLayout()
        self.text_vectorization_selection_widget = QWidget()
        self.text_vectorization_selection_widget.setLayout(text_vectorization_selection_layout)

        text_vectorization_selection_label = QLabel("Метод векторизации текста")
        text_vectorization_selection_layout.addWidget(text_vectorization_selection_label)

        text_vectorization_selection_combo_box = QComboBox()
        text_vectorization_selection_combo_box.addItems([e.value for e in TextVectorizationType])
        text_vectorization_selection_combo_box.currentTextChanged.connect(self.on_text_vectorization_changed)
        text_vectorization_selection_layout.addWidget(text_vectorization_selection_combo_box)

        config_layout.addWidget(self.text_vectorization_selection_widget)

        # -------------------------

        # WORD2VEC MODEL SELECTION WIDGET

        word2vec_model_selection_layout = QHBoxLayout()
        word2vec_model_selection_widget = QWidget()
        word2vec_model_selection_widget.setLayout(word2vec_model_selection_layout)

        word2vec_model_selection_label = QLabel("Модель Word2vec")
        word2vec_model_selection_layout.addWidget(word2vec_model_selection_label)

        word2vec_model_selection_combo_box = QComboBox()
        word2vec_model_selection_combo_box.addItems([e.value for e in Word2VecModelType])
        word2vec_model_selection_combo_box.currentTextChanged.connect(self.on_word2vec_model_changed)
        word2vec_model_selection_layout.addWidget(word2vec_model_selection_combo_box)

        config_layout.addWidget(word2vec_model_selection_widget)

        # -------------------------

        return config_widget

    def build_base_widget(self):
        base_layout = QVBoxLayout()
        base_widget = QWidget()
        base_widget.setLayout(base_layout)

        text_input_widget = self.build_text_input_widget()
        base_layout.addWidget(text_input_widget)

        output_widget = self.build_output_widget()
        base_layout.addWidget(output_widget)

        config_widget = self.build_config_widget()
        base_layout.addWidget(config_widget)

        return base_widget

    def __init__(self):
        super().__init__()

        self.dataset_type = DatasetType.BANKS
        self.classifier_type = ClassifierType.LOGISTIC_REGRESSION
        self.word_vectorization_type = WordVectorizationType.OHE
        self.text_vectorization_type = TextVectorizationType.BOW
        self.word2vec_model_type = None

        self.vectorizer = None
        self.classifier = None

        self.setWindowTitle("Анализ тональности текста")
        self.setFixedSize(QSize(400, 500))

        base_widget = self.build_base_widget()
        self.load_words()
        self.setCentralWidget(base_widget)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()