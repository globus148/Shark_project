from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, current_user, logout_user, login_user
import smtplib
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from random import randint
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import torch
import os
import nltk
import re
from nltk.tokenize import word_tokenize
import number_converter
import random
import time
import qrcode
import io
import base64
from flask import send_file
import pytz
from sqlalchemy import func
from sqlalchemy.orm import Session

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'hardsecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dbase.db'
app.config['TEMPLATES_AUTO_RELOAD'] = True
db = SQLAlchemy(app)

IMAGES_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

CODE = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Глобальные переменные для моделей
paraphraser_model = None
paraphraser_tokenizer = None
headliner_model = None
headliner_tokenizer = None


class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    login_time = db.Column(db.DateTime)
    logout_time = db.Column(db.DateTime)
    session_duration = db.Column(db.Integer, nullable=False, default=0)


class Achievement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    description = db.Column(db.Text)
    icon = db.Column(db.String(100))  # Путь к иконке достижения
    condition = db.Column(db.String(100))  # Условие для получения (например: 'messages_count:10')


class UserAchievement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    achievement_id = db.Column(db.Integer, db.ForeignKey('achievement.id'))
    earned_at = db.Column(db.DateTime, default=datetime.now)
    __table_args__ = (db.UniqueConstraint('user_id', 'achievement_id', name='unique_user_achievement'),)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(256))
    name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime)


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'))
    content = db.Column(db.Text)
    is_user = db.Column(db.Boolean)
    timestamp = db.Column(db.DateTime, default=datetime.now)


class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'))
    created_at = db.Column(db.DateTime, default=datetime.now)


class Term(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    term = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)
    __table_args__ = (db.UniqueConstraint('user_id', 'term', name='unique_user_term'),)


def initialize_database():
    """Функция для инициализации базы данных"""
    if not os.path.exists('instance/dbase.db'):  # Проверяем, существует ли БД
        with app.app_context():
            db.create_all()

            # Добавляем начальные достижения только если их нет
            if Achievement.query.count() == 0:
                achievements = [
                    Achievement(
                        name="Новичок",
                        description="Отправить первое сообщение",
                        icon="achievement1.png",
                        condition="messages_count:1"
                    ),
                    Achievement(
                        name="Активный пользователь",
                        description="Отправить 10 сообщений",
                        icon="achievement2.png",
                        condition="messages_count:10"
                    ),
                    Achievement(
                        name="Ветеран",
                        description="Отправить 100 сообщений",
                        icon="achievement3.png",
                        condition="messages_count:100"
                    ),
                    Achievement(
                        name="Коллекционер",
                        description="Добавить 5 терминов",
                        icon="achievement4.png",
                        condition="terms_count:5"
                    ),
                    Achievement(
                        name="Любитель закладок",
                        description="Добавить 3 сообщения в избранное",
                        icon="achievement5.png",
                        condition="favorites_count:3"
                    ),
                    Achievement(
                        name="Долгожитель",
                        description="Провести на сайте 60 минут",
                        icon="achievement6.png",
                        condition="session_time:60"
                    ),
                    Achievement(
                        name="Неделя с нами",
                        description="Зарегистрироваться 7 дней назад",
                        icon="achievement7.png",
                        condition="registration_days:7"
                    )
                ]
                db.session.add_all(achievements)
                db.session.commit()
            print("База данных инициализирована")


with app.app_context():
    db.create_all()


def send_email(message, adress):
    sender = "gulovskiu@gmail.com"
    password = "nwjcfhzloyluetwv"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    try:
        server.login(sender, password)
        msg = MIMEText(message)
        msg["Subject"] = "Подтверждение почты"
        server.sendmail(sender, adress, msg.as_string())
        return "The message was sent successfully!"
    except Exception as _ex:
        return f"{_ex}\nCheck your login or password please!"


def check_achievements(user_id):
    user = User.query.get(user_id)
    if not user:
        return

    # Получаем статистику пользователя
    messages_count = Message.query.join(Chat).filter(Chat.user_id == user_id).count()
    chats_count = Chat.query.filter_by(user_id=user_id).count()
    favorites_count = Favorite.query.filter_by(user_id=user_id).count()
    terms_count = Term.query.filter_by(user_id=user_id).count()
    session_time = db.session.query(func.sum(UserActivity.session_duration)).filter_by(user_id=user_id).scalar() or 0

    # Все возможные достижения
    achievements = Achievement.query.all()

    for achievement in achievements:
        # Проверяем, есть ли уже это достижение у пользователя
        if not UserAchievement.query.filter_by(user_id=user_id, achievement_id=achievement.id).first():
            condition_type, condition_value = achievement.condition.split(':')
            condition_value = int(condition_value)

            earned = False
            if condition_type == 'messages_count' and messages_count >= condition_value:
                earned = True
            elif condition_type == 'chats_count' and chats_count >= condition_value:
                earned = True
            elif condition_type == 'favorites_count' and favorites_count >= condition_value:
                earned = True
            elif condition_type == 'terms_count' and terms_count >= condition_value:
                earned = True
            elif condition_type == 'session_time' and session_time >= condition_value:
                earned = True
            elif condition_type == 'registration_days' and (
                    datetime.now() - user.created_at).days >= condition_value:
                earned = True

            if earned:
                user_achievement = UserAchievement(
                    user_id=user_id,
                    achievement_id=achievement.id
                )
                db.session.add(user_achievement)

    db.session.commit()


def correct_spelling(text, max_length=4000):
    MODEL_NAME = 'UrukHan/t5-russian-spell'
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    task_prefix = "Spell correct: "
    input_sequences = [text] if type(text) is not list else text

    encoded = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predicts = model.generate(**encoded.to(device), max_length=max_length)
    results = tokenizer.batch_decode(predicts, skip_special_tokens=True)
    return results[0] if isinstance(text, str) else results


def load_headliner():
    tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_sum_gazeta", legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/rut5_base_sum_gazeta").to(device)
    return model, tokenizer


def should_generate_headline(text):
    """Определяет, нужно ли генерировать заголовок"""
    words = word_tokenize(text)
    return len(words) >= 20  # Только для текстов от 20 слов


def generate_short_headline(text, model, tokenizer):
    """Генерирует краткий осмысленный заголовок"""
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        max_length=1000,
        truncation=True
    ).input_ids.to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=20,
        min_length=5,
        num_beams=4,
        repetition_penalty=3.0,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    headline = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    headline = headline.split(".")[0].strip()
    words = [w for w in headline.split() if len(w) > 2][:7]  # Фильтрация коротких слов
    return " ".join(words).capitalize()


def is_meaningful_headline(headline):
    """Проверяет качество заголовка"""
    words = headline.lower().split()
    if len(words) < 3:
        return False
    if len(set(words)) < len(words) / 2:
        return False
    return True


def load_paraphraser():
    MODEL_PATH = "cointegrated/rut5-base-paraphraser"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, legacy=True)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
        return model, tokenizer
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        exit()


def paraphrase_text(text, model, tokenizer, max_length=4000):
    # Добавляем небольшой случайный разброс к параметрам
    temp = 1.7 + random.uniform(-0.2, 0.2)  # 1.5-1.9
    top_k = random.randint(55, 65)  # 55-65
    top_p = 0.92 + random.uniform(-0.02, 0.02)  # 0.9-0.94

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            do_sample=True,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=2.5,
            early_stopping=True
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_paraphrase_output(decoded_output)


def clean_paraphrase_output(text):
    text = re.sub(r'^(перефразируй:|перефразируя:|подробнее:|дополнительно:)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def generate_qr_base64(data):
    """Генерирует QR-код и возвращает его как base64 строку"""
    img_byte_arr = generate_qr_code(data)
    return "data:image/png;base64," + base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')


def generate_qr_code(data):
    """Генерирует QR-код из переданных данных"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Сохраняем изображение в байтовый поток
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr


@app.route('/generate_qr')
@login_required
def generate_qr():
    data = request.args.get('data', '')
    if not data:
        return "No data provided", 400

    img_byte_arr = generate_qr_code(data)
    return send_file(img_byte_arr, mimetype='image/png')


@app.route('/stats')
@login_required
def stats():
    try:
        # 1. Определяем часовой пояс (из сессии или по умолчанию Москва)
        user_tz = session.get('user_timezone', 'Europe/Moscow')
        timezone = pytz.timezone(user_tz)

        # 2. Получаем все сообщения пользователя с конвертацией времени
        messages = db.session.query(
            Message.id,
            Message.content,
            Message.timestamp
        ).join(Chat).filter(
            Chat.user_id == current_user.id
        ).all()

        # 3. Подсчёт сообщений по часам (с учётом часового пояса)
        hour_counts = {h: 0 for h in range(24)}
        for msg in messages:
            if msg.timestamp:
                local_time = msg.timestamp.replace(tzinfo=pytz.UTC).astimezone(timezone)
                hour = local_time.hour
                hour_counts[hour] += 1

        # 4. Подготовка данных для графика
        time_labels = [f"{h:02d}:00" for h in range(24)]
        message_data = [hour_counts[h] for h in range(24)]

        # 5. Топ-3 слова (без изменений)
        top_words = db.session.query(
            Term.term,
            func.count(Term.term).label('count')
        ).filter_by(
            user_id=current_user.id
        ).group_by(
            Term.term
        ).order_by(
            func.count(Term.term).desc()
        ).limit(3).all()

        total_time = db.session.query(func.sum(UserActivity.session_duration)) \
                         .filter_by(user_id=current_user.id).scalar() or 0

        return render_template('stats.html',
                               time_labels=time_labels,
                               message_data=message_data,
                               top_words=top_words,
                               registration_date=current_user.created_at.strftime('%d.%m.%Y %H:%M'),
                               total_time=total_time,
                               last_login=current_user.last_login.strftime(
                                   '%d.%m.%Y %H:%M') if current_user.last_login else 'Нет данных'
                               )

    except Exception as e:
        app.logger.error(f"Error in stats route: {str(e)}")
        return render_template('error.html', message="Ошибка при формировании статистики"), 500


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.route('/new_password', methods=['GET', 'POST'])
def new_password():
    email = request.args.get('email')
    if request.method == 'POST':
        psw1 = request.form['psw1']
        psw2 = request.form['psw2']
        if not psw1 or not psw2:
            return render_template('new_password.html', err='Заполните все поля', psw1=psw1, psw2=psw2)
        if len(psw1) < 8:
            return render_template('new_password.html', err='Пароль слишком маленький', psw1=psw1, psw2=psw2)
        if psw1 != psw2:
            return render_template('new_password.html', err='Пароли различаются', psw1=psw1, psw2=psw2)
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(psw1)
            db.session.commit()
        return redirect(url_for('login'))
    return render_template('new_password.html', psw1='', psw2='')


@app.route('/send', methods=['GET', 'POST'])
def send():
    email = request.args.get('email')
    name = request.args.get('name')
    password = request.args.get('password')
    reset = int(request.args.get('reset'))
    if request.method == 'POST':
        global CODE
        email = request.form['mail']
        unic_code = request.form['unik_cod']
        if email == '':
            return render_template('check_email.html', flag=False, err="Введите почту")
        if unic_code == '':
            CODE = randint(1000, 9999)
            message = f'''Здравствуйте!
            Вы получили это письмо, потому что мы получили запрос на подтверждения почты для вашей учетной записи.
            Специальный код: {CODE}
            Если вы не запрашивали код, никаких дальнейших действий не требуется.

            С Уважением,
            команда "Вот они слева направо".'''
            send_email(message=message, adress=email)
            return render_template('check_email.html', flag=True, err="Код отправлен", email=email)
        else:
            if int(unic_code) == CODE:
                if reset == 0:
                    CODE = 0
                    session['email'] = email
                    new_user = User(name=name, email=email, password=generate_password_hash(password))
                    db.session.add(new_user)
                    db.session.commit()
                    login_user(new_user)
                    return redirect(url_for('index'))
                elif reset == 1:
                    return redirect(url_for('new_password', email=email))
    return render_template('check_email.html', flag=False, err="", email=email)


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        if form_type == 'register':
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            errors = []
            if not all([name, email, password, confirm_password]):
                errors.append("Все поля обязательны для заполнения")
            elif len(password) < 8:
                errors.append("Пароль должен содержать минимум 8 символов")
            elif password != confirm_password:
                errors.append("Пароли не совпадают")
            elif not email or '@' not in email:
                errors.append("Введите корректный email")
            elif User.query.filter_by(email=email).first():
                errors.append("Пользователь с таким email уже существует")
            if errors:
                return render_template('entrance.html', register_errors=errors, name=name,
                                       email=email, password=password, confirm_password=confirm_password,
                                       active_form='register')
            return redirect(url_for('send', name=name, email=email, password=password, reset=0))
        elif form_type == 'login':
            email = request.form.get('email')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False
            user = User.query.filter_by(email=email).first()
            if not user or not check_password_hash(user.password, password):
                return render_template('entrance.html', login_errors=["Неверный email или пароль"],
                                       login_email=email,
                                       login_password=password,
                                       active_form='login')
            login_user(user, remember=remember)
            user.last_login = datetime.datetime.now()
            activity = UserActivity(user_id=user.id, login_time=datetime.datetime.now())
            db.session.add(activity)
            db.session.commit()
            return redirect(url_for('index'))
    return render_template('entrance.html', active_form='register')


def init_models():
    """Инициализация моделей при запуске приложения"""
    global paraphraser_model, paraphraser_tokenizer, headliner_model, headliner_tokenizer

    print("Инициализация моделей...")
    paraphraser_model, paraphraser_tokenizer = load_paraphraser()
    headliner_model, headliner_tokenizer = load_headliner()
    print("Модели успешно загружены")


@app.route('/main', methods=['GET', 'POST'])
@login_required
def index():
    global paraphraser_model, paraphraser_tokenizer, headliner_model, headliner_tokenizer
    if request.method == 'POST':
        try:
            start_time = time.time()
            print(f"\n[{datetime.now()}] Получено новое сообщение")

            chat_id = request.form.get('chat_id')

            # Получение сообщения
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                if file and file.filename.endswith('.txt'):
                    message_content = file.read().decode('utf-8').strip()
                else:
                    message_content = request.form['message'].strip()
            else:
                message_content = request.form['message'].strip()

            print(f"[{datetime.now()}] Текст сообщения: {message_content[:50]}...")

            # Создание нового чата при необходимости
            if chat_id == 'new':
                new_chat = Chat(user_id=current_user.id, title=f"Чат {datetime.now().strftime('%d.%m %H:%M')}")
                db.session.add(new_chat)
                db.session.commit()
                chat_id = new_chat.id
                print(f"[{datetime.now()}] Создан новый чат с ID: {chat_id}")

            # Сохранение сообщения пользователя
            user_message = Message(
                chat_id=int(chat_id),
                content=message_content,
                is_user=True,
                timestamp=datetime.utcnow()
            )
            db.session.add(user_message)

            # Проверка инициализации моделей
            if None in [paraphraser_model, paraphraser_tokenizer, headliner_model, headliner_tokenizer]:
                init_models()

            # AJAX обработка
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                db.session.commit()
                return jsonify({
                    'success': True,
                    'user_message': message_content,
                    'waiting_message': "Ожидайте...",
                    'chat_id': chat_id
                })

            waiting_message = Message(
                chat_id=int(chat_id),
                content="Ожидайте...",
                is_user=False,
                timestamp=datetime.utcnow()
            )
            db.session.add(waiting_message)
            db.session.commit()

            # Обработка текста
            processing_stages = {}

            # 1. Коррекция орфографии
            stage_start = time.time()
            corrected_text = correct_spelling(message_content)
            processing_stages['spell_correction'] = time.time() - stage_start

            # 2. Парафраз
            stage_start = time.time()
            paraphrased_text = paraphrase_text(corrected_text, paraphraser_model, paraphraser_tokenizer)
            processing_stages['paraphrasing'] = time.time() - stage_start

            # 3. Обработка чисел
            stage_start = time.time()
            final_text = number_converter.replace_numbers_with_digits(paraphrased_text)[0]
            processing_stages['number_processing'] = time.time() - stage_start

            # Формирование ответа
            if should_generate_headline(corrected_text):
                stage_start = time.time()
                headline = generate_short_headline(corrected_text, headliner_model, headliner_tokenizer)
                if is_meaningful_headline(headline):
                    ai_response = f'{headline}\n\n{final_text}'
                    processing_stages['headline_generation'] = time.time() - stage_start
                else:
                    ai_response = final_text
            else:
                ai_response = final_text

            # AJAX ответ
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                db.session.delete(Message.query.filter_by(content="Ожидайте...", chat_id=chat_id).first())
                ai_message = Message(
                    chat_id=int(chat_id),
                    content=ai_response,
                    is_user=False,
                    timestamp=datetime.utcnow()
                )
                db.session.add(ai_message)
                db.session.commit()

                return jsonify({
                    'success': True,
                    'ai_response': ai_response,
                    'processing_time': {k: f"{v:.2f} сек" for k, v in processing_stages.items()}
                })

            # Обычный ответ
            db.session.delete(waiting_message)
            ai_message = Message(
                chat_id=int(chat_id),
                content=ai_response,
                is_user=False,
                timestamp=datetime.utcnow()
            )
            db.session.add(ai_message)
            db.session.commit()

            print(f"\n[{datetime.now()}] Обработка завершена за {time.time() - start_time:.2f} сек")
            return redirect(url_for('index', chat_id=chat_id))

        except Exception as e:
            print(f"[{datetime.now()}] Ошибка: {str(e)}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': str(e)}), 500
            raise

    # GET обработчик
    chat_id = request.args.get('chat_id')
    user_chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()

    if not chat_id and user_chats:
        chat_id = user_chats[0].id

    messages = []
    if chat_id:
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp.asc()).all()

    qr_data = f"chat:{chat_id}" if chat_id else ""
    qr_code = generate_qr_base64(qr_data) if qr_data else None

    total_time = db.session.query(func.sum(UserActivity.session_duration)) \
                     .filter_by(user_id=current_user.id).scalar() or 0

    return render_template('index.html',
                           chats=user_chats,
                           messages=messages,
                           current_chat_id=chat_id,
                           qr_code=qr_code,
                           total_time=total_time,  # Добавьте это
                           is_ajax=request.headers.get('X-Requested-With') == 'XMLHttpRequest')


@app.route('/favorite/<int:message_id>', methods=['POST'])
@login_required
def add_to_favorite(message_id):
    favorite = Favorite.query.filter_by(user_id=current_user.id, message_id=message_id).first()
    if not favorite:
        new_favorite = Favorite(
            user_id=current_user.id,
            message_id=message_id
        )
        db.session.add(new_favorite)
        db.session.commit()
        return jsonify({'success': True, 'action': 'added'})
    else:
        db.session.delete(favorite)
        db.session.commit()
        return jsonify({'success': True, 'action': 'removed'})


@app.route('/new_chat', methods=['GET', 'POST'])
@login_required
def new_chat():
    if request.method == 'POST':
        title = request.form.get('title')
        if not title:
            title = f"Чат {datetime.datetime.now().strftime('%d.%m %H:%M')}"
        new_chat = Chat(
            user_id=current_user.id,
            title=title
        )
        db.session.add(new_chat)
        db.session.commit()

        return jsonify({
            'success': True,
            'chat_id': new_chat.id,
            'chat_title': new_chat.title
        })

    return render_template('index.html')


@app.route('/edit_chat/<int:chat_id>', methods=['POST'])
@login_required
def edit_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    if chat.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    new_title = request.form.get('title')
    if new_title:
        chat.title = new_title
        db.session.commit()
        return jsonify({'success': True, 'chat_title': chat.title})
    return jsonify({'success': False, 'error': 'No title provided'}), 400


@app.route('/delete_chat/<int:chat_id>', methods=['POST'])
@login_required
def delete_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    if chat.user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    Message.query.filter_by(chat_id=chat_id).delete()
    db.session.delete(chat)
    db.session.commit()

    return jsonify({'success': True})


@app.route('/profile')
@login_required
def profile():
    favorites = Favorite.query.filter_by(user_id=current_user.id).all()
    favorite_messages = [Message.query.get(fav.message_id) for fav in favorites if Message.query.get(fav.message_id)]
    terms = Term.query.filter_by(user_id=current_user.id).order_by(Term.created_at.desc()).all()

    # Вычисляем общее время в системе
    total_seconds = db.session.query(func.sum(UserActivity.session_duration)) \
                        .filter_by(user_id=current_user.id).scalar() or 0
    hours = total_seconds // 60
    minutes = total_seconds % 60
    time_spent = f"{hours} ч {minutes} мин"

    return render_template('profile.html',
                           name=current_user.name,
                           email=current_user.email,
                           image='static/image/profile_rev.png',
                           favorite_messages=favorite_messages,
                           terms=terms,
                           time_spent=time_spent)


@app.route('/edit_name', methods=['POST'])
@login_required
def edit_name():
    new_name = request.form.get('new_name')
    if not new_name:
        return redirect(url_for('profile'))
    current_user.name = new_name
    db.session.commit()
    return redirect(url_for('profile'))


@app.route('/remove_favorite/<int:message_id>', methods=['POST'])
@login_required
def remove_favorite(message_id):
    favorite = Favorite.query.filter_by(user_id=current_user.id, message_id=message_id).first()
    if favorite:
        db.session.delete(favorite)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 404


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Инициализация БД при запуске
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    with app.app_context():
        initialize_database()

if __name__ == '__main__':
    init_models()
    app.run(debug=True)
