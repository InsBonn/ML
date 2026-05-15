import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from x_russian_cities import is_valid_russian_city, normalize_city, get_valid_city
from x_it_keywords import is_it_candidate, IT_KEYWORDS_SET, NON_IT_POSITIONS

# ===================================================================
# Загрузка модели (выполняется единожды при запуске скрипта)
# Семантическое сравнение текста вакансии и резюме
# ===================================================================

MODEL_NAME = "ai-forever/sbert_large_nlu_ru"

model = SentenceTransformer(MODEL_NAME)

# ===================================================================
# ФОРМАТИРОВАНИЕ ТЕКСТА РЕЗЮМЕ ДЛЯ НЕЙРОСЕТИ
#
# словарь с данными -> в структурированный текст -> в ИИ для семантического анализа
# ===================================================================

def build_resume_text(resume: Dict) -> str:
    parts = []
    
    # Основная информация
    if name := resume.get('name'):
        parts.append(f"Имя: {name.strip()}")
    
    # Возраст (предпочитается уже распарсенный)
    age = resume.get('parsed_age') or resume.get('age')
    if age:
        parts.append(f"Возраст: {age}")
    
    city = resume.get('city')
    if city:
        parts.append(f"Город: {city.strip()}")
    
    position = resume.get('desired_position')
    if position:
        parts.append(f"Желаемая должность: {position.strip()}")
    
    # Опыт (предпочитается распарсенный)
    exp = resume.get('parsed_experience') or resume.get('experience')
    if exp is not None:
        parts.append(f"Опыт работы: {exp} лет")
    
    # Навыки
    skills = resume.get('skills') or resume.get('Навыки')
    if skills:
        parts.append(f"Навыки: {skills.strip()}")
    
    education = resume.get('education') or resume.get('Образование')
    if education:
        parts.append(f"Образование: {education.strip()}")
    
    comment = resume.get('comment') or resume.get('Комментарий')
    if comment:
        parts.append(f"Комментарий: {comment.strip()}")
    
    # Последнее место работы
    last_job = resume.get('last_job') or resume.get('Последнее место работы')
    if last_job:
        parts.append(f"Последнее место работы: {last_job.strip()}")
    
    resume_text = "\n".join(parts)  # сбор всего в один текст
    
    return resume_text.strip()

import re
from typing import List, Dict, Optional, Tuple

# ===================================================================
# ЗАГРУЗКА И ПАРСИНГ РЕЗЮМЕ ИЗ ФАЙЛА
#
# Умный парсер, который читает файл resumes.txt -> разделяет на отдельные резюме
# и извлекает все поля + работает с шумными резюме
# ===================================================================

def load_resumes_from_file(filename: str) -> List[Dict]:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Разделяем файл на отдельные резюме
    blocks = re.split(r'(=== Резюме кандидата №\d+ ===)', content)
    
    resumes = []
    resume_number = 0
    
    for i in range(1, len(blocks), 2):  # берём заголовки и следующие за ними блоки
        header = blocks[i].strip()
        block = blocks[i+1] if i+1 < len(blocks) else ""
        
        # Извлекаем номер резюме
        match = re.search(r'№(\d+)', header)
        if match:
            resume_number = int(match.group(1))
        else:
            resume_number += 1  # fallback
        
        resume = {
            'resume_number': resume_number,   # ← НОВОЕ ПОЛЕ
            'name': '',
            'age': None,
            'experience': None,
            'city': '',
            'desired_position': '',
            'skills': '',
            'education': '',
            'salary': '',
            'last_job': '',
            'comment': ''
        }
        
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        for line in lines:
            if line.startswith('Имя:'):
                resume['name'] = line[4:].strip()
            elif line.startswith('Возраст:'):
                resume['age'] = line[8:].strip()
            elif line.startswith('Город:'):
                resume['city'] = line[6:].strip()
            elif line.startswith('Желаемая должность:'):
                resume['desired_position'] = line[19:].strip()
            elif line.startswith('Опыт (лет):') or line.startswith('Опыт:'):
                resume['experience'] = line.split(':', 1)[1].strip()
            elif line.startswith('Навыки:'):
                resume['skills'] = line[7:].strip()
            elif line.startswith('Образование:'):
                resume['education'] = line[12:].strip()
            elif line.startswith('Ожидаемая зарплата:'):
                resume['salary'] = line[19:].strip()
            elif line.startswith('Последнее место работы:'):
                resume['last_job'] = line[24:].strip()
            elif line.startswith('Комментарий:'):
                resume['comment'] = line[12:].strip()
        
        # === Умная дополнительная обработка ===
        if not resume['name'] or re.match(r'^\d{1,2}\s*лет?$', resume['name'].strip()):
            resume['name'] = "Неизвестный кандидат"
        
        # Поиск возраста и опыта в других полях
        if not resume.get('age'):
            for field in ['name', 'desired_position', 'city', 'comment', 'experience']:
                if resume.get(field):
                    extracted = extract_age(resume[field])
                    if extracted:
                        resume['age'] = str(extracted)
                        break
        
        if not resume.get('experience'):
            for field in ['experience', 'name', 'comment']:
                if resume.get(field):
                    extracted = extract_experience(resume[field])
                    if extracted is not None:
                        resume['experience'] = str(extracted)
                        break
        
        resumes.append(resume)
    
    print(f"Загружено и обработано резюме: {len(resumes)}")
    return resumes

# ===================================================================
# ПАРСЕР ВОЗРАСТА
# Извлекает возраст кандидата из текста. Очень устойчив к разным форматам написания.
# ===================================================================

def extract_age(text: str) -> Optional[int]:
    if not text:
        return None
    
    text_str = str(text).strip().lower()
    
    if text_str in ['неизвестен', 'unknown', '-', '', 'нет', ' ']:
        return None
    
    # Убираем "год", "лет", "года" и всё после
    text_str = re.sub(r'\s*(год|лет|года|лет)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    
    # Ищем все числа
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        age = int(m)
        if 16 <= age <= 65:          # разумный диапазон для кандидатов
            return age
    
    # Если число 15 или меньше — иногда это школьники, но оставляем (можно отфильтровать позже)
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        age = int(m)
        if 15 <= age <= 70:
            return age
    
    return None

# ===================================================================
# ПАРСЕР ОПЫТА РАБОТЫ
# Извлекает количество лет опыта из текста. Устойчив к шумным данным.
# ===================================================================

def extract_experience(text: str) -> Optional[int]:
    if not text:
        return None
    
    text_str = str(text).strip().lower()
    
    if text_str in ['неизвестен', 'unknown', 'нет', '∞', '-', '', '0 (minecraft)', 'нет']:
        return None
    
    # Убираем телефоны, "год", "лет", "года"
    text_str = re.sub(r'\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}', '', text_str)
    text_str = re.sub(r'\s*(год|лет|года)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        exp = int(m)
        if 0 <= exp <= 50:
            return exp
    return None

# ===================================================================
# ВАЛИДАЦИЯ ИМЕНИ КАНДИДАТА
# Проверяет, является ли строка настоящим русским ФИО.
# Защищает систему от мусора, email и перепутанных полей.
# ===================================================================

def is_valid_name(name: str) -> bool:
    if not name:
        return False
    
    name = name.strip()
    if len(name) < 5:  # слишком коротко для нормального ФИО
        return False
    
    # Убираем лишние пробелы
    name = re.sub(r'\s+', ' ', name)
    
    # Основные запреты
    forbidden = {
        'test@mail.ru', 'неизвестный', 'неизвестен', 'unknown', 'anonymous',
        'java', 'python', 'spring', 'docker', 'sql', 'html', 'css'
    }
    lower_name = name.lower()
    if any(word in lower_name for word in forbidden):
        return False
    
    # Не должно содержать цифры или email-подобные конструкции
    if re.search(r'\d|@|\.ru|\.com', name):
        return False
    
    # Разбиваем на слова
    words = name.split()
    
    # Русское ФИО обычно состоит из 2–4 слов
    if len(words) < 2 or len(words) > 4:
        return False
    
    # Каждое слово должно начинаться с заглавной русской буквы
    for word in words:
        if not re.match(r'^[А-ЯЁ]', word):   # начинается с заглавной
            return False
        # Слово должно состоять только из русских букв, дефиса и апострофа
        if not re.match(r'^[А-Яа-яЁё-]+$', word):
            return False
        if len(word) < 2:  # слишком короткие слова не допускаем
            return False
    
    # Дополнительная проверка: не должно быть одного очень длинного слова (типа города)
    if len(words) == 1 and len(name) > 12:
        return False
    
    # Защита от названий городов в одном слове (дополнительный эвристический фильтр)
    single_word_lower = words[0].lower()
    if len(words) == 1 and len(single_word_lower) > 8 and re.match(r'^[а-яё]+$', single_word_lower):
        # Если одно слово и оно выглядит как типичный город — отсеиваем
        return False
    
    return True

# ===================================================================
# ГЛАВНАЯ ФУНКЦИЯ РАНЖИРОВАНИЯ КАНДИДАТОВ
# Выполняет полную обработку: жёсткую валидацию, применение фильтров HR 
# и семантическое ранжирование с помощью нейросети. Возвращает топ кандидатов.
# ===================================================================

def rank_candidates(
    vacancy_text: str,
    resumes: List[Dict],
    filters: dict = None,
    use_filters: bool = True,
    min_score: float = 0.0,
    top_k: int = 10
) -> List[Dict]:
    
    if filters is None:
        filters = {}

    filtered_resumes = []
    removed_stats = {
        "invalid_name": 0,
        "invalid_city": 0,
        "no_age": 0,
        "no_experience": 0,
        "invalid_data": 0,
        "filter_age": 0,
        "filter_experience": 0,
        "filter_city": 0,
        "non_it": 0,           # ← добавлено
        "low_score": 0
    }

    print("Начинаем валидацию и фильтрацию резюме...\n")

    for resume in resumes:
        reason = None
        raw_name = str(resume.get('name', '')).strip()
        raw_city = str(resume.get('city', '')).strip()

        # Парсинг возраста и опыта
        age = extract_age(resume.get('age')) if resume.get('age') else None
        experience = extract_experience(resume.get('experience')) if resume.get('experience') else None

        # ====================== ЖЁСТКАЯ ВАЛИДАЦИЯ ======================
        if not raw_name or not is_valid_name(raw_name):
            removed_stats["invalid_name"] += 1
            reason = f"Некорректное имя: '{raw_name}'"

        elif age is None or not (16 <= age <= 70):
            removed_stats["no_age"] += 1
            reason = f"Некорректный возраст (было: {resume.get('age')})"

        elif experience is None or not (0 <= experience <= 50):
            removed_stats["no_experience"] += 1
            reason = f"Некорректный опыт (было: {resume.get('experience')})"

        elif not is_valid_russian_city(raw_city):
            removed_stats["invalid_city"] += 1
            reason = f"Неизвестный или некорректный город: '{raw_city}'"

        elif len(raw_name.split()) <= 1 and is_valid_russian_city(raw_name):
            removed_stats["invalid_name"] += 1
            reason = "Имя и город, скорее всего, перепутаны местами"

        # ====================== IT-ФИЛЬТР (самое важное исправление) ======================
        elif not is_it_candidate(resume):
            removed_stats["non_it"] += 1
            reason = f"Не IT-кандидат (должность: {resume.get('desired_position', '—')})"

        # ====================== ФИЛЬТРЫ HR ======================
        if reason is None and use_filters:
            # Возраст
            if filters.get('min_age') is not None and age < filters['min_age']:
                removed_stats["filter_age"] += 1
                reason = f"Возраст {age} ниже минимального"
            if filters.get('max_age') is not None and age > filters['max_age']:
                removed_stats["filter_age"] += 1
                reason = f"Возраст {age} выше максимального"

            # Опыт
            if filters.get('min_experience') is not None and experience < filters['min_experience']:
                removed_stats["filter_experience"] += 1
                reason = f"Опыт {experience} лет меньше требуемого"

            # Город
            if filters.get('city'):
                if not is_valid_russian_city(filters['city']):
                    pass  # игнорируем неверный фильтр
                elif filters['city'].lower() not in normalize_city(raw_city):
                    removed_stats["filter_city"] += 1
                    reason = "Город не соответствует фильтру HR"

        # ====================== ИТОГ ======================
        if reason is None:
            resume['parsed_age'] = age
            resume['parsed_experience'] = experience
            filtered_resumes.append(resume)
        else:
            print(f"Отсеяно | {raw_name[:35]:35} | Город: {raw_city[:25]:25} | Причина: {reason}")

    # ====================== ОТЧЁТ ======================
    total_removed = sum(removed_stats.values())
    
    print("\n" + "="*100)
    print(f"Результат фильтрации: {len(filtered_resumes)} валидных резюме из {len(resumes)}")
    print(f"Отсеяно: {total_removed} резюме")
    
    if total_removed > 0:
        print("\nПричины отсева:")
        reason_map = {
            "invalid_name": "Некорректное имя",
            "invalid_city": "Неверный / неизвестный город",
            "no_age": "Проблема с возрастом",
            "no_experience": "Проблема с опытом",
            "non_it": "Не IT-кандидат",
            "filter_age": "Не прошёл фильтр по возрасту",
            "filter_experience": "Не прошёл фильтр по опыту",
            "filter_city": "Не прошёл фильтр по городу HR",
            "low_score": "Низкий семантический score"
        }
        for key, count in removed_stats.items():
            if count > 0:
                print(f"   • {reason_map.get(key, key)}: {count}")
    print("="*100)

    if not filtered_resumes:
        print("После строгой фильтрации кандидатов не осталось.")
        return []

    # ====================== СЕМАНТИЧЕСКОЕ РАНЖИРОВАНИЕ ======================
    print(f"\nВыполняется семантическое ранжирование {len(filtered_resumes)} кандидатов...\n")

    vacancy_embedding = model.encode(vacancy_text, convert_to_tensor=True)
    results = []

    for resume in filtered_resumes:
        resume_text = build_resume_text(resume)
        if len(resume_text) < 30:
            continue

        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            vacancy_embedding.unsqueeze(0), resume_embedding.unsqueeze(0)
        ).item()

        score_percent = round(similarity * 100, 1)

        if score_percent < min_score:
            removed_stats["low_score"] += 1
            continue

        results.append({
            'resume_number': resume.get('resume_number'),
            'name': resume.get('name'),
            'age': resume.get('parsed_age'),
            'experience': resume.get('parsed_experience'),
            'city': resume.get('city', '-'),
            'desired_position': resume.get('desired_position', '-'),
            'salary': resume.get('salary') or '-',
            'score': score_percent
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# ===================================================================
# ЗАПУСК СИСТЕМЫ СКРОЛЛИНГА КАНДИДАТОВ
# Определяет текст вакансии, настраивает фильтры HR и запускает полный цикл обработки.
# В конце выводит топ кандидатов с процентом соответствия.
# ===================================================================

if __name__ == "__main__":
    # ====================== ТЕКСТ ВАКАНСИИ ======================
    vacancy_text = """
    Ищем Senior Data Scientist в продуктовую команду финтеха.
    Требования: опыт от 4 лет, Python, SQL, TensorFlow/PyTorch, Pandas, Scikit-learn.
    Опыт построения и внедрения ML-моделей в продакшн.
    """

    # ====================== ФИЛЬТРЫ ДЛЯ HR ======================
    filters = {
        'min_age': 23,
        'max_age': 45,
        'min_experience': 4,        # минимум 4 года опыта
        'city': None,
        'max_salary': None,         # можно добавить позже
    }

    MIN_SCORE = 62.0                # повысили порог
    TOP_K = 30

    # ====================== ЗАПУСК ======================
    resumes = load_resumes_from_file("resumes.txt")
    print(f"Загружено резюме: {len(resumes)}\n")

    top_candidates = rank_candidates(
        vacancy_text=vacancy_text,
        resumes=resumes,
        filters=filters,
        min_score=MIN_SCORE,
        top_k=TOP_K
    )

    # ====================== ВЫВОД ======================
    print("=" * 100)
    print(f"ТОП-{TOP_K} КАНДИДАТОВ (после IT-фильтра и фильтров HR)")
    print("=" * 100)

    for i, cand in enumerate(top_candidates, 1):
        print(f"{i:2d}. {cand['name']:32} | "
              f"Возраст: {cand['age']:3} | "
              f"Опыт: {cand['experience']:3} лет | "
              f"Город: {cand['city']:15} | "
              f"З/п: {cand['salary']:10} | "
              f"Соответствие: {cand['score']:6.1f}% "
              f"(резюме №{cand['resume_number']})")

    print("=" * 100)