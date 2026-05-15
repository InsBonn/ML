import torch
from sentence_transformers import SentenceTransformer
import re
import warnings
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')

# ====================== ИМПОРТ СПЕЦИФИЧЕСКИХ ФУНКЦИЙ ======================
# Предполагается, что файл russian_cities.py находится в той же папке
from x_russian_cities import is_valid_russian_city, normalize_city, get_valid_city


# ====================== ЗАГРУЗКА МОДЕЛИ ======================
MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
print(f"Загружается модель: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Модель успешно загружена!\n")


# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def build_resume_text(resume: Dict) -> str:
    """Собирает качественный текст резюме из всех доступных полей."""
    parts = []
    
    if name := resume.get('name'):
        parts.append(f"Имя: {name.strip()}")
    
    age = resume.get('parsed_age') or resume.get('age')
    if age:
        parts.append(f"Возраст: {age}")
    
    if city := resume.get('city'):
        parts.append(f"Город: {city.strip()}")
    
    if position := resume.get('desired_position'):
        parts.append(f"Желаемая должность: {position.strip()}")
    
    exp = resume.get('parsed_experience') or resume.get('experience')
    if exp is not None:
        parts.append(f"Опыт работы: {exp} лет")
    
    skills = resume.get('skills') or resume.get('Навыки')
    if skills:
        parts.append(f"Навыки: {skills.strip()}")
    
    education = resume.get('education') or resume.get('Образование')
    if education:
        parts.append(f"Образование: {education.strip()}")
    
    comment = resume.get('comment') or resume.get('Комментарий')
    if comment:
        parts.append(f"Комментарий: {comment.strip()}")
    
    last_job = resume.get('last_job') or resume.get('Последнее место работы')
    if last_job:
        parts.append(f"Последнее место работы: {last_job.strip()}")
    
    return "\n".join(parts).strip()


def extract_age(text: str) -> Optional[int]:
    """Устойчивый парсер возраста"""
    if not text:
        return None
    
    text_str = str(text).strip().lower()
    
    if text_str in ['неизвестен', 'unknown', '-', '', 'нет', ' ']:
        return None
    
    # Убираем слова "год", "лет" и всё после
    text_str = re.sub(r'\s*(год|лет|года|лет)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        age = int(m)
        if 16 <= age <= 70:
            return age
    
    for m in matches:
        age = int(m)
        if 15 <= age <= 70:
            return age
    
    return None


def extract_experience(text: str) -> Optional[int]:
    """Улучшенный парсер опыта"""
    if not text:
        return None
    
    text_str = str(text).strip().lower()
    
    if text_str in ['неизвестен', 'unknown', 'нет', '∞', '-', '', '0 (minecraft)']:
        return None
    
    text_str = re.sub(r'\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}', '', text_str)
    text_str = re.sub(r'\s*(год|лет|года)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        exp = int(m)
        if 0 <= exp <= 50:
            return exp
    return None


def is_valid_name(name: str) -> bool:
    """Проверка, похоже ли поле на настоящее русское ФИО"""
    if not name:
        return False
    
    name = name.strip()
    if len(name) < 5:
        return False
    
    name = re.sub(r'\s+', ' ', name)
    lower_name = name.lower()
    
    forbidden = {'test@mail.ru', 'неизвестный', 'неизвестен', 'unknown', 'anonymous',
                 'java', 'python', 'spring', 'docker', 'sql', 'html', 'css'}
    
    if any(word in lower_name for word in forbidden):
        return False
    
    if re.search(r'\d|@|\.ru|\.com', name):
        return False
    
    words = name.split()
    if len(words) < 2 or len(words) > 4:
        return False
    
    for word in words:
        if not re.match(r'^[А-ЯЁ]', word):
            return False
        if not re.match(r'^[А-Яа-яЁё-]+$', word):
            return False
        if len(word) < 2:
            return False
    
    if len(words) == 1 and len(name) > 12:
        return False
    
    return True


def load_resumes_from_file(filename: str) -> List[Dict]:
    """Умный парсер файла resumes.txt"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'=== Резюме кандидата №\d+ ===', content)
    
    resumes = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        resume = {
            'name': '', 'age': None, 'experience': None, 'city': '',
            'desired_position': '', 'skills': '', 'education': '',
            'salary': '', 'last_job': '', 'comment': ''
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
            elif line.startswith(('Опыт (лет):', 'Опыт:')):
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
        
        # Дополнительная обработка
        if not resume['name'] or re.match(r'^\d{1,2}\s*лет?$', resume['name'].strip()):
            resume['name'] = "Неизвестный кандидат"
        
        # Поиск возраста и опыта в других полях
        if not resume['age']:
            for field in ['name', 'desired_position', 'city', 'comment', 'experience']:
                if resume.get(field):
                    extracted = extract_age(resume[field])
                    if extracted:
                        resume['age'] = str(extracted)
                        break
        
        if not resume['experience']:
            for field in ['experience', 'name', 'comment']:
                if resume.get(field):
                    extracted = extract_experience(resume[field])
                    if extracted is not None:
                        resume['experience'] = str(extracted)
                        break
        
        resumes.append(resume)
    
    print(f"Загружено и обработано резюме: {len(resumes)}")
    return resumes


def rank_candidates(
    vacancy_text: str,
    resumes: List[Dict],
    filters: dict = None,
    use_filters: bool = True,
    min_score: float = 0.0,
    top_k: int = 10
) -> List[Dict]:
    """Основная функция ранжирования кандидатов с жёсткой валидацией"""
    if filters is None:
        filters = {}

    filtered_resumes = []
    removed_stats = {
        "invalid_name": 0, "invalid_city": 0, "no_age": 0, "no_experience": 0,
        "invalid_data": 0, "filter_age": 0, "filter_experience": 0,
        "filter_city": 0, "low_score": 0
    }

    print("Начинаем валидацию и фильтрацию резюме...\n")

    for resume in resumes:
        reason = None
        raw_name = str(resume.get('name', '')).strip()
        raw_city = str(resume.get('city', '')).strip()

        age = extract_age(resume.get('age')) if resume.get('age') else None
        experience = extract_experience(resume.get('experience')) if resume.get('experience') else None

        # Жёсткая валидация
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

        # Фильтры HR
        if reason is None and use_filters:
            if filters.get('min_age') is not None and age < filters['min_age']:
                removed_stats["filter_age"] += 1
                reason = f"Возраст {age} ниже минимального"
            if filters.get('max_age') is not None and age > filters['max_age']:
                removed_stats["filter_age"] += 1
                reason = f"Возраст {age} выше максимального"
            if filters.get('experience') is not None and experience < filters['experience']:
                removed_stats["filter_experience"] += 1
                reason = f"Опыт {experience} лет меньше требуемого"
            if filters.get('city') and filters['city'].lower() not in normalize_city(raw_city).lower():
                removed_stats["filter_city"] += 1
                reason = "Город не соответствует фильтру HR"

        if reason is None:
            resume['parsed_age'] = age
            resume['parsed_experience'] = experience
            filtered_resumes.append(resume)
        else:
            print(f"Отсеяно → {raw_name[:35]:35} | Город: {raw_city[:25]:25} | Причина: {reason}")

    # Отчёт по фильтрации
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
            "filter_age": "Не прошёл фильтр по возрасту",
            "filter_experience": "Не прошёл фильтр по опыту",
            "filter_city": "Не прошёл фильтр по городу HR",
            "low_score": "Низкий семантический score"
        }
        for key, count in removed_stats.items():
            if count > 0:
                print(f"   • {reason_map.get(key, key)}: {count}")

    if not filtered_resumes:
        print("После строгой фильтрации кандидатов не осталось.")
        return []

    # Семантическое ранжирование
    print(f"\nВыполняется семантическое ранжирование {len(filtered_resumes)} кандидатов...\n")

    vacancy_embedding = model.encode(vacancy_text, convert_to_tensor=True)
    results = []

    for resume in filtered_resumes:
        resume_text = build_resume_text(resume)
        if len(resume_text) < 30:
            continue

        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            vacancy_embedding.unsqueeze(0),
            resume_embedding.unsqueeze(0)
        ).item()

        score_percent = round(similarity * 100, 1)

        if score_percent < min_score:
            removed_stats["low_score"] += 1
            continue

        results.append({
            'name': resume.get('name'),
            'age': resume.get('parsed_age'),
            'experience': resume.get('parsed_experience'),
            'city': resume.get('city', '-'),
            'desired_position': resume.get('desired_position', '-'),
            'salary': resume.get('salary') or resume.get('Ожидаемая зарплата', '-'),
            'score': score_percent
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ====================== ОСНОВНОЙ ЗАПУСК ======================
if __name__ == "__main__":
    # Текст вакансии
    vacancy_text = """
    Ищем Senior Data Scientist в продуктовую команду финтеха.
    Требования: опыт от 4 лет, Python, SQL, TensorFlow/PyTorch, Pandas, Scikit-learn.
    Опыт построения и внедрения ML-моделей в продакшн.
    """

    # Фильтры HR
    filters = {
        'min_age': 0,
        'max_age': 100,
        'experience': 0,
        'city': None,
    }

    MIN_SCORE = 0.0
    TOP_K = 100

    # Запуск
    resumes = load_resumes_from_file("resumes.txt")
    print(f"Загружено резюме: {len(resumes)}\n")

    top_candidates = rank_candidates(
        vacancy_text=vacancy_text,
        resumes=resumes,
        filters=filters,
        min_score=MIN_SCORE,
        top_k=TOP_K
    )

    # Вывод результатов
    print("=" * 100)
    print(f"ТОП-{TOP_K} КАНДИДАТОВ ПОСЛЕ ФИЛЬТРАЦИИ")
    print("=" * 100)

    for i, cand in enumerate(top_candidates, 1):
        print(f"{i:2d}. {cand['name']:32} | "
              f"Возраст: {cand['age']:3} | "
              f"Опыт: {cand['experience']:3} лет | "
              f"Город: {cand['city']:15} | "
              f"З/п: {cand['salary']:10} | "
              f"Соответствие: {cand['score']:6.1f}%")

    print("=" * 100)