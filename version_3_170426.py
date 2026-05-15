import torch
import re
import warnings
import numpy as np
from scipy.special import softmax
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings('ignore')

# Кастомные библиотеки
from x_russian_cities import is_valid_russian_city, normalize_city
from x_it_keywords import is_it_candidate

# ========================= ЗАГРУЗКА МОДЕЛЕЙ =========================
print("Загрузка моделей...")
model = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    max_length=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("Модели успешно загружены.\n")


# ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========================
def build_resume_text(resume: Dict) -> str:
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
    if skills := resume.get('skills'):
        parts.append(f"Навыки: {skills.strip()}")
    if education := resume.get('education'):
        parts.append(f"Образование: {education.strip()}")
    if last_job := resume.get('last_job'):
        parts.append(f"Последнее место работы: {last_job.strip()}")
    if comment := resume.get('comment'):
        parts.append(f"Комментарий: {comment.strip()}")
    return "\n".join(parts).strip()


def extract_age(text: str) -> Optional[int]:
    if not text:
        return None
    text_str = str(text).strip().lower()
    if text_str in ['неизвестен', 'unknown', '-', '', 'нет', ' ']:
        return None
    text_str = re.sub(r'\s*(год|лет|года|лет)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        age = int(m)
        if 16 <= age <= 65:
            return age
    for m in matches:
        age = int(m)
        if 15 <= age <= 70:
            return age
    return None


def extract_experience(text: str) -> Optional[int]:
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
    if not name:
        return False
    name = name.strip()
    if len(name) < 5:
        return False
    name = re.sub(r'\s+', ' ', name)
    forbidden = {
        'test@mail.ru', 'неизвестный', 'неизвестен', 'unknown', 'anonymous',
        'java', 'python', 'spring', 'docker', 'sql', 'html', 'css'
    }
    lower_name = name.lower()
    if any(word in lower_name for word in forbidden):
        return False
    if re.search(r'\d|@|\.ru|\.com', name):
        return False
    words = name.split()
    if len(words) < 2 or len(words) > 4:
        return False
    for word in words:
        if not re.match(r'^[А-ЯЁ]', word) or not re.match(r'^[А-Яа-яЁё-]+$', word) or len(word) < 2:
            return False
    if len(words) == 1 and len(name) > 12:
        return False
    single_word_lower = words[0].lower()
    if len(words) == 1 and len(single_word_lower) > 8 and re.match(r'^[а-яё]+$', single_word_lower):
        return False
    return True


def load_resumes_from_file(filename: str) -> List[Dict]:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'(=== Резюме кандидата №\d+ ===)', content)
    resumes = []
    resume_number = 0

    for i in range(1, len(blocks), 2):
        header = blocks[i].strip()
        block = blocks[i + 1] if i + 1 < len(blocks) else ""

        match = re.search(r'№(\d+)', header)
        resume_number = int(match.group(1)) if match else resume_number + 1

        resume = {
            'resume_number': resume_number,
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
            elif line.startswith('Опыт') and ':' in line:
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

        if not resume['name'] or re.match(r'^\d{1,2}\s*лет?$', resume['name'].strip()):
            resume['name'] = "Неизвестный кандидат"

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

    return resumes


def rerank_candidates(vacancy_text: str, candidates: List[Dict], top_k: int = 20) -> List[Dict]:
    if not candidates:
        return []

    pairs = []
    for cand in candidates:
        resume_dict = {
            'name': cand.get('name', ''),
            'parsed_age': cand.get('age'),
            'parsed_experience': cand.get('experience'),
            'city': cand.get('city', ''),
            'desired_position': cand.get('desired_position', ''),
            'skills': cand.get('skills', ''),
            'education': cand.get('education', ''),
            'last_job': cand.get('last_job', ''),
            'comment': cand.get('comment', '')
        }
        pairs.append([vacancy_text, build_resume_text(resume_dict)])

    raw_scores = reranker.predict(pairs, show_progress_bar=False)
    scores_array = np.array(raw_scores)
    normalized = softmax(scores_array) * 100

    for i, cand in enumerate(candidates):
        cand['rerank_score'] = float(raw_scores[i])
        cand['rerank_score_percent'] = round(float(normalized[i]), 1)

    return sorted(candidates, key=lambda x: x.get('rerank_score_percent', 0), reverse=True)[:top_k]


# ========================= ОСНОВНАЯ ФУНКЦИЯ =========================
def rank_candidates(
    vacancy_text: str,
    resumes: List[Dict],
    filters: dict = None,
    use_filters: bool = True,
    use_reranking: bool = True,
    min_score: float = 15.0,
    top_k: int = 20,
    rerank_top: int = 40
) -> List[Dict]:
    
    if filters is None:
        filters = {}

    filtered_resumes = []
    removed_stats = {
        "invalid_name": 0, "invalid_city": 0, "no_age": 0, "no_experience": 0,
        "non_it": 0, "filter_age": 0, "filter_experience": 0, "filter_city": 0,
        "filter_position": 0, "filter_skills": 0, "low_score": 0
    }

    print("Начинаем валидацию и фильтрацию резюме...\n")

    for resume in resumes:
        reason = None
        raw_name = str(resume.get('name', '')).strip()
        raw_city = str(resume.get('city', '')).strip()
        raw_position = str(resume.get('desired_position', '')).lower()
        raw_skills = str(resume.get('skills', '')).lower()

        age = extract_age(resume.get('age')) if resume.get('age') else None
        experience = extract_experience(resume.get('experience')) if resume.get('experience') else None

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
        elif not is_it_candidate(resume):
            removed_stats["non_it"] += 1
            reason = f"Не IT-кандидат (должность: {resume.get('desired_position', '—')})"

        if reason is None and use_filters:
            if filters.get('min_age') and age < filters['min_age']:
                removed_stats["filter_age"] += 1
                reason = f"Возраст {age} ниже минимального"
            if filters.get('max_age') and age > filters['max_age']:
                removed_stats["filter_age"] += 1
                reason = f"Возраст {age} выше максимального"
            if filters.get('min_experience') and experience < filters['min_experience']:
                removed_stats["filter_experience"] += 1
                reason = f"Опыт {experience} лет меньше требуемого"
            if filters.get('city') and filters['city'].lower() not in normalize_city(raw_city):
                removed_stats["filter_city"] += 1
                reason = f"Город не соответствует фильтру ({filters['city']})"
            if filters.get('position_keywords'):
                pos_match = any(kw.lower() in raw_position for kw in filters['position_keywords'])
                if not pos_match:
                    removed_stats["filter_position"] += 1
                    reason = f"Должность не содержит ключевых слов: {filters['position_keywords']}"
            if filters.get('skills_keywords'):
                skill_match = any(kw.lower() in raw_skills for kw in filters['skills_keywords'])
                if not skill_match:
                    removed_stats["filter_skills"] += 1
                    reason = f"Навыки не содержат ключевых слов: {filters['skills_keywords']}"

        if reason is None:
            resume['parsed_age'] = age
            resume['parsed_experience'] = experience
            filtered_resumes.append(resume)
        else:
            print(f"Отсеяно | {raw_name[:35]:35} | Город: {raw_city[:25]:25} | Причина: {reason}")

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
            "filter_city": "Не прошёл фильтр по городу",
            "filter_position": "Не прошёл фильтр по должности",
            "filter_skills": "Не прошёл фильтр по навыкам",
            "low_score": "Низкий семантический score"
        }
        for key, count in removed_stats.items():
            if count > 0:
                print(f"   • {reason_map.get(key, key)}: {count}")
    print("="*100)

    if not filtered_resumes:
        print("После строгой фильтрации кандидатов не осталось.")
        return []

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
            'skills': resume.get('skills', ''),
            'education': resume.get('education', ''),
            'last_job': resume.get('last_job', ''),
            'comment': resume.get('comment', ''),
            'score': score_percent
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    if use_reranking and len(results) > 3:
        print(f"Запускаем reranking топ-{min(rerank_top, len(results))} кандидатов...")
        results = rerank_candidates(vacancy_text, results[:rerank_top], top_k)

    return results[:top_k]


# ========================= ЗАПУСК =========================
if __name__ == "__main__":
    vacancy_text = """
    Ищем Senior Data Scientist в продуктовую команду финтеха.
    Требования: опыт от 4 лет, Python, SQL, TensorFlow/PyTorch, Pandas, Scikit-learn.
    Опыт построения и внедрения ML-моделей в продакшн.
    """

    filters = {
        'min_age': 23,
        'max_age': 45,
        'min_experience': 4,
        'max_salary': None,
        'city': None,
        'position_keywords': ["Data Scientist", "ML Engineer", "Machine Learning", "Data"],
        'skills_keywords': ["Python", "PyTorch", "TensorFlow", "SQL", "Pandas"]
    }

    USE_FILTERS = True
    USE_RERANKING = True
    MIN_SCORE = 15.0
    TOP_K = 20

    resumes = load_resumes_from_file("resumes_generated.txt")

    top_candidates = rank_candidates(
        vacancy_text=vacancy_text,
        resumes=resumes,
        filters=filters,
        use_filters=USE_FILTERS,
        use_reranking=USE_RERANKING,
        min_score=MIN_SCORE,
        top_k=TOP_K
    )

    print("="*130)
    print(f"ТОП-{TOP_K} КАНДИДАТОВ | Фильтры: {'ВКЛ' if USE_FILTERS else 'ВЫКЛ'} | Reranking: {'ВКЛ' if USE_RERANKING else 'ВЫКЛ'}")
    print("="*130)

    for i, cand in enumerate(top_candidates, 1):
        retrieval = cand.get('score', 0)
        rerank = cand.get('rerank_score_percent')
        score_str = f"Retrieval: {retrieval:5.1f}% → Rerank: {rerank:5.1f}%" if rerank is not None else f"Retrieval: {retrieval:5.1f}%"
        print(f"{i:2d}. {cand['name']:38} | Возраст: {cand['age']:3} | "
              f"Опыт: {cand['experience']:2} лет | Город: {cand['city']:15} | "
              f"З/п: {cand['salary']:8} | {score_str} (№{cand.get('resume_number')})")

    print("="*130)