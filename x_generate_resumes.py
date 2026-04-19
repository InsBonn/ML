import random
import json
import os

# ===================== БАЗОВЫЕ ДАННЫЕ ДЛЯ ГЕНЕРАЦИИ =====================

first_names_male = ['Александр', 'Дмитрий', 'Максим', 'Сергей', 'Андрей', 'Евгений', 'Михаил', 'Владимир', 'Алексей', 'Николай']
first_names_female = ['Анна', 'Елена', 'Мария', 'Ольга', 'Татьяна', 'Наталья', 'Ирина', 'Екатерина', 'Юлия', 'Анастасия']
last_names = ['Иванов', 'Смирнов', 'Кузнецов', 'Попов', 'Васильев', 'Петров', 'Соколов', 'Михайлов', 'Новиков', 'Федоров', 'Морозов', 'Волков', 'Алексеев', 'Лебедев', 'Семенов', 'Егоров', 'Павлов', 'Козлов']
middle_names_male = ['Александрович', 'Дмитриевич', 'Максимович', 'Сергеевич', 'Андреевич', 'Евгеньевич', 'Михайлович', 'Владимирович', 'Алексеевич']
middle_names_female = ['Александровна', 'Дмитриевна', 'Максимовна', 'Сергеевна', 'Андреевна', 'Евгеньевна', 'Михайловна', 'Владимировна', 'Алексеевна']

cities = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань', 'Нижний Новгород', 'Челябинск', 'Самара', 'Омск', 'Ростов-на-Дону', 'Уфа', 'Красноярск', 'Пермь', 'Воронеж', 'Волгоград', 'Краснодар', 'Саратов', 'Тюмень']

positions = [
    ('Data Scientist', ['Python', 'SQL', 'TensorFlow', 'PyTorch', 'Pandas', 'Scikit-learn', 'NumPy', 'Matplotlib', 'Keras', 'MLflow']),
    ('Data Engineer', ['Python', 'Airflow', 'Spark', 'Hadoop', 'SQL', 'dbt', 'Kafka', 'AWS', 'Redshift', 'BigQuery']),
    ('Data Analyst', ['SQL', 'Tableau', 'Excel', 'Python', 'Power BI', 'Looker', 'Statistics', 'A/B testing']),
    ('Backend Developer', ['Java', 'Spring', 'PostgreSQL', 'Docker', 'Kubernetes', 'Go', 'Python', 'FastAPI', 'Redis', 'RabbitMQ']),
    ('Frontend Developer', ['JavaScript', 'TypeScript', 'React', 'Redux', 'Next.js', 'Vue.js', 'Tailwind', 'CSS', 'HTML']),
    ('Fullstack Developer', ['React', 'Node.js', 'TypeScript', 'PostgreSQL', 'Docker', 'MongoDB', 'Express', 'Prisma']),
    ('DevOps Engineer', ['Linux', 'Bash', 'Ansible', 'Terraform', 'Jenkins', 'GitLab CI', 'Prometheus', 'Grafana', 'AWS', 'Kubernetes']),
    ('Product Manager', ['Product analytics', 'A/B testing', 'Roadmap', 'SQL', 'Figma', 'Jira', 'Confluence', 'Agile', 'Scrum']),
    ('Project Manager', ['Agile', 'Scrum', 'Jira', 'Confluence', 'MS Project', 'Risk management', 'Budgeting', 'Team leadership']),
    ('QA Engineer', ['Manual testing', 'Jira', 'TestRail', 'Postman', 'Selenium', 'Pytest', 'API testing', 'SQL']),
    ('QA Lead', ['Python', 'Pytest', 'Selenium', 'JUnit', 'Load Testing', 'TestRail', 'Team management', 'CI/CD']),
    ('System Analyst', ['UML', 'BPMN', 'SQL', 'REST API', 'Postman', 'Draw.io', 'Requirements gathering', 'Technical documentation']),
    ('Security Analyst', ['Kali Linux', 'Metasploit', 'Wireshark', 'Burp Suite', 'Python', 'Network security', 'Penetration testing']),
    ('HR Generalist', ['1C:ZUP', 'Recruiting', 'Onboarding', 'HR administration', 'Performance reviews', 'Sourcing', 'Interviews']),
    ('UX/UI Designer', ['Figma', 'Sketch', 'Adobe XD', 'Prototyping', 'User Testing', 'Tilda', 'Miro', 'Illustrator']),
    ('iOS Developer', ['Swift', 'UIKit', 'SwiftUI', 'CoreData', 'Combine', 'Firebase', 'XCTest']),
    ('Android Developer', ['Kotlin', 'Java', 'Android SDK', 'Jetpack Compose', 'Coroutines', 'Room', 'Firebase']),
    ('Game Developer', ['C#', 'Unity', 'Unreal Engine', 'C++', 'Blender', 'Git', 'Shader programming']),
    ('Tech Support Engineer', ['Windows', 'Linux', 'Active Directory', 'Zendesk', 'Jira', 'Bash', 'Networking']),
    ('Sales Manager', ['B2B sales', 'Negotiation', 'CRM', 'Cold calling', 'Key account management', 'Contract negotiation']),
    ('Business Analyst', ['SQL', 'Jira', 'Confluence', 'BPMN', 'UML', 'Requirements gathering', 'Data analysis']),
    ('Machine Learning Engineer', ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'Kubernetes', 'MLOps', 'SQL', 'AWS']),
]

educations = [
    'Высшее (МГУ, прикладная математика)',
    'Высшее (СПбГУ, информатика)',
    'Высшее (МФТИ, прикладная математика и физика)',
    'Высшее (ВШЭ, программная инженерия)',
    'Высшее (ИТМО, информационные технологии)',
    'Высшее (НГУ, программирование)',
    'Высшее (УрФУ, менеджмент)',
    'Высшее (КФУ, системное администрирование)',
    'Среднее профессиональное',
    'Неоконченное высшее',
    'Высшее (экономическое)',
    'Высшее (психологическое)',
    'Высшее (лингвистическое)'
]

english_levels = ['A1', 'A2', 'Pre-Intermediate', 'Intermediate', 'Upper-Intermediate', 'Advanced', 'Fluent']
companies = ['Яндекс', 'Ozon', 'Wildberries', 'Tinkoff', 'Сбер', 'VK', 'Avito', 'X5 Group', 'Магнит', 'Ростелеком', 'Газпром нефть', 'Лаборатория Касперского', 'Райффайзенбанк', 'Skillbox', 'Нетология', 'МТС', 'Билайн', 'Росатом']

comments = [
    '', 'Люблю свою работу.', 'Всегда учусь новому.', 'Работаю на результат.', '', '', 
    'Легко нахожу общий язык с командой.', 'Стрессоустойчив.', 'Готов к переработкам.',
    'Хочу развиваться в IT.', '', 'Могу обучать джуниоров.', 'Участвовал в крупных проектах.',
    'Ищу долгосрочные отношения с компанией.', '', '', 'Пунктуальный и ответственный.',
    'Быстро осваиваю новые технологии.', 'Коммуникабельный.'
]

def generate_phone():
    """Генерирует случайный номер телефона"""
    return f"+7 (9{random.randint(10, 99)}) {random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(10, 99)}"

def generate_name():
    """Генерирует случайное ФИО"""
    is_male = random.choice([True, False])
    if is_male:
        first = random.choice(first_names_male)
        middle = random.choice(middle_names_male)
    else:
        first = random.choice(first_names_female)
        middle = random.choice(middle_names_female)
    
    last = random.choice(last_names)
    
    # Иногда без отчества
    if random.random() > 0.7:
        return f"{first} {last}"
    return f"{first} {middle} {last}"

def generate_resume(num):
    """Генерирует одно резюме"""
    full_name = generate_name()
    city = random.choice(cities)
    position, skill_list = random.choice(positions)
    
    # Опыт в зависимости от позиции
    if 'Lead' in position or 'Senior' in position:
        experience = random.randint(5, 12)
    elif 'Junior' in position:
        experience = random.randint(0, 2)
    else:
        experience = random.randint(2, 8)
    
    # Навыки - от 3 до 8 штук
    num_skills = random.randint(4, 8)
    selected_skills = random.sample(skill_list, min(num_skills, len(skill_list)))
    skills = ', '.join(selected_skills)
    
    # Возраст
    age = 22 + experience + random.randint(-2, 3)
    age = max(20, min(60, age))
    
    education = random.choice(educations)
    english = random.choice(english_levels)
    
    # Зарплата
    if 'Lead' in position or 'Senior' in position:
        salary = random.randint(250000, 400000)
    elif 'Junior' in position:
        salary = random.randint(60000, 120000)
    else:
        salary = random.choice([80000, 100000, 130000, 150000, 180000, 200000, 220000, 250000])
    
    # Корректировка по опыту
    salary = int(salary * (0.7 + experience / 25))
    salary = min(max(salary, 40000), 450000)
    
    company = random.choice(companies)
    relocate = random.choice(['да', 'нет', 'только крупные города'])
    comment = random.choice(comments)
    
    # Добавляем случайные комментарии
    if random.random() < 0.15:
        comment = random.choice([
            'Имею опыт работы в международной компании.',
            'Участвовал в запуске 5+ продуктов.',
            'Есть рекомендации с предыдущих мест работы.',
            'Прошел курсы повышения квалификации.',
            'Владею английским на уровне носителя.',
            'Ищу работу с возможностью релокации.',
            'Готов к командировкам.',
            'Имею собственные pet-проекты на GitHub.'
        ])
    
    company_position = position.split()[0] if position.split() else position
    last_work = f"{company} ({company_position})" if random.random() > 0.3 else company
    
    return f"""=== Резюме кандидата №{num} ===
Имя: {full_name}
Возраст: {age}
Город: {city}
Желаемая должность: {position}
Опыт (лет): {experience}
Навыки: {skills}
Образование: {education}
Уровень английского: {english}
Ожидаемая зарплата: {salary}
Готов к переезду: {relocate}
Последнее место работы: {last_work}
Телефон: {generate_phone()}
Комментарий: {comment if comment else ''}"""

def generate_dataset(count=200):
    """Генерирует dataset из count резюме"""
    print(f"Генерация {count} резюме...")
    resumes = []
    for i in range(1, count + 1):
        resume = generate_resume(i)
        resumes.append(resume)
        if i % 50 == 0:
            print(f"  Сгенерировано {i} из {count}")
    return resumes

def save_to_file(resumes, filename='resumes_generated.txt'):
    """Сохраняет резюме в текстовый файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(resumes))
    print(f"[OK] Сохранено {len(resumes)} резюме в {filename}")

def save_to_json(resumes, filename='resumes_generated.json'):
    """Сохраняет резюме в JSON файл"""
    data = []
    for resume in resumes:
        lines = resume.split('\n')
        record = {}
        for line in lines[1:]:  # пропускаем === Резюме ===
            if ': ' in line:
                k, v = line.split(': ', 1)
                record[k.strip()] = v.strip()
        data.append(record)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Сохранено {len(data)} резюме в {filename}")

def main():
    print("=" * 60)
    print("ГЕНЕРАТОР РЕЗЮМЕ ДЛЯ ДАТАСЕТА")
    print("=" * 60)
    
    # Генерация
    resumes = generate_dataset(200)
    
    # Сохранение
    save_to_file(resumes, 'resumes_generated.txt')
    save_to_json(resumes, 'resumes_generated.json')
    
    # Показываем примеры
    print("\n" + "=" * 60)
    print("ПРИМЕР СГЕНЕРИРОВАННОГО РЕЗЮМЕ:")
    print("=" * 60)
    print(resumes[0])
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print("Файлы созданы:")
    print("  - resumes_generated.txt (для загрузки в нейросеть)")
    print("  - resumes_generated.json (для анализа)")
    
    return resumes

if __name__ == "__main__":
    # Устанавливаем кодировку для Windows
    import sys
    import io
    
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    resumes = main()