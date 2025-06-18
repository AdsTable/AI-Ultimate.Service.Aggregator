## 📁 AI-Ultimate.Service.Aggregator Complete File System Structure

AI-Ultimate.Service.Aggregator/
├── 📁 src/
│   ├── 📁 core/
│   │   ├── __init__.py
│   │   ├── 📄 service_extractor.py          # Основной класс извлечения
│   │   ├── 📄 base_extractor.py             # Базовый абстрактный класс
│   │   └── 📄 exceptions.py                 # Кастомные исключения
│   │
│   ├── 📁 extractors/
│   │   ├── __init__.py
│   │   ├── 📄 web_extractor.py              # Веб-скрапинг логика
│   │   ├── 📄 api_extractor.py              # API интеграции
│   │   └── 📄 hybrid_extractor.py           # Комбинированный подход
│   │
│   ├── 📁 processors/
│   │   ├── __init__.py
│   │   ├── 📄 html_processor.py             # HTML парсинг
│   │   ├── 📄 content_analyzer.py           # AI анализ контента
│   │   └── 📄 plan_normalizer.py            # Нормализация данных
│   │
│   ├── 📁 storage/
│   │   ├── __init__.py
│   │   ├── 📄 database_manager.py           # Управление БД
│   │   ├── 📄 cache_manager.py              # Кэширование
│   │   └── 📄 models.py                     # Модели данных
│   │
│   ├── 📁 utils/
│   │   ├── __init__.py
│   │   ├── 📄 browser_utils.py              # Утилиты браузера
│   │   ├── 📄 stealth_utils.py              # Анти-детекция
│   │   ├── 📄 retry_utils.py                # Механизмы повторов
│   │   └── 📄 validation.py                 # Валидация данных
│   │
│   └── 📁 config/
│       ├── __init__.py
│       ├── 📄 settings.py                   # Основные настройки
│       ├── 📄 browser_config.py             # Конфигурация браузера
│       └── 📄 provider_configs.py           # Настройки провайдеров
│
├── 📁 tests/
│   ├── 📁 unit/
│   ├── 📁 integration/
│   └── 📁 fixtures/
│
├── 📁 docs/
├── 📁 scripts/
├── 📄 main.py                               # Streamlit приложение
├── 📄 requirements.txt
├── 📄 setup.py
└── 📄 README.md