# Face AntiSpoofing Project

Репозиторий для проекта системы антиспуфинга лиц.

## Обучение модели

Код для обучения моедли находится [здесь](https://github.com/MkSerdyuk/Face-Anti-Spoofing/blob/dev/notebooks/FaceAntiSpoofing.ipynb)

## Инструкция по установке

1) Клонирование репозитория:
    ```
    git clone https://github.com/MkSerdyuk/Face-Anti-Spoofing/.git 
    cd Face-Anti-Spoofing
    ```

2) Создание и запуск вирутальной среды:
    ```
    python -m venv venv
    venv/scripts/activate
    ```

3) Импорт зависимостей:
    ```
    pip install -r requirements.txt
    ```

4) Запуск демонстрации:
    ``` 
    python main.py
    ```