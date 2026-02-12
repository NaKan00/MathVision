# MathVision — Setup Guide

## 1. Клонирование репозитория

```bash
git clone https://github.com/NaKan00/MathVision.git
cd MathVision 

## 2. Создание виртуального окружения

python3 -m venv .venv
source .venv/bin/activate


python -m venv .venv
.venv\Scripts\activate

## 3. Установка зависимостей

pip install -r requirements.txt

## 4. Датасет

https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k


MathVision/
└── datasets/
    └── im2latex/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── images/
            └── formula_images_processed/


## 5. Проверка установки 


python -m im2latex.quick_check