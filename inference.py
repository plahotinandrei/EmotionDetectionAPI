from transformers import pipeline

classifier = pipeline("text-classification", model="plakhotin94/emotion-bert")

LABELS = {
    "LABEL_0": "Радость",
    "LABEL_1": "Грусть",
    "LABEL_2": "Удивление",
    "LABEL_3": "Страх",
    "LABEL_4": "Гнев",
    "LABEL_5": "Нейтрально"
}

def predict_emotion(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    return LABELS.get(label, label), score

if __name__ == "__main__":
    test_texts = [
        "Я так счастлив, что не могу уснуть!",
        "Мне очень грустно расставаться с вами.",
        "Как ты это сделал? Я потрясён!",
        "Я боюсь завтрашнего экзамена.",
        "Меня бесит, когда опаздывают!",
        "Сегодня обычный день, ничего особенного."
    ]
    for t in test_texts:
        emotion, conf = predict_emotion(t)
        print(f"Текст: {t}\nЭмоция: {emotion} (уверенность: {conf:.3f})\n")