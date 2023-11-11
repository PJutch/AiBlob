from transformers import pipeline

# DeepPavlov/rubert-base-cased
# DeepPavlov/xlm-roberta-large-en-ru-mnli

classifier = pipeline(task="zero-shot-classification",
                      model="DeepPavlov/xlm-roberta-large-en-ru-mnli",
                      multi_label=True)
pred = classifier(
    "Только сегодня, заберите свой выигрыш!",
    ["срочно", "не срочно", "телефон", "компьютер", "планшет", "спам", "полиция", "лотерея"],
)
print(pred)
