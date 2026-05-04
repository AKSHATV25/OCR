from paddleocr import PaddleOCR

images = {
    "English": "C:/Users/aksha/OneDrive/Desktop/Multilingual_OCR_Project/multilingual_test/english.png",
    "Hindi": "C:/Users/aksha/OneDrive/Desktop/Multilingual_OCR_Project/multilingual_test/hindi.jpg",
    "Chinese": "C:/Users/aksha/OneDrive/Desktop/Multilingual_OCR_Project/multilingual_test/chinese.jpg"
}

langs = {
    "English": "en",
    "Hindi": "hi",
    "Chinese": "ch"
}

results = []

for lang_name in images:

    print("\nTesting:", lang_name)

    ocr = PaddleOCR(lang=langs[lang_name])

    result = ocr.predict(images[lang_name])

    for line in result:
        texts = line['rec_texts']
        scores = line['rec_scores']

        for t, s in zip(texts, scores):
            print("Text:", t)
            print("Confidence:", round(s,3))

            results.append([lang_name, t, round(s,3)])

file = open("language_results.txt","w")

for r in results:
    file.write(str(r)+"\n")

file.close()