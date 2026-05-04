from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(lang='en')

img = cv2.imread("test.png")

result = ocr.predict("test.png")

for line in result:

    boxes = line['rec_boxes']
    texts = line['rec_texts']

    for box, text in zip(boxes, texts):

        x1, y1, x2, y2 = box

        # draw bounding box
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        # draw text
        cv2.putText(img,text,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        # SAVE TEXT HERE
        file = open("output_text.txt","a")
        file.write(text+"\n")
        file.close()

cv2.imwrite("output.png",img)

print("OCR Done")