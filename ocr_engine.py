from paddleocr import PaddleOCR
import cv2

# Initialize OCR
ocr = PaddleOCR(lang='cv')

# Read image
img = cv2.imread("test1.jpg")

# Run OCR
result = ocr.predict("test1.jpg")

for line in result:

    boxes = line['rec_boxes']
    texts = line['rec_texts']

    for box, text in zip(boxes, texts):

        x1, y1, x2, y2 = box

        # draw rectangle
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        # add text
        cv2.putText(img, text, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

# Save output
cv2.imwrite("output.png", img)

print("OCR completed. Check output.png")