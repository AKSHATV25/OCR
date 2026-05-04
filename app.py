import streamlit as st
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("Multilingual OCR System")

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:

    extracted_text = []

    # -------- PDF Handling --------
    if uploaded_file.type == "application/pdf":

        pages = convert_from_bytes(
            uploaded_file.read(),
            poppler_path=r"C:\Users\aksha\poppler-25.12.0\Library\bin"
        )

        for page in pages:
            img_array = np.array(page)
            result = ocr.predict(img_array)

            for line in result:
                texts = line['rec_texts']
                for text in texts:
                    extracted_text.append(text)

        st.success("PDF Processed Successfully")

    # -------- Image Handling --------
    else:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)

        # Show original image
        st.image(img, caption="Uploaded Image")

        result = ocr.predict(img_array)

        # Draw bounding boxes
        for line in result:
            boxes = line['rec_boxes']
            texts = line['rec_texts']

            for box, text in zip(boxes, texts):
                x1, y1, x2, y2 = box

                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_array, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                extracted_text.append(text)

        # Show detected image with boxes
        st.image(img_array, caption="Detected Text with Bounding Boxes")

    # -------- Dynamic Table --------
    st.subheader("Structured Output Table")

    cols = 6

    table_data = [
        extracted_text[i:i+cols]
        for i in range(0, len(extracted_text), cols)
    ]

    df = pd.DataFrame(table_data)

    st.dataframe(df)