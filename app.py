import streamlit as st
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd

st.set_page_config(page_title="Multilingual OCR", layout="wide")

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

st.title("📄 Multilingual OCR System")

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:

    extracted_text = []

    try:
        # ================= PDF =================
        if uploaded_file.type == "application/pdf":

            pages = convert_from_bytes(uploaded_file.read())

            st.success(f"PDF contains {len(pages)} page(s)")

            for page_num, page in enumerate(pages, start=1):

                img_array = np.array(page)

                result = ocr.predict(img_array)

                for line in result:
                    texts = line["rec_texts"]

                    for text in texts:
                        extracted_text.append(text)

            st.success("PDF Processed Successfully")

        # ================= IMAGE =================
        else:

            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img)

            st.image(img, caption="Uploaded Image", use_container_width=True)

            result = ocr.predict(img_array)

            for line in result:

                boxes = line["rec_boxes"]
                texts = line["rec_texts"]

                for box, text in zip(boxes, texts):

                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(
                        img_array,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        img_array,
                        text,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    extracted_text.append(text)

            st.image(
                img_array,
                caption="Detected Text",
                use_container_width=True
            )

        # ================= TEXT OUTPUT =================
        st.subheader("Extracted Text")

        if extracted_text:
            st.text_area(
                "OCR Output",
                "\n".join(extracted_text),
                height=250
            )

        # ================= TABLE OUTPUT =================
        st.subheader("Structured Table")

        cols = 6

        table_data = [
            extracted_text[i:i + cols]
            for i in range(0, len(extracted_text), cols)
        ]

        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")
