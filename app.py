import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pdf2image import convert_from_bytes

st.set_page_config(page_title="Multilingual OCR", layout="wide")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

st.title("📄 Multilingual OCR System")

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:

    extracted_text = []

    try:
        # PDF
        if uploaded_file.type == "application/pdf":

            pages = convert_from_bytes(uploaded_file.read())

            for page in pages:
                img_array = np.array(page)

                results = reader.readtext(img_array)

                for res in results:
                    extracted_text.append(res[1])

            st.success("PDF Processed Successfully")

        # IMAGE
        else:

            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            st.image(image, caption="Uploaded Image")

            results = reader.readtext(img_array)

            for (bbox, text, prob) in results:

                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                cv2.rectangle(
                    img_array,
                    top_left,
                    bottom_right,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img_array,
                    text,
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                extracted_text.append(text)

            st.image(
                img_array,
                caption="Detected Text",
                use_container_width=True
            )

        st.subheader("Extracted Text")

        st.text_area(
            "OCR Output",
            "\n".join(extracted_text),
            height=250
        )

        st.subheader("Structured Table")

        cols = 5

        rows = [
            extracted_text[i:i + cols]
            for i in range(0, len(extracted_text), cols)
        ]

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
