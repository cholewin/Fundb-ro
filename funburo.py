import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
import json
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
DB_FILE = "fundbuero.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# MODEL LADEN
# ---------------------------
@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH, compile=False)
    class_names = open(LABELS_PATH, "r").readlines()
    return model, class_names

model, class_names = load_my_model()

# ---------------------------
# BILD VORBEREITEN
# ---------------------------
def prepare_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# ---------------------------
# DATABASE
# ---------------------------
def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []

def save_database(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ---------------------------
# UI
# ---------------------------
st.title("🏫 Digitales Fundbüro")

menu = st.sidebar.selectbox(
    "Menü",
    ["Gegenstand hochladen", "Durchsuchen"]
)

# ===========================
# UPLOAD
# ===========================
if menu == "Gegenstand hochladen":
    st.header("📸 Neuen Gegenstand melden")

    finder_name = st.text_input("Dein Name")
    location = st.text_input("Fundort (z.B. Aula, Sporthalle)")
    description = st.text_area("Beschreibung (optional)")

    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Kategorie automatisch erkennen"):
            data = prepare_image(image)
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = float(prediction[0][index])

            st.success(f"Erkannte Kategorie: {class_name}")
            st.write(f"Sicherheit: {round(confidence_score*100, 2)} %")

            if st.button("Im Fundbüro speichern"):
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                image.save(filepath)

                db = load_database()
                db.append({
                    "id": len(db) + 1,
                    "category": class_name,
                    "confidence": confidence_score,
                    "image_path": filepath,
                    "finder": finder_name,
                    "location": location,
                    "description": description,
                    "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                    "status": "Offen",
                    "messages": []
                })

                save_database(db)
                st.success("Gegenstand wurde gespeichert!")

# ===========================
# SUCHEN
# ===========================
elif menu == "Durchsuchen":
    st.header("🔎 Gefundene Gegenstände")

    db = load_database()

    if len(db) == 0:
        st.info("Noch keine Gegenstände vorhanden.")
    else:
        categories = list(set(item["category"] for item in db))
        selected_category = st.selectbox("Kategorie", ["Alle"] + categories)

        status_filter = st.selectbox("Status", ["Alle", "Offen", "Abgeholt"])

        for item in db:
            if (selected_category == "Alle" or item["category"] == selected_category) and \
               (status_filter == "Alle" or item["status"] == status_filter):

                st.image(item["image_path"], width=250)
                st.write(f"📂 Kategorie: {item['category']}")
                st.write(f"📍 Fundort: {item['location']}")
                st.write(f"📝 Beschreibung: {item['description']}")
                st.write(f"👤 Finder: {item['finder']}")
                st.write(f"📅 Datum: {item['date']}")
                st.write(f"📌 Status: {item['status']}")

                # Kontaktfunktion
                with st.expander("📬 Nachricht hinterlassen"):
                    message = st.text_area(f"Nachricht für {item['finder']}", key=f"msg_{item['id']}")
                    if st.button("Senden", key=f"send_{item['id']}"):
                        item["messages"].append({
                            "text": message,
                            "date": datetime.now().strftime("%d.%m.%Y %H:%M")
                        })
                        save_database(db)
                        st.success("Nachricht gespeichert!")

                # Nachrichten anzeigen
                if item["messages"]:
                    with st.expander("📨 Nachrichten anzeigen"):
                        for msg in item["messages"]:
                            st.write(f"{msg['date']} - {msg['text']}")

                # Abgeholt Button
                if item["status"] == "Offen":
                    if st.button("✅ Als abgeholt markieren", key=f"done_{item['id']}"):
                        item["status"] = "Abgeholt"
                        save_database(db)
                        st.success("Status geändert!")

                st.write("---")
