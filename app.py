# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import pickle
# # import requests
# # from PIL import Image
# # from io import BytesIO


# # # Load the trained model
# # with open('rfc.pkl', 'rb') as file:
# #     model = pickle.load(file)


# # # Creating web app
# # st.title("Forest Cover Type Prediction")

# # # Image URL
# # url = "https://media.springernature.com/lw1200/springer-static/image/art%3A10.1038%2Fs41598-023-50863-1/MediaObjects/41598_2023_50863_Fig7_HTML.png"

# # # Fetch and open image correctly
# # response = requests.get(url)
# # img = Image.open(BytesIO(response.content))

# # st.image(img, caption='Forest Cover Types', use_column_width=True)
# # user_input = str.text_input("Enter the following features to predict the forest cover type:")

# # if user_input:
# #     user_input = user_input.split(',')
# #     features = np.array([user_input], dtype=np.float64).reshape(1, -1)
# #     prediction = model.predict(features)
# #     st.write(f"The predicted forest cover type is: {prediction[0]}")
# #     prediction = int(prediction[0])


# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# import requests
# from PIL import Image
# from io import BytesIO

# # Load the trained model
# with open('rfc.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Creating web app
# st.title("Forest Cover Type Prediction")

# # Image URL
# url = "https://media.springernature.com/lw1200/springer-static/image/art%3A10.1038%2Fs41598-023-50863-1/MediaObjects/41598_2023_50863_Fig7_HTML.png"

# # Fetch and display image safely
# try:
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     st.image(img, caption='Forest Cover Types', use_column_width=True)
# except:
#     st.warning("Could not load image.")

# # User input
# user_input = st.text_input(
#     "Enter the features separated by commas (example: 2596,51,3,258,0,510,221,232,148,6279,...):"
# )

# if user_input:
#     try:
#         # Convert input string into list of floats
#         user_input_list = [float(x.strip()) for x in user_input.split(',')]

#         # Convert to numpy array
#         features = np.array(user_input_list).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(features)
#         predicted_class = int(prediction[0])

#         st.success(f"The predicted forest cover type is: {predicted_class}")

#     except ValueError:
#         st.error("Please enter only numeric values separated by commas.")
#     except Exception as e:
#         st.error(f"Error during prediction: {e}")

# cover_image_dict = {
#     1: {
#         'name': 'Spruce/Fir',
#         'image_url': 'https://images.unsplash.com/photo-1501785888041-af3ef285b470'
#     },
#     2: {
#         'name': 'Lodgepole Pine',
#         'image_url': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e'
#     },
#     3: {
#         'name': 'Ponderosa Pine',
#         'image_url': 'https://images.unsplash.com/photo-1470770903676-69b98201ea1c'
#     },
#     4: {
#         'name': 'Cottonwood/Willow',
#         'image_url': 'https://images.unsplash.com/photo-1502082553048-f009c37129b9'
#     },
#     5: {
#         'name': 'Aspen',
#         'image_url': 'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee'
#     },
#     6: {
#         'name': 'Douglas-fir',
#         'image_url': 'https://images.unsplash.com/photo-1448375240586-882707db888b'
#     },
#     7: {
#         'name': 'Krummholz',
#         'image_url': 'https://images.unsplash.com/photo-1469474968028-56623f02e42e'
#     }
# }

# if user_input:
#     predicted_cover = cover_image_dict.get(predicted_class, None)
#     if predicted_cover:
#         st.subheader(f"Predicted Cover Type: {predicted_cover['name']}")
#         try:
#             response = requests.get(predicted_cover['image_url'])
#             img = Image.open(BytesIO(response.content))
#             st.image(img, caption=predicted_cover['name'], use_column_width=True)
#         except:
#             st.warning("Could not load cover type image.")

#------------------------------------------------------------------------------------
# import streamlit as st
# import numpy as np
# import pickle
# import re

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="Forest Cover Prediction",
#     page_icon="🌲",
#     layout="wide"
# )

# # ---------------- LOAD MODEL ----------------
# @st.cache_resource
# def load_model():
#     with open("rfc.pkl", "rb") as f:
#         return pickle.load(f)

# model = load_model()

# # ✅ Automatically detect feature count from model
# EXPECTED_FEATURES = model.n_features_in_

# # ---------------- HEADER ----------------
# st.markdown(
#     "<h1 style='text-align:center;'>🌲 Forest Cover Type Prediction</h1>",
#     unsafe_allow_html=True
# )
# st.markdown(
#     "<p style='text-align:center;'>Production-Ready ML Deployment</p>",
#     unsafe_allow_html=True
# )

# st.divider()

# # ---------------- IMAGE DICTIONARY ----------------
# cover_image_dict = {
#     1: {"name": "Spruce/Fir", "image_url": "https://images.unsplash.com/photo-1501785888041-af3ef285b470"},
#     2: {"name": "Lodgepole Pine", "image_url": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e"},
#     3: {"name": "Ponderosa Pine", "image_url": "https://images.unsplash.com/photo-1470770903676-69b98201ea1c"},
#     4: {"name": "Cottonwood/Willow", "image_url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9"},
#     5: {"name": "Aspen", "image_url": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee"},
#     6: {"name": "Douglas-fir", "image_url": "https://images.unsplash.com/photo-1448375240586-882707db888b"},
#     7: {"name": "Krummholz", "image_url": "https://images.unsplash.com/photo-1469474968028-56623f02e42e"},
# }

# # ---------------- INPUT AREA ----------------
# raw_input = st.text_area(
#     f"📥 Paste comma-separated feature values (Model expects {EXPECTED_FEATURES} features):",
#     placeholder="You can paste values with spaces, line breaks, or trailing commas..."
# )

# # ---------------- SMART PARSING ----------------
# if raw_input:

#     # Extract only numeric values safely
#     values = re.findall(r"-?\d+\.?\d*", raw_input)
#     values = [float(v) for v in values]

#     st.info(f"📊 Detected Features: {len(values)} / {EXPECTED_FEATURES}")

#     # Auto-trim extra values
#     if len(values) > EXPECTED_FEATURES:
#         values = values[:EXPECTED_FEATURES]
#         st.warning("Extra values trimmed automatically.")

#     # Auto-pad missing values
#     if len(values) < EXPECTED_FEATURES:
#         values += [0.0] * (EXPECTED_FEATURES - len(values))
#         st.warning("Missing values padded with 0.")

#     # ---------------- PREDICT BUTTON ----------------
#     if st.button("🔍 Predict", use_container_width=True):

#         try:
#             features = np.array(values).reshape(1, -1)

#             prediction = model.predict(features)[0]

#             # Confidence score (if available)
#             if hasattr(model, "predict_proba"):
#                 probability = np.max(model.predict_proba(features)) * 100
#                 probability = round(probability, 2)
#             else:
#                 probability = None

#             predicted_cover = cover_image_dict.get(prediction)

#             if predicted_cover:

#                 st.divider()

#                 col1, col2 = st.columns([1, 2])

#                 # LEFT COLUMN - RESULT CARD
#                 with col1:
#                     st.markdown("### 🌳 Prediction Result")
#                     st.markdown(
#                         f"""
#                         <div style="
#                             background-color:#f5f7fa;
#                             padding:30px;
#                             border-radius:15px;
#                             text-align:center;
#                             box-shadow: 0 6px 18px rgba(0,0,0,0.15);">
#                             <h2 style='color:#2e7d32;'>{predicted_cover['name']}</h2>
#                             <p style='font-size:18px;'>
#                             Confidence: {probability if probability else "N/A"}%
#                             </p>
#                         </div>
#                         """,
#                         unsafe_allow_html=True
#                     )

#                 # RIGHT COLUMN - RESPONSIVE IMAGE
#                 with col2:
#                     st.image(
#                         predicted_cover["image_url"] + "?auto=format&fit=crop&w=1400&q=80",
#                         use_column_width=True
#                     )

#             else:
#                 st.error("Prediction class not found in dictionary.")

#         except Exception as e:
#             st.error(f"Prediction Error: {e}")

# # ---------------- FOOTER ----------------
# st.divider()
# st.markdown(
#     "<p style='text-align:center;color:gray;'>Built with ❤️ | Random Forest Classifier | Streamlit Deployment</p>",
#     unsafe_allow_html=True
# )

#------------------------------------------------------------------------------------


import streamlit as st
import numpy as np
import pandas as pd
import requests
import re
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Forest Cover Prediction", page_icon="🌲", layout="wide")

st.title("🌲 Forest Cover Type Prediction")
st.caption("Frontend connected to FastAPI backend")

# ---------------- COVER IMAGE DICTIONARY ----------------
cover_image_dict = {
    1: {"name": "Spruce/Fir", "image_url": "https://images.unsplash.com/photo-1501785888041-af3ef285b470"},
    2: {"name": "Lodgepole Pine", "image_url": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e"},
    3: {"name": "Ponderosa Pine", "image_url": "https://images.unsplash.com/photo-1470770903676-69b98201ea1c"},
    4: {"name": "Cottonwood/Willow", "image_url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9"},
    5: {"name": "Aspen", "image_url": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee"},
    6: {"name": "Douglas-fir", "image_url": "https://images.unsplash.com/photo-1448375240586-882707db888b"},
    7: {"name": "Krummholz", "image_url": "https://images.unsplash.com/photo-1469474968028-56623f02e42e"},
}

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- GET MODEL INFO ----------------
try:
    response = requests.get(f"{API_URL}/model-info")
    model_info = response.json()
    EXPECTED_FEATURES = model_info["expected_features"]
    FEATURE_NAMES = model_info["feature_names"]
except:
    EXPECTED_FEATURES = 55
    FEATURE_NAMES = [f"Feature_{i+1}" for i in range(EXPECTED_FEATURES)]

# ---------------- SHOW FEATURE NAMES ----------------
with st.expander("📋 View Feature Names"):
    st.write(pd.DataFrame({"Feature Names": FEATURE_NAMES}))

# ---------------- INPUT ----------------
raw_input = st.text_area(
    f"Paste comma-separated values (Model expects {EXPECTED_FEATURES} features)"
)

if raw_input:

    values = re.findall(r"-?\d+\.?\d*", raw_input)
    values = [float(v) for v in values]

    st.info(f"Detected Features: {len(values)} / {EXPECTED_FEATURES}")

    if len(values) > EXPECTED_FEATURES:
        values = values[:EXPECTED_FEATURES]
        st.warning("Extra values trimmed.")

    if len(values) < EXPECTED_FEATURES:
        values += [0.0] * (EXPECTED_FEATURES - len(values))
        st.warning("Missing values padded with 0.")

    if st.button("🔍 Predict"):

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": values}
            )

            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                prediction = result["prediction"]
                probability = result["confidence"]

                predicted_cover = cover_image_dict.get(prediction)

                if predicted_cover:

                    st.divider()

                    col1, col2 = st.columns([1, 2])

                    # LEFT COLUMN - RESULT CARD
                    with col1:
                        st.markdown("### 🌳 Prediction Result")

                        st.markdown(
                            f"""
                            <div style="
                                background-color:#f5f7fa;
                                padding:30px;
                                border-radius:15px;
                                text-align:center;
                                box-shadow: 0 6px 18px rgba(0,0,0,0.15);">
                                <h2 style='color:#2e7d32;'>{predicted_cover['name']}</h2>
                                <p style='font-size:18px;'>
                                Confidence: {round(probability,2) if probability else "N/A"}%
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # RIGHT COLUMN - RESPONSIVE IMAGE
                    with col2:
                        st.image(
                            predicted_cover["image_url"] + "?auto=format&fit=crop&w=1400&q=80",
                            use_column_width=True
                        )

                # Save history
                st.session_state.history.append({
                    "Prediction": prediction,
                    "Forest Type": predicted_cover["name"] if predicted_cover else prediction,
                    "Confidence (%)": probability
                })

        except Exception as e:
            st.error(f"API Connection Error: {e}")

# ---------------- HISTORY TABLE ----------------
if st.session_state.history:
    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# ---------------- FEATURE IMPORTANCE ----------------
if st.button("📊 Show Feature Importance"):

    try:
        response = requests.get(f"{API_URL}/feature-importance")
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame({
                "Feature": data["feature_names"],
                "Importance": data["importance"]
            }).sort_values(by="Importance", ascending=False).head(15)

            fig, ax = plt.subplots()
            ax.barh(df["Feature"], df["Importance"])
            ax.invert_yaxis()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not fetch feature importance: {e}")