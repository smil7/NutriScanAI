import streamlit as st

# Title of the app
st.title("Capture or Upload a Photo")

# Create two columns: one for each button
col1, col2 = st.columns(2)

# Button for taking a photo in the first column
with col1:
    if st.button('Take a Photo'):
        photo = st.camera_input("Take a picture")

        if photo is not None:
            st.image(photo, caption="Captured Image", use_column_width=True)
            st.write("Photo successfully captured!")

# Button for uploading an image in the second column
with col2:
    if st.button('Upload an Image'):
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("Image successfully uploaded!")
