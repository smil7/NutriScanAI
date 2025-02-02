import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page title
st.title("🚀 Hi I'm NutriScan AI!")

# Sidebar for user input
st.text_input("User Input")
name = st.sidebar.text_input("Enter your name:")
age = st.sidebar.slider("Select your age", 0, 100, 25)

# Display personalized message
st.write(f"Hello, {name}! You are {age} years old.")

# Generate random data
st.subheader("📊 Random Data Visualization")
data = pd.DataFrame(
    np.random.randn(20, 3), columns=["A", "B", "C"]
)

# Show the dataframe
st.dataframe(data)

# Plot the data
fig, ax = plt.subplots()
ax.plot(data["A"], label="A")
ax.plot(data["B"], label="B")
ax.plot(data["C"], label="C")
ax.legend()
st.pyplot(fig)

# File Upload
st.subheader("📂 Upload a File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df)

# Button Example
if st.button("Click Me!"):
    st.success("You clicked the button! 🎉")

# Display an image
st.subheader("🖼️ Display an Image")
st.image("https://picsum.photos/400", caption="Random Image")

# Camera input
st.subheader("📸 Take a Picture")
picture = st.camera_input("Take a photo")
if picture:
    st.image(picture, caption="Captured Image")
