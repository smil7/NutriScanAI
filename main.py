import os
import streamlit as st
#import geneai as genai
from PIL import Image

#title
st.title("NutriScan AI")

#init session variable at the start once
# if 'model' not in st.session_state:
#     st.session_state['model'] = genai(api_key=os.getenv('GENEAI_API_KEY'))

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Move everything to the sidebar
with st.sidebar:
    st.sidebar.title("NutriScan AI")
    st.markdown("Image Upload or Capture App")
    # Let the user choose between uploading or capturing
    option = st.radio("Choose an option:", ("Upload an Image", "Capture a Photo"))

    # Handle the chosen option
    if option == "Upload an Image":
        # File uploader widget
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Save the uploaded image
            save_dir = "uploaded_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image_path = os.path.join(save_dir, uploaded_image.name)
            image = Image.open(uploaded_image)
            image.save(image_path)
            # st.success(f"Image saved at: `{image_path}`")
            store_button = st.button("Store")

    elif option == "Capture a Photo":
        # Camera input widget
        captured_image = st.camera_input("Take a photo")
        
        if captured_image is not None:
            # Save the captured image
            save_dir = "captured_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image_path = os.path.join(save_dir, "captured_photo.jpg")
            with open(image_path, "wb") as f:
                f.write(captured_image.getvalue())
            #st.success(f"Image saved at: `{image_path}`")
            store_button = st.button("Store")
    
    




#update the interface with previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#create chat interface
if prompt := st.text_input("Chat with us"):
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

#get response from the model
# with st.chat_message('assistant'):
#     client = st.session_state['model']
#     stream = client.chat.completions.create(
#         model='gpt-3.5-turbo',
#         messages=[
#             {"role": message['role'], "content": message["content"]} for message in st.session_state['messages']
#         ],
#         stream = True
#     )
# response = st.write_stream(stream)
# st.session_state['messages'].append({'role': 'assistant', 'content': response})


#handle message overflow based on the model size
