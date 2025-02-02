import os
import streamlit as st
import google.generativeai as genai
import utils
import faiss
from PIL import Image

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#title
st.title("NutriScan AI")

#init session variable at the start once
# if 'model' not in st.session_state:
#     st.session_state['model'] = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ''

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

if 'count' not in st.session_state:
	st.session_state.count = 1

def increment_counter():
	st.session_state.count += 1

def fetch_gemini_response(user_query):
    # Use the session's model to generate a response
    response = st.session_state.messages.model.generate_content(user_query)
    print(f"Gemini's Response: {response}")
    return response.parts[0].text

# Move everything to the sidebar
with st.sidebar:
    st.sidebar.title("NutriScan AI")
    st.markdown("Image Upload or Capture App")
    # Let the user choose between uploading or capturing
    option = st.radio("Choose an option:", ("Upload an Image", "Capture a Photo"))

    # Handle the chosen option
    if option == "Upload an Image":
        # File uploader widget
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_image is not None:
            # Save the uploaded image
            save_dir = "uploaded_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # directory_path = '\uploaded_images'
            images = [img for img in uploaded_image]
            #print(images)
            # st.write(len(uploaded_image))
            for img in images:
                image_path = os.path.join(save_dir, img.name)
                image = Image.open(img)
                image.save(image_path)
        
            directory_path = "./uploaded_images"
            print("Files: ")
            files = os.listdir(directory_path)
            print(files)
            print('From uploaded image condition: ')
            files_path = [os.path.join(directory_path, file).replace('\\', '/') for file in os.listdir(directory_path)]
            print(files_path)
            
            if st.button("Store"):
                txts_list = utils.extract_text_from_image(files_path)
                for txt in txts_list:
                    st.session_state["extracted_text"] += txt

    elif option == "Capture a Photo":
        st.button('Increment', on_click=increment_counter)
        st.write('Count = ', st.session_state.count)
        # Camera input widget
        captured_image = st.camera_input("Take a photo")

        if captured_image is not None:
            # Save the captured image
            save_dir = "captured_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image_path = os.path.join(save_dir, "captured_photo_" + str(st.session_state.count) + ".jpg")

            with open(image_path, "wb") as f:
                f.write(captured_image.getvalue())

            directory_path = "./captured_images"
            print("Files: ")
            files = os.listdir(directory_path)
            print(files)
            print('From capture a photo condition: ')
            files_path = [os.path.join(directory_path, file).replace('\\', '/') for file in os.listdir(directory_path)]
            print(files_path)
           
            if st.button("Store"):
                txts_list = utils.extract_text_from_image(files_path)
                for txt in txts_list:
                    st.session_state["extracted_text"] += txt

gemini_model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction="""You are an expert at analyzing nutrition facts from products and providing a recommendations for the user.
    Your task is to engage in conversations about nutritions, dietry and symptoms and answer questions.
    Use the nutrition facts that are stored in the vector database as well as the prompt the user has prompted
    to provide a specific recommendation for their usecase. TWO IMPORTANT NOTEs:
    1) If there was a prompt that the user entered is outside your expertise are then inform the user that this query is not related and out of context
    2) Please make your response consie and clear""",
)

print("EXtraacted texts: ")
print(st.session_state["extracted_text"])

if 'messages' not in st.session_state:
    st.session_state.messages = gemini_model.start_chat(history=[])

#update the interface with previous messages
for message in st.session_state.messages.history:
    with st.chat_message(utils.map_role(message['role'])):
        st.markdown(message['content'])

all_input = st.session_state["extracted_text"]
#create chat interface
if prompt := st.text_input("Chat with us"):
    
    all_input += "\n" + prompt
    st.chat_message('user').markdown(prompt)
    gemini_response = fetch_gemini_response(all_input)

    #get response from the model
    with st.chat_message('assistant'):
        st.markdown(gemini_response)

        st.session_state.messages.history.append({"role": "user", "content": all_input})
        st.session_state.messages.history.append({"role": "model", "content": gemini_response})

