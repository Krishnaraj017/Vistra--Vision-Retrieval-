import streamlit as st
import requests
import json
import base64
from PIL import Image
import io

# Configure the application
st.set_page_config(
    page_title="MultiModal RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Set the FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000/"

def main():
    # st.title("MultiModal Document Assistant")
    st.subheader("Upload PDFs and ask questions about your documents")
    
    # Sidebar
    with st.sidebar:
        # st.header("Controls")
        
        # PDF Upload
        st.subheader("Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        # Process PDF button
        if uploaded_file and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                process_pdf(uploaded_file)
        
        # # Clear memory/conversation
        # if st.button("Clear Conversation History"):
        #     clear_memory()
        #     st.session_state.messages = []
        #     st.success("Conversation history cleared!")
        
        # Options for retrieval
        # st.subheader("Retrieval Options")
        # return_sources = st.checkbox("Show sources", value=False)
        # return_images = st.checkbox("Include images", value=True)
        
        # View all images
        if st.button("View All Document Images"):
            view_all_images()
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "images" in message and message["images"]:
                st.write(message["content"])
                # Display images in a grid using columns
                cols = st.columns(min(3, len(message["images"])))
                for i, img_data in enumerate(message["images"]):
                    with cols[i % 3]:
                        try:
                            # Decode base64 image
                            img_bytes = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = ask_question(prompt, False, False)
                print(response)
                # Display text response
                st.write(response["text_response"])
                
                # Display images if any
                # if return_images and "images" in response and response["images"]:
                #     # Display images in a grid using columns
                #     cols = st.columns(min(3, len(response["images"])))
                #     for i, img_data in enumerate(response["images"]):
                #         with cols[i % 3]:
                #             try:
                #                 # Decode base64 image
                #                 img_bytes = base64.b64decode(img_data["data"])
                #                 img = Image.open(io.BytesIO(img_bytes))
                #                 st.image(img, use_column_width=True)
                #             except Exception as e:
                #                 st.error(f"Error displaying image: {str(e)}")
                
                # # Display sources if requested
                # if return_sources and "sources" in response and response["sources"]:
                #     with st.expander("Sources"):
                #         st.json(response["sources"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["text_response"],
            "images": [img["data"] for img in response["images"]] if "images" in response and response["images"] else []
        })

def process_pdf(pdf_file):
    """Upload and process a PDF file"""
    try:
        files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
        response = requests.post(f"{FASTAPI_URL}/upload-pdf/", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"PDF processed successfully! Found {result['stats']['text_chunks']} text chunks, {result['stats']['tables']} tables, and {result['stats']['images']} images.")
            return True
        else:
            st.error(f"Error processing PDF: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error communicating with server: {str(e)}")
        return False

def ask_question(question, return_sources=False, return_images=True):
    """Send a question to the RAG system"""
    try:
        payload = {
            "question": question,
            "return_sources": return_sources,
            "return_images": return_images
        }
        
        response = requests.post(
            f"{FASTAPI_URL}/ask-question/",
            json=payload
        )
        
        if response.status_code == 200:
            print("___________________________________")
            print(response)
            return response.json()
        else:
            st.error(f"Error processing question: {response.text}")
            return {"text_response": "Error processing your question. Please try again."}
    except Exception as e:
        st.error(f"Error communicating with server: {str(e)}")
        return {"text_response": f"Server communication error: {str(e)}"}

def clear_memory():
    """Clear the conversation memory"""
    try:
        response = requests.get(f"{FASTAPI_URL}/clear-memory/")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error clearing memory: {str(e)}")
        return False

def view_all_images():
    """View all images in the document"""
    try:
        response = requests.get(f"{FASTAPI_URL}/list-images/")
        
        if response.status_code == 200:
            data = response.json()
            
            # Create a new tab/window for images
            with st.expander("Document Images", expanded=True):
                st.write(f"Found {data['image_count']} images in the document")
                
                if data['image_count'] > 0:
                    # If there are image descriptions, display them
                    if data['image_descriptions'] and len(data['image_descriptions']) > 0:
                        for i, desc in enumerate(data['image_descriptions']):
                            st.write(f"Image {i+1}: {desc}")
                    else:
                        st.write("No image descriptions available.")
                else:
                    st.write("No images found in the document.")
        else:
            st.error(f"Error retrieving images: {response.text}")
    except Exception as e:
        st.error(f"Error communicating with server: {str(e)}")

if __name__ == "__main__":
    main()

