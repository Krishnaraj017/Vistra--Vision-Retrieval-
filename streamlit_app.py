import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import pandas as pd
import plotly.express as px

# Configure the application
st.set_page_config(
    page_title="MultiModal RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Set the FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000/"

def main():
    st.title("MultiModal Document Assistant")
    st.subheader("Upload PDFs and ask questions about your documents")
    
    # Sidebar
    with st.sidebar:
        st.subheader("Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        if uploaded_file and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                process_pdf(uploaded_file)
        
        if st.button("View All Document Images"):
            view_all_images()
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:  # assistant
                # Check if this is a JSON response that needs parsing
                if "raw_json" in message:
                    render_json_response(message["raw_json"])
                else:
                    st.write(message.get("content", ""))
                    
                    # Display images if any
                    if "images" in message and message["images"]:
                        display_images(message["images"])
    
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
                response = ask_question(prompt)
                
                # Try to parse as JSON to see if it's a structured response
                try:
                    if isinstance(response, dict):
                        # Already a dict, no need to parse
                        json_response = response
                    else:
                        # Try to parse string as JSON
                        json_response = json.loads(response)
                        
                    # Render the JSON response in a visual format
                    render_json_response(json_response)
                    
                    # Save both raw text and parsed JSON for history
                    message_data = {
                        "role": "assistant",
                        "raw_json": json_response
                    }
                except (json.JSONDecodeError, TypeError):
                    # Not JSON or couldn't parse, just display as text
                    st.write(response)
                    message_data = {
                        "role": "assistant",
                        "content": response
                    }
                
                # Add assistant response to chat history
                st.session_state.messages.append(message_data)

def render_json_response(json_data):
    """Render a JSON response in a visual format"""
    # First, handle the main answer text
    if "answer" in json_data:
        st.markdown(f"### {json_data['answer']}")
    elif "text_response" in json_data:
        st.markdown(f"### {json_data['text_response']}")
    
    # Display key points if available
    if "details" in json_data and "key_points" in json_data["details"]:
        st.subheader("Key Points")
        for point in json_data["details"]["key_points"]:
            st.markdown(f"- {point}")
    
    # Display table data if available
    if "table_analysis" in json_data and "raw_table" in json_data["table_analysis"]:
        st.subheader("Table Data")
        
        table_data = json_data["table_analysis"]["raw_table"]
        try:
            if isinstance(table_data, dict) and "columns" in table_data and "data" in table_data:
                # Format the data as a DataFrame
                columns = table_data["columns"]
                data = table_data["data"]
                
                # Clean data: replace None with empty strings and flatten lists if needed
                cleaned_data = []
                for row in data:
                    cleaned_row = []
                    for cell in row:
                        if cell is None:
                            cleaned_row.append("")
                        else:
                            cleaned_row.append(cell)
                    cleaned_data.append(cleaned_row)
                
                df = pd.DataFrame(cleaned_data, columns=columns)
                st.dataframe(df, use_container_width=True)
            elif isinstance(table_data, list):
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.json(table_data)
        except Exception as e:
            st.error(f"Error displaying table: {e}")
            st.json(table_data)
    
    # Display visualizations if available
    if "visualization" in json_data and "visualizations" in json_data["visualization"]:
        st.subheader("Visualizations")
        
        for viz in json_data["visualization"]["visualizations"]:
            create_visualization(viz)

def create_visualization(viz_data):
    """Create a visualization from the provided data"""
    try:
        title = viz_data.get("title", "Chart")
        viz_type = viz_data.get("type", "line")
        
        st.markdown(f"**{title}**")
        
        # Create dataframe for plotting
        df = pd.DataFrame()
        
        # Handle different data structures
        if "x_axis" in viz_data and ("y_axis" in viz_data or "y_series" in viz_data):
            x_axis = viz_data.get("x_axis", [])
            
            # Add x-axis data
            df["x"] = x_axis
            
            # Process y-series if available (preferred format)
            if "y_series" in viz_data and isinstance(viz_data["y_series"], list):
                for series in viz_data["y_series"]:
                    series_name = series.get("name", "Series")
                    values = series.get("values", [])
                    
                    # Clean values (convert strings to float)
                    clean_values = []
                    for val in values:
                        try:
                            clean_values.append(float(val) if val else None)
                        except (ValueError, TypeError):
                            clean_values.append(None)
                    
                    # Make sure the values list is the right length
                    if len(clean_values) < len(x_axis):
                        clean_values.extend([None] * (len(x_axis) - len(clean_values)))
                    elif len(clean_values) > len(x_axis):
                        clean_values = clean_values[:len(x_axis)]
                    
                    df[series_name] = clean_values
            
            # Process simple y_axis if y_series is not available
            elif "y_axis" in viz_data:
                y_values = viz_data.get("y_axis", [])
                series_name = "Value"
                
                # Clean values
                clean_values = []
                for val in y_values:
                    try:
                        clean_values.append(float(val) if val else None)
                    except (ValueError, TypeError):
                        clean_values.append(None)
                
                df[series_name] = clean_values[:len(x_axis)]
        
        # Create chart based on type
        if viz_type.lower() in ["bar", "column"]:
            # Bar chart
            if len(df.columns) > 1:  # If we have data
                # Melt the dataframe if multiple y columns
                if len(df.columns) > 2:
                    df_melted = df.melt(id_vars="x", var_name="Series", value_name="Value")
                    fig = px.bar(df_melted, x="x", y="Value", color="Series", title=title)
                else:
                    # Simple bar chart if just one y column
                    y_col = df.columns[1]
                    fig = px.bar(df, x="x", y=y_col, title=title)
                
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Value",
                    legend_title="Series"
                )
                
                # Handle long x-axis labels
                if max([len(str(x)) for x in df["x"]]) > 15:
                    fig.update_layout(
                        xaxis=dict(
                            tickangle=45,
                            tickmode='array',
                            tickvals=list(range(len(df["x"]))),
                            ticktext=[str(x)[:15] + "..." if len(str(x)) > 15 else str(x) for x in df["x"]]
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander("Show Data"):
                    st.dataframe(df)
        
        elif viz_type.lower() == "line":
            # Line chart
            if len(df.columns) > 1:  # If we have data
                fig = px.line(df, x="x", y=[col for col in df.columns if col != "x"],
                             title=title, markers=True)
                
                fig.update_layout(
                    xaxis_title="X Axis",
                    yaxis_title="Value",
                    legend_title="Series"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander("Show Data"):
                    st.dataframe(df)
        
        else:
            # Fall back to showing raw data
            st.warning(f"Visualization type '{viz_type}' not supported. Showing raw data instead.")
            st.json(viz_data)
                
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.write("Error details:", str(e))
        st.json(viz_data)  # Fall back to showing the JSON
def display_images(images_data):
    """Display images in a grid"""
    if not images_data:
        return
    
    cols = st.columns(min(3, len(images_data)))
    for i, img_data in enumerate(images_data):
        with cols[i % 3]:
            try:
                # Handle either base64 string directly or dict with data field
                img_b64 = img_data if isinstance(img_data, str) else img_data.get("data", "")
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")

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

def ask_question(question):
    """Send a question to the RAG system"""
    try:
        payload = {
            "question": question,
            "return_sources": True,
            "return_images": True
        }
        
        response = requests.post(
            f"{FASTAPI_URL}/ask-question/",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error processing question: {response.text}")
            return {"text_response": "Error processing your question. Please try again."}
    except Exception as e:
        st.error(f"Error communicating with server: {str(e)}")
        return {"text_response": f"Server communication error: {str(e)}"}

def view_all_images():
    """View all images in the document"""
    try:
        response = requests.get(f"{FASTAPI_URL}/list-images/")
        
        if response.status_code == 200:
            data = response.json()
            
            with st.expander("Document Images", expanded=True):
                st.write(f"Found {data['image_count']} images in the document")
                
                if data['image_count'] > 0 and data.get('image_descriptions'):
                    for i, desc in enumerate(data['image_descriptions']):
                        st.write(f"Image {i+1}: {desc}")
                else:
                    st.write("No image descriptions available.")
        else:
            st.error(f"Error retrieving images: {response.text}")
    except Exception as e:
        st.error(f"Error communicating with server: {str(e)}")

if __name__ == "__main__":
    main()