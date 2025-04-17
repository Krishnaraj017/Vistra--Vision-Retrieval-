import os
import pickle
import uuid
import base64
from typing import List, Dict, Any, Union
from IPython.display import Image, display

# Import necessary libraries - Updated imports to avoid deprecation warnings
from unstructured.partition.pdf import partition_pdf
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import torch



# LangGraph imports for memory
from langgraph.graph import MessagesState

from langchain_community.storage import SQLStore
import sqlite3
class MultiModalRAG:
    """
    A Retrieval Augmented Generation system that processes PDFs with text, tables, and images.
    """
    
    def __init__(
        self, 
        embedding_model="huggingface", 
        llm_model="groq",
        groq_api_key=None,
        google_api_key=None,
        output_path="./content/",
      

    ):
        """
        Initialize the RAG system with the specified models.
        
        Args:
            embedding_model: The model to use for embeddings ("huggingface" or other supported models)
            llm_model: The model to use for text generation ("groq" or other supported models)
            groq_api_key: API key for Groq (if using Groq models)
            google_api_key: API key for Google Generative AI
            output_path: Directory to store output files
            mongodb_uri: Connection string for MongoDB
            mongodb_db: MongoDB database name
            mongodb_collection: MongoDB collection name for document storage
        """
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBPx7f5voGYLDyoh2muB7T7eTu-YGxh3TA"
        os.environ["GROQ_API_KEY"] = "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"
        # Set API keys if provided
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        # Initialize embedding model
        if embedding_model == "huggingface":
            # Detect if CUDA (GPU) is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.embedding_function = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={'device': device},  # Use CUDA if available, otherwise fall back to CPU
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"Using {device} for embeddings")
        else:
            raise ValueError(f"Embedding model {embedding_model} not supported")
            
        # Initialize LLM models with explicit API keys
        if llm_model == "groq":
            self.text_llm = ChatGroq(
                temperature=0, 
                model="llama3-70b-8192",
                api_key=groq_api_key or os.environ.get("GROQ_API_KEY")
            )
            # self.text_llm = ChatGoogleGenerativeAI(
            #     temperature=0,
            #     model="gemini-2.0-flash",  # Gemini model with multimodal capabilities
            #     google_api_key=google_api_key or os.environ.get("GOOGLE_API_KEY")
            # )
            self.vision_llm = ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-2.0-flash",  # Gemini model with multimodal capabilities
                google_api_key=google_api_key or os.environ.get("GOOGLE_API_KEY")
            )
        else:
            raise ValueError(f"LLM model {llm_model} not supported")
            
        # Initialize vector store and retriever
        self.vectorstore = Chroma(
            collection_name="multi_modal_rag", 
            embedding_function=self.embedding_function,
            # persist_directory="./chroma_store",  # or some permanent path
            # persist_directory="./chroma_store"  # or some permanent path
        )
        # try:
        #     # Create MongoDB client
        #     self.mongo_client = MongoClient(mongodb_uri)
            
        #     # Initialize MongoDB store
        #     self.store = MongoDBStore(
        #         client=self.mongo_client,
        #         db_name=mongodb_db,
        #         collection_name=mongodb_collection
        #     )
        #     print(f"Successfully connected to MongoDB at {mongodb_uri}")
            
        # except Exception as e:
        #     print(f"Error connecting to MongoDB: {e}")
        #     print("Falling back to InMemoryStore")
        #     self.store = InMemoryStore()

        # conn = sqlite3.connect("./rag_data.db")

        # self.store = SQLStore(
        #     namespace="multi_modal_rag_documents",
        #     db_url="sqlite:///./rag_data.db",  # SQLite database file
        #     # engine_kwargs={"pool_size": 5},  # Optional connection pool settings
        # )

        # self.store.create_schema()

        self.store = InMemoryStore()

        self.id_key = "doc_id"
        
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )
        
        # Initialize data containers
        self.texts = []
        self.tables = []
        self.images = []
        self.text_summaries = []
        self.table_summaries = []
        self.image_summaries = []
        
        # Initialize memory with LangGraph MemorySaver
        self.history_manager = ConversationHistoryManager()
        self.thread_id = str(uuid.uuid4())
    def serialize_documents(docs):
        """Serialize documents to bytes before storage"""
        return [(doc_id, pickle.dumps(doc)) for doc_id, doc in docs]

    def set_thread_id(self, thread_id=None):
        """
        Set or generate a new thread ID for conversation tracking.
        
        Args:
            thread_id: Optional custom thread ID. If None, generates a new UUID.
        """
        self.thread_id = thread_id if thread_id else str(uuid.uuid4())
        return self.thread_id

    def process_pdf(self, pdf_path, chunking_strategy="by_title", max_characters=10000):
        """
        Process a PDF file to extract text, tables, and images.
        
        Args:
            pdf_path: Path to the PDF file
            chunking_strategy: Strategy for chunking text ("by_title" or "basic")
            max_characters: Maximum number of characters per chunk
        """
        file_path = pdf_path if os.path.isabs(pdf_path) else os.path.join(self.output_path, pdf_path)
        
        # Extract content from PDF
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image","Table"],
            extract_image_block_to_payload=True,
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
            combine_text_under_n_chars=1000,
            new_after_n_chars=5000,
        )
        
        # Separate chunks by type
        self.tables = []
        self.texts = []
        
        for chunk in chunks:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(str(type(chunk)))
            if "Table" in str(type(chunk)):
                self.tables.append(chunk)
            if "CompositeElement" in str(type((chunk))):
                self.texts.append(chunk)
                
        # Extract images from CompositeElements
        self.images = self._get_images_base64(chunks)
        
        print(f"Extracted {len(self.texts)} text chunks, {len(self.tables)} tables, and {len(self.images)} images")
        
        return self

    def _get_images_base64(self, chunks):
        """Extract base64-encoded images from chunks"""
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64

    def display_image(self, index=0):
        """Display an image from the extracted images"""
        if index >= len(self.images):
            print(f"No image at index {index}. Only {len(self.images)} images available.")
            return
            
        image_data = base64.b64decode(self.images[index])
        display(Image(data=image_data))

    def generate_summaries(self):
        """Generate summaries for text, tables, and images"""
        # Define summarization prompt
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Your job is to provide a detailed yet concise summary that includes every important piece of information from the input.

        Ensure that all relevant data points, facts, and key elements are captured in the summary.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}
        """

        summarize_prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | summarize_prompt | self.text_llm | StrOutputParser()
        
        # Summarize text
        self.text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 3})
        
        # Summarize tables
        tables_html = [table.metadata.text_as_html for table in self.tables]
        self.table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
        
        # Generate image descriptions
        prompt_template =  """
        Describe the financial chart or graph in detail. The image is part of a fund factsheet or financial report.
        
        Be specific about:
        1. The type of chart (line chart, bar chart, pie chart, etc.)
        2. The title and what financial data it's displaying
        3. The time period covered (if applicable)
        4. The numerical values shown, including any highs, lows, or trends
        5. Any comparison benchmarks included
        6. Any legends or color coding that indicates different assets or categories
        7. Any annotations or callouts highlighting specific financial events
        8. The generated description is used to compare various funds and their performance.
        For financial data visualization, precision is essential. Provide exact figures where visible.
        """
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        
        image_prompt = ChatPromptTemplate.from_messages(messages)
        image_chain = image_prompt | self.vision_llm | StrOutputParser()
        
        self.image_summaries = image_chain.batch(self.images)
      
        print(f"Generated {len(self.text_summaries)} text summaries, " 
              f"{len(self.table_summaries)} table summaries, and "
              f"{len(self.image_summaries)} image descriptions")
        
        # file_path = "document_chunks.txt"
        # # write splits into the file
        # with open(file_path, "w", encoding="utf-8") as f:
        #     # Extract the text content from each document object
        #     text_contents = self.text_summaries # Adjust attribute if needed
        #     image_contents = self.image_summaries
        #     f.write("\n".join(text_contents) + "\n")
        #     f.write("\n".join(image_contents) + "\n")
        return self

    def index_content(self):
        """Index all content in the vector store"""
        # Add texts (only if there are any)
        if self.texts and self.text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in self.texts]
            summary_texts = [
                Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) 
                for i, summary in enumerate(self.text_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids,self.texts)))
            print(f"Indexed {len(doc_ids)} texts")
        
        # Add tables (only if there are any)
        if self.tables and self.table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in self.tables]
            summary_tables = [
                Document(page_content=summary, metadata={self.id_key: table_ids[i]}) 
                for i, summary in enumerate(self.table_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, self.tables)))
            print(f"Indexed {len(table_ids)} tables")
        
        # Add image summaries (only if there are any)
        if self.images and self.image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in self.images]
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]}) 
                for i, summary in enumerate(self.image_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_img)
            self.retriever.docstore.mset(list(zip(img_ids, self.images)))
            print(f"Indexed {len(img_ids)} images")
        
        return self

    def _parse_docs(self, docs):
        """Split retrieved documents into base64-encoded images and texts"""
        b64_images = []
        text_docs = []
        
        for doc in docs:
            # Check if it's a base64 image
            try:
                base64.b64decode(doc)
                b64_images.append(doc)
            except:
                text_docs.append(doc)
                
        return {"images": b64_images, "texts": text_docs}

    def _get_message_history(self):
        """Retrieve conversation history"""
        return self.history_manager.get_history(self.thread_id)

    def _build_prompt(self, kwargs):
        """Build a prompt with text context, images, and conversation history"""
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]
        chat_history = kwargs.get("chat_history", [])
        
        # Compile text context
        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text
        
        # Format conversation history
        history_text = ""
        if chat_history:
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    history_text += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_text += f"AI: {message.content}\n"
        print(context_text)
        # Construct prompt with history
        prompt_template = f"""
            Answer the question based only on the following context, which can include text, tables, and images.

            Your response MUST be a single JSON object with the following structure:
            {{
                "answer": "A direct and complete in detail answer based strictly on the provided context",
                "details": {{
                    "key_points": ["List of key information extracted from context"],
                    "source_references": ["Any reference identifiers found in context"]
                }},
                "table_analysis": {{
                    "structure": "Description of table structure or null if no table present",
                    "headers": ["List of table headers or null if no table"],
                    "row_count": "Number of data rows or null if no table",
                    "key_metrics": ["Important metrics extracted from table"],
                    "patterns": ["Patterns or trends observed in tabular data"],
                    "raw_table": {{
                        "columns": ["Column names exactly as presented in the table"],
                        "data": [
                            ["Row 1 values in order of columns"],
                            ["Row 2 values in order of columns"],
                            ["And so on..."]
                        ]
                    }}
                }},
                "comparison": {{
                    "compared_values": ["List of compared items or null if not applicable"],
                    "basis": "On what basis comparison is made or null if not applicable",
                    "result": "Summary of comparison or null if not applicable",
                    "graph_type": "multi_comparison|single_comparison|trend_analysis|null"
                }},
                "visualization": {{
                    "visualizations": [
                        {{
                            "title": "Suitable graph title",
                            "type": "bar|line|pie|scatter|stacked_bar|radar|bubble|heatmap|table",
                            "x_axis": ["List of labels or categories"],
                            "y_axis": ["List of corresponding values"],
                            "data_labels": ["Optional labels for data points"],
                            "y_series": [
                                {{"name": "Series name", "values": [series values]}},
                                {{"name": "Series name", "values": [series values]}}
                            ],
                            "table_data": {{
                                "headers": ["Column headers for table visualization"],
                                "rows": [
                                    ["Row 1 values"],
                                    ["Row 2 values"]
                                ]
                            }},
                            "description": "Brief explanation of why this visualization type fits the data"
                        }}
                    ]
                }}
            }}

            Rules:
            - Return ONLY the JSON object, with no additional text before or after.
            - Every property must be populated, use null for inapplicable properties.
            - Provide all values exactly as found in the context. Do NOT shorten or paraphrase any values.
            - For tables: preserve exact cell values, maintain row/column relationships, and capture all table content completely.
            - For tables: include the raw_table property with the exact structure and values from the original table.
            - For tabular data: identify patterns, summarize distributions, calculate totals/averages when relevant.
            - Use the conversation history only when relevant.
            - If the answer cannot be determined based on the provided context, set "answer" to "Unable to determine from provided context" and use null for all other fields.
            - Remove all newline characters (`\\n`) and triple quotes (`\"\"\"`) from the JSON output.
            - Ensure the JSON output is compact (no unnecessary whitespace) and properly escaped.
            - Keep the answere dont know if question is not related to the document.
            - If the question is like Can i invest in this fund, politely answere i'm not authorized to answere this question.
            - If there is no context ,jsut say "I'm unable to answere to this question".

           
            Previous conversation:
            {history_text}

            Context:
            {context_text}

            Current question:
            {user_question}
            """
        prompt_content = [{"type": "text", "text": prompt_template}]

                    # Add images to prompt if available
        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )
        
        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def setup_chain(self):
        """Set up the RAG chain for answering questions with memory"""
        self.chain = (
            {
                "context": self.retriever | RunnableLambda(self._parse_docs),
                "question": RunnablePassthrough(),
                "chat_history": lambda _: self._get_message_history()
            }
            | RunnableLambda(self._build_prompt)
            | self.vision_llm
            | StrOutputParser()
        )
        
        # Chain with sources for debugging
        self.chain_with_sources = {
            "context": self.retriever | RunnableLambda(self._parse_docs),
            "question": RunnablePassthrough(),
            "chat_history": lambda _: self._get_message_history()
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self._build_prompt)
                | self.vision_llm
                | StrOutputParser()
            )
        )
        
        return self


    def _save_to_memory(self, question, response):
        """Save a message exchange to memory"""
        return self.history_manager.save_exchange(self.thread_id, question, response)


    def ask(self, question: str, return_sources: bool = False):
        """
        Ask a question to the RAG system.
        
        Args:
            question: The question to ask
            return_sources: Whether to return the sources used for the answer
            
        Returns:
            The answer or a dict with answer and sources
        """
        if return_sources:
            result = self.chain_with_sources.invoke(question)
            # Update memory with the new exchange
            self._save_to_memory(question, result["response"])
            return result
        else:
            response = self.chain.invoke(question)
            # Update memory with the new exchange
            self._save_to_memory(question, response)
            return response
            
    def ask_with_images(self, question, max_images=3):
        """Ask a question and display images if requested"""
        response = self.ask(question)
        
        # Check if the query is about showing images
        if "show" in question.lower() and "image" in question.lower():
            print("\nShowing images from the document:")
            num_to_display = min(max_images, len(self.images))
            for i in range(num_to_display):
                print(f"\nImage {i+1}:")
                self.display_image(i)
        
        return response
    
    def ask_and_save_images(self, question, output_dir="./saved_images"):
        """Ask a question and save images to files if the query is about showing images"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        response = self.ask(question)
        print(response)
        print(f"\nQuestion: {question}")
        print(f"Answer: {response}")
        
        # Check if the query is asking to show images
        if "show" in question.lower() and "image" in question.lower():
            print("\nSaving first 3 images from the document...")
            num_to_save = len(self.images)
            for i in range(num_to_save):
                if i < len(self.images):
                    image_data = base64.b64decode(self.images[i])
                    file_path = os.path.join(output_dir, f"image_{i+1}.png")
                    
                    with open(file_path, "wb") as img_file:
                        img_file.write(image_data)
                    
                    print(f"Image {i+1} saved successfully")
            
            print(f"All images have been saved to the '{output_dir}' directory")
        
        return response
            
    def clear_memory(self):
        """Clear the conversation memory"""
        self.history_manager.clear_history(self.thread_id)
        print("Conversation memory has been cleared")

class ConversationHistoryManager:
    """A simple Python-based conversation history manager."""
    
    def __init__(self):
        """Initialize an empty conversation history store."""
        self.conversations = {}  # Dictionary to store conversations by thread_id
    
    def get_history(self, thread_id):
        """
        Retrieve conversation history for a specific thread.
        
        Args:
            thread_id: The unique identifier for the conversation thread
            
        Returns:
            A list of message objects or an empty list if no history exists
        """
        return self.conversations.get(thread_id, [])
    
    def save_exchange(self, thread_id, question, response):
        """
        Save a message exchange to the conversation history.
        
        Args:
            thread_id: The unique identifier for the conversation thread
            question: The human's question/message
            response: The AI's response
            
        Returns:
            The updated list of messages
        """
        # Get existing messages or create a new list
        messages = self.conversations.get(thread_id, [])
        
        # Add new messages
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=response))
        
        # Update the conversation store
        self.conversations[thread_id] = messages
        
        return messages
    
    def clear_history(self, thread_id):
        """
        Clear the conversation history for a specific thread.
        
        Args:
            thread_id: The unique identifier for the conversation thread
        """
        if thread_id in self.conversations:
            self.conversations[thread_id] = []