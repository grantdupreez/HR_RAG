import streamlit as st
import os
import anthropic
import tempfile
import uuid
import shutil
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
import qdrant_client
import hmac

def check_password():
    """Returns `True` if the user had a correct password and sets user role."""
    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            
            # Check if user is admin and set role
            if st.session_state["username"] == "admin":  # You can define admin users in secrets too
                st.session_state["user_role"] = "hr_admin"
            else:
                st.session_state["user_role"] = "employee"
                
            del st.session_state["password"]  # Don't store the password
            del st.session_state["username"]  # Don't store the username
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("🔒 User not known or password incorrect")
    return False

# Logout function
def logout():
    # Clear the session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Check if logged in
if not check_password():
    st.stop()

# Initialize session state for user role if it doesn't exist
if "user_role" not in st.session_state:
    st.session_state.user_role = "employee"  # Default to standard employee

# Set up API keys
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # For embeddings

# Qdrant server configuration - from secrets
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_COLLECTION = st.secrets["QDRANT_COLLECTION"]

# Initialize the Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Define paths for uploads
UPLOAD_DIR = "hr_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to get Qdrant client
def get_qdrant_client():
    if QDRANT_API_KEY:
        return qdrant_client.QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    else:
        return qdrant_client.QdrantClient(url=QDRANT_URL)

# Function to save uploaded PDFs
def save_uploaded_files(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        # Create a unique file name
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # Write the file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        file_paths.append(file_path)
    
    return file_paths

# Function to process PDFs and create/update Qdrant vector database
def process_documents(file_paths, collection_name):
    documents = []
    
    # Load all PDFs
    for file_path in file_paths:
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error loading {os.path.basename(file_path)}: {str(e)}")
    
    if not documents:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Set up Qdrant client (server mode)
    qdrant_client_instance = get_qdrant_client()
    
    # Check if collection exists
    try:
        collections = qdrant_client_instance.get_collections().collections
        collection_exists = any(collection.name == collection_name for collection in collections)
    except Exception as e:
        st.error(f"Error connecting to Qdrant server: {str(e)}")
        st.info("Please check your Qdrant server URL and API key in secrets.")
        return None
    
    # Set up embeddings
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings()
    
    # Create or update vector store
    if collection_exists:
        # Get existing vectorstore and add new documents
        vectorstore = Qdrant(
            client=qdrant_client_instance,
            collection_name=collection_name,
            embeddings=embeddings
        )
        vectorstore.add_documents(chunks)
        st.info(f"Added new documents to existing collection: {collection_name}")
    else:
        # Create new vectorstore with server URL
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            collection_name=collection_name,
            force_recreate=True
        )
        st.success(f"Created new collection: {collection_name}")
    
    return vectorstore, qdrant_client_instance

# Function to generate response using Claude with RAG and conversation memory
def generate_response(user_query, context, conversation_history):
    # Create a system prompt with context - HR focused
    system_prompt = f"""You are CareerVertex, a helpful HR assistant that answers employee questions about workplace benefits, 
    company policies, and HR procedures based on the provided context from company policy documents.
    
    As an HR assistant, your goal is to help employees understand their benefits, rights, and company policies 
    accurately and clearly. Use a professional but approachable tone.
    
    You should provide information based strictly on the provided context from policy documents.
    If you don't know the answer or if the information is not in the context, politely explain that 
    you cannot find this specific information in the available documents and suggest contacting the 
    HR department for more details.

    Use British English spelling and terminology (organisation, centre, programme, etc.)

    Context information is below:
    {context}

    When answering, reference specific information from the HR documents when possible.
    """

    # Format conversation history - Initialize as empty list
    claude_messages = [] 

    # Add a limited number of previous messages to avoid token limits
    # Using the most recent 5 exchanges (10 messages)
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
    for message in recent_history:
        # Ensure only user and assistant roles are added
        if message["role"] in ["user", "assistant"]:
             claude_messages.append(message)

    # Add the current user query (make sure it's not duplicated if already last message)
    if not claude_messages or claude_messages[-1]["role"] != "user":
         claude_messages.append({"role": "user", "content": user_query})

    # Generate response from Claude
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        system=system_prompt,      
        messages=claude_messages   
    )

    return response.content[0].text

# Function to list available collections
def get_collections():
    try:
        qdrant_client_instance = get_qdrant_client()
        collections = qdrant_client_instance.get_collections().collections
        return [collection.name for collection in collections]
    except Exception as e:
        st.error(f"Error getting collections: {str(e)}")
        return []

# Function to get collection info
def get_collection_info(collection_name):
    try:
        qdrant_client_instance = get_qdrant_client()
        collection_info = qdrant_client_instance.get_collection(collection_name=collection_name)
        points_count = qdrant_client_instance.count(collection_name=collection_name).count
        return {
            "vectors_count": points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "created": "Available on Qdrant server"
        }
    except Exception as e:
        st.error(f"Error getting collection info: {str(e)}")
        return None

# Streamlit interface with CareerVertex branding
st.title("🚀 CareerVertex: HR Policy & Benefits Assistant")
st.markdown("*Your guide to workplace policies, benefits, and HR procedures*")

# Header container with user info and logout button
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    # Display user role badge
    if st.session_state.user_role == "hr_admin":
        st.markdown("**:red[HR Admin Mode]**")
    else:
        st.markdown("**:blue[Employee Mode]**")
        
with header_col2:
    # Logout button
    if st.button("Logout", key="logout_button"):
        logout()

# Admin sidebar - Only visible to HR admin users
if st.session_state.user_role == "hr_admin":
    with st.sidebar:
        st.subheader("HR Admin Controls")
        # Test Qdrant connection
        try:
            qdrant_client_instance = get_qdrant_client()
            st.success("✅ Connected to Knowledge Base")
            st.write(f"Server URL: {QDRANT_URL}")
        except Exception as e:
            st.error(f"❌ Failed to connect to Knowledge Base: {str(e)}")
            st.write("Please check your Qdrant server configuration in secrets.")

        st.header("HR Document Management")
        
        # File uploader for multiple PDFs
        uploaded_files = st.file_uploader("Upload HR Policy Documents (PDF)", type="pdf", accept_multiple_files=True)
        
        # Collection name for persistence
        collection_options = ["Create New Policy Collection"] + get_collections()
        collection_choice = st.selectbox("Select Policy Collection", collection_options)
        
        if collection_choice == "Create New Policy Collection":
            collection_name = st.text_input("New Collection Name:", f"hr_policies_{uuid.uuid4().hex[:8]}")
        else:
            collection_name = collection_choice
        
        # Process button
        col1, col2 = st.columns(2)
        process_button = col1.button("Process HR Documents")
        
        # Clear conversation button
        if col2.button("Clear Chat History"):
            # Preserve login and role information when clearing chat
            user_role = st.session_state.user_role
            password_correct = st.session_state.password_correct
            
            # Clear conversation and messages
            st.session_state.conversation = []
            st.session_state.messages = []
            
            # Restore login and role information
            st.session_state.user_role = user_role
            st.session_state.password_correct = password_correct
            
            st.rerun()
        
        # Collection management
        st.subheader("Policy Collection Management")
        delete_collection = st.selectbox("Select Collection to Delete", [""] + get_collections())
        if delete_collection and st.button("Delete Collection"):
            try:
                qdrant_client_instance = get_qdrant_client()
                qdrant_client_instance.delete_collection(delete_collection)
                if "collection_name" in st.session_state and st.session_state.collection_name == delete_collection:
                    if "vectorstore" in st.session_state:
                        del st.session_state.vectorstore
                    if "collection_name" in st.session_state:
                        del st.session_state.collection_name
                st.success(f"Collection '{delete_collection}' deleted successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting collection: {str(e)}")
        
        # Display database info if available
        if "vectorstore" in st.session_state and "collection_name" in st.session_state:
            st.success("Knowledge Base Status: Connected")
            st.info(f"Active Policy Collection: {st.session_state.collection_name}")
            
            # Show collection details
            collection_info = get_collection_info(st.session_state.collection_name)
            if collection_info:
                st.write(f"Document Vectors: {collection_info['vectors_count']}")
                st.write(f"Vector size: {collection_info['vector_size']}")
else:
    # For employees, show which collection they're using in a small info message
    if "collection_name" in st.session_state:
        st.info(f"Using HR policy collection: {st.session_state.collection_name}")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Auto-connect to the default Qdrant collection on startup
if ("vectorstore" not in st.session_state and 
    "collection_name" not in st.session_state):
    
    try:
        # Get list of available collections
        qdrant_client_obj = get_qdrant_client()
        available_collections = get_collections()
        
        # For HR admins, guide them to create a collection if none exist
        if st.session_state.user_role == "hr_admin":
            if not available_collections:
                st.sidebar.warning("⚠️ No HR policy collections found. Please upload and process HR documents to create a collection.")
            else:
                st.sidebar.info(f"Available collections: {', '.join(available_collections)}")
                st.sidebar.info("Select a collection and click 'Process HR Documents' to connect.")
                
        # For employees, show appropriate message if no collections exist
        else:
            if not available_collections:
                st.warning("⚠️ No HR policy collections are currently available. Please contact your HR administrator.")
            else:
                # Try to connect to the first available collection for employees
                try:
                    first_collection = available_collections[0]
                    st.info(f"Connecting to HR policy collection: {first_collection}...")
                    
                    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                    embeddings = OpenAIEmbeddings()
                    
                    vectorstore = Qdrant(
                        client=qdrant_client_obj,
                        collection_name=first_collection,
                        embeddings=embeddings
                    )
                    
                    # Save to session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qdrant_client_obj = qdrant_client_obj
                    st.session_state.collection_name = first_collection
                    
                    st.success(f"Connected to HR policy collection: {first_collection}")
                except Exception as e:
                    st.error(f"Error connecting to HR policy collection: {str(e)}")
                    st.info("Please contact your HR administrator for assistance.")
    
    except Exception as e:
        # Connection error to Qdrant server
        error_message = f"Error connecting to HR knowledge base: {str(e)}"
        if st.session_state.user_role == "hr_admin":
            st.sidebar.error(error_message)
        else:
            st.error(error_message)

# Process documents when requested (HR admin only)
if st.session_state.user_role == "hr_admin" and process_button and uploaded_files:
    with st.spinner("Processing HR policy documents..."):
        try:
            # Save uploaded files
            file_paths = save_uploaded_files(uploaded_files)
            
            # Process documents and create vector store
            result = process_documents(file_paths, collection_name)
            
            if result:
                vectorstore, qdrant_client_obj = result
                st.session_state.vectorstore = vectorstore
                st.session_state.qdrant_client_obj = qdrant_client_obj
                st.session_state.collection_name = collection_name
                st.success(f"Successfully processed {len(file_paths)} HR policy documents")
            else:
                st.error("No documents were processed. Check if the uploaded files are valid PDFs.")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
# If HR admin selected an existing collection without uploads
elif st.session_state.user_role == "hr_admin" and process_button and collection_choice != "Create New Policy Collection":
    try:
        # Connect to existing collection
        qdrant_client_obj = get_qdrant_client()
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = OpenAIEmbeddings()
        
        vectorstore = Qdrant(
            client=qdrant_client_obj,
            collection_name=collection_choice,
            embeddings=embeddings
        )
        
        st.session_state.vectorstore = vectorstore
        st.session_state.qdrant_client_obj = qdrant_client_obj
        st.session_state.collection_name = collection_choice
        st.success(f"Connected to HR policy collection: {collection_choice}")
    except Exception as e:
        st.error(f"Error connecting to collection: {str(e)}")

# Chat interface
chat_container = st.container()

# Add example HR questions for employees - only visible when no messages yet
if not st.session_state.messages and st.session_state.user_role == "employee":
    st.markdown("### Example questions you can ask:")
    col1, col2 = st.columns(2)
    
    with col1:
        example_q1 = "What is our company's maternity leave policy?"
        example_q2 = "How many holidays am I entitled to per year?"
        
        if st.button(example_q1):
            user_query = example_q1
        if st.button(example_q2):
            user_query = example_q2
            
    with col2:
        example_q3 = "What's the process for requesting flexible working?"
        example_q4 = "How does our pension scheme work?"
        
        if st.button(example_q3):
            user_query = example_q3
        if st.button(example_q4):
            user_query = example_q4

# Chat input (placed outside and after the message display logic)
user_query = st.chat_input("Ask a question about workplace policies, benefits, or procedures...")

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle user input and generate response
    if user_query and "vectorstore" in st.session_state:
        # Add user message to the UI
        with st.chat_message("user"):
            st.write(user_query)
        
        # Add user message to conversation and message history
        st.session_state.conversation.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Search for relevant context
        with st.spinner("Searching HR policies..."):
            docs = st.session_state.vectorstore.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response with spinner
        with st.spinner("Finding your answer..."):
            response = generate_response(user_query, context, st.session_state.conversation)
        
        # Add assistant response to conversation and message history
        st.session_state.conversation.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
    elif user_query:
        st.warning("Please wait for HR documents to be loaded before asking questions.")
    elif not "vectorstore" in st.session_state:
        if st.session_state.user_role == "hr_admin":
            st.info("To start the HR assistant, upload policy documents and process them using the sidebar.")
        else:
            st.info("No HR policy documents are currently loaded. Please contact your HR administrator to set up the knowledge base.")

# Add a footer with HR contact info
st.markdown("---")
st.markdown("**Need more help?** Contact the HR team at hr@yourcompany.co.uk")

# Auto-scroll to bottom of chat
if st.session_state.messages:
    st.components.v1.html(
        """
        <script>
            window.scrollTo(0, document.body.scrollHeight);
        </script>
        """,
        height=0
    )
