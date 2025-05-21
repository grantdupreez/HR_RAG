import streamlit as st
import os
import anthropic
import tempfile
import uuid
import hmac
import json
import re
import pandas as pd
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import io
import qdrant_client
from qdrant_client.http import models

# Page configuration
st.set_page_config(
    page_title="CareerVertex HR Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- AUTHENTICATION -----------------

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

# ----------------- ANTHROPIC CLIENT SETUP -----------------

# Set up API keys from Streamlit secrets
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

# Initialize the Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Define paths for uploads
HR_DOCS_DIR = "hr_documents"
os.makedirs(HR_DOCS_DIR, exist_ok=True)

# ----------------- QDRANT SETUP -----------------

# Get Qdrant client
def get_qdrant_client():
    """Get a Qdrant client configured for cloud or local persistence."""
    try:
        if "QDRANT_URL" in st.secrets and st.secrets["QDRANT_URL"]:
            # Cloud or remote Qdrant
            if "QDRANT_API_KEY" in st.secrets and st.secrets["QDRANT_API_KEY"]:
                # With API key
                return qdrant_client.QdrantClient(
                    url=st.secrets["QDRANT_URL"],
                    api_key=st.secrets["QDRANT_API_KEY"]
                )
            else:
                # Without API key
                return qdrant_client.QdrantClient(
                    url=st.secrets["QDRANT_URL"]
                )
        else:
            # Local Qdrant (for development)
            return qdrant_client.QdrantClient(":memory:")
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {str(e)}")
        # Fallback to in-memory
        return qdrant_client.QdrantClient(":memory:")

# Get Claude embeddings for texts
def get_claude_embeddings(texts):
    """Get embeddings from Claude for a list of texts."""
    if not texts:
        return []
        
    embeddings = []
    for text in texts:
        # Skip empty texts
        if not text or not text.strip():
            embeddings.append([0] * 1536)  # Default embedding size
            continue
            
        try:
            response = anthropic_client.embeddings.create(
                model="claude-3-7-sonnet-20250219",
                input=text
            )
            embeddings.append(response.embedding)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            # Return a zero embedding as fallback
            embeddings.append([0] * 1536)
    
    return embeddings

# Create or get collection
def get_or_create_collection(client, collection_name):
    """Create a new collection or get an existing one."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(collection.name == collection_name for collection in collections)
        
        if not exists:
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Claude embeddings size
                    distance=models.Distance.COSINE
                )
            )
        
        return collection_name
    
    except Exception as e:
        st.error(f"Error with Qdrant collection: {str(e)}")
        return None

# ----------------- DOCUMENT PROCESSING -----------------

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text() + "\n"
    return text

def save_uploaded_file(uploaded_file):
    """Save uploaded file to disk and return the path."""
    file_path = os.path.join(HR_DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_document_with_claude(file_content, file_name):
    """Use Claude to extract structured content from HR documents."""
    system_prompt = """
    You are an HR document processor. Extract the following from this HR policy document:
    1. Policy title
    2. Category (e.g., Leave, Benefits, Conduct, etc.)
    3. Effective date (if mentioned)
    4. Target audience (which employees this applies to)
    5. Key points (summarized in 3-5 bullet points)
    
    Also, segment the document into logical chunks that represent distinct policies or sections.
    Each chunk should be self-contained and meaningful on its own.
    
    Format your response as JSON with the following structure:
    {
        "metadata": {
            "title": "Policy title",
            "category": "Policy category",
            "effective_date": "Date or Unknown",
            "audience": "Target audience",
            "key_points": ["point 1", "point 2", "point 3"]
        },
        "chunks": [
            {
                "title": "Section title",
                "content": "Full text of this section",
                "summary": "Brief summary of this section"
            },
            ...
        ]
    }
    """
    
    # Truncate content if too long
    max_length = 100000  # Claude has a context limit
    truncated_content = file_content[:max_length] if len(file_content) > max_length else file_content
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Document name: {file_name}\n\nContent:\n{truncated_content}"}]
        )
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content[0].text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content[0].text
        
        # Parse the JSON
        try:
            extracted_data = json.loads(json_str)
            return extracted_data
        except json.JSONDecodeError:
            st.warning("Could not parse Claude's JSON output. Falling back to simple chunking.")
            return fallback_chunking(file_content, file_name)
            
    except Exception as e:
        st.error(f"Error processing document with Claude: {str(e)}")
        return fallback_chunking(file_content, file_name)

def fallback_chunking(text, file_name):
    """Fall back to simple chunking if Claude processing fails."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Create a basic metadata structure
    result = {
        "metadata": {
            "title": file_name,
            "category": "Unknown",
            "effective_date": "Unknown",
            "audience": "All employees",
            "key_points": ["Automatically chunked document"]
        },
        "chunks": []
    }
    
    # Add chunks
    for i, chunk in enumerate(chunks):
        result["chunks"].append({
            "title": f"Section {i+1}",
            "content": chunk,
            "summary": "Automatically generated chunk"
        })
    
    return result

def add_document_to_qdrant(client, collection_name, processed_doc, document_id):
    """Add processed document chunks to Qdrant."""
    points = []
    
    # Extract document metadata
    doc_metadata = processed_doc["metadata"]
    
    # Process each chunk
    for i, chunk in enumerate(processed_doc["chunks"]):
        # Create a unique ID for the point
        point_id = f"{document_id}_{i}"
        
        # Create metadata for this chunk
        metadata = {
            "doc_id": document_id,
            "doc_title": doc_metadata["title"],
            "category": doc_metadata["category"],
            "effective_date": doc_metadata["effective_date"],
            "audience": doc_metadata["audience"],
            "chunk_title": chunk["title"],
            "chunk_summary": chunk["summary"]
        }
        
        # Get embeddings for the chunk
        chunk_embeddings = get_claude_embeddings([chunk["content"]])
        
        if chunk_embeddings:
            # Create a point
            point = models.PointStruct(
                id=point_id,
                vector=chunk_embeddings[0],
                payload={
                    "metadata": metadata,
                    "content": chunk["content"]
                }
            )
            
            points.append(point)
    
    # Add points to Qdrant
    if points:
        try:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            return len(points)
        except Exception as e:
            st.error(f"Error adding points to Qdrant: {str(e)}")
            return 0
    
    return 0

# ----------------- RETRIEVAL FUNCTIONS -----------------

def retrieve_relevant_chunks(client, collection_name, query, top_k=5):
    """Retrieve relevant chunks based on the query."""
    try:
        # Get embeddings for the query
        query_embedding = get_claude_embeddings([query])[0]
        
        # Search in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Extract and format results
        documents = []
        metadatas = []
        scores = []
        
        for result in search_results:
            documents.append(result.payload["content"])
            metadatas.append(result.payload["metadata"])
            scores.append(result.score)
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "scores": scores
        }
    
    except Exception as e:
        st.error(f"Error searching Qdrant: {str(e)}")
        return {
            "documents": [],
            "metadatas": [],
            "scores": []
        }

def rerank_with_claude(query, search_results):
    """Use Claude to rerank search results by relevance."""
    if not search_results["documents"]:
        return search_results
    
    system_prompt = """
    You are an HR document retrieval expert. Rank these document chunks by their relevance
    to the employee's question about HR policies. For each document, provide a score from 0-10 
    where 10 is most relevant. Return a JSON array with document indices and scores.
    Example: [{"index": 0, "score": 8}, {"index": 1, "score": 3}, ...]
    """
    
    message = f"Question: {query}\n\nDocument chunks to rank:\n"
    for i, (doc, meta) in enumerate(zip(search_results["documents"], search_results["metadatas"])):
        message += f"\n--- Document Chunk {i} ---\n"
        message += f"Title: {meta.get('doc_title', 'Untitled')} - {meta.get('chunk_title', 'Section')}\n"
        message += f"Category: {meta.get('category', 'Uncategorized')}\n"
        message += f"Content: {doc}\n"
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": message}]
        )
        
        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content[0].text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content[0].text
        
        # Parse rankings
        try:
            rankings = json.loads(json_str)
            
            # Sort the results by score (descending)
            sorted_indices = [item["index"] for item in sorted(rankings, key=lambda x: x["score"], reverse=True)]
            
            # Reorder the results
            reranked_results = {
                "documents": [search_results["documents"][i] for i in sorted_indices],
                "metadatas": [search_results["metadatas"][i] for i in sorted_indices],
                "scores": [search_results["scores"][i] for i in sorted_indices] if "scores" in search_results else []
            }
            
            return reranked_results
            
        except json.JSONDecodeError:
            # If parsing fails, return original results
            return search_results
            
    except Exception as e:
        st.error(f"Error during reranking: {str(e)}")
        return search_results

# ----------------- RESPONSE GENERATION -----------------

def generate_hr_response(query, search_results):
    """Generate an HR-specific response using Claude."""
    if not search_results["documents"]:
        return "I don't have information about this in our HR policy documents. Please contact the HR team directly for assistance."
    
    system_prompt = """
    You are CareerVertex, a helpful HR assistant that answers employee questions about workplace benefits, 
    company policies, and HR procedures based on the provided context from company policy documents.
    
    As an HR assistant, your goal is to help employees understand their benefits, rights, and company policies 
    accurately and clearly. Use a professional but approachable tone.
    
    Your responses should:
    1. Begin with a direct answer to the question
    2. Reference specific policy details and dates when available
    3. Note any approval requirements or exceptions
    4. End with next steps if action is required
    
    Use British English spelling and terminology (organisation, centre, programme, etc.)
    
    Only provide information based strictly on the provided context from policy documents.
    If you don't know the answer or if the information is not complete in the context, clearly state 
    this limitation and suggest contacting the HR department for more details.
    
    Include relevant policy references when possible.
    """
    
    # Format context from search results
    context = "HR POLICY CONTEXT:\n\n"
    for i, (doc, meta) in enumerate(zip(search_results["documents"], search_results["metadatas"])):
        context += f"--- Policy {i+1} ---\n"
        context += f"Title: {meta.get('doc_title', 'Untitled')}\n"
        context += f"Section: {meta.get('chunk_title', 'General')}\n"
        context += f"Category: {meta.get('category', 'Policy')}\n"
        if meta.get('effective_date') and meta.get('effective_date') != 'Unknown':
            context += f"Effective Date: {meta.get('effective_date')}\n"
        context += f"Content: {doc}\n\n"
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system=system_prompt,
            messages=[{
                "role": "user", 
                "content": f"Employee Question: {query}\n\n{context}"
            }]
        )
        
        return response.content[0].text
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I encountered an error while trying to answer your question. Please try again or contact HR directly."

# ----------------- ANALYTICS FUNCTIONS -----------------

def categorize_question(query):
    """Categorize HR questions using Claude."""
    categories = [
        "Leave Policy", "Compensation", "Benefits", "Performance", 
        "Working Hours", "Conduct", "Training", "Onboarding", "Offboarding", "Other"
    ]
    
    system_prompt = f"""
    You are an HR question classifier. Categorize this employee question 
    into exactly one of these categories: {', '.join(categories)}
    Return only the category name, nothing else.
    """
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            system=system_prompt,
            messages=[{"role": "user", "content": query}]
        )
        
        # Extract category name
        category = response.content[0].text.strip()
        if category not in categories:
            category = "Other"
        
        return category
        
    except Exception as e:
        return "Other"

def track_interaction(query, response, category):
    """Track user interactions for analytics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize analytics in session state if needed
    if "analytics" not in st.session_state:
        st.session_state.analytics = []
    
    # Add interaction data
    st.session_state.analytics.append({
        "timestamp": timestamp,
        "query": query,
        "category": category,
        "response_length": len(response),
        "feedback": None  # Will be updated if user provides feedback
    })
    
    # Limit size of analytics in session state
    if len(st.session_state.analytics) > 100:
        st.session_state.analytics = st.session_state.analytics[-100:]

def update_feedback(index, feedback):
    """Update feedback for a specific interaction."""
    if "analytics" in st.session_state and index < len(st.session_state.analytics):
        st.session_state.analytics[index]["feedback"] = feedback

# ----------------- ADMIN DASHBOARD -----------------

def show_admin_dashboard():
    """Show the admin dashboard."""
    st.header("HR Admin Dashboard")
    
    # Document management tab
    tab1, tab2 = st.tabs(["Document Management", "Analytics"])
    
    with tab1:
        st.subheader("HR Document Management")
        
        # Get Qdrant client
        try:
            qdrant_client_obj = get_qdrant_client()
            
            # Get collection status
            try:
                collections = qdrant_client_obj.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if collections:
                    st.success(f"✅ Connected to Qdrant ({len(collections)} collections)")
                else:
                    st.info("✅ Connected to Qdrant (No collections yet)")
            except Exception as e:
                st.warning(f"⚠️ Connected to Qdrant but failed to get collections: {str(e)}")
                collection_names = []
            
            # Collection selection or creation
            col1, col2 = st.columns(2)
            
            with col1:
                if collection_names:
                    collection_option = st.selectbox(
                        "Select Collection", 
                        ["Create New Collection"] + collection_names
                    )
                else:
                    collection_option = "Create New Collection"
                    st.info("No collections found. Create your first collection.")
            
            with col2:
                if collection_option == "Create New Collection":
                    new_collection_name = st.text_input(
                        "New Collection Name", 
                        value=f"hr_policies_{uuid.uuid4().hex[:8]}"
                    )
                    selected_collection = new_collection_name
                else:
                    selected_collection = collection_option
            
            # File uploader for multiple PDFs
            uploaded_files = st.file_uploader(
                "Upload HR Policy Documents (PDF)", 
                type="pdf", 
                accept_multiple_files=True
            )
            
            # Process button
            if st.button("Process HR Documents") and uploaded_files:
                with st.spinner("Processing HR policy documents..."):
                    try:
                        # Get or create collection
                        collection_name = get_or_create_collection(qdrant_client_obj, selected_collection)
                        
                        if collection_name:
                            total_chunks = 0
                            
                            for uploaded_file in uploaded_files:
                                # Display progress
                                st.write(f"Processing: {uploaded_file.name}")
                                
                                # Generate a unique ID for this document
                                doc_id = f"doc_{uuid.uuid4().hex}"
                                
                                # Extract text from PDF
                                file_bytes = io.BytesIO(uploaded_file.getvalue())
                                text = extract_text_from_pdf(file_bytes)
                                
                                # Process with Claude
                                processed_doc = process_document_with_claude(text, uploaded_file.name)
                                
                                # Add to Qdrant
                                chunks_added = add_document_to_qdrant(
                                    qdrant_client_obj, 
                                    collection_name, 
                                    processed_doc, 
                                    doc_id
                                )
                                total_chunks += chunks_added
                                
                                st.write(f"Added {chunks_added} chunks from {uploaded_file.name}")
                            
                            # Save file to disk as well
                            for uploaded_file in uploaded_files:
                                save_uploaded_file(uploaded_file)
                            
                            st.success(f"Successfully processed {len(uploaded_files)} HR policy documents with {total_chunks} total chunks")
                        else:
                            st.error("Failed to create or access the collection")
                    
                    except Exception as e:
                        st.error(f"Error during document processing: {str(e)}")
            
            # Collection management
            if collection_names:
                st.subheader("Collection Management")
                delete_collection = st.selectbox(
                    "Select Collection to Delete", 
                    [""] + collection_names
                )
                if delete_collection and st.button("Delete Collection"):
                    try:
                        qdrant_client_obj.delete_collection(collection_name=delete_collection)
                        st.success(f"Collection '{delete_collection}' deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting collection: {str(e)}")
        
        except Exception as e:
            st.error(f"Error connecting to Qdrant: {str(e)}")
            st.info("Please configure Qdrant connection in secrets.")
    
    with tab2:
        st.subheader("Usage Analytics")
        
        if "analytics" in st.session_state and st.session_state.analytics:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(st.session_state.analytics)
            
            # Display total interactions
            st.metric("Total Interactions", len(df))
            
            # Display category breakdown
            if not df.empty and "category" in df.columns:
                st.subheader("Question Categories")
                category_counts = df["category"].value_counts()
                st.bar_chart(category_counts)
            
            # Display feedback summary if any
            if "feedback" in df.columns and df["feedback"].notna().any():
                st.subheader("Feedback Summary")
                feedback_counts = df["feedback"].value_counts()
                st.bar_chart(feedback_counts)
            
            # Show recent interactions
            st.subheader("Recent Interactions")
            st.dataframe(df[["timestamp", "query", "category", "feedback"]].tail(10))
        else:
            st.info("No analytics data available yet.")

# ----------------- EMPLOYEE CHAT INTERFACE -----------------

def show_employee_chat():
    """Show the employee chat interface."""
    st.header("🚀 CareerVertex: HR Policy & Benefits Assistant")
    st.markdown("*Your guide to workplace policies, benefits, and HR procedures*")
    
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant" and "feedback" not in message:
                col1, col2, col3 = st.columns([1, 1, 5])
                with col1:
                    if st.button("👍 Helpful", key=f"helpful_{i}"):
                        st.session_state.messages[i]["feedback"] = "Helpful"
                        update_feedback(i // 2, "Helpful")  # Every other message is from assistant
                        st.rerun()
                with col2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                        st.session_state.messages[i]["feedback"] = "Not Helpful"
                        update_feedback(i // 2, "Not Helpful")
                        st.rerun()
    
    # Show example questions for new users
    if not st.session_state.messages:
        st.info("Ask me any question about HR policies, benefits, or workplace procedures.")
        
        st.markdown("### Example questions you can ask:")
        col1, col2 = st.columns(2)
        
        with col1:
            example_q1 = "What is our company's maternity leave policy?"
            example_q2 = "How many holidays am I entitled to per year?"
            
            if st.button(example_q1):
                process_user_query(example_q1)
                st.rerun()
                
            if st.button(example_q2):
                process_user_query(example_q2)
                st.rerun()
                
        with col2:
            example_q3 = "What's the process for requesting flexible working?"
            example_q4 = "How does our pension scheme work?"
            
            if st.button(example_q3):
                process_user_query(example_q3)
                st.rerun()
                
            if st.button(example_q4):
                process_user_query(example_q4)
                st.rerun()
    
    # Chat input
    user_query = st.chat_input("Ask a question about workplace policies, benefits, or procedures...")
    
    if user_query:
        process_user_query(user_query)

def process_user_query(query):
    """Process a user query and generate a response."""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Categorize the question for analytics
    category = categorize_question(query)
    
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    
    # Create empty assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching HR policies...")
    
    try:
        # Get Qdrant client
        qdrant_client_obj = get_qdrant_client()
        
        # Check if any collections exist
        collections = qdrant_client_obj.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if not collection_names:
            response = "I don't have any HR policy documents loaded yet. Please ask your HR administrator to upload some policy documents."
        else:
            # Use the first collection for employee queries
            collection_name = collection_names[0]
            
            # Retrieve relevant chunks
            with st.spinner("Searching HR policies..."):
                search_results = retrieve_relevant_chunks(qdrant_client_obj, collection_name, query)
                
                # Rerank results if we have more than one result
                if len(search_results["documents"]) > 1:
                    search_results = rerank_with_claude(query, search_results)
            
            # Generate response with Claude
            with st.spinner("Finding your answer..."):
                response = generate_hr_response(query, search_results)
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        response = "I encountered an error while trying to answer your question. Please try again later or contact HR directly."
    
    # Update assistant message
    message_placeholder.markdown(response)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Track interaction for analytics
    track_interaction(query, response, category)
    
    # Add footer reference
    if "No HR policy documents" not in response:
        st.caption("Based on your company's HR policy documents")

# ----------------- MAIN APP -----------------

# Display header
st.sidebar.title("CareerVertex")
st.sidebar.markdown("HR Policy & Benefits Assistant")

# Header container with user info and logout button
header_col1, header_col2 = st.sidebar.columns([3, 1])
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

# Show appropriate interface based on user role
if st.session_state.user_role == "hr_admin":
    show_admin_dashboard()
else:
    show_employee_chat()

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Need help? Contact HR at hr@yourcompany.co.uk")

# Add auto-scroll (works in some browsers)
if st.session_state.get("messages", []):
    st.components.v1.html(
        """
        <script>
            window.scrollTo(0, document.body.scrollHeight);
            const observer = new MutationObserver((mutations) => {
                window.scrollTo(0, document.body.scrollHeight);
            });
            observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """,
        height=0
    )
