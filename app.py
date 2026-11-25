import streamlit as st
import os
import tempfile
import re
import uuid
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

COLLECTION_NAME = "my_knowledge_base"
VECTOR_SIZE = 384
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
LM_STUDIO_URL = "http://localhost:1234/v1"

class RecursiveTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", r"(?<=\. )", " ", ""]

    def split_text(self, text):
        text = text.replace("\n\n", "||PARAGRAF||")
        text = text.replace("\n", " ") 
        text = text.replace("||PARAGRAF||", "\n\n")
        text = re.sub(r'\s+', ' ', text).strip()
        
        splits = self._split_text(text, self.separators)
        return self._merge_splits(splits)

    def _split_text(self, text, separators):
        final_splits = []
        if not text: return []
        separator = separators[0]
        new_separators = separators[1:] if len(separators) > 1 else []
        
        if separator == "":
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        try:
            pattern = f"({re.escape(separator)})" if "\\" not in separator and "(" not in separator else f"({separator})"
            splits = re.split(pattern, text)
        except:
            splits = text.split(separator)

        splits = [s for s in splits if s]
        for s in splits:
            is_separator = (s == separator) or (separator == " " and s == " ")
            if len(s) > self.chunk_size and not is_separator:
                if new_separators:
                    final_splits.extend(self._split_text(s, new_separators))
                else:
                    final_splits.append(s)
            else:
                final_splits.append(s)
        return final_splits

    def _merge_splits(self, splits):
        final_docs = []
        current_doc = []
        current_len = 0
        for d in splits:
            doc_len = len(d)
            if current_len + doc_len > self.chunk_size:
                if current_len > 0:
                    doc = "".join(current_doc).strip()
                    if doc: final_docs.append(doc)
                while current_len > self.chunk_overlap:
                    if len(current_doc) <= 1: break
                    removed = current_doc.pop(0)
                    current_len -= len(removed)
            current_doc.append(d)
            current_len += len(d)
        if current_doc:
            doc = "".join(current_doc).strip()
            if doc: final_docs.append(doc)
        return final_docs

def load_pdf_final(file_path):
    try:
        text_pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1) 
                if text: text_pages.append(text)
        return "\n\n".join(text_pages)
    except Exception as e:
        return None

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

@st.cache_resource
def get_lm_studio_client():
    return OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

embed_model = load_embedding_model()
qdrant_client = get_qdrant_client()
llm_client = get_lm_studio_client()

with st.sidebar:
    st.header("📂 Belge Yükle")
    uploaded_file = st.file_uploader("Bir PDF seçin", type="pdf")
    
    if uploaded_file and st.button("Qdrant'a Yükle"):
        with st.spinner("PDF işleniyor..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            raw_text = load_pdf_final(tmp_path)
            if raw_text:
                splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(raw_text)
                
                if qdrant_client.collection_exists(COLLECTION_NAME):
                    qdrant_client.delete_collection(COLLECTION_NAME)
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                
                vectors = embed_model.encode(chunks).tolist()
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=v, 
                        payload={"text": t}
                    ) for t, v in zip(chunks, vectors)
                ]
                
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                st.success(f"✅ {len(points)} parça başarıyla yüklendi!")
                
            os.remove(tmp_path)

st.title("🤖 RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        st.warning("Önce sol menüden bir PDF yükleyin.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching & Thinking...")
        
        query_vector = embed_model.encode(prompt).tolist()
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
        ).points
        
        context_text = "\n".join([f"- {hit.payload['text']}" for hit in search_results])
        
        full_response = ""
        
        try:
            stream = llm_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Use the Context to answer the Question."},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"}
                ],
                temperature=0.7,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"LM Studio Hatası: {e}")
            full_response = "Hata oluştu."

    st.session_state.messages.append({"role": "assistant", "content": full_response})