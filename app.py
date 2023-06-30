import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks, model):
    if model == 'OpenAI':
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    else:
         embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
         return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    '''llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={
            "temperature": 0.5,
            "max_length": 512
        }
    )'''
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def prepare_template(text, isRobot):
    if isRobot:
        return bot_template.replace("{{MSG}}", text)
    else:
        return user_template.replace("{{MSG}}", text)

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(
                prepare_template(message.content, False),
                unsafe_allow_html=True
            )
        else:
            st.write(
                prepare_template(message.content, True),
                unsafe_allow_html=True
            )



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.title('Tech Talk 30/06 :blue[IA] :sunglasses:')
    st.header("Chat :books:")
    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Qual gerador dos embeddings:")
        vector_choose = st.radio(
            "Escolha um",
            ('OpenAI', 'Instructor')
        )
        st.subheader("Sua documentação")
        pdf_docs = st.file_uploader("Insira seus PDFs e clique em 'Ok'", accept_multiple_files=True)
        if st.button("Ok") and vector_choose is not None:
            with st.spinner("Carregando..."):
                # text
                raw_text = get_pdf_text(pdf_docs)
                # text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # vectors
                vectorstore = get_vectorstore(text_chunks, vector_choose)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success('Base de conhecimento pronta!', icon="✅")

    


if __name__ == '__main__':
    main()