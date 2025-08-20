from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, filter_to_minimal_docs, text_split
from src.prompt import system_prompt
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

docs = load_pdf_file("data")  
minimal_docs = filter_to_minimal_docs(docs)
chunks = text_split(minimal_docs)

embeddings = download_hugging_face_embeddings()

docsearch = Chroma.from_documents(chunks, embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.environ.get('groq_api_key'))
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
