!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf

from google.colab import drive
drive. mount ("/content/drive")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain

loader = PyPDFDirectoryLoader("/content/drive/MyDrive/Colab Notebooks")
docs = loader.load()

len(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents (docs)

len(chunks)

import os

os. environ ['HUGGINGFACEHUB_API_TOKEN' ] = "" # Enter your API key

embedding=SentenceTransformerEmbeddings (model_name="NeuML/pubmedbert-base-embeddings")

vectorstore = Chroma.from_documents(chunks,embedding)

query = "Who is at risk of heart disease?"
search_results = vectorstore.similarity_search(query)

search_results

retriever = vectorstore.as_retriever(search_kwargs={'k':5})
retriever.get_relevant_documents (query)

llm = LlamaCpp (
  model_path="/content/drive/MyDrive/BioMistral-7B.Q2_K.gguf",
  temperature=0.2,
  max_tokens = 2048,
  top_p=1
)

template = """
<|context|>
You are an Medical Assistant that follows the instructions and generate the accurate
response based
on the query and
the
context provided.
Please be truthful and give direct answers.
</S>
<|user |>
{query}
</s>
<|assistant |>
"""

from langchain.schema. runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
  {"context": retriever, "query": RunnablePassthrough()}
  |prompt
  | llm
  | StrOutputParser()
)

response=rag_chain.invoke(query)

response

import sys
while True:
  user_input = input (f"Input query: ")
  if user_input == 'exit':
    print ("Exiting...")
    sys.exit()
  if user_input=="":
    continue
  result = rag_chain. invoke(user_input)
  print("Answer: ", result)
