import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.vectorstores import Pinecone
from uuid import uuid4

# TODO:以下の関数をクラスにまとめて、重複している共通データをインスタンス変数とかにする

#　環境変数の読み込み
load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


# embeddingsの初期化
model_name = 'text-embedding-ada-002'#TODO env化
embeddings = OpenAIEmbeddings(model=model_name)


def create_index_or_pass():
    # すでに作成しているインデックスネームのリストを取得して存在確認
    index_name_list = pinecone.list_indexes()

    if INDEX_NAME not in index_name_list:
        pinecone.create_index(
            INDEX_NAME,
            dimension=1536,
            metric="cosine",
            pod_type="s1"
            )

        print(pinecone.describe_index(INDEX_NAME))

    return



def upsert_vector_from_documents(documents):

    # インデックス初期化
    index = pinecone.Index(INDEX_NAME)
    print(index.describe_index_stats())

    # TODO:namespaceをユーザーから受け取るかenv変数化
    vectorstore = Pinecone(index, embeddings, 'testname')

    #　documentsを追加
    vectorstore.add_documents(documents)


def pinecone_retriever():
    index = pinecone.Index(INDEX_NAME)
    # TODO:namespaceをユーザーから受け取るかenv変数化　
    vectorstore = Pinecone(index, embeddings.embed_query,'testname')
    retriever = vectorstore.as_retriever(search_type="mmr",serch_kwargs={"k":1})
    return retriever





