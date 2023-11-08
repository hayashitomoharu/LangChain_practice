import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.vectorstores import Pinecone

# 　環境変数の読み込み
load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME 環境変数が設定されていません。")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY 環境変数が設定されていません。")
if not PINECONE_ENV:
    raise ValueError("PINECONE_ENV 環境変数が設定されていません。")

# pinecone clientの初期化
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# インデックス初期化
index = pinecone.Index(INDEX_NAME)

# embeddingsの初期化
model_name = "text-embedding-ada-002"  # TODO env化
embeddings = OpenAIEmbeddings(model=model_name)

# TODO:以下の関数をクラスにまとめて、重複している共通データをインスタンス変数とかにする


def create_index_or_pass():
    # pyrightでエラー出るのでつけてます。#TODO:refactor
    if not INDEX_NAME:
        raise ValueError("INDEX_NAME 環境変数が設定されていません。")

    # すでに作成しているインデックスネームのリストを取得して存在確認
    index_name_list = pinecone.list_indexes()

    if INDEX_NAME not in index_name_list:
        pinecone.create_index(
            INDEX_NAME, dimension=1536, metric="cosine", pod_type="s1"
        )

    return


def upsert_vector_from_documents(documents):
    # TODO:namespaceをユーザーから受け取るかenv変数化
    vectorstore = Pinecone(index, embeddings, "testname")

    # 　documentsを追加
    vectorstore.add_documents(documents)
