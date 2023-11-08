from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from git import Repo
import os
import shutil
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone_client.vector_db import (
    index,
    create_index_or_pass,
    upsert_vector_from_documents,
    embeddings,
)
from langchain.vectorstores import Pinecone

# git_clone or 存在確認
load_dotenv()


# ダウンロード先repository
REPO_URL = os.getenv("REPO_URL")
BRANCH = os.getenv("BRANCH")

# githubからダウンロードされるものが配置されるディレクトリ
LOCAL_STRAGE_PATH = os.getenv("LOCAL_STRAGE_PATH")

# 解析するファイルのパス
ANALIZE_TARGET_PATH = os.getenv("ANALIZE_TARGET_PATH")

# pyright　通らないためNoneにならないことを確定させる
if not REPO_URL:
    raise ValueError("REPO_URL 環境変数が設定されていません。")
if not BRANCH:
    raise ValueError("BRANCH 環境変数が設定されていません。")
if not LOCAL_STRAGE_PATH:
    raise ValueError("LOCAL_STRAGE_PATH 環境変数が設定されていません。")
if not ANALIZE_TARGET_PATH:
    raise ValueError("ANALIZE_TARGET_PATH 環境変数が設定されていません。")


if os.path.exists(LOCAL_STRAGE_PATH):
    pass
else:
    print("\ndownloading...\n")
    Repo.clone_from(REPO_URL, to_path=LOCAL_STRAGE_PATH, branch=BRANCH)
    print("complete!\n")


# リポジトリの読み込みと、テキストの分割
loader = GenericLoader.from_filesystem(
    LOCAL_STRAGE_PATH + ANALIZE_TARGET_PATH,
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)

documents = loader.load()
print(f"分割されたドキュメントの総数は:{len(documents)} 個\n")


# 全てのファイルを読み込んで、テキストを分割する
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
documents = python_splitter.split_documents(documents)

# インデックス作成
create_index_or_pass()

# ベクトル化とベクターDBに保存
upsert_vector_from_documents(documents)

# retriever定義
vectorstore = Pinecone(index, embeddings.embed_query, "testname")
retriever = vectorstore.as_retriever(search_type="mmr", serch_kwargs={"k": 1})

# pinecone retrieverで会話するための初期化
llm = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

# 現状from_llmなので、これつけるとベクトル検索のためのプロンプトが、不安定になるのでプロンプトテンプレートを使用して対策する。search_type="mmr",serch_kwargs={"k":1}も最適にする
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# memory設定と問い合わせ
question = input("このリポジトリに対する質問を入力してください。\n")
result = qa(question)
print(result["answer"])

# ファイルが存在するか確認してから削除
if os.path.exists(LOCAL_STRAGE_PATH):
    os.chmod(LOCAL_STRAGE_PATH, 0o777)
    shutil.rmtree(LOCAL_STRAGE_PATH)
    print(f"{LOCAL_STRAGE_PATH} が削除されました。")
else:
    print(f"{LOCAL_STRAGE_PATH} は存在しません。")
