import os
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI

from langchain.memory import ConversationBufferMemory

# .envからAPI_KEYの読み込み
load_dotenv()
temp = os.getenv("temperature")


# llmとagent定義

# TODO:refactor (Tehre can define in one function)
llm = OpenAI(temperature=temp)
tools = load_tools(["llm-math", "python_repl"], llm=llm)
agent = initialize_agent(tools, llm, AgentType="zero-shot-react-description")


memory = ConversationBufferMemory(
    human_prefix="User", ai_prefix="Bot", memory_key="history", return_messages=True
)


# 会話を開始

# TODO: feat (Need other way that can break loop (timeout, something looplimit, etc..))
while True:
    user_input = input("You: ")
    memory.chat_memory.add_user_message(user_input)

    if user_input == "exit":
        break

    response = agent.run(input=user_input)
    memory.chat_memory.add_ai_message(response)

    print(f"AI: {response}")

    print("------------------")

    # 保存されたメッセージを変数に読み込む
    buffer = memory.load_memory_variables({})
    print(buffer["history"])
