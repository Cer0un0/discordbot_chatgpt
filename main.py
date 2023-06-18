import re
import os
from dotenv import load_dotenv
import openai
from discord import Client, Intents, Message, Thread
from langchain.agents import ZeroShotAgent, AgentExecutor, load_tools
from langchain import OpenAI, LLMChain
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
# Google Search API
from langchain.utilities import GoogleSearchAPIWrapper
# Memory
from langchain.memory import ConversationBufferMemory

load_dotenv()
TOKEN = os.environ['TOKEN']  # discord botのtoken
openai.api_key = os.environ['OPENAI_API_KEY']  # openaiのapi key

LIMIT = 200  # openaiにAPIで送る会話の上限

client: Client = Client(intents=Intents.all())

async def search(search_str: str):
#    search_str += '\n日本語で回答してください'

    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prefix = os.environ['PREFIX']
    suffix = os.environ['SUFFIX']

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    # tool, memory, llm を設定して agent を作成
    agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, prefix=prefix, suffix=suffix)
    '''
    tools = load_tools(["google-search"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    result = agent.run(search_str)
    '''
    result = agent_chain.run(input=search_str)
    return result

async def chatgpt(message: Message):
    """ chatGPTによるreplyを行う"""

    # メッセージから@を除去
    escape_content: str = re.sub(r"<@(everyone|here|[!&]?[0-9]{17,20})> ", "", message.content)

    if type(message.channel) is Thread:  # messageがスレッド内の場合
        thread: Thread = message.channel

        # openaiに送るメッセージ
        messages: list[dict[str, str]] = []

        # スレッドの履歴をすべて取得
        async for mes in thread.history():
            # botとユーザーの区別
            role: str = "assistant" if mes.author == client.user else "user"

            messages.append({"role": role, "content": mes.content})

        # 上限と時系列を考慮したopenaiに送るメッセージリスト
        messages = messages[:200][::-1]
    else:  # messageがスレッド外の場合（@で呼ばれた場合）
        # スレッドを作成
        name: str = escape_content[:20]  # スレッド名
        thread: Thread = await message.create_thread(name=name)

        # openaiに送るメッセージ
        messages: list[dict[str, str]] = [{"role": "user", "content": escape_content}]

    # 結果を取得
#    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages
    print(messages[-1]['content'])
    try:
        response = await search(messages[-1]['content'])
    except:
        response = 'もう少し具体的に質問すると返してくれることがあるよ'
    # Reply
#    reply: str = "\n".join([choice["message"]["content"] for choice in response["choices"]])
    reply: str = response
    await thread.send(reply)


@client.event
async def on_message(message: Message):
    """ メッセージを受信したときのイベント"""
    
    if client.user in message.mentions or type(message.channel) is Thread:

        # bot自身のメッセージは無視
        if message.author == client.user:
            return

        # Reply
        await chatgpt(message)


client.run(TOKEN)
