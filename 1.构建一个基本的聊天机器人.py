"""
基于LangGraph构建的简单聊天机器人
使用OpenRouter API作为大语言模型后端
"""

# 导入必要的库
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from IPython.display import Image, display

# 导入LangGraph相关组件
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# 导入OpenAI客户端（用于OpenRouter API）
from langchain_openai import ChatOpenAI

# 加载环境变量（从.env文件）
load_dotenv()

# 定义状态类型


class State(TypedDict):
    """聊天机器人的状态类型定义"""
    messages: Annotated[list, add_messages]  # 消息列表，使用add_messages注解


# 初始化状态图构建器
graph_builder = StateGraph(State)

# 配置OpenRouter API客户端
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),  # 从环境变量获取模型名称
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),  # 从环境变量获取API密钥
    openai_api_base=os.getenv("OPENROUTER_API_URL"),  # 从环境变量获取API基础URL
    temperature=0.7,  # 设置温度参数，控制输出的随机性
)


def chatbot(state: State):
    """
    聊天机器人节点函数

    参数:
        state: 当前状态，包含消息历史

    返回:
        包含新消息的状态更新
    """
    return {"messages": [llm.invoke(state["messages"])]}


# 构建图结构
# 添加聊天机器人节点
graph_builder.add_node("chatbot", chatbot)
# 设置入口点
graph_builder.set_entry_point("chatbot")
# 设置结束点
graph_builder.set_finish_point("chatbot")
# 编译图
graph = graph_builder.compile()


# 尝试显示图的可视化（需要额外依赖，可选功能）
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # 如果无法显示图，则忽略错误
    pass


def stream_graph_updates(user_input: str):
    """
    流式处理用户输入并获取助手回复

    参数:
        user_input: 用户输入的文本
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# 主交互循环
if __name__ == "__main__":
    print("欢迎使用基于LangGraph的聊天机器人！输入'quit'、'exit'或'q'退出。")

    while True:
        try:
            # 获取用户输入
            user_input = input("User: ")

            # 检查是否退出
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break

            # 处理用户输入并显示回复
            stream_graph_updates(user_input)
        except:
            # 如果input()不可用，使用默认输入（用于Jupyter环境等）
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
