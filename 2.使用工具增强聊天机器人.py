#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用工具增强的聊天机器人
基于LangGraph框架实现，集成了Tavily搜索工具
"""

from typing import Annotated, Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os

load_dotenv()


def pretty_print(messages: List[BaseMessage]) -> None:
    """优化后的对话打印函数"""
    for msg in messages:
        role = {
            "human": "User",
            "ai": "Assistant",
            "tool": "Tool"
        }.get(msg.type, msg.type.capitalize())

        color = {
            "User": "\033[94m",     # 蓝色
            "Assistant": "\033[92m",  # 绿色
            "Tool": "\033[93m"      # 黄色
        }.get(role, "\033[0m")

        content = msg.content
        if hasattr(msg, 'tool_calls'):
            content += f"\nTool Calls: {msg.tool_calls}"

        print(f"{color}{role}:\033[0m {content}")


def get_llm():
    return ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base=os.getenv("ZHIPU_API_URL"),
    )


class State(TypedDict):
    """
    定义聊天机器人的状态类型

    属性:
        messages: 消息列表，使用add_messages注解增强功能
    """
    messages: Annotated[List[BaseMessage], add_messages]


def create_chatbot_graph() -> Any:
    """
    创建并配置聊天机器人图

    返回:
        编译后的图对象，可用于执行聊天流程
    """
    # 初始化图构建器
    graph_builder = StateGraph(State)

    # 配置搜索工具
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # 初始化LLM模型
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    # 定义聊天机器人节点函数
    def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
        """
        聊天机器人节点函数，处理用户输入并生成回复

        参数:
            state: 当前状态，包含消息历史

        返回:
            包含新消息的字典
        """
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 添加聊天机器人节点
    graph_builder.add_node("chatbot", chatbot)

    # 添加工具节点
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    # 添加条件边：当需要工具时从聊天机器人到工具
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # 添加从工具到聊天机器人的边：工具执行后返回聊天机器人
    graph_builder.add_edge("tools", "chatbot")

    # 设置入口点为聊天机器人
    graph_builder.set_entry_point("chatbot")

    # 设置结束点为聊天机器人
    graph_builder.set_finish_point("chatbot")

    # 编译图
    return graph_builder.compile()


def stream_graph_updates(user_input: str):
    """
    流式处理用户输入并获取助手回复

    参数:
        user_input: 用户输入的文本
    """

    # 创建聊天机器人图
    graph = create_chatbot_graph()
    for events in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in events.values():
            print("Assistant:", value["messages"][-1].content)


# 主交互循环
if __name__ == "__main__":
    stream_graph_updates("特斯拉最新股价多少？")
