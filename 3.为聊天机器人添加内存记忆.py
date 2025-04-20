#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聊天机器人记忆功能实现
该模块实现了一个具有记忆功能的聊天机器人，使用LangGraph框架构建，
能够记住对话历史并在多轮对话中保持上下文连贯性。
"""

import os
from typing import Annotated, Dict, List, Any

from IPython.display import Image, display
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


def get_llm() -> ChatOpenAI:
    """
    获取语言模型实例

    返回:
        ChatOpenAI: 配置好的语言模型实例
    """
    return ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base=os.getenv("ZHIPU_API_URL"),
    )


class State(TypedDict):
    """
    定义聊天机器人的状态类型

    属性:
        messages: 消息列表，使用add_messages注解标记
    """
    messages: Annotated[List[Dict[str, Any]], add_messages]


def create_chat_graph() -> StateGraph:
    """
    创建聊天机器人的状态图

    返回:
        StateGraph: 配置好的状态图实例
    """
    # 初始化状态图
    graph_builder = StateGraph(State)

    # 配置搜索工具
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # 获取语言模型并绑定工具
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

    # 添加条件边
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # 添加从工具到聊天机器人的边
    graph_builder.add_edge("tools", "chatbot")

    # 设置入口点
    graph_builder.set_entry_point("chatbot")

    return graph_builder


def visualize_graph(graph: StateGraph) -> None:
    """
    可视化状态图并保存为PNG文件

    参数:
        graph: 状态图实例
    """
    try:
        graph_png = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(graph_png)
    except Exception:
        # 可视化需要额外的依赖，这是可选的
        pass


def run_conversation(graph: StateGraph, user_input: str, config: Dict[str, Any]) -> None:
    """
    运行一轮对话

    参数:
        graph: 状态图实例
        user_input: 用户输入的消息
        config: 配置参数
    """
    # 注意：config是stream()或invoke()的第二个位置参数
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )

    # 打印对话结果
    for event in events:
        event["messages"][-1].pretty_print()


def main():
    """
    主函数，运行聊天机器人示例
    """
    # 创建状态图
    graph_builder = create_chat_graph()

    # 配置内存检查点
    # 注意：这里使用了内存检查点，适合教程演示
    # 在生产环境中，可以改用SqliteSaver或PostgresSaver，连接到自己的数据库
    memory = MemorySaver()

    # 编译状态图
    # 检查点比简单的聊天记忆功能强大得多
    # 它允许随时保存和恢复复杂的状态，用于错误恢复、人机交互、时间旅行交互等
    graph = graph_builder.compile(checkpointer=memory)

    # 可视化状态图
    visualize_graph(graph)

    # 配置对话参数
    config = {"configurable": {"thread_id": "1"}}

    # 第一轮对话
    user_input = "Hi there! My name is Will."
    run_conversation(graph, user_input, config)

    # 第二轮对话，测试记忆功能
    user_input = "Remember my name?"
    run_conversation(graph, user_input, config)


if __name__ == "__main__":
    main()
