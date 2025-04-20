#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph流式输出示例
该模块演示了如何使用LangGraph框架实现流式输出功能，
通过构建一个简单的中文笑话生成器来展示流式处理的能力。
"""

import os
from typing import Dict, Any, TypedDict, Generator, Tuple

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START


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
    定义状态类型

    属性:
        topic: 笑话主题
        joke: 生成的笑话内容
    """
    topic: str
    joke: str


def refine_topic(state: State) -> Dict[str, str]:
    """
    优化主题，在原始主题后添加"和猫"

    参数:
        state: 当前状态，包含原始主题

    返回:
        包含优化后主题的字典
    """
    return {"topic": state["topic"] + " 和猫"}


def generate_joke(state: State) -> Dict[str, str]:
    """
    根据主题生成中文笑话

    参数:
        state: 当前状态，包含优化后的主题

    返回:
        包含生成笑话的字典
    """
    llm_response = llm.invoke(
        [
            {"role": "user",
             "content": f"请生成一个关于{state['topic']}的中文笑话，要求：\n1. 笑话要简短有趣\n2. 使用中文回答\n3. 直接给出笑话内容，不要加任何前缀"}
        ]
    )
    return {"joke": llm_response.content}


def create_joke_graph() -> StateGraph:
    """
    创建笑话生成的状态图

    返回:
        StateGraph: 配置好的状态图实例
    """
    return (
        StateGraph(State)
        .add_node("refine_topic", refine_topic)
        .add_node("generate_joke", generate_joke)
        .add_edge(START, "refine_topic")
        .add_edge("refine_topic", "generate_joke")
        .compile()
    )


def stream_joke(topic: str) -> Generator[Tuple[BaseMessage, Dict[str, Any]], None, None]:
    """
    流式生成关于指定主题的中文笑话

    参数:
        topic: 笑话主题

    返回:
        生成器，产生消息块和元数据
    """
    graph = create_joke_graph()
    return graph.stream(
        {"topic": topic},
        stream_mode="messages",
    )


def main():
    """
    主函数，运行中文笑话生成示例
    """
    # 初始化语言模型
    global llm
    llm = get_llm()

    # 设置初始主题
    initial_topic = "兔子"

    # 流式输出笑话生成过程
    print(f"正在生成关于'{initial_topic}'的中文笑话...")
    for message_chunk, metadata in stream_joke(initial_topic):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print("\n笑话生成完成！")


if __name__ == "__main__":
    main()
