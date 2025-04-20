#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph流式输出示例 - FastAPI异步实现
该模块演示了如何使用LangGraph框架和FastAPI实现异步流式输出功能，
通过构建一个简单的中文笑话生成器API来展示流式处理的能力。
"""

import os
import asyncio
import json
from typing import Dict, Any, TypedDict, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
import uvicorn


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


async def stream_joke_async(topic: str) -> AsyncGenerator[str, None]:
    """
    异步流式生成关于指定主题的中文笑话

    参数:
        topic: 笑话主题

    返回:
        异步生成器，产生笑话内容
    """
    graph = create_joke_graph()
    stream = graph.stream(
        {"topic": topic},
        stream_mode="messages",
    )

    for message_chunk, _ in stream:
        if message_chunk.content:
            yield message_chunk.content
            # 添加小延迟，使流式效果更明显
            await asyncio.sleep(0.05)


# 创建FastAPI应用
app = FastAPI(
    title="中文笑话生成API",
    description="使用LangGraph和FastAPI实现的异步流式中文笑话生成服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 全局变量存储LLM实例
llm: Optional[ChatOpenAI] = None


@app.get("/")
async def root():
    """
    API根路径，返回欢迎信息
    """
    return {"message": "欢迎使用中文笑话生成API，请访问 /joke?topic=主题 来生成笑话"}


@app.get("/joke")
async def generate_joke_api(topic: str = "兔子"):
    """
    生成中文笑话的API端点

    参数:
        topic: 笑话主题，默认为"兔子"

    返回:
        流式响应，包含生成的笑话内容
    """
    return StreamingResponse(
        stream_joke_async(topic),
        media_type="text/plain"
    )


async def sse_stream_joke(topic: str) -> AsyncGenerator[str, None]:
    """
    使用SSE格式流式生成笑话

    参数:
        topic: 笑话主题

    返回:
        异步生成器，产生SSE格式的数据
    """
    async for chunk in stream_joke_async(topic):
        # 将每个字符作为单独的SSE事件发送，实现打字机效果
        for char in chunk:
            yield f"data: {json.dumps({'content': char})}\n\n"
            await asyncio.sleep(0.05)  # 控制打字速度


@app.get("/joke/sse")
async def generate_joke_sse(topic: str = "兔子"):
    """
    使用SSE生成中文笑话的API端点

    参数:
        topic: 笑话主题，默认为"兔子"

    返回:
        SSE流式响应，包含生成的笑话内容
    """
    return StreamingResponse(
        sse_stream_joke(topic),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """
    提供演示页面
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>中文笑话生成器</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .input-group {
                margin-bottom: 20px;
            }
            input[type="text"] {
                width: 70%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                min-height: 100px;
                white-space: pre-wrap;
                font-size: 18px;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <h1>中文笑话生成器</h1>
        <div class="container">
            <div class="input-group">
                <input type="text" id="topic" placeholder="输入笑话主题" value="兔子">
                <button onclick="generateJoke()">生成笑话</button>
            </div>
            <div id="result"></div>
        </div>

        <script>
            function generateJoke() {
                const topic = document.getElementById('topic').value;
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = '';
                
                // 创建EventSource连接
                const eventSource = new EventSource(`/joke/sse?topic=${encodeURIComponent(topic)}`);
                
                // 监听消息事件
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    resultDiv.textContent += data.content;
                };
                
                // 监听错误事件
                eventSource.onerror = function() {
                    eventSource.close();
                };
            }
        </script>
    </body>
    </html>
    """


def main():
    """
    主函数，启动FastAPI服务器
    """
    # 初始化语言模型
    global llm
    llm = get_llm()

    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    print("http://localhost:8000/demo")
    main()
