#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终修复版中文笑话生成器 - 修复所有前端报错
"""

import os
import asyncio
import json
from typing import Dict, Any, TypedDict, AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
import uvicorn


class StepEvent(TypedDict):
    """步骤事件类型定义"""
    type: str
    stage: Optional[str] = None
    status: Optional[str] = None
    result: Optional[str] = None
    content: Optional[str] = None


def get_llm() -> ChatOpenAI:
    """获取配置好的语言模型实例"""
    return ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base=os.getenv("ZHIPU_API_URL"),
        streaming=True
    )


class State(TypedDict):
    """流程状态类型定义"""
    topic: str
    joke: str


def refine_topic(state: State) -> Dict[str, str]:
    """主题优化处理"""
    return {"topic": state["topic"] + " 和猫"}


async def stream_joke_async(topic: str) -> AsyncGenerator[StepEvent, None]:
    """增强异常处理的流式生成"""
    try:
        # 第一阶段：优化主题
        yield StepEvent(type="step", stage="refine", status="start")
        refined_topic = refine_topic({"topic": topic})["topic"]
        yield StepEvent(
            type="step",
            stage="refine",
            status="complete",
            result=refined_topic
        )

        # 第二阶段：生成笑话
        yield StepEvent(type="step", stage="generate", status="start")
        llm = get_llm()

        try:
            async for chunk in llm.astream([
                {"role": "user", "content": f"请生成一个关于{refined_topic}的中文笑话，要求：\n1. 简短有趣\n2. 使用中文\n3. 直接输出内容"}
            ]):
                if chunk.content:
                    yield StepEvent(type="content", content=chunk.content)
        finally:
            # 确保最终发送完成事件
            yield StepEvent(type="step", stage="generate", status="complete")

    except Exception as e:
        yield StepEvent(type="error", content=str(e))
        # 异常时也发送完成事件
        yield StepEvent(type="step", stage="generate", status="complete")

app = FastAPI(
    title="最终版中文笑话生成API",
    description="修复所有前端报错版本",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>中文笑话生成器 - 最终版</title>
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
                margin-bottom: 30px;
            }
            .container {
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            .input-group {
                margin-bottom: 25px;
                display: flex;
                gap: 10px;
            }
            input[type="text"] {
                flex: 1;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #4CAF50;
            }
            button {
                padding: 12px 25px;
                background: linear-gradient(135deg, #4CAF50, #45a049);
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            #status {
                margin: 20px 0;
                padding: 15px;
                background: #f8f8f8;
                border-radius: 8px;
            }
            .status-item {
                margin: 12px 0;
                padding: 10px;
                background: white;
                border-left: 4px solid #4CAF50;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }
            #result {
                min-height: 120px;
                padding: 20px;
                background: #fff9f9;
                border-radius: 8px;
                border: 2px solid #eee;
                font-size: 18px;
                line-height: 1.6;
                white-space: pre-wrap;
                transition: background 0.3s;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .typing-effect {
                animation: fadeIn 0.1s ease-in;
            }
            .error {
                border-color: #ff4444 !important;
                background: #fff0f0;
            }
        </style>
    </head>
    <body>
        <h1>📚 中文笑话生成器 - 最终版</h1>
        <div class="container">
            <div class="input-group">
                <input type="text" id="topic" placeholder="请输入笑话主题" value="兔子">
                <button onclick="generateJoke()">🚀 生成笑话</button>
            </div>
            
            <div id="status"></div>
            <div id="result"></div>
        </div>

        <script>
            function generateJoke() {
                let isProcessCompleted = false;  // 新增完成状态标志
                const topic = document.getElementById('topic').value;
                const resultDiv = document.getElementById('result');
                const statusDiv = document.getElementById('status');
                
                // 初始化显示
                resultDiv.innerHTML = '';
                statusDiv.innerHTML = `
                    <div class="status-item" id="refine_status">📝 等待开始...</div>
                    <div class="status-item" id="generate_status">📝 等待开始...</div>
                `;

                const eventSource = new EventSource(`/joke/sse?topic=${encodeURIComponent(topic)}`);
                
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'step' && data.status === 'complete' && data.stage === 'generate') {
                        isProcessCompleted = true;
                    }
                    
                    switch(data.type) {
                        case 'step':
                            if (data.status === 'complete') {
                                isProcessCompleted = true;  // 标记流程完成
                            }
                            handleStepEvent(data);
                            break;
                        case 'content':
                            appendContent(data.content);
                            break;
                        case 'error':
                            showError(data.content);
                            eventSource.close();
                            break;
                    }
                };

                eventSource.onerror = function() {
                    eventSource.close();

                    // 仅当流程未完成时显示错误
                    if (!isProcessCompleted) {
                        const generateStatus = document.getElementById('generate_status');
                        if (generateStatus) {
                            generateStatus.innerHTML = '❌ 连接意外中断';
                            generateStatus.style.color = '#ff4444';
                        }
                    }

                    // 修复报错：直接操作DOM而不是调用未定义的updateStatus
                    const generateStatus = document.getElementById('generate_status');
                    if (generateStatus) {
                        generateStatus.innerHTML = '❌ 连接意外中断';
                        generateStatus.style.color = '#ff4444';
                    }
                };
            }

            function handleStepEvent(data) {
                const element = document.getElementById(`${data.stage}_status`);
                if (!element) {
                    console.error('状态元素未找到:', data.stage);
                    return;
                }

                const statusMap = {
                    start: { icon: '🔄', color: '#666', text: '进行中...' },
                    complete: { icon: '✅', color: '#4CAF50', text: '已完成' },
                    error: { icon: '❌', color: '#ff4444', text: '失败' }
                };

                const statusInfo = statusMap[data.status] || {};
                element.innerHTML = `
                    ${statusInfo.icon || ''} ${stageToText(data.stage)} ${statusInfo.text || ''}
                    ${data.result ? `<br><small style="color: #666;">优化结果：${data.result}</small>` : ''}
                `;
                element.style.color = statusInfo.color || '#666';
            }

            function stageToText(stage) {
                return {
                    'refine': '主题优化',
                    'generate': '笑话生成'
                }[stage] || stage;
            }

            function appendContent(content) {
                const resultDiv = document.getElementById('result');
                const span = document.createElement('span');
                span.className = 'typing-effect';
                span.textContent = content;
                resultDiv.appendChild(span);
                resultDiv.scrollTop = resultDiv.scrollHeight;
            }

            function showError(message) {
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML += `
                    <div class="status-item error">
                        ❌ 错误：${message}
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    """


@app.get("/joke/sse")
async def joke_stream(topic: str = "兔子"):
    """SSE流式接口"""
    return StreamingResponse(
        sse_generator(topic),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def sse_generator(topic: str) -> AsyncGenerator[str, None]:
    """SSE格式转换"""
    async for event in stream_joke_async(topic):
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.03)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
