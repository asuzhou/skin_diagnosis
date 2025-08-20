import os
from volcenginesdkarkruntime import Ark
import streamlit as st

class DoubaoChat:
    def __init__(self, model="doubao-pro-32k-241215"):
        '''
        kimi-k2-250711
        doubao-pro-32k-241215
        doubao-pro-4k-240515
        deepseek-r1-250528
        doubao-1-5-lite-32k-250115
        '''
        # 初始化豆包客户端
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=st.secrets["doubao_api_key"],  # 替换为你的API Key
        )
        self.model = model
        # 初始化对话历史（包含系统提示）
        self.chat_history = [
            {"role": "system", "content": "你是一个皮肤病诊断医生，根据模型推测的结果分析病情，关于模型的分析不需要告知患者。"}
        ]
        self.json_text=''

    def reset_chat(self):
        """重置对话历史（保留系统提示）"""
        self.chat_history = self.chat_history[:1]  # 只保留系统提示

    def stream_chat(self, user_input):
        """
        发送用户输入并流式返回响应
        :param user_input: 用户输入的文本
        :return: 生成器，逐步返回响应内容
        """
        # 将用户输入添加到对话历史
        self.chat_history.append({"role": "user", "content": user_input})

        # 发送流式请求
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,  # 发送完整对话历史
            extra_headers={'x-is-encrypted': 'true'},
            temperature=0.7,
            top_p=0.7,
            max_tokens=4096,
            stream=True,  # 启用流式输出
        )

        # 收集模型响应（用于更新对话历史）
        full_response = ""
        json_flag=False
        self.json_text=""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if ('json' in content) or ('```json' in content) or json_flag:
                    json_flag=True  
                    self.json_text += content
                else:
                    full_response += content
                    yield content  # 流式返回每一段内容
                
        # 将模型响应添加到对话历史（维护上下文）
        self.chat_history.append({"role": "assistant", "content": full_response})
