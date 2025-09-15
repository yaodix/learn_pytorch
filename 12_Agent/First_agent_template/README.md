---
title: First Agent Template
emoji: ⚡
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
tags:
- smolagents
- agent
- smolagent
- tool
- agent-course
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

错误处理：

1. httpx.RemoteProtocolError: Server disconnected without sending a response

全部关闭vscode 重启vscode后成功运行

3.Error in generating model output:
'NoneType' object has no attribute 'content'， 模型没有加载成功。

模型本地使用TransformersModel

3. 报错KeyError: 'final_answer'吗

使用最新版本smolagents, 更新对应prompt
