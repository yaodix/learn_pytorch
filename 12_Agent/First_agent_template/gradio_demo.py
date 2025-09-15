import gradio as gr

# 核心逻辑：输入 "hello" 返回 "hi"，其他输入返回 "I don't understand"
def respond(message):
    if message.lower() == "hello":
        return "hi"
    else:
        return "I don't understand"

# 创建 Gradio 界面
demo = gr.Interface(
    fn=respond,                      # 处理函数
    inputs=gr.Textbox(label="输入"),  # 输入框
    outputs=gr.Textbox(label="输出"), # 输出框
    title="Hello-Hi 演示",            # 标题
    description="输入 'hello' 试试看！" # 描述
)

# 启动本地服务（自动打开浏览器）
demo.launch(server_name="0.0.0.0", server_port=8880)
