#E:/Anaconda/envs/skin_env/python.exe -m streamlit run app.py
import streamlit as st
from PIL import Image
import os
# 导入你的模型推理类和豆包对话类
from infer import SkinInferencer
from chat import DoubaoChat

region_map={"手臂": "ARM", "颈部": "NECK", "面部": "FACE", "手部": "HAND", "前臂": "FOREARM", 
            "胸部": "CHEST", "鼻部": "NOSE", "大腿": "THIGH", "头皮": "SCALP",
              "耳部": "EAR", "背部": "BACK", "足部": "FOOT", "腹部": "ABDOMEN", "唇部": "LIP"}

diagnose_map={"NEV": "色素痣", "BCC": "基底细胞癌", "ACK": "光化性角化病",
               "SEK": "脂溢性角化病", "SCC": "鳞状细胞癌", "MEL": "黑色素瘤"}

# 初始化 Streamlit 页面配置
st.set_page_config(page_title="皮肤病辅助诊断", layout="wide")

# 缓存模型和豆包客户端（避免重复加载）
@st.cache_resource
def init_resources():
    # 初始化分类模型推理器
    inferencer = SkinInferencer(
        model_path="config/best_fusion_model_skin.pth",
        meta_preprocessor_path="config/meta_preprocessor.pkl",
        label_encoder_path="config/label_encoder.pkl"
    )
    # 初始化豆包对话客户端
    doubao_chat = DoubaoChat()
    return inferencer, doubao_chat

# 加载资源
inferencer, doubao_chat = init_resources()

# 页面标题
st.title("皮肤病辅助诊断工具")

# 侧边栏：用户输入区域
with st.sidebar:
    st.header("输入信息")
    uploaded_img = st.file_uploader("上传皮肤图像", type=["png", "jpg", "jpeg"])
    age = st.number_input("患者年龄", min_value=0, max_value=120, value=20)
    region = st.selectbox(
        "病变区域",
        ["手臂", "颈部", "面部", "手部", "前臂", "胸部", "鼻部", "大腿", "头皮",
          "耳部", "背部", "足部", "腹部", "唇部"]
    )
    diagnose_btn = st.button("开始诊断", use_container_width=True)
    reset_btn = st.button("重置对话", use_container_width=True)

# 重置对话逻辑
if reset_btn:
    doubao_chat.reset_chat()
    # 清空显示区域
    st.session_state["messages"] = []
    st.success("对话已重置")

# 初始化会话状态（存储对话消息）
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示历史对话
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 诊断逻辑
if diagnose_btn and uploaded_img is not None:
    try:
        # 1. 显示上传的图片
        image = Image.open(uploaded_img).convert("RGB")
        with st.chat_message("user"):
            st.image(image, caption="上传的图像", width=300)
            st.write(f"年龄：{age}，病变区域：{region}")

        # 2. 调用分类模型获取诊断结果
        with st.spinner("正在分析图像..."):
            pred_result = inferencer.predict(uploaded_img, age, region_map[region])  # 注意：这里需要修改infer.py的predict方法支持直接传入UploadedFile
            # 构造给豆包的提示（包含分类结果）
            prompt = f"""
            请基于以下皮肤病诊断模型的结果，用通俗语言给用户解释：
            - 预测类别：{diagnose_map[pred_result['predicted_diagnostic']]}
            - 概率分布：{ {diagnose_map[k]: round(v, 3) for k, v in pred_result['probabilities'].items()} }
            请说明可能的症状、注意事项和建议，适当使用专业术语。
            """

        # 3. 调用豆包API并流式输出结果
        with st.chat_message("assistant"):
            # 创建一个占位符用于动态更新内容
            message_placeholder = st.empty()
            full_response = ""
            # 流式获取响应并更新界面
            for chunk in doubao_chat.stream_chat(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")  # 显示光标动画
            # 最终替换为完整响应（去除光标）
            message_placeholder.markdown(full_response)

        # 4. 保存对话到会话状态
        st.session_state["messages"].append({
            "role": "user",
            "content": f"上传了图像（年龄：{age}，区域：{region}）"
        })
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response
        })

    except Exception as e:
        st.error(f"处理失败：{str(e)}")

# 支持后续对话（用户可继续提问）
if user_input := st.chat_input("有其他问题请提问..."):
    # 显示用户输入
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 流式获取豆包响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in doubao_chat.stream_chat(user_input):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
