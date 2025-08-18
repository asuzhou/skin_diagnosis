#E:/Anaconda/envs/skin_env/python.exe -m streamlit run app.py
import streamlit as st
from PIL import Image
import os
# 导入你的模型推理类和豆包对话类
from infer import SkinInferencer
from chat import DoubaoChat
from infer2 import SkinDiseaseInference
from infer3 import SkinDiseaseInference as SkinDiseaseInference2

region_map={"手臂": "ARM", "颈部": "NECK", "面部": "FACE", "手部": "HAND", "前臂": "FOREARM", 
            "胸部": "CHEST", "鼻部": "NOSE", "大腿": "THIGH", "头皮": "SCALP",
              "耳部": "EAR", "背部": "BACK", "足部": "FOOT", "腹部": "ABDOMEN", "唇部": "LIP"}

diagnose_map={"NEV": "色素痣", "BCC": "基底细胞癌", "ACK": "光化性角化病",
               "SEK": "脂溢性角化病", "SCC": "鳞状细胞癌", "MEL": "黑色素瘤"}

diagnose_map2={
                "hailey hailey disease": "家族性良性天疱疮",
                "papilomatosis confluentes and reticulate": "融合性网状乳头瘤病",
                "scabies": "疥疮",
                "tuberous sclerosis": "结节性硬化症",
                "keloid": "瘢痕疙瘩",
                "tungiasis": "潜蚤病",
                "dermatofibroma": "皮肤纤维瘤",
                "lupus erythematosus": "红斑狼疮",
                "dariers disease": "达里埃病（毛囊角化病）",
                "neutrophilic dermatoses": "嗜中性皮病",
                "seborrheic dermatitis": "脂溢性皮炎",
                "photodermatoses": "光敏感性皮肤病",
                "cheilitis": "唇炎",
                "mucinosis": "黏蛋白沉积症",
                "myiasis": "蝇蛆病",
                "sarcoidosis": "结节病",
                "nematode infection": "线虫感染",
                "erythema nodosum": "结节性红斑",
                "keratosis pilaris": "毛发角化病",
                "lichen simplex": "单纯性苔藓",
                "scleromyxedema": "硬化性黏液水肿",
                "pityriasis rosea": "玫瑰糠疹",
                "pityriasis rubra pilaris": "毛发红糠疹",
                "melanoma": "黑色素瘤",
                "squamous cell carcinoma": "鳞状细胞癌",
                "lichen amyloidosis": "苔藓样淀粉样变",
                "mycosis fungoides": "蕈样肉芽肿",
                "scleroderma": "硬皮病",
                "porokeratosis actinic": "光化性汗孔角化症",
                "allergic contact dermatitis": "过敏性接触性皮炎",
                "naevus comedonicus": "黑头粉刺痣",
                "folliculitis": "毛囊炎",
                "vitiligo": "白癜风",
                "erythema annulare centrifigum": "离心性环状红斑",
                "dyshidrotic eczema": "汗疱疹",
                "neurofibromatosis": "神经纤维瘤病",
                "lichen planus": "扁平苔藓",
                "calcinosis cutis": "皮肤钙质沉着症",
                "pediculosis lids": "眼睑虱病",
                "basal cell carcinoma": "基底细胞癌",
                "granuloma annulare": "环状肉芽肿",
                "fixed eruptions": "固定性药疹",
                "psoriasis": "银屑病（牛皮癣）",
                "fordyce spots": "福代斯斑点（异位皮脂腺）",
                "porokeratosis of mibelli": "米贝利汗孔角化症",
                "syringoma": "汗管瘤",
                "granuloma pyogenic": "化脓性肉芽肿",
                "telangiectases": "毛细血管扩张",
                "erythema multiforme": "多形红斑",
                "pityriasis lichenoides chronica": "慢性苔藓样糠疹",
                "factitial dermatitis": "人工性皮炎",
                "kaposi sarcoma": "卡波西肉瘤",
                "ehlers danlos syndrome": "埃勒斯 - 当洛斯综合征",
                "prurigo nodularis": "结节性痒疹",
                "neurotic excoriations": "神经官能性表皮剥脱",
                "actinic keratosis": "光化性角化病",
                "acrodermatitis enteropathica": "肠病性肢端皮炎",
                "juvenile xanthogranuloma": "幼年性黄色肉芽肿",
                "stasis edema": "淤滞性水肿",
                "acanthosis nigricans": "黑棘皮病",
                "urticaria": "荨麻疹",
                "aplasia cutis": "皮肤发育不全",
                "lymphangioma": "淋巴管瘤",
                "acne": "痤疮",
                "langerhans cell histiocytosis": "朗格汉斯细胞组织细胞增生症",
                "ichthyosis vulgaris": "寻常型鱼鳞病",
                "behcets disease": "贝赫切特病",
                "drug induced pigmentary changes": "药物性色素异常",
                "necrobiosis lipoidica": "类脂质渐进性坏死",
                "pilar cyst": "毛发囊肿",
                "mucous cyst": "黏液囊肿",
                "porphyria": "卟啉病",
                "erythema elevatum diutinum": "持久性隆起性红斑",
                "milia": "粟丘疹",
                "hidradenitis": "汗腺炎",
                "xeroderma pigmentosum": "着色性干皮病",
                "rhinophyma": "鼻赘",
                "urticaria pigmentosa": "色素性荨麻疹",
                "dermatomyositis": "皮肌炎",
                "striae": "皮肤条纹（萎缩纹）",
                "eczema": "湿疹",
                "tick bite": "蜱虫叮咬",
                "pilomatricoma": "毛母质瘤",
                "lupus subacute": "亚急性红斑狼疮",
                "paronychia": "甲沟炎",
                "congenital nevus": "先天性色素痣",
                "lyme disease": "莱姆病",
                "epidermal nevus": "表皮痣",
                "perioral dermatitis": "口周皮炎",
                "rosacea": "玫瑰痤疮"
                }


diagnose_map3={"genodermatoses": "遗传性皮肤病", "inflammatory": "炎症性皮肤病", 
               "benign dermal": "良性真皮肿瘤", "malignant melanoma": "恶性黑色素瘤", 
               "malignant epidermal": "恶性表皮肿瘤", "malignant cutaneous lymphoma": "恶性皮肤淋巴瘤",
                 "benign epidermal": "良性表皮肿瘤", "malignant dermal": "恶性真皮肿瘤", 
                 "benign melanocyte": "良性黑素细胞肿瘤"}
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
    inferencer2 = SkinDiseaseInference2('config3/skin_model2.pth','config3')
    inferencer3 = SkinDiseaseInference('config2/skin_model2.pth','config2')
    # 初始化豆包对话客户端
    doubao_chat = DoubaoChat()
    return inferencer, doubao_chat,inferencer2,inferencer3

# 加载资源
inferencer, doubao_chat ,inferencer2 ,inferencer3= init_resources()

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
            pred_result2 = inferencer2.predict_single_image(uploaded_img)
            # 构造给豆包的提示（包含分类结果）
            prompt = f"""
            请基于以下皮肤病诊断模型的结果，给用户解释：（模型的对比分析过程不要告知患者）
            - 模型1预测类别：{diagnose_map[pred_result['predicted_diagnostic']]}
            - 模型1预测概率分布：{ {diagnose_map[k]: round(v, 3) for k, v in pred_result['probabilities'].items()} }
            - 模型2预测类别：{diagnose_map3[pred_result2['label']]}，
            - 置信度: {pred_result2['confidence']:.2f}
            结合两个模型的预测结果，分析可能属于的皮肤病类型，
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
