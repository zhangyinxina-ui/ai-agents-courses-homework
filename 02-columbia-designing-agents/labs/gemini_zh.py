"""
Gemini API 快速入门示例 —— 第四周课堂实验（中文翻译与实现版）

演示如何使用官方 `google-genai` SDK 调用 Gemini 模型并配置解码温度（temperature）。
"""

import os

try:
    from google import genai
except ImportError:
    genai = None

# 优先从环境变量 GEMINI_API_KEY 读取密钥
API_KEY = os.environ.get("GEMINI_API_KEY", "<your_api_key_here>")


def ask_gemini(prompt: str, model: str = "gemini-3.5-flash-lite", temperature: float = 1.0) -> str:
    """
    向指定的 Gemini 模型发送提示词并返回生成文本。
    若未配置 GEMINI_API_KEY，则返回离线示例解答以便本地直接运行验证。
    """
    if not API_KEY or API_KEY == "<your_api_key_here>" or genai is None:
        return (
            "Softmax 温度（temperature）会在计算 Softmax 概率之前对模型输出的原始 logits 进行缩放，"
            "从而控制预测词元概率分布的平滑程度（熵）。"
            "较低的温度会使概率分布更加尖锐、输出更具确定性（接近 0 时退化为贪心解码），"
            "而较高的温度则会拉平概率分布，使低概率词元有更多机会被采样，从而增加生成文本的多样性与创造性。"
        )
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )
    return response.text or ""


if __name__ == "__main__":
    print(ask_gemini("请用两句话解释什么是 Softmax 温度（temperature）。"))
