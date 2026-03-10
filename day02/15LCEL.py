from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 1. 定义模型 (Model)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-9a5b614d3e224da098180d6fcf1aa4c4", 
    openai_api_base="https://api.deepseek.com"
)

# 2. 定义提示词模板 (Prompt)
# system: 设定 AI 的角色
# user: 用户的具体输入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的技术专家，擅长用通俗易懂的'大白话'解释复杂的科技概念。你的解释应该包含一个生动的比喻。"),
    ("user", "{concept}")
])

# 3. 定义输出解析器 (Output Parser)
# 将模型的 Message 对象直接转为纯字符串
parser = StrOutputParser()

# 4. 构建链 (Chain) - 使用管道符 | 连接
# 流程：输入字典 -> 填充 Prompt -> 发给 LLM -> 解析输出
chain = prompt | llm | parser

# 5. 调用链
concept_to_explain = "遗忘学习"
print(f"正在解释概念：{concept_to_explain}...\n")

result = chain.invoke({"concept": concept_to_explain})

print("--- DeepSeek 回答 ---")
print(result)