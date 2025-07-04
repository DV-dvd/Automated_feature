import pandas as pd
import json

# 1. 读取 Excel 文件
df = pd.read_excel("中间表.xlsx")

# 2. 按 "汇总" 列分组，并转换为字典
result = {}
for group, data in df.groupby("汇总"):
    result[group] = {
        "含义": data["含义"].tolist(),
        "变量名": data["变量名"].tolist()
    }

# 3. 保存为 JSON 文件
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("JSON 文件已生成：output.json")