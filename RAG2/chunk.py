# 文档分片模块，采用基于标题的分块策略，保留每个章节的标题信息，以便在检索时保持上下文完整性。

def read_data() -> str:
    with open("data.md", "r", encoding="utf-8") as f:
        return f.read()

def get_chunks() -> list[str]:
    """
    将文档分割为多个chunk

    分块策略：
    1. 按空行（\n\n）分割文档
    2. 识别以 # 开头的标题行，累积到 header 变量中
    3. 非标题内容与当前 header 合并，形成完整上下文的 chunk
    4. 每个 chunk 都包含其前面的所有标题，便于后续检索时理解上下文

    Returns:
        list[str]: 包含标题上下文的 chunk 列表
    """
    content = read_data()
    chunks = content.split("\n\n")

    result = []
    header = ""
    for c in chunks:                        # 遍历每个chunk
        if c.startswith("#"):               # 如果chunk以#开头，则将其添加到header中
            header += f"{c}\n"
        else:
            result.append(f"{header}{c}")   # 否则，将header和chunk合并，并添加到result中
            header = ""

    return result


if __name__ == "__main__":
    chunks = get_chunks()
    for c in chunks:
        print(c)
        print("--------------")


