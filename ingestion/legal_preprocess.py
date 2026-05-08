import re
#re模块用于正则表达式处理，帮助我们识别和清洗法律文本中的结构和噪声。

# 1. 正则表达式定义（结构识别）
# 编、章、节 (LINE_PART/CHAPTER/SECTION_PATTERN)：识别如“第一编”、“第三章”等标题。

# 条目 (LINE_ARTICLE_PATTERN)：专门识别类似“第一百零二条 ”开头的行。
# 注意它要求条号后必须有空格，这是为了防止把正文中提到的“根据第十条规定”误判为新标题。
# 行内切分 (INLINE_ARTICLE_BREAK_PATTERN)：处理那种没有换行、直接连在一起写的法条。

LINE_PART_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+编\s*.*$")
LINE_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+章\s*.*$")
LINE_SECTION_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+节\s*.*$")
LINE_ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+\s*条[\s　]+")
INLINE_ARTICLE_BREAK_PATTERN = re.compile(r"(?<!\n)(第[一二三四五六七八九十百千零〇0-9]+\s*条)(?=[\s　])")

# 2. 定位正文起点
# 函数 preprocess_legal_text 开始后，第一步是“切除头部噪声”：
# 代码会搜索全文，找到第一个“第X编”或者第一个“第X条”出现的位置。
# 目的：法律文件的开头通常有很长的目录、起草说明或通过日期，这些对 RAG 检索干扰很大，直接切掉，只保留正文。
def preprocess_legal_text(text: str) -> str:
	"""清洗法律文本中的目录/页眉等噪声，保留正文结构。"""
	
	if not text:
		return text # 如果输入文本为空，直接返回空字符串。

	normalized = text.replace("\r\n", "\n")# 统一换行符，方便后续正则处理。

	# 优先从首个“编”标题开始，其次从首个法条开始。
	part_start = re.search(r"\n第[一二三四五六七八九十百千零〇0-9]+编", normalized)
	if part_start:
		normalized = normalized[part_start.start() + 1 :]
	else:
		article_start = re.search(r"\n第[一二三四五六七八九十百千零〇0-9]+\s*条", normalized)
		if article_start:
			normalized = normalized[article_start.start() + 1 :]

	noise_patterns = [
		r"^目\s*录\s*$",
		r"^中华人民共和国刑法\s*$",
		r"^\（.*?\）$",
	]
	noise_res = [re.compile(p) for p in noise_patterns]# 预编译正则表达式，提高匹配效率。

# 3. 行级噪声过滤
# 代码将文本按行切分，并进行遍历：去除空白：删掉每一行的前后空格和纯空行。
# 黑名单过滤：通过 noise_patterns 删掉特定的无用信息，比如：
# 单独出现的“目录”字样。
# 重复出现的法律名称（如“中华人民共和国刑法”）。
	cleaned_lines = []
	for raw in normalized.split("\n"):
		line = raw.strip()
		if not line:
			continue
		if any(pattern.match(line) for pattern in noise_res):
			continue
		cleaned_lines.append(line)

	# 删除目录区块污染：仅保留正文首条法条前最近一次出现的编/章/节上下文。
	if cleaned_lines:
		pre_context = {"part": "", "chapter": "", "section": ""}
		body_lines = []
		found_first_article = False
		for line in cleaned_lines:
			if not found_first_article:
				if LINE_PART_PATTERN.match(line):
					pre_context["part"] = line
					pre_context["chapter"] = ""
					pre_context["section"] = ""
					continue
				if LINE_CHAPTER_PATTERN.match(line):
					pre_context["chapter"] = line
					pre_context["section"] = ""
					continue
				if LINE_SECTION_PATTERN.match(line):
					pre_context["section"] = line
					continue
				if LINE_ARTICLE_PATTERN.match(line):
					found_first_article = True
					if pre_context["part"]:
						body_lines.append(pre_context["part"])
					if pre_context["chapter"]:
						body_lines.append(pre_context["chapter"])
					if pre_context["section"]:
						body_lines.append(pre_context["section"])
					body_lines.append(line)
			else:
				body_lines.append(line) # 一旦进入正文，就不再更新 pre_context，直接保留所有后续行。

		if body_lines:
			cleaned_lines = body_lines

	cleaned = "\n".join(cleaned_lines)
	cleaned = re.sub(r"\n{2,}", "\n", cleaned)# 将多余的连续空行压缩成一个，保持文本整洁。
	return cleaned.strip()

#对外接口，提供预处理函数和相关正则表达式模式，以便其他模块调用和测试。
__all__ = [
	"preprocess_legal_text",
	"LINE_PART_PATTERN",
	"LINE_CHAPTER_PATTERN",
	"LINE_SECTION_PATTERN",
	"LINE_ARTICLE_PATTERN",
	"INLINE_ARTICLE_BREAK_PATTERN",
]
