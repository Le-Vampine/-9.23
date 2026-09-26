# -*- coding: utf-8 -*-
"""把 06_appendix/appendix.pdf 合并进 main.pdf：替换原第 57 页（B+占位C），输出 main_full.pdf"""
import re
import pymupdf

ROOT = r'D:\数学建模\数学建模'
main = pymupdf.open(ROOT + r'\main.pdf')
appx = pymupdf.open(ROOT + r'\06_appendix\appendix.pdf')

print('原 main.pdf 页数:', main.page_count)
old_toc = main.get_toc()
print('原书签数:', len(old_toc))

# 1) 删除原第 57 页（索引 56，含 B 与 C 占位）
main.delete_page(56)
print('删除后页数:', main.page_count)

# 2) 追加附录（接在第 56 页之后）
main.insert_pdf(appx)
print('合并后页数:', main.page_count)

# 3) 重建书签：保留原前 56 页的条目，追加 B / C / C.x
keep = [e for e in old_toc if e[2] <= 56 and 'B 不确定性' not in e[1] and e[1] != 'C 核心代码']
keep = [e for e in keep if not (e[0] >= 1 and e[1].strip() in ('B 不确定性与拒识能力分析（补充实验）', 'C 核心代码'))]
new_toc = keep + [
    [1, 'B 不确定性与拒识能力分析（补充实验）', 57],
    [1, 'C 核心代码', 57],
]
# 在合并后的文档里定位 C.1–C.13 的页码
pages_text = [main[i].get_text() for i in range(main.page_count)]
for n in range(1, 14):
    pat = re.compile(r'C\.%d\s' % n)
    found = None
    for i in range(56, main.page_count):
        if pat.search(pages_text[i]):
            found = i + 1
            break
    if found:
        # 取标题行作为书签名
        m = re.search(r'C\.%d\s+([^\n]{4,50})' % n, pages_text[found - 1])
        title = ('C.%d %s' % (n, m.group(1).strip())) if m else ('C.%d' % n)
        new_toc.append([2, title, found])
main.set_toc(new_toc)
print('新书签数:', len(new_toc))

# 4) 元数据
md = main.metadata
md['title'] = '复杂场景下多模态情感预测的数学建模与算法设计'
main.set_metadata(md)

out = ROOT + r'\main_full.pdf'
main.save(out, garbage=3, deflate=True)
print('已保存:', out)

# 5) 自检
chk = pymupdf.open(out)
print('输出页数:', chk.page_count)
for p in [56, 57, 58, 129, 130]:
    if p <= chk.page_count:
        t = chk[p - 1].get_text()
        tail = t.strip().split('\n')[-1] if t.strip() else ''
        head = t.strip().split('\n')[0][:40] if t.strip() else ''
        print(f'  p{p}: 首行[{head}] 末行[数字:{tail[:6]}]')
garbled = 0
for i in range(56, chk.page_count):
    if '\ufffd' in chk[i].get_text() or '\u25a1' in chk[i].get_text():
        garbled += 1
print('新增页乱码页数:', garbled)
