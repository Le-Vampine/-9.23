# -*- coding: utf-8 -*-
"""Extract text from a .docx into a UTF-8 .txt file (paragraph-per-line)."""
import sys
import zipfile
import re
import xml.etree.ElementTree as ET

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def para_text(p):
    parts = []
    for node in p.iter():
        tag = node.tag
        if tag == W + 't':
            parts.append(node.text or '')
        elif tag == W + 'tab':
            parts.append('\t')
        elif tag == W + 'br':
            parts.append('\n')
    return ''.join(parts)


def walk(container, out):
    for child in container:
        if child.tag == W + 'p':
            out.append(para_text(child))
        elif child.tag == W + 'tbl':
            for row in child.findall(W + 'tr'):
                cells = []
                for cell in row.findall(W + 'tc'):
                    cell_lines = []
                    walk(cell, cell_lines)
                    cells.append(' '.join(x.strip() for x in cell_lines if x.strip()))
                out.append(' | '.join(cells))
        else:
            walk(child, out)


def main(src, dst):
    with zipfile.ZipFile(src) as z:
        xml = z.read('word/document.xml')
    root = ET.fromstring(xml)
    body = root.find(W + 'body')
    out = []
    walk(body, out)
    text = '\n'.join(out)
    text = re.sub(r'\n{3,}', '\n\n', text)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)
    print('OK paragraphs=%d chars=%d -> %s' % (len(out), len(text), dst))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
