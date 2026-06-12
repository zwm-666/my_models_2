from docx import Document
from pathlib import Path
import re, sys, json
p=Path(sys.argv[1])
doc=Document(p)
paras=[pa.text for pa in doc.paragraphs]
ref_idx=None
for i,t in enumerate(paras):
    if t.strip()=='References': ref_idx=i
print('doc', p)
print('paragraphs', len(paras), 'ref_idx', ref_idx)
body='\n'.join(paras[:ref_idx] if ref_idx is not None else paras)
refs=[t.strip() for t in paras[ref_idx+1:] if t.strip()] if ref_idx is not None else []
print('references_count', len(refs))
# collect citations
cits=[]
for m in re.finditer(r'\[([^\]]+)\]', body):
    raw=m.group(1)
    if re.fullmatch(r'\d+(?:\s*[,;，、-]\s*\d+)*', raw):
        nums=[]
        range_or_multi=False
        for part in re.split(r'[,;，、]\s*', raw):
            part=part.strip()
            if '-' in part:
                a,b=map(int,re.split(r'\s*-\s*',part))
                nums.extend(range(a,b+1)); range_or_multi=True
            elif part:
                nums.append(int(part))
        if len(nums)>1: range_or_multi=True
        cits.append((m.group(0), nums, m.start(), range_or_multi))
print('citation_occurrences', len(cits))
print('first_citations', [c[0] for c in cits[:120]])
first=[]; seen=set()
for raw, nums, pos, multi in cits:
    for n in nums:
        if n not in seen:
            seen.add(n); first.append(n)
print('first_unique_order', first)
print('is_first_order_1_to_N', first==list(range(1, max(first)+1)) if first else True)
print('max_cited', max(first) if first else 0)
print('range_or_multi', [c[0] for c in cits if c[3]])
print('doi_lower_count', sum('doi:' in r.lower() for r in refs))
print('https_count', sum('http://' in r.lower() or 'https://' in r.lower() for r in refs))
print('blank_refs', [i+1 for i,r in enumerate(refs) if not r.strip()])
for i,r in enumerate(refs[:5],1): print(f'REF{i}', r[:180])
for i,r in enumerate(refs[-5:],len(refs)-4): print(f'REF{i}', r[:180])
