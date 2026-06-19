import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

def read_auto(path):
    for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']:
        try:
            df = pd.read_csv(path, encoding=enc)
            if any('?' in str(c) for c in df.columns):
                continue
            return df, enc
        except Exception:
            pass
    return None, None

pairs = [
    ('data/2024_1_data.csv', 'crawl/2024-1/2024_1_data_소프트웨어학과.csv', '2024-1'),
    ('data/2024_2_data.csv', 'crawl/2024-2/2024_2_data_소프트웨어학과.csv', '2024-2'),
]

for old_path, new_path, label in pairs:
    print(f'\n{"="*60}')
    print(f'[{label}]')
    old, enc1 = read_auto(old_path)
    new, enc2 = read_auto(new_path)

    if old is None or new is None:
        print('  [ERROR] 파일 읽기 실패')
        continue

    print(f'  old({enc1}): {len(old)}행  /  new({enc2}): {len(new)}행')

    # 과목명 기준으로 비교 (수강인원)
    old_map = dict(zip(old['교과목명'].str.strip(), old['수강인원']))
    new_map = dict(zip(new['교과목명'].str.strip(), new['수강인원']))

    common = set(old_map) & set(new_map)
    only_old = set(old_map) - set(new_map)
    only_new = set(new_map) - set(old_map)

    mismatch = []
    for name in sorted(common):
        o_val = old_map[name]
        n_val = new_map[name]
        if o_val != n_val:
            mismatch.append((name, o_val, n_val))

    print(f'\n  공통 과목: {len(common)}개')
    print(f'  old에만 있는 과목: {len(only_old)}개')
    if only_old:
        for n in sorted(only_old): print(f'    - {n}')
    print(f'  new에만 있는 과목: {len(only_new)}개')
    if only_new:
        for n in sorted(only_new): print(f'    + {n}')

    print(f'\n  수강인원 불일치: {len(mismatch)}개')
    if mismatch:
        print(f'  {"과목명":<30} {"old":>8} {"new":>8}')
        print(f'  {"-"*50}')
        for name, o, n in mismatch:
            print(f'  {name:<30} {o:>8} {n:>8}')
    else:
        print('  → 일치!')
