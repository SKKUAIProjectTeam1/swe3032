"""
campus_snapshot.py
crawl/ 폴더의 학과별 CSV → 학기별 요일×시각×건물 스냅샷 CSV

사용법:
  python campus_snapshot.py              # 4개 학기 전체
  python campus_snapshot.py 2025-1      # 특정 학기만
"""
import sys, re, os, glob
import pandas as pd

SLOT_MIN  = 30
DAY_START = '08:00'
DAY_END   = '22:00'
DAY_ORDER = ['월', '화', '수', '목', '금', '토']

CRAWL_DIR = os.path.join(os.path.dirname(__file__), 'crawl')

SEMESTER_FOLDERS = {
    '2024-1': '2024-1',
    '2024-2': '2024-2',
    '2025-1': '2025-1',
    '2025-2': '2025-2',
}


# ── 시간 유틸 ──────────────────────────────────────────────────────────────
def _t2m(t: str) -> int:
    h, m = map(int, t.split(':'))
    return h * 60 + m

def _m2t(m: int) -> str:
    return f'{m // 60:02d}:{m % 60:02d}'


# ── 세션 파싱 ──────────────────────────────────────────────────────────────
def parse_sessions(time_str: str, fallback_bld: str = '') -> list[dict]:
    """
    '화10:30-11:45【21502】,목09:00-10:15【21502】'
    → [{'day':'화','start':'10:30','end':'11:45','building':'21'}, ...]
    미지정·온라인 등은 스킵.
    """
    if not time_str or pd.isna(time_str):
        return []

    results = []
    for seg in str(time_str).split(','):
        seg = seg.strip()
        m = re.match(r'([월화수목금토일])(\d{1,2}:\d{2})-(\d{1,2}:\d{2})', seg)
        if not m:
            continue
        day, start, end = m.group(1), m.group(2), m.group(3)

        bld = None
        for rm_m in re.finditer(r'[【\[]([^】\]]+)[】\]]', seg):
            r = rm_m.group(1).strip()
            if not r or any(x in r for x in ('미지정', '온라인', '원격', 'ON', 'OFF', 'h(')):
                continue
            nb = re.match(r'(\d{2})\d+', r)
            if nb:
                bld = nb.group(1)
                break
            if re.match(r'모듈[A-Za-z]', r):
                bld = r[:3]
                break

        if bld is None and fallback_bld and fallback_bld != '0':
            bld = fallback_bld

        if bld:
            results.append({'day': day, 'start': start, 'end': end, 'building': bld})

    return results


# ── CSV 머지 + 전처리 ──────────────────────────────────────────────────────
def load_semester(folder_key: str) -> pd.DataFrame:
    """
    crawl/{folder_key}/*.csv 전체 머지 후:
    1. building=0 제외
    2. (과목명, 수업시간대) 기준 중복 제거  ← 학과간 공통과목 중복
    3. 수강인원 / 개설강좌수  ← 분반 균등분배
    """
    pattern = os.path.join(CRAWL_DIR, folder_key, '*.csv')
    files = glob.glob(pattern)
    if not files:
        print(f'  [WARN] {pattern} - 파일 없음')
        return pd.DataFrame()

    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp, encoding='utf-8-sig'))
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')

    df = pd.concat(frames, ignore_index=True)
    print(f'  머지: {len(files)}개 파일, {len(df)}행')

    # 컬럼명 표준화 (혹시 다를 경우 대비)
    df.columns = df.columns.str.strip()

    # building=0 제외
    df = df[df['building'].astype(str) != '0'].copy()

    # (교과목명, 수업시간대) dedup — 같은 강의가 여러 학과 파일에 중복
    before = len(df)
    df = df.drop_duplicates(subset=['교과목명', '수업시간대'], keep='first')
    print(f'  dedup: {before} -> {len(df)}행 ({before - len(df)}건 중복 제거)')

    # 개설강좌수로 균등분배
    df['개설강좌수'] = pd.to_numeric(df['개설강좌수'], errors='coerce').fillna(1).clip(lower=1)
    df['수강인원']   = pd.to_numeric(df['수강인원'],   errors='coerce').fillna(0)
    df['분반인원']   = (df['수강인원'] / df['개설강좌수']).round().astype(int)

    return df


# ── 스냅샷 빌드 ────────────────────────────────────────────────────────────
def build_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """요일×시각 슬롯마다 건물별 체류 인원수."""
    events = []
    for _, row in df.iterrows():
        sessions = parse_sessions(
            row.get('수업시간대', ''),
            str(row.get('building', ''))
        )
        enr = int(row.get('분반인원', 0))
        for s in sessions:
            events.append({**s, 'enr': enr})

    if not events:
        print('  [WARN] 파싱된 세션 없음')
        return pd.DataFrame()

    buildings = sorted(
        set(e['building'] for e in events),
        key=lambda x: (0, int(x), '') if x.isdigit() else (1, 0, x)
    )
    days  = [d for d in DAY_ORDER if any(e['day'] == d for e in events)]
    slots = range(_t2m(DAY_START), _t2m(DAY_END), SLOT_MIN)

    records = []
    for day in days:
        day_ev = [e for e in events if e['day'] == day]
        for slot in slots:
            row_out = {'요일': day, '시각': _m2t(slot)}
            for b in buildings:
                row_out[b] = sum(
                    e['enr'] for e in day_ev
                    if e['building'] == b
                    and _t2m(e['start']) <= slot < _t2m(e['end'])
                )
            records.append(row_out)

    return pd.DataFrame(records)


# ── 메인 ──────────────────────────────────────────────────────────────────
def main(target: str | None = None):
    targets = [target] if target else list(SEMESTER_FOLDERS.keys())

    for key in targets:
        folder = SEMESTER_FOLDERS.get(key)
        if not folder:
            print(f'[ERROR] 알 수 없는 학기: {key}')
            continue

        print(f'\n=== {key} ===')
        df = load_semester(folder)
        if df.empty:
            continue

        snap = build_snapshot(df)
        if snap.empty:
            continue

        out_path = os.path.join(CRAWL_DIR, folder, f'snapshot_{folder.replace("-","_")}.csv')
        snap.to_csv(out_path, index=False, encoding='utf-8-sig')

        bld_cols = [c for c in snap.columns if c not in ('요일', '시각')]
        active   = snap[snap[bld_cols].sum(axis=1) > 0]
        print(f'  저장: {out_path}')
        print(f'  슬롯: {len(snap)}개 전체 / {len(active)}개 수업 있음')
        print(f'  건물: {bld_cols}')


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
