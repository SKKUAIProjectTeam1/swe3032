"""
campus_snapshot.py
동일 양식의 xlsx → 요일×시각×건물 스냅샷 CSV

사용법: python campus_snapshot.py 2025_1.xlsx
       python campus_snapshot.py 2025_2.xlsx
"""
import sys, re
import pandas as pd

SLOT_MIN  = 30       # 스냅샷 간격 (분)
DAY_START = '08:00'
DAY_END   = '22:00'
DAY_ORDER = ['월', '화', '수', '목', '금', '토']


def _t2m(t: str) -> int:
    h, m = map(int, t.split(':'))
    return h * 60 + m

def _m2t(m: int) -> str:
    return f'{m // 60:02d}:{m % 60:02d}'


def parse_sessions(time_str: str, fallback_bld: str = '') -> list[dict]:
    """
    '화10:30-11:45【21502】,목09:00-10:15【21502】'
    → [{'day':'화','start':'10:30','end':'11:45','building':'21'}, ...]

    건물번호: 【21502】→ '21', 【모듈D】→ '모듈D'
    미지정·온라인 등은 스킵, fallback_bld로 대체
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
        for rm_m in re.finditer(r'【([^】]+)】', seg):
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


def build_snapshot(df: pd.DataFrame, slot_min: int = SLOT_MIN) -> pd.DataFrame:
    """
    각 시간 슬롯마다 건물별 체류 인원수 계산.
    체류: start <= slot_time < end 인 수업의 수강인원 합산.
    """
    df = df[df['building'].astype(str) != '0'].copy()

    # 수업 세션 이벤트 수집
    events = []
    for _, row in df.iterrows():
        sessions = parse_sessions(
            row.get('수업시간대', ''),
            str(row.get('building', ''))
        )
        enr = int(row['수강인원']) if pd.notna(row.get('수강인원')) else 0
        for s in sessions:
            events.append({**s, 'enr': enr})

    if not events:
        print('[WARN] 파싱된 세션 없음 — 수업시간대 컬럼 확인 필요')
        return pd.DataFrame()

    # 등장한 건물만, 숫자→정수 오름차순, 모듈* 뒤
    buildings = sorted(
        set(e['building'] for e in events),
        key=lambda x: (0, int(x), '') if x.isdigit() else (1, 0, x)
    )

    days  = [d for d in DAY_ORDER if any(e['day'] == d for e in events)]
    slots = range(_t2m(DAY_START), _t2m(DAY_END), slot_min)

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


def main(xlsx_path: str):
    print(f'[읽기] {xlsx_path}')
    df = pd.read_excel(xlsx_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    print(f'  {len(df)}행, 컬럼: {df.columns.tolist()}')

    snap = build_snapshot(df)
    if snap.empty:
        return

    out_path = xlsx_path.replace('.xlsx', '_snapshot.csv')
    snap.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'[저장] {out_path}  ({snap.shape[0]}행 × {snap.shape[1]}열)')

    # 수업이 있는 슬롯만 출력 (확인용)
    bld_cols = [c for c in snap.columns if c not in ('요일', '시각')]
    active = snap[snap[bld_cols].sum(axis=1) > 0]
    print(f'\n[활성 슬롯 {len(active)}개]')
    print(active.to_string(index=False))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '2025_1.xlsx'
    main(path)
