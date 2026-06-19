"""
campus_snapshot_merge.py
crawl/{semester}/snapshot_*.csv (21개 건물) → 14개 노드로 병합/정리해서
crawl/snapshot/ 폴더에 저장

처리 규칙:
  - 제외        : 05
  - 코드 흡수   : 62B08 -> 83 (제2종합연구동)
  - 노드 병합   : {25,26,27} / {51,61,62} / {31,32}  (점유인원 합산)
"""
import os, glob
import pandas as pd

CRAWL_DIR = os.path.join(os.path.dirname(__file__), 'crawl')
OUT_DIR   = os.path.join(CRAWL_DIR, 'snapshot')
SEMESTERS = ['2024-1', '2024-2', '2025-1', '2025-2']

EXCLUDE = {'05'}

# 최종 노드명으로의 매핑 (명시 안 된 건물은 자기 자신)
BUILDING_MAP = {
    '25': '25_26_27', '26': '25_26_27', '27': '25_26_27',
    '51': '51_61_62', '61': '51_61_62', '62': '51_61_62',
    '31': '31_32',    '32': '31_32',
    '62B08': '83',
}


def merge_semester(folder: str) -> pd.DataFrame:
    fp = glob.glob(os.path.join(CRAWL_DIR, folder, f'snapshot_{folder.replace("-", "_")}.csv'))[0]
    df = pd.read_csv(fp, encoding='utf-8-sig')

    bld_cols = [c for c in df.columns if c not in ('요일', '시각')]
    bld_cols = [c for c in bld_cols if c not in EXCLUDE]

    out = df[['요일', '시각']].copy()
    targets = sorted(set(BUILDING_MAP.get(c, c) for c in bld_cols),
                     key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    for t in targets:
        members = [c for c in bld_cols if BUILDING_MAP.get(c, c) == t]
        out[t] = df[members].sum(axis=1)

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for sem in SEMESTERS:
        merged = merge_semester(sem)
        out_path = os.path.join(OUT_DIR, f'snapshot_{sem.replace("-", "_")}.csv')
        merged.to_csv(out_path, index=False, encoding='utf-8-sig')

        bld_cols = [c for c in merged.columns if c not in ('요일', '시각')]
        active = merged[merged[bld_cols].sum(axis=1) > 0]
        print(f'=== {sem} ===')
        print(f'  저장: {out_path}')
        print(f'  노드 {len(bld_cols)}개: {bld_cols}')
        print(f'  슬롯: {len(merged)}개 전체 / {len(active)}개 수업 있음')


if __name__ == '__main__':
    main()
