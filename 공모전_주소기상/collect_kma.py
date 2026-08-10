"""기상자료개방포털에서 지점별 폭염일수 / 열대야일수 / 한파일수를 수집한다.

로그인·API키 없이 공개 CSV 다운로드 엔드포인트를 사용한다.
"""
import io
import json
import re
import time

import pandas as pd
import requests

BASE = "https://data.kma.go.kr"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": BASE}

SPECS = {
    "폭염일수": dict(
        url=f"{BASE}/climate/heatWave/selectHeatWaveDownload.do",
        extra={"menuNo": "549", "maxTa": "33.0", "mddlClssCd": "SFC01", "selectType": "1"},
    ),
    "열대야일수": dict(
        url=f"{BASE}/climate/tropicalNight/selectTropicalNightDownload.do",
        extra={"menuNo": "550", "minTa": "25.0", "mddlClssCd": "SFC01", "selectType": "2"},
    ),
    "한파일수": dict(
        url=f"{BASE}/climate/cdwv/selectCdwvDownload.do",
        extra={"menuNo": "1400", "mddlClssCd": "WWRPT07", "selectType": "1"},
    ),
}

START_YEAR, END_YEAR = 2016, 2026


def fetch(spec, stn_id, session):
    data = {
        "fileType": "csv",
        # pgmNo는 필수이나 세 지표 모두 674로 통과한다 (빈 값이면 에러 페이지 반환)
        "pgmNo": "674",
        "pageIndex": "",
        "stnGroupSns": "",
        "serviceSe": "F00101",
        "stdStart": "",
        "stdEnd": "",
        "stdYearCnt": "",
        "schType": "2",
        "schStnId": stn_id,
        "firstLoading": "N",
        "startYear": str(START_YEAR),
        "endYear": str(END_YEAR),
    }
    data.update(spec["extra"])
    r = session.post(spec["url"], data=data, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.content.decode("euc-kr", errors="replace")


def parse_annual(text):
    """CSV 앞부분의 '연도 ... 연합계' 표에서 연도별 합계를 뽑는다."""
    lines = [ln.strip() for ln in text.splitlines()]
    try:
        head = next(i for i, ln in enumerate(lines) if ln.startswith("연도,") and "연합계" in ln)
    except StopIteration:
        return {}
    cols = lines[head].split(",")
    total_idx = cols.index("연합계")
    out = {}
    for ln in lines[head + 1 :]:
        # 폭염·열대야 CSV는 헤더 다음에 빈 줄이 하나 있다. 빈 줄로 끊지 말 것.
        if ln.startswith("평균"):
            break
        parts = ln.split(",")
        if not re.fullmatch(r"\d{4}", parts[0]):
            continue
        val = parts[total_idx] if total_idx < len(parts) else ""
        out[int(parts[0])] = pd.to_numeric(val, errors="coerce")
    return out


def main():
    stations = json.load(open("stations.json", encoding="utf-8"))
    session = requests.Session()
    records = []
    for n, (stn_id, meta) in enumerate(stations.items(), 1):
        row = {"stn_id": stn_id, "지점명": meta["name"], "시도": meta["sido"]}
        for name, spec in SPECS.items():
            try:
                annual = parse_annual(fetch(spec, stn_id, session))
            except Exception as exc:  # 지점별로 미제공인 항목이 있다
                print(f"  ! {meta['name']} {name}: {exc}")
                annual = {}
            for year, val in annual.items():
                records.append({**row, "연도": year, "지표": name, "일수": val})
            time.sleep(0.25)
        print(f"[{n}/{len(stations)}] {meta['sido']} {meta['name']}")

    df = pd.DataFrame(records)
    df.to_csv("kma_climate_days.csv", index=False, encoding="utf-8-sig")
    print(df.shape)
    print(df.groupby("지표")["일수"].describe())


if __name__ == "__main__":
    main()
