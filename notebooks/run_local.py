# VS Code 셀 실행용 (README의 Colab 셀을 로컬 경로로 옮긴 것).
# `# %%` 줄마다 셀 하나: 셀 위 "Run Cell" 또는 Shift+Enter. 커널은 .venv 선택.
# FINNHUB_API_KEY는 VS Code를 켜기 전에 `setx`로 설정한다(코드·파일에 쓰지 않음).

# %%
# 셀 1: 준비 (커널 시작 시 한 번)
import os, sys, importlib
from pathlib import Path

# Interactive 창의 작업 폴더는 이 파일 위치(notebooks/)라서 src/main.py가 있는 곳까지 올라간다.
REPO = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "main.py").exists())
os.chdir(REPO)
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "data_cache"
OUTPUT = REPO / "real_data_output"
PANEL = OUTPUT / "panel.parquet"
OUTPUT.mkdir(exist_ok=True)

print("repo:", REPO)
print("cache:", CACHE, "(있음)" if CACHE.exists() else "(없음: Drive의 stock_valuation/data_cache 복사 필요)")
print("FINNHUB_API_KEY:", "설정됨" if os.environ.get("FINNHUB_API_KEY") else "없음 (build에만 필요)")

# %%
# 셀 2: 모듈 불러오기 / 코드 수정·git pull 후 다시 실행해 새 코드 반영
import universe, config, data, features, fair_value, screening, main
for m in (universe, config, data, features, fair_value, screening, main):
    importlib.reload(m)
main.OUTPUT_DIR = OUTPUT

# %%
# 셀 3: 데이터 수집 + 패널 생성 (새 데이터가 필요할 때만)
main.build(cache_dir=CACHE, panel_path=PANEL)

# %%
# 셀 4: 후보 변수 비교
main.compare_features(PANEL)

# %%
# 셀 5: 모델 평가 + 수익률 가설 검정
main.evaluate(PANEL)

# %%
# 셀 6: 최신 리포트 (CSV 저장)
report = main.screen(PANEL)

# %%
# 셀 7: 한 종목 설명
main.screen(PANEL, ticker="AAPL", save=False);
