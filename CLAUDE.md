# CLAUDE.md

미국 상장 종목 280여 개의 **적정 밸류에이션 스크리닝** 도구다(개인 프로젝트 "Stock_Prediction"). 재무 지표로 시장이 보통 쳐주는 PER·PBR(적정 배수)을 학습하고, 실제 배수와의 괴리로 저평가/고평가를 **서술**한다. 구조, 결과, 이력은 `README.md`에 있다.

## 현재 방향 (먼저 읽을 것)

- 수익률 예측은 2026-09-18에 포기했다. FDR 보정 후 0/60 유의였고, 모델이 섹터 평균 기준선을 넘지 못했다.
- 2026-09-23부터는 적정가 모델(`src/fair_value.py`)이 라벨을 정한다. 섹터 대비 단순 순위(`sector_valuation_rank`)는 참고 컬럼으로만 남아 있다.
- 적정가 괴리와 이후 수익률의 관계도 검정했지만 유의하지 않았다(FDR 후 0/24). 괴리를 **수익률 예측처럼 표현하지 말 것.**

## 핵심 규칙

1. **Train/Val/Test는 절대 랜덤 split 금지.** 항상 `src/config.py`의 고정 날짜(`TRAIN_END`, `VAL_END`)로 나눈다.
   - 적정가 모델은 한 시점의 종목 단면 안에서만 학습한다. 종목 단위 out-of-fold이고, 미래 행은 쓰지 않는다.
   - 모델 설정(변수, 범위, 기준)은 Train/Val 결과만 보고 정한다. Test는 최종 보고용이다.
2. **look-ahead bias 방지가 최우선 원칙이다.** "시점 T에 무엇을 알 수 있었나"는 `src/features.py`에서만 결정한다(모듈 docstring 참고).
   - 재무 데이터는 분기 말 + 45일(`REPORTING_LAG_DAYS`)부터 알 수 있다고 본다.
   - 가격은 as_of 당일까지만 쓴다.
   - `fwd_return_*`는 검정용 정답이지 절대 입력 변수가 아니다.
   - 새 변수를 추가하면 `tests/test_pipeline.py::test_no_look_ahead`가 계속 통과해야 한다. 이 테스트는 미래 데이터를 뒤섞어도 과거 행이 바뀌지 않는지 확인한다.
   - 가격에서 나온 지표(모멘텀, 52주 고점 대비)는 적정가 모델에 넣지 않는다. 주가가 빠진 것을 모델이 "원래 싸야 할 종목"으로 설명해 버린다.
3. **통계 검증과 "쓸모있는 도구인가"는 별개의 질문이다.**
   - 서술할 때 "예측 정확도가 좋다"는 표현은 쓰지 않는다.
   - 적정가 모델의 R²는 "현재 배수를 재무 지표로 얼마나 설명하나"이지 미래 예측력이 아니다. 비교 기준은 항상 섹터 중앙값 기준선이다.
   - 수익률 관련 검정에는 겹치는 기간 보정(Newey-West)과 FDR 보정을 적용한다.
4. **데이터 소스**
   - Finnhub 무료 API(**분당 50콜 제한**): 분기 재무. 공시일은 제공하지 않고, PER/PBR은 분기 말 주가 기준이라 as_of 주가로 보정한다(`RESCALE_MULTIPLES_TO_AS_OF_PRICE`).
   - yfinance: 수정주가.
   - 수집 데이터는 `data_cache/`에, 결과는 `real_data_output/`에 저장된다. 둘 다 `.gitignore`로 제외되어 있다.
   - API 키는 환경변수 `FINNHUB_API_KEY`로 넘긴다. 코드에 하드코딩하거나 커밋하지 않는다.
   - 데이터 품질 검사에서 "no price history"가 나오면 먼저 상장폐지나 티커 변경인지 확인한다(`universe.py` docstring).

## 실행

```bash
python tests/test_pipeline.py        # 코드 수정 후 항상 실행
python src/main.py build             # FINNHUB_API_KEY 필요
python src/main.py evaluate
python src/main.py screen [--ticker AAPL]
```

- Windows 콘솔에서는 한글 출력 때문에 `PYTHONIOENCODING=utf-8` 설정이 필요할 수 있다.
- `src/` 내부 import는 `from config import ...`처럼 평평한 형태다. Colab에서 모든 `.py`를 한 폴더에 올려 쓰기 때문이다.
- `evaluate`와 `screen`은 저장된 패널로 적정가 모델을 매번 다시 계산한다. 패널에는 원시 지표만 저장된다.
