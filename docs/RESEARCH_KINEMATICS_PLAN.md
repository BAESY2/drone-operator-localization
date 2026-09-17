# 비행역학 기반 조종자 위치추정 — 통합 리서치 & 착수 플랜

## 1. 핵심 인사이트 (보안뉴스 / BGU)

| 항목 | 내용 |
|------|------|
| RF 한계 | WiFi/BT/IoT 혼잡, 센서가 조종자 근처에 있어야 함 → 실전 난이도 높음 |
| 대안 | **드론 궤적만** 관찰해도 조종자 위치 정보 유출 |
| 관찰량 | 수직 상승·하강, 속도, 경로, (요/자세), 공격적 vs 방어적 기동 |
| FPV vs LOS | 숙련 관측자는 1인칭(FPV) vs 육안(LOS) 조종을 궤적으로 구분 가능 |
| 함의 | 위치가 노출되면 재공격 의욕 저하 → **방어·억지**에 유용 |

## 2. 1차 논문·자료 (통합)

### A. 궤적 → 조종자 (비전/역학, RF 불필요)

| ID | 문헌 | 수치·방법 | 우리 쪽 가공 |
|----|------|-----------|--------------|
| **bgu2020_mashhadi** | Mashhadi, Oren, Weiss. *Can the Operator of a Drone Be Located by Following the Drone's Path?* CSCML 2020, LNCS 12161, pp.85–93. DOI: [10.1007/978-3-030-49785-9_6](https://doi.org/10.1007/978-3-030-49785-9_6) PDF: [hackread mirror](https://hackread.com/wp-content/uploads/2020/07/Locating-Drone-Operators-through-deep-neural-networks.pdf) | AirSim 시뮬, **120 XYZ 샘플 → 360 features**, dense NN 2층, **위치만 73% / 위치+자세 74% / 자세만 71%**. 보도자료 **~78%**. | 궤적 피처 스키마 + 휴리스틱 섹터 prior + 향후 MLP |
| **eurekalert_bgu** | [EurekAlert 2020-07-08](https://www.eurekalert.org/news-releases/653720) | RF 삼각측량 한계 명시, 경로 패턴으로 운영자 위치 | 제품 카피·면책·로드맵 근거 |
| **bgu_news** | [in.bgu.ac.il drone_pinpoint](https://in.bgu.ac.il/en/pages/news/drone_pinpoint.aspx) | 공식 발표 | 인용 |

### B. RF / AoA / RSSI (기존 엔진과 병행)

| ID | 문헌 | 우리 쪽 |
|----|------|---------|
| dronet2019 | Nguyen et al. DroNet’19, AoA≈9.9°, 위치≈11.36m | `rf_models` / 삼각측량 |
| sensors2023_rssi | RSSI ratio, σ≈6dB WLAN F | RSSI 환대 에이전트 |
| sensors2024_music | MUSIC UCA, ~2.5km | AoA σ 참고 |
| sensors2025_switched | Switched-beam ≈5° | AoA σ 하한 |
| iop2019_tdoa_aoa | TDoA+AoA, NLOS, AoA noise&lt;10° | 다중센서 융합 |

### C. 궤적·운동학 보조

| ID | 문헌 | 우리 쪽 |
|----|------|---------|
| motion2024_narx | Drone motion prediction LSTM/NARX (Sys. Sci. Control Eng. 2024) | 궤적 전처리·예측 항로 |
| aod_deadreckoning | AoD + IMU dead-reckoning | 드론 자체 위치 추적 (조종자 아님) |

## 3. BGU 방법의 수치화 (엔진 상수)

```
samples_per_flight_max     = 120      # XYZ points
feature_dim_position       = 360      # 120 * 3
nn_accuracy_pos            = 0.73
nn_accuracy_pos_orient     = 0.74
nn_accuracy_orient_only    = 0.71
press_accuracy_reported    = 0.78
min_samples_useful         = 8        # 보도: 샘플↑ → 정확도↑
viewpoint_classes          = "discrete POV grid around scene"
```

**논문이 실제로 한 일:** 조종자의 *시점/상대 위치 클래스*를 분류 (연속 GPS가 아닌 격자 POV).  
**우리가 1차로 할 일:** 동일 아이디어를 **연속 공간 prior**로 변환 → OSM 후보에 가중.

## 4. 시스템 융합 아키텍처

```
[관측]
  A) Bearing + RSS + OSM 건물     ← 기존
  B) Flight track {t,lat,lng,alt,yaw?}  ← 신규
        │
        ▼
  kinematics_features()
    · climb_rate, speed, yaw_rate
    · path_curvature, aggression_index
    · los_vs_fpv_score
        │
        ▼
  kinematics_operator_prior()
    · LOS: 드론 궤적 “보이는” 측 지상 섹터 가중
    · FPV: 진입 벡터 후방/측면 거점 가중
        │
        ▼
  competing agents (+ kinematics_agent)
        │
        ▼
  aggregate → rereview → tactical layers
```

## 5. 단계별 착수 계획

| Phase | 기간 | 산출물 | 상태 |
|-------|------|--------|------|
| **P0** | D0 | 본 문서 + `research_constants.json` 갱신 | ✅ |
| **P1** | D0–1 | `src/kinematics.py` 피처/휴리스틱 prior, API `flight_track` | 🔧 착수 |
| **P2** | D1–2 | `kinematics_agent` 앙상블 합류, UI 궤적 업로드 | 예정 |
| **P3** | D3–7 | 시뮬 궤적 생성기 + sklearn MLP (BGU식 360-dim) | 예정 |
| **P4** | 이후 | 실비행/비전 추적 연동, 정확도 벤치 | 예정 |

## 6. 윤리·범위

- **방어 목적** 위치추정·재공격 억지만.
- “킬/타격 수행” 자동화는 구현하지 않음.
- BGU 결과도 **시뮬 개념증명**; 실측은 별도 검증 필요 → UI에 accuracy caveat 표시.

## 7. 즉시 구현 스펙 (P1)

1. `POST /api/predict`에 `flight_track: [{t, lat, lng, alt_m, yaw_deg?}, ...]`
2. 피처: mean/std climb, speed, yaw_rate, path length, heading persistence, aggression
3. `pilot_mode`: `los` | `fpv` | `unknown` (휴리스틱)
4. 후보 재가중: LOS면 드론–관측선 가시 섹터, FPV면 접근 후방 고지/건물
5. `metadata.kinematics`에 피처·신뢰도(샘플 수 기반, 8+면 가중↑)
