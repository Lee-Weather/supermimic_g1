# 接地方案对比（同一批 clip，指标均由 mjlab 的 G1 模型测得）

| clip | 方案 | 脚底 p10 | p50 | p90 | 触地帧(≤2cm) | 根高中位 | 根高 std |
|---|---|---|---|---|---|---|---|
| walk1_subject1 | baseline | +1.8 cm | +2.6 cm | +3.3 cm | 16.1% | 0.793 m | 2.8 cm |
| walk1_subject1 | rolling(0.3s) | +0.0 cm | +0.4 cm | +1.2 cm | 99.3% | 0.772 m | 2.6 cm |
| walk3_subject3 | baseline | +4.9 cm | +5.9 cm | +10.5 cm | 0.3% | 0.811 m | 18.1 cm |
| walk3_subject3 | rolling(0.3s) | +0.0 cm | +0.3 cm | +1.2 cm | 96.6% | 0.759 m | 20.0 cm |
| walk4_subject1 | baseline | +4.6 cm | +5.2 cm | +5.7 cm | 0.3% | 0.787 m | 2.1 cm |
| walk4_subject1 | GMR scale x0.89 | +5.8 cm | +6.9 cm | +8.2 cm | 0.2% | 0.772 m | 2.1 cm |
| walk4_subject1 | GMR offset_to_ground | +5.2 cm | +5.8 cm | +6.4 cm | 0.3% | 0.793 m | 2.1 cm |
| walk4_subject1 | rolling(0.3s) | +0.0 cm | +0.2 cm | +0.7 cm | 99.4% | 0.738 m | 1.9 cm |
