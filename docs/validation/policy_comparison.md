# Policy options: measured trade-offs

Frozen thresholds applied to held-out exports; nothing is tuned here. Rates are class-conditional. 95% grouped bootstrap intervals in brackets.

## cal

| Metric | A_alpha0.05_margin0.05 | B_alpha0.02_margin0.05 | B2_alpha0.02_margin0.01 |
|---|---|---|---|
| n (positives / negatives) | 1310 (851 / 459) | 1310 (851 / 459) | 1310 (851 / 459) |
| Real fire/smoke labelled NO_VISIBLE_FIRE (miss) | 0.0141 [0.007, 0.0222] | 0.0 [0.0, 0.0] | 0.0047 [0.0011, 0.0093] |
| Flames head FNR | 0.036 [0.0211, 0.054] | 0.0106 [0.0031, 0.0208] | 0.0106 [0.0031, 0.0208] |
| Smoke head FNR | 0.0393 [0.0257, 0.0522] | 0.0127 [0.0057, 0.0196] | 0.0127 [0.0057, 0.0196] |
| Negatives labelled ACTIVE_FIRE/SMOKE_ONLY (false alarm) | 0.024 [0.0102, 0.0401] | 0.0523 [0.0361, 0.0722] | 0.146 [0.1158, 0.1841] |
| Negatives at HIGH priority | 0.0479 [0.0284, 0.0692] | 1.0 [1.0, 1.0] | 0.5969 [0.5535, 0.6455] |
| UNCERTAIN share (all) | 0.042 [0.0321, 0.0538] | 0.5748 [0.5515, 0.6031] | 0.2664 [0.2439, 0.2932] |
| HIGH-priority share (all) | 0.6573 [0.6282, 0.679] | 1.0 [1.0, 1.0] | 0.8557 [0.8362, 0.8763] |
| Flames flag precision | 0.8922 | 0.5845 | 0.5845 |
| Smoke flag precision | 0.9607 | 0.9274 | 0.9274 |
| Per 1000 reports at 5% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 94.8 / 49.3 / 0.7 / 25.4 | 1000.0 / 50.0 / 0.0 / 919.0 | 616.8 / 49.8 / 0.2 / 436.8 |
| Per 1000 reports at 20% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 235.5 / 197.2 / 2.8 / 29.5 | 1000.0 / 200.0 / 0.0 / 832.9 | 676.6 / 199.1 / 0.9 / 394.2 |
| Per 1000 reports at 50% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 516.9 / 492.9 / 7.0 / 37.9 | 1000.0 / 500.0 / 0.0 / 660.7 | 796.1 / 497.6 / 2.4 / 308.9 |

## val

| Metric | A_alpha0.05_margin0.05 | B_alpha0.02_margin0.05 | B2_alpha0.02_margin0.01 |
|---|---|---|---|
| n (positives / negatives) | 1310 (893 / 417) | 1310 (893 / 417) | 1310 (893 / 417) |
| Real fire/smoke labelled NO_VISIBLE_FIRE (miss) | 0.0291 [0.0178, 0.0407] | 0.0 [0.0, 0.0] | 0.0112 [0.0046, 0.018] |
| Flames head FNR | 0.022 [0.0107, 0.0376] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Smoke head FNR | 0.0462 [0.0325, 0.0596] | 0.0255 [0.0157, 0.0354] | 0.0255 [0.0157, 0.0354] |
| Negatives labelled ACTIVE_FIRE/SMOKE_ONLY (false alarm) | 0.012 [0.0024, 0.0231] | 0.0288 [0.0132, 0.046] | 0.1295 [0.0962, 0.1599] |
| Negatives at HIGH priority | 0.0168 [0.0051, 0.0288] | 1.0 [1.0, 1.0] | 0.4796 [0.44, 0.5238] |
| UNCERTAIN share (all) | 0.0237 [0.0164, 0.0317] | 0.5802 [0.5568, 0.605] | 0.2458 [0.2248, 0.2691] |
| HIGH-priority share (all) | 0.6672 [0.6378, 0.692] | 1.0 [1.0, 1.0] | 0.8267 [0.8061, 0.8458] |
| Flames flag precision | 0.89 | 0.5709 | 0.5709 |
| Smoke flag precision | 0.9561 | 0.9271 | 0.9271 |
| Per 1000 reports at 5% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 64.5 / 48.5 / 1.5 / 6.2 | 1000.0 / 50.0 / 0.0 / 942.5 | 505.1 / 49.4 / 0.6 / 342.5 |
| Per 1000 reports at 20% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 207.6 / 194.2 / 5.8 / 10.3 | 1000.0 / 200.0 / 0.0 / 856.5 | 581.4 / 197.8 / 2.2 / 319.5 |
| Per 1000 reports at 50% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 493.9 / 485.4 / 14.6 / 18.6 | 1000.0 / 500.0 / 0.0 / 684.4 | 734.2 / 494.4 / 5.6 / 273.6 |

## test

| Metric | A_alpha0.05_margin0.05 | B_alpha0.02_margin0.05 | B2_alpha0.02_margin0.01 |
|---|---|---|---|
| n (positives / negatives) | 1310 (829 / 481) | 1310 (829 / 481) | 1310 (829 / 481) |
| Real fire/smoke labelled NO_VISIBLE_FIRE (miss) | 0.0121 [0.0054, 0.02] | 0.0 [0.0, 0.0] | 0.0012 [0.0, 0.0047] |
| Flames head FNR | 0.0245 [0.0119, 0.039] | 0.0041 [0.0, 0.01] | 0.0041 [0.0, 0.01] |
| Smoke head FNR | 0.0597 [0.0442, 0.0745] | 0.0233 [0.0137, 0.0339] | 0.0233 [0.0137, 0.0339] |
| Negatives labelled ACTIVE_FIRE/SMOKE_ONLY (false alarm) | 0.0166 [0.0064, 0.0286] | 0.0437 [0.0279, 0.0632] | 0.1164 [0.087, 0.1438] |
| Negatives at HIGH priority | 0.0291 [0.0154, 0.0452] | 1.0 [1.0, 1.0] | 0.4595 [0.4212, 0.5068] |
| UNCERTAIN share (all) | 0.0313 [0.0214, 0.0412] | 0.5519 [0.5275, 0.5813] | 0.2198 [0.2008, 0.2443] |
| HIGH-priority share (all) | 0.6359 [0.6057, 0.6584] | 1.0 [1.0, 1.0] | 0.8008 [0.779, 0.8218] |
| Flames flag precision | 0.8966 | 0.5946 | 0.5946 |
| Smoke flag precision | 0.959 | 0.9377 | 0.9377 |
| Per 1000 reports at 5% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 77.0 / 49.4 / 0.6 / 14.0 | 1000.0 / 50.0 / 0.0 / 924.3 | 486.5 / 49.9 / 0.1 / 333.3 |
| Per 1000 reports at 20% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 220.9 / 197.6 / 2.4 / 18.4 | 1000.0 / 200.0 / 0.0 / 828.5 | 567.4 / 199.8 / 0.2 / 304.1 |
| Per 1000 reports at 50% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 508.5 / 493.9 / 6.0 / 27.3 | 1000.0 / 500.0 / 0.0 / 636.8 | 729.1 / 499.4 / 0.6 / 245.7 |

## xsrc_aof

| Metric | A_alpha0.05_margin0.05 | B_alpha0.02_margin0.05 | B2_alpha0.02_margin0.01 |
|---|---|---|---|
| n (positives / negatives) | 8428 (3385 / 5043) | 8428 (3385 / 5043) | 8428 (3385 / 5043) |
| Real fire/smoke labelled NO_VISIBLE_FIRE (miss) | 0.3734 [0.3245, 0.4266] | 0.0 [0.0, 0.0] | 0.1938 [0.1557, 0.2351] |
| Flames head FNR | 0.2946 [0.2092, 0.3982] | 0.1379 [0.0784, 0.2099] | 0.1379 [0.0784, 0.2099] |
| Smoke head FNR | 0.5154 [0.4693, 0.562] | 0.3456 [0.3017, 0.3897] | 0.3456 [0.3017, 0.3897] |
| Negatives labelled ACTIVE_FIRE/SMOKE_ONLY (false alarm) | 0.2082 [0.1666, 0.2403] | 0.052 [0.0391, 0.0683] | 0.3555 [0.308, 0.401] |
| Negatives at HIGH priority | 0.2471 [0.2029, 0.2847] | 1.0 [1.0, 1.0] | 0.5768 [0.5383, 0.6184] |
| UNCERTAIN share (all) | 0.0464 [0.04, 0.0535] | 0.8612 [0.8343, 0.8886] | 0.1973 [0.1778, 0.2188] |
| HIGH-priority share (all) | 0.3995 [0.3616, 0.4394] | 1.0 [1.0, 1.0] | 0.669 [0.6372, 0.699] |
| Flames flag precision | 0.8417 | 0.5759 | 0.5759 |
| Smoke flag precision | 0.5597 | 0.5108 | 0.5108 |
| Per 1000 reports at 5% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 266.1 / 31.3 / 18.7 / 39.8 | 1000.0 / 50.0 / 0.0 / 937.2 | 588.3 / 40.3 / 9.7 / 218.3 |
| Per 1000 reports at 20% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 323.0 / 125.3 / 74.7 / 42.6 | 1000.0 / 200.0 / 0.0 / 904.8 | 622.7 / 161.2 / 38.8 / 209.4 |
| Per 1000 reports at 50% real: HIGH items / real among them / real fire labelled NO_VISIBLE_FIRE / UNCERTAIN | 436.9 / 313.3 / 186.7 / 48.2 | 1000.0 / 500.0 / 0.0 / 839.9 | 691.5 / 403.1 / 96.9 / 191.5 |
