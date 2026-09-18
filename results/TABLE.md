| Setting | Conditioning − projection | Refinement − (BayesFP + projection) | Raw BayesFP unsafe |
|---|---:|---:|---:|
| Pendulum negative | +56.37 [+51.11, +61.54] | +0.38 [-0.99, +2.06] | 71/512 |
| Pendulum positive | +47.77 [+43.08, +52.37] | -1.20 [-2.44, +0.21] | 92/512 |
| Walker2d 0.72 | -17.57 [-29.68, -5.94] | +21.35 [+11.06, +31.82] | 74/128 |
| Walker2d 0.75 | -6.24 [-12.45, -0.41] | -4.09 [-95.29, +32.03] | 25/128 |
| HalfCheetah 0.7 | +1.19 [-30.11, +35.40] | +14.01 [-21.90, +49.63] | 49/128 |
| HalfCheetah 0.72 | -0.73 [-8.99, +9.38] | +30.61 [+5.44, +57.42] | 23/128 |

Intervals: conditioning–projection is descriptive 95% for Pendulum and planned 99.375% for MuJoCo. Refinement–BayesFP+projection is 98.75% for Pendulum and 99.375% for MuJoCo. They correct different predeclared comparison families. Native-return units differ by task. All hard-output arms had zero chunk violations and zero rejection refusals.
