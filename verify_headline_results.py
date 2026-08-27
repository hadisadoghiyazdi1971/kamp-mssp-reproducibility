from pathlib import Path
import csv, math
D=Path(__file__).resolve().parent/'results'/'paper_data'
def rows(name): return list(csv.DictReader(open(D/name,encoding='utf-8')))
# R11 headline
r11={r['solver']:r for r in rows('R11_onebit.csv')}
assert abs(float(r11['KAMP']['nmse_median'])-2.3132688042178593)<1e-12
assert abs(float(r11['ISTA']['nmse_median'])-7.343567283734143)<1e-12
# R15 boundary
r15=rows('R15_jakes_channel.csv')
assert float(r15[0]['kamp_nmse']) < float(r15[0]['static_nmse'])
assert float(r15[1]['kamp_nmse']) > float(r15[1]['static_nmse'])
assert float(r15[2]['kamp_nmse']) > float(r15[2]['static_nmse'])
print('Headline numerical checks passed.')
