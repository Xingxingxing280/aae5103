"""Reproduce E's analysis directly from the existing A/B handoff ZIPs.

Python 3 standard library only. No training, network access, or model loading.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--a-zip', type=Path, default=ROOT.parent/'A_handoff.zip')
parser.add_argument('--b-zip', type=Path, default=ROOT.parent/'b.zip')
parser.add_argument('--output-dir', type=Path, default=ROOT/'02_results')
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)

def rows(archive, name):
    with archive.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')))

def save(name, data):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

bands, scenarios, checks = [], [], {}
with ZipFile(args.a_zip) as az, ZipFile(args.b_zip) as bz:
    a_path = 'data/processed/'
    b_path = 'AAEwed/B_models/'
    policy = json.loads(bz.read(b_path+'models/risk_policy.json'))
    a_clean = az.read(a_path+'cleaned_flights.csv')
    b_clean = bz.read('AAEwed/A_handoff/'+a_path+'cleaned_flights.csv')
    assert hashlib.sha256(a_clean).digest() == hashlib.sha256(b_clean).digest()
    clean = list(csv.DictReader(io.StringIO(a_clean.decode('utf-8-sig'), newline='')))
    byid = {r['record_id']: r for r in clean}
    assert len(byid) == len(clean)
    del a_clean, b_clean
    for split in ['validation', 'test']:
        data = rows(bz, b_path+f'data/predictions/{split}.csv')
        assert len({r['record_id'] for r in data}) == len(data)
        assert {r['record_id'] for r in data} == {
            r['record_id'] for r in rows(az, a_path+f'metadata_{split}.csv')}
        groups = defaultdict(list)
        for r in data:
            a = byid[r['record_id']]
            for field in ['FlightDate', 'Origin', 'Airline', 'Dest', 'CRSDepTime', 'TargetDelayed15']:
                assert r[field] == a[field], (field, r['record_id'])
            probability = float(r['delay_probability'])
            assert 0 <= probability <= 1
            risk = ('High' if probability >= policy['high_cut'] else
                    'Medium' if probability >= policy['medium_cut'] else 'Low')
            assert risk == r['risk_level']
            r['p'], r['y'] = probability, int(r['TargetDelayed15'])
            groups[(r['Origin'], r['FlightDate'], a['ScheduledHour'])].append(r)
        total_delayed = sum(r['y'] for r in data)
        checks[split] = dict(n=len(data), delayed=total_delayed,
                             airport_date_hour_groups=len(groups), ids_labels_risk_verified=True)
        for risk in ['Low', 'Medium', 'High']:
            group = [r for r in data if r['risk_level'] == risk]
            delayed = sum(r['y'] for r in group)
            bands.append(dict(split=split, risk=risk, n=len(group), delayed=delayed,
                              observed_rate=delayed/len(group),
                              mean_probability=sum(r['p'] for r in group)/len(group)))
        for capacity in [0.05, 0.1, 0.2]:
            for method in ['risk_priority', 'schedule_order', 'random_expected']:
                slots, selected_delayed = 0, 0
                for group in groups.values():
                    k = math.floor(capacity*len(group))
                    slots += k
                    if method == 'random_expected':
                        selected_delayed += k*sum(r['y'] for r in group)/len(group)
                    else:
                        key = ((lambda r: (-r['p'], int(r['CRSDepTime']), r['record_id']))
                               if method == 'risk_priority' else
                               (lambda r: (int(r['CRSDepTime']), r['record_id'])))
                        ranked = sorted(group, key=key)
                        selected_delayed += sum(r['y'] for r in ranked[:k])
                scenarios.append(dict(split=split, capacity=capacity, method=method,
                                      flights=len(data), slots=slots,
                                      selected_delayed=selected_delayed,
                                      selected_nondelayed=slots-selected_delayed,
                                      precision_at_capacity=selected_delayed/slots,
                                      recall_at_capacity=selected_delayed/total_delayed,
                                      actual_capacity=slots/len(data)))

save('risk_bands.csv', bands)
save('resource_scenarios.csv', scenarios)
(OUT/'verification.json').write_text(json.dumps(checks, indent=2))
(OUT/'analysis_data.json').write_text(json.dumps(dict(checks=checks, bands=bands, scenarios=scenarios), indent=2))
print(json.dumps(checks, indent=2))
print(f'Results saved to {OUT}')
