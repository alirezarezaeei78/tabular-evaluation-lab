"""Shared authored RFM cleaning and independent arithmetic from the5November study.
No provider rows or customer identifiers are embedded.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import math
import numpy as np
import pandas as pd
FEATURES = ['R', 'F', 'M']


def clean_transactions(frame):
    required = {'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice'}
    if not required <= set(frame.columns):
        raise ValueError('Missing required transaction columns')
    customer = pd.to_numeric(frame.CustomerID, errors='coerce')
    quantity = pd.to_numeric(frame.Quantity, errors='coerce')
    price = pd.to_numeric(frame.UnitPrice, errors='coerce')
    dates = pd.to_datetime(frame.InvoiceDate, errors='coerce')
    if dates.dt.tz is not None:
        raise ValueError('Use the declared naive source timestamp convention')
    invoice = frame.InvoiceNo.astype(str).str.strip()
    checks = [
        ('invalid_customer_id', np.isfinite(customer) & (customer > 0) & (customer % 1 == 0)),
        ('invalid_timestamp_or_invoice', dates.notna() & frame.InvoiceNo.notna() & invoice.ne('')),
        ('cancelled_invoice', ~invoice.str.upper().str.startswith('C')),
        ('invalid_or_nonpositive_quantity', np.isfinite(quantity) & (quantity > 0)),
        ('invalid_or_nonpositive_price', np.isfinite(price) & (price > 0))]
    keep = pd.Series(True, index=frame.index)
    counts = {}
    for name, valid in checks:
        counts[name] = int((keep & ~valid).sum())
        keep &= valid
    cleaned = pd.DataFrame({'CustomerID': customer[keep].astype(np.int64).astype(str),
        'InvoiceNo': invoice[keep], 'InvoiceDate': dates[keep],
        'Quantity': quantity[keep], 'UnitPrice': price[keep]})
    if not np.isfinite((cleaned.Quantity * cleaned.UnitPrice).to_numpy()).all():
        raise ValueError('Purchase amounts overflow')
    counts.update(input_rows=len(frame), retained_rows=len(cleaned),
        exact_duplicate_input_rows_retained_by_policy=int(frame.duplicated().sum()))
    assert sum(counts[name] for name, _ in checks) + len(cleaned) == len(frame)
    return cleaned, counts

def rfm(frame, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end:
        raise ValueError('Require start before end')
    rows = frame.loc[(frame.InvoiceDate >= start) & (frame.InvoiceDate < end)].copy()
    if rows.empty:
        raise ValueError('No retained purchases in the requested window')
    rows['amount'] = rows.Quantity * rows.UnitPrice
    grouped = rows.groupby('CustomerID', sort=True).agg(last=('InvoiceDate', 'max'),
        F=('InvoiceNo', 'nunique'), M=('amount', 'sum'))
    grouped['R'] = (end - grouped['last']).dt.total_seconds() / 86400
    result = grouped[FEATURES].astype(float)
    assert result.index.is_unique and np.isfinite(result.to_numpy()).all()
    assert (result.R > 0).all() and (result.F >= 1).all() and (result.M > 0).all()
    return result, rows

def scalar_rfm_check(features, rows, end):
    """Independent sets/latest timestamp/fsum, without pandas grouping."""
    end = pd.Timestamp(end)
    values = defaultdict(lambda: {'invoices': set(), 'latest': None, 'amounts': []})
    for row in rows.itertuples(index=False):
        value = values[row.CustomerID]
        value['invoices'].add(row.InvoiceNo)
        value['latest'] = row.InvoiceDate if value['latest'] is None else max(value['latest'], row.InvoiceDate)
        value['amounts'].append(float(row.Quantity) * float(row.UnitPrice))
    assert set(values) == set(features.index)
    max_error = 0.0
    for customer, value in values.items():
        expected = np.array([(end-value['latest']).total_seconds()/86400,
            len(value['invoices']), math.fsum(value['amounts'])])
        actual = features.loc[customer].to_numpy()
        max_error = max(max_error, float(np.max(np.abs(expected-actual))))
        assert np.allclose(actual, expected, rtol=1e-12, atol=2e-8)
    return max_error

def exact_ari(left, right):
    """Independent exact-rational contingency oracle; label values are arbitrary."""
    if len(left) != len(right):
        raise ValueError('Partitions must have equal length')
    if len(left) < 2:
        return 1.0
    joint, a, b = Counter(zip(left, right)), Counter(left), Counter(right)
    choose = lambda values: sum(n * (n-1) // 2 for n in values)
    index, rows, columns = choose(joint.values()), choose(a.values()), choose(b.values())
    expected = Fraction(rows * columns, len(left) * (len(left)-1) // 2)
    denominator = Fraction(rows + columns, 2) - expected
    return 1.0 if denominator == 0 else float((index-expected)/denominator)

def direct_predict(x, centers):
    return ((x[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

def profiles(raw, labels):
    result = []
    for label in sorted(set(map(int, labels))):
        selected = raw.loc[np.asarray(labels) == label]
        result.append({'label': label, 'customers': len(selected),
            'median_rfm': selected.median().to_dict() if len(selected) >= 10 else None,
            'small_profile_suppressed': len(selected) < 10})
    return result
