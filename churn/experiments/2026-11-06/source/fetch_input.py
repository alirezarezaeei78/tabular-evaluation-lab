"""Fetch public UCI input locally, verify its exact declared hash, never overwrite."""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL='https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip'
SHA='90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1'

def main(destination):
    path=Path(destination)
    if path.exists():raise FileExistsError('Preserve existing input; choose a new destination')
    with urllib.request.urlopen(URL,timeout=60) as response:
        raw=response.read(1_000_001)
    if len(raw)>1_000_000:raise ValueError('Provider archive exceeds declared bound')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        data=archive.read('Customer Churn.csv')
    if hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('Provider input hash changed')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(data)
    print('Verified public UCI input saved locally; provider records are excluded from public release.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination');main(parser.parse_args().destination)
