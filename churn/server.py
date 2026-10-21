"""Loopback-only local runner. User CSV contents are processed in memory, never logged."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from churn_lab import FEATURES, LIMITS, export_csv, score_csv

HERE=Path(__file__).resolve().parent


def make_handler(bundle, report_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, body, content_type='application/json'):
            if not isinstance(body, bytes):
                body=json.dumps(body,allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',content_type+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def local_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.local_host():return self.send(403,{'error':'Local host required'})
            if self.path=='/':return self.send(200,(HERE/'app.html').read_bytes(),'text/html')
            if self.path=='/api/schema':return self.send(200,{'features':FEATURES,'limits':LIMITS})
            if self.path=='/api/demo':return self.send(200,{'csv':(HERE/'examples/demo.csv').read_text(encoding='utf-8'), 'scope':'Authored synthetic workflow fixture; not evaluation data'})
            if self.path=='/api/report':return self.send(200,json.loads(report_path.read_text(encoding='utf-8')))
            return self.send(404,{'error':'Not found'})

        def do_POST(self):
            if not self.local_host():return self.send(403,{'error':'Local host required'})
            if self.headers.get('Origin') not in [None,'http://'+self.headers.get('Host','')]:
                return self.send(403,{'error':'Same-origin requests only'})
            if self.path!='/api/score':return self.send(404,{'error':'Not found'})
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                return self.send(415,{'error':'JSON request required'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=1_100_000:raise ValueError('Request limit is 1.1 MB')
                payload=json.loads(self.rfile.read(n))
                if not isinstance(payload,dict):raise ValueError('JSON object required')
                with threadpool_limits(limits=1):
                    report=score_csv(payload.get('csv'),bundle,payload.get('model','boosting_sigmoid'),float(payload.get('capacity',.1)))
                return self.send(200,{'report':report,'csv':export_csv(report)})
            except (ValueError,TypeError,UnicodeDecodeError,KeyError) as exc:
                return self.send(400,{'error':str(exc)})
    return Handler


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8766)
    p.add_argument('--report',type=Path,default=HERE/'experiments/2026-10-21/results.json');args=p.parse_args()
    model=HERE/'.local/model.joblib'
    if not model.exists():p.error('Run churn/run_benchmark.py first to train the local model. Never load untrusted joblib files.')
    httpd=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(joblib.load(model),args.report))
    print(f'Churn Decision Lab: http://127.0.0.1:{args.port} (local-only; Ctrl+C stops)',flush=True)
    httpd.serve_forever()
