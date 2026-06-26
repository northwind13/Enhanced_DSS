"""
DisasterAware Web UI Dashboard (thesis Section 4 architecture).
Flask backend: Simulation Core + DSS Core exposed as a local web service.
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import os, sys, warnings, webbrowser, threading
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'core'))
from flask import Flask, jsonify, request, render_template
from engine import Session

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log=logging.getLogger('disasteraware')
app=Flask(__name__); S=Session()
log.info('DisasterAware session ready')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/layers')
def api_layers(): return jsonify(S.layers())

@app.route('/api/reset', methods=['POST'])
def api_reset():
    p=request.get_json(force=True) or {}
    clean={}
    for k in ('H','W','seed','regions','n_responders','event_trigger'):
        if k in p: clean[k]=int(p[k])
    for k in ('fuel_density','forest_frac','wind_speed','wind_dir','eta','eps','humidity','spotting'):
        if k in p: clean[k]=float(p[k])
    if 'dss_enabled' in p: clean['dss_enabled']=bool(p['dss_enabled'])
    S.reset(clean)
    return jsonify(dict(layers=S.layers(), view=S.view(), params=S.params))

@app.route('/api/clear_fire', methods=['POST'])
def api_clear(): S.clear_fire(); return jsonify(S.view())

@app.route('/api/params', methods=['POST'])
def api_params():
    p=request.get_json(force=True) or {}; clean={}
    for k in ('wind_speed','wind_dir','eta','eps','humidity','spotting'):
        if k in p: clean[k]=float(p[k])
    if 'n_responders' in p: clean['n_responders']=int(p['n_responders'])
    if 'event_trigger' in p: clean['event_trigger']=int(p['event_trigger'])
    if 'dss_enabled' in p: clean['dss_enabled']=bool(p['dss_enabled'])
    if 'dss_enabled' in p: clean['dss_enabled']=bool(p['dss_enabled'])
    S.set_params(clean); return jsonify(dict(ok=True, params=S.params))

@app.route('/api/ignite', methods=['POST'])
def api_ignite():
    p=request.get_json(force=True) or {}
    S.ignite([(int(p['y']),int(p['x']))], radius=int(p.get('radius',1)))
    return jsonify(S.view())

@app.route('/api/step', methods=['POST'])
def api_step():
    p=request.get_json(force=True) or {}
    S.step(int(p.get('n',1))); return jsonify(S.view())

@app.route('/api/goto', methods=['POST'])
def api_goto():
    p=request.get_json(force=True) or {}
    return jsonify(S.view(int(p.get('i',-1))))

@app.route('/api/state')
def api_state(): return jsonify(S.view())


@app.route('/api/inspect', methods=['POST'])
def api_inspect():
    p=request.get_json(force=True) or {}
    return jsonify(S.inspect(int(p['y']),int(p['x'])))

@app.route('/api/concept', methods=['POST'])
def api_concept():
    p=request.get_json(force=True) or {}
    return jsonify(S.concept_layer(p.get('name','threat')))

@app.route('/api/checkpoint', methods=['POST'])
def api_checkpoint(): return jsonify(dict(checkpoint=S.set_checkpoint()))

@app.route('/api/restore', methods=['POST'])
def api_restore(): return jsonify(S.restore_checkpoint())

@app.route('/api/preset', methods=['POST'])
def api_preset():
    p=request.get_json(force=True) or {}
    return jsonify(dict(view=S.preset(p.get('name','lightning')), layers=S.layers(), params=S.params))

@app.route('/api/export_csv')
def api_export_csv():
    from flask import Response
    return Response(S.export_csv(), mimetype='text/csv',
                    headers={'Content-Disposition':'attachment; filename=disasteraware_run.csv'})

@app.route('/api/save_scenario')
def api_save():
    from flask import Response
    return Response(__import__('json').dumps(S.save_scenario(),indent=2), mimetype='application/json',
                    headers={'Content-Disposition':'attachment; filename=disasteraware_scenario.json'})

@app.route('/api/load_scenario', methods=['POST'])
def api_load():
    sc=request.get_json(force=True) or {}
    return jsonify(dict(view=S.load_scenario(sc), layers=S.layers(), params=S.params))

def _open():
    try: webbrowser.open('http://127.0.0.1:5000')
    except Exception: pass

if __name__=='__main__':
    if os.environ.get('OPEN_BROWSER','1')=='1':
        threading.Timer(1.2,_open).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
