#!/usr/bin/env python3
"""
inference_cascade_lstm_v90-4-5-tau14.py
二段式推理：v90-2-11 作物分类 → v90-4-5-tau14（Delta T τ=14）作物候预测
★ vs v90-4-5：唯一改动 encode_delta_t τ=10→14
"""
import os, re, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

BANDS = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
CROP_NAMES = ['corn','soybean','rice']
PHASE_NAMES = ['Greenup','MidGreenup','Maturity','Peak','Senescence','MidSenescence','Dormancy']
NUM_CROPS, NUM_PHASES, NUM_FOLDS = 3, 7, 5
INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 36, 192, 2, 0.0
GLOBAL_LON_MIN, GLOBAL_LON_MAX = 124.10, 134.64

# ★ v90-4-5-tau14: Delta T 尺度 τ=14
DELTA_TAU = 14

INPUT_DIR = Path(os.environ.get("DATA_DIR", "/rs"))
REGION_TEST_DIR = INPUT_DIR
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
OUTPUT_JSON = OUTPUT_DIR / "result.json"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))

MODEL_PATHS_V11 = [MODEL_DIR / f"best_multitask_lstm_fold-v90-2-11-{i}.pth" for i in range(1,6)]
MODEL_PATHS_V45T14 = [MODEL_DIR / f"best_pheno_lstm_fold-v90-4-5-tau14-{i}.pth" for i in range(1,6)]

# /rs 下文件名如 96_2018-10-04.tif， region 编号可能带或不带 region 前缀
FILE_PATTERN = re.compile(
    r"^(?:region)?(\d+)_(\d{4}-\d{2}-\d{2})\.(tiff|tif)$", re.IGNORECASE)
FALLBACK_CROP = 'rice'

def make_key(lon_raw, lat_raw, obs_date_raw): return f"{lon_raw}_{lat_raw}_{obs_date_raw}"
def parse_obs_date(s):
    m = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', str(s).strip())
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d).timetuple().tm_yday
    raise ValueError(f"无法解析日期: {s}")
def date_to_doy(s): return datetime.strptime(s, "%Y-%m-%d").timetuple().tm_yday

def calc_indices(bv):
    b = np.array(bv, dtype=np.float32); b02,b03,b04,b05,b07 = b[1],b[2],b[3],b[4],b[6]; b08,b8a,b11 = b[7],b[8],b[10]
    eps=1e-6; ndvi=(b08-b04)/(b08+b04+eps); evi=2.5*(b08-b04)/(b08+6*b04-7.5*b02+1+eps)
    lswi=(b8a-b11)/(b8a+b11+eps); gcvi=np.clip(b08/(b04+eps)-1,-1,10); ndre1=(b8a-b05)/(b8a+b05+eps)
    mndwi=(b03-b8a)/(b03+b8a+eps); gndvi=(b08-b03)/(b08+b03+eps); mtci=np.clip((b08-b05)/(b05-b04+eps),-10,50)
    ireci=np.clip((b08-b04)*b07/(b05+eps),-10,50); ci_rededge=np.clip(b08/(b05+eps)-1,-1,20)
    nirv=ndvi*b08; dvi=b08-b04; wdrvi=(0.2*b08-b04)/(0.2*b08+b04+eps); ndre2=(b07-b05)/(b07+b05+eps)
    ndmi=(b8a-b11)/(b8a+b11+eps)
    return [ndvi,evi,lswi,gcvi,ndre1,mndwi,gndvi,mtci,ireci,ci_rededge,nirv,dvi,wdrvi,ndre2,ndmi]

def encode_doy(doy): r=2*np.pi*doy/365; return [np.sin(r),np.cos(r),doy/365]
def encode_location_crop(lon,lat):
    lr,nr=np.radians(lon),np.radians(lat)
    return [np.sin(lr),np.cos(lr),np.sin(nr),np.cos(nr),0.0]
def encode_location_pheno_global(lon, lat):
    lat_rad=np.radians(lat); lat_norm=(lat-47.5)/7.5
    lon_norm=(lon-GLOBAL_LON_MIN)/(GLOBAL_LON_MAX-GLOBAL_LON_MIN+1e-6)
    return [np.sin(lat_rad),np.cos(lat_rad),lat_norm,lon_norm,0.0]
# ★ τ=14
def encode_delta_t(doy, target_doy):
    return np.exp(-abs(doy - target_doy) / DELTA_TAU)

class MultiTaskLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_lstm = nn.LSTM(35,HIDDEN_DIM,NUM_LAYERS,batch_first=True,
                                   dropout=DROPOUT if NUM_LAYERS>1 else 0,bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(HIDDEN_DIM*2,HIDDEN_DIM),nn.Tanh(),nn.Linear(HIDDEN_DIM,1))
        self.crop_classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM*2),nn.Linear(HIDDEN_DIM*2,HIDDEN_DIM),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(HIDDEN_DIM,3))
        self.pheno_classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM*2),nn.Linear(HIDDEN_DIM*2,HIDDEN_DIM),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(HIDDEN_DIM,7))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self,seq,mask):
        packed = pack_padded_sequence(seq,mask.sum(dim=1).cpu(),batch_first=True,enforce_sorted=False)
        out,_ = pad_packed_sequence(self.shared_lstm(packed)[0],batch_first=True)
        aw = F.softmax(self.attention(out).squeeze(-1).masked_fill(~mask,float('-inf')),dim=1)
        pooled = (out*aw.unsqueeze(-1)).sum(dim=1)
        return self.crop_classifier(self.dropout(pooled)), self.pheno_classifier(self.dropout(out))

class PhenoLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM,HIDDEN_DIM,NUM_LAYERS,batch_first=True,
                            dropout=DROPOUT if NUM_LAYERS>1 else 0,bidirectional=True)
        self.ph = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM*2),nn.Linear(HIDDEN_DIM*2,HIDDEN_DIM),
            nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(HIDDEN_DIM,7))
        self.dp = nn.Dropout(DROPOUT)
    def forward(self,seq,mask):
        packed = pack_padded_sequence(seq,mask.sum(dim=1).cpu(),batch_first=True,enforce_sorted=False)
        out,_ = pad_packed_sequence(self.lstm(packed)[0],batch_first=True)
        return self.ph(self.dp(out))

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

COMPETITION_PATTERN = re.compile(
    r"^(region\d+)_(\d{4}-\d{2}-\d{2})-\d{2}-\d{2}_.*_(L1C|L2A)_([A-Z0-9]+)_\(Raw\)\.(tiff|tif)$", re.IGNORECASE)

def scan_test_images():
    """自适应两种数据格式:
    比赛格式 region17_2018-04-24-..._B01_(Raw).tiff → rf[rid][date][band] = 文件 (每波段一个文件)
    平台格式 13_2018-10-04.tif 单文件12波段        → rf[rid][date] = 文件
    """
    rf_multi = defaultdict(lambda: defaultdict(dict))
    rf_single = defaultdict(dict)
    sd = REGION_TEST_DIR if REGION_TEST_DIR.exists() else INPUT_DIR
    for f in sd.iterdir():
        if f.suffix.lower() not in ['.tiff','.tif']: continue
        mc = COMPETITION_PATTERN.match(f.name)
        if mc:
            rid, ds, band = mc.group(1), mc.group(2), mc.group(4).upper()
            if band in BANDS: rf_multi[rid][ds][band] = str(f)
            continue
        ms = FILE_PATTERN.match(f.name)
        if ms:
            rid, ds = f"region{ms.group(1)}", ms.group(2)
            rf_single[rid][ds] = str(f)
    if rf_multi:
        return dict(rf_multi), 'multi'
    if rf_single:
        return dict(rf_single), 'single'
    return {}, 'single'

def build_region_bounds(rf):
    bm = {}
    for rid, dd in rf.items():
        for val in dd.values():
            fp = next(iter(val.values())) if isinstance(val, dict) else val
            try:
                with rasterio.open(fp) as src: bm[rid] = src.bounds; break
            except: continue
    return bm

def find_region(lon, lat, bm, pid=None, rf=None):
    # 平台格式: 文件名第一段即 point_id, 优先精确匹配
    if pid is not None and rf is not None:
        rid = f"region{pid}"
        if rid in rf and rid in bm:
            b = bm[rid]
            if b.left <= lon <= b.right and b.bottom <= lat <= b.top: return rid
    for rid, b in bm.items():
        if b.left <= lon <= b.right and b.bottom <= lat <= b.top: return rid
    return None

ANOMALY_MAP = {}
def scan_anomalies(rf):
    am, ck = {}, set()
    for rid, dd in rf.items():
        for ds, val in dd.items():
            k = (rid, ds)
            if k in ck: continue
            ck.add(k)
            try:
                if isinstance(val, dict):
                    fp = val.get('B08') or next(iter(val.values()))
                    with rasterio.open(fp) as src:
                        h, w = src.height, src.width
                        d = src.read(1, window=Window(max(0, w//2-50), max(0, h//2-50), 100, 100))
                else:
                    with rasterio.open(val) as src:
                        if src.count < 8: continue
                        h, w = src.height, src.width
                        d = src.read(8, window=Window(max(0, w//2-50), max(0, h//2-50), 100, 100))
                v = d[d > 0]
                if v.size > 0 and float(np.mean(v)) > 2:
                    am.setdefault(rid, {})[ds] = 255.0
                    print(f"  [异常] {rid} {ds}: B08均值={float(np.mean(v)):.1f} → ÷255")
            except: continue
    return am


BAND_TO_IDX = {b: i+1 for i, b in enumerate(BANDS)}

def extract_dualsource(lon, lat, rid, rf, td=None):
    if rid is None or rid not in rf: return None, None, None
    s3, s1, dl = [], [], []
    for ds in sorted(rf[rid].keys()):
        val = rf[rid][ds]
        nd = ANOMALY_MAP.get(rid, {}).get(ds)
        bv3, bv1, ok = [], [], True
        try:
            if isinstance(val, dict):
                # 比赛格式: 每波段一个文件
                for b in BANDS:
                    fp = val.get(b)
                    if fp is None: ok = False; break
                    with rasterio.open(fp) as src:
                        py, px = src.index(lon, lat)
                        r = list(src.sample([(lon, lat)]))
                        if not r: ok = False; break
                        v1 = float(r[0][0])
                        if (src.nodata is not None and v1 == src.nodata) or v1 <= 0: ok = False; break
                        d = src.read(1, window=Window(px-1, py-1, 3, 3), boundless=True, fill_value=0)
                        vd = d[(d != src.nodata) & (d > 0)] if src.nodata is not None else d[d > 0]
                        if vd.size == 0: ok = False; break
                        v3 = float(np.mean(vd))
                        if nd is not None: v1 /= nd; v3 /= nd
                        bv1.append(v1); bv3.append(v3)
            else:
                # 平台格式: 单文件12波段
                with rasterio.open(val) as src:
                    if src.count < len(BANDS):
                        ok = False
                    else:
                        py, px = src.index(lon, lat)
                        for bi in range(1, len(BANDS)+1):
                            r = list(src.sample([(lon, lat)], indexes=bi))
                            if not r: ok = False; break
                            v1 = float(r[0][0])
                            if (src.nodata is not None and v1 == src.nodata) or v1 <= 0: ok = False; break
                            d = src.read(bi, window=Window(px-1, py-1, 3, 3), boundless=True, fill_value=0)
                            vd = d[(d != src.nodata) & (d > 0)] if src.nodata is not None else d[d > 0]
                            if vd.size == 0: ok = False; break
                            v3 = float(np.mean(vd))
                            if nd is not None: v1 /= nd; v3 /= nd
                            bv1.append(v1); bv3.append(v3)
        except: continue
        if not ok or len(bv3) != len(BANDS): continue
        if bv3[3] == 0 and bv3[7] == 0 and bv3[1] == 0: continue
        i3, i1 = calc_indices(bv3), calc_indices(bv1)
        s3.append(bv3+i3); s1.append(bv1+i1); dl.append(date_to_doy(ds))
    if len(s3) < 1: return None, None, None
    od = np.array(dl); f3 = np.array(s3, dtype=np.float32); f1 = np.array(s1, dtype=np.float32)
    if td is not None and td not in od:
        if td < od[0]: i3, i1, fp = f3[0].copy(), f1[0].copy(), 0
        elif td > od[-1]: i3, i1, fp = f3[-1].copy(), f1[-1].copy(), len(od)
        else:
            ix = max(0, min(np.searchsorted(od, td)-1, len(od)-2))
            a = (td-od[ix])/(od[ix+1]-od[ix]+1e-6)
            i3 = f3[ix]*(1-a)+f3[ix+1]*a; i1 = f1[ix]*(1-a)+f1[ix+1]*a; fp = ix+1
        od = np.insert(od, fp, td); f3 = np.insert(f3, fp, i3, axis=0); f1 = np.insert(f1, fp, i1, axis=0)
    sq3 = np.array([list(f3[i])+encode_doy(d) for i, d in enumerate(od)], dtype=np.float32)
    sq1 = np.array([list(f1[i])+encode_doy(d) for i, d in enumerate(od)], dtype=np.float32)
    return sq3, sq1, od

def extract_point(args):
    pid,lon,lat,lr,la,obs_date,obs_doy,rid,rf = args
    if rid is None or rid not in rf:
        return None,{'pid':pid,'lon':lon,'lat':lat,'lon_raw':lr,'lat_raw':la,'obs_date':obs_date,'obs_doy':obs_doy,'reason':'no_region'}
    s3,s1,d = extract_dualsource(lon,lat,rid,rf,obs_doy)
    if s3 is None or len(s3)<1:
        return None,{'pid':pid,'lon':lon,'lat':lat,'lon_raw':lr,'lat_raw':la,'obs_date':obs_date,'obs_doy':obs_doy,
                     'reason':f'too_short({0 if s3 is None else len(s3)})'}
    return {'pid':pid,'lon_raw':lr,'lat_raw':la,'obs_date':obs_date,'obs_doy':obs_doy,
            'lon':lon,'lat':lat,'base_seq_3x3':s3,'base_seq_1x1':s1,'doys':d,'length':len(s3)},None

def fallback_phase(doy):
    return PHASE_NAMES[np.argmin(np.abs(doy-np.array([120,145,175,200,240,265,295])))]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    test_csv = INPUT_DIR / "index.csv"
    if not test_csv.exists(): test_csv = INPUT_DIR / "test_point.csv"
    if not test_csv.exists(): test_csv = INPUT_DIR / "points_test.csv"
    df = pd.read_csv(test_csv); print(f"测试点: {len(df)}")

    print("\n扫描影像..."); rf, dmode = scan_test_images()
    print(f" {len(rf)} region (格式: {dmode})"); bm = build_region_bounds(rf); print(" bounds OK")
    global ANOMALY_MAP; ANOMALY_MAP = scan_anomalies(rf)

    print("\n提取数据...")
    al,nd = [],[]
    for _,r in df.iterrows():
        pid,lon,lat = r['point_id'],float(r['Longitude']),float(r['Latitude'])
        od,lonr,latr = str(r['phenophase_date']),r['Longitude'],r['Latitude']
        try: doy = parse_obs_date(od)
        except: nd.append({'pid':pid,'lon':lon,'lat':lat,'lon_raw':lonr,'lat_raw':latr,'obs_date':od,'obs_doy':180,'reason':'parse'}); continue
        rid = find_region(lon,lat,bm,pid=pid,rf=rf)
        al.append((pid,lon,lat,lonr,latr,od,doy,rid,rf))
    nw = min(cpu_count(),8)
    sp,nd2 = [],[]
    with Pool(nw) as p:
        for r,ndat in tqdm(p.imap_unordered(extract_point,al,chunksize=10),total=len(al),desc=" 提取"):
            if r: sp.append(r)
            if ndat: nd2.append(ndat)
    nd+=nd2; print(f" 有效:{len(sp)}, 无影像:{len(nd)}")

    print("\n加载 v90-2-11 (作物分类)...")
    m11,m11m = [],[]
    for p in MODEL_PATHS_V11:
        if not p.exists(): continue
        c = torch.load(p,map_location=device,weights_only=False)
        m = MultiTaskLSTM().to(device); m.load_state_dict(c['model']); m.eval()
        m11.append(m); m11m.append((c.get('lon_min',GLOBAL_LON_MIN),c.get('lon_max',GLOBAL_LON_MAX)))
    print(f" {len(m11)} 模型")

    print(f"加载 v90-4-5-tau14 (Delta T τ={DELTA_TAU})...")
    m45t14 = []
    for p in MODEL_PATHS_V45T14:
        if not p.exists(): continue
        c = torch.load(p,map_location=device,weights_only=False)
        m = PhenoLSTM().to(device); m.load_state_dict(c['model']); m.eval()
        m45t14.append(m)
    print(f" {len(m45t14)} 模型")

    if not m11 or not m45t14: raise RuntimeError("无模型")
    res = {}
    ph_cache = {}

    if sp:
        print(f"\n[推理] 二段式: v90-2-11 作物 → v90-4-5-tau14 (τ={DELTA_TAU}, INPUT_DIM={INPUT_DIM})...")
        for s in tqdm(sp,desc=" 推理"):
            da = s['doys']; td = s['obs_doy']
            mi = np.where(da==td)[0]
            ap_crop = []; crop_pred = None

            for m,(lomin,lomax) in zip(m11,m11m):
                b3=s['base_seq_3x3']; L=len(b3)
                cl=encode_location_crop(s['lon'],s['lat'])
                cs=np.concatenate([b3,np.array([cl]*L,dtype=np.float32)],axis=1)
                mk=torch.ones(1,L,dtype=torch.bool).to(device)
                ct=torch.tensor(cs,dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    cr,_=m(ct,mk)
                    if crop_pred is None: crop_pred=torch.argmax(cr).item()
                    ap_crop.append(F.softmax(cr,dim=-1))
            crop_pred=max(0,min(NUM_CROPS-1,torch.argmax(torch.stack(ap_crop).mean(dim=0)).item()))

            ap_ph=[]
            for m in m45t14:
                b1=s['base_seq_1x1']; L=len(b1)
                pl=encode_location_pheno_global(s['lon'],s['lat'])
                ps=np.concatenate([b1,np.array([pl]*L,dtype=np.float32)],axis=1)
                dt=np.array([encode_delta_t(d,td) for d in da],dtype=np.float32).reshape(-1,1)
                ps=np.concatenate([ps,dt],axis=1)
                mk=torch.ones(1,L,dtype=torch.bool).to(device)
                pt=torch.tensor(ps,dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    ph=m(pt,mk)
                    if len(mi)>0: idx=min(mi[0],ph.size(1)-1); sl=ph[0,idx]
                    else:
                        ib=max(0,np.searchsorted(da,td)-1); ia=min(ib+1,L-1)
                        sl=ph[0,min(ib,ph.size(1)-1)] if ib==ia else ph[0,ib]*(1-np.clip((td-da[ib])/(da[ia]-da[ib]+1e-6),0,1))+ph[0,ia]*np.clip((td-da[ib])/(da[ia]-da[ib]+1e-6),0,1)
                    ap_ph.append(F.softmax(sl,dim=-1))
            mean_probs = torch.stack(ap_ph).mean(dim=0)
            pp = torch.argmax(mean_probs).item()
            pp = max(0, min(NUM_PHASES-1, pp))
            key = make_key(s['lon_raw'], s['lat_raw'], s['obs_date'])
            res[key] = [CROP_NAMES[crop_pred], PHASE_NAMES[pp]]
            ph_cache[key] = np.asarray(mean_probs.detach().cpu().tolist(), dtype=np.float64)

    fc=0
    for p in nd:
        res[make_key(p['lon_raw'],p['lat_raw'],p['obs_date'])]=[FALLBACK_CROP,fallback_phase(p['obs_doy'])]
        fc+=1
    fm=0
    for _,r in df.iterrows():
        k=make_key(r['Longitude'],r['Latitude'],r['phenophase_date'])
        if k not in res:
            try: d=parse_obs_date(str(r['phenophase_date']))
            except: d=180
            res[k]=[FALLBACK_CROP,fallback_phase(d)]; fm+=1

    # ================================================================
    # 后处理阶段1：物候重复检测（同点不同日期出现相同物候→低置信度改第二高）
    # ================================================================
    print("\n[后处理1] 检查重复物候...")
    from datetime import datetime
    from collections import Counter
    def point_key(k): return '_'.join(k.split('_')[:2])
    def date_from_key(k):
        ds = '_'.join(k.split('_')[2:])
        for f in ['%Y/%m/%d','%Y-%m-%d']:
            try: return datetime.strptime(ds.strip(), f).timetuple().tm_yday
            except: continue
        return 0
    pg = {}
    for k, v in res.items():
        if v[0] != 'rice':
            continue
        pk = point_key(k)
        pg.setdefault(pk, []).append((k, date_from_key(k), v[1]))
    n_fix_dup = 0
    for pk, entries in pg.items():
        entries.sort(key=lambda x: x[1])
        phases = [e[2] for e in entries]
        dup_phases = [ph for ph, cnt in Counter(phases).items() if cnt >= 2]
        if not dup_phases:
            continue
        dup_phase = dup_phases[0]
        di = [i for i, p in enumerate(phases) if p == dup_phase]
        i1, i2 = di[0], di[1]
        k1, k2 = entries[i1][0], entries[i2][0]
        p1 = ph_cache.get(k1)
        p2 = ph_cache.get(k2)
        if p1 is None or p2 is None:
            continue
        dup_idx = PHASE_NAMES.index(dup_phase)
        conf1, conf2 = p1[dup_idx], p2[dup_idx]
        def fmt_probs(probs):
            parts = [f"{PHASE_NAMES[i]}={probs[i]:.4f}" for i in range(7)]
            return '[' + ', '.join(parts) + ']'
        print(f"\n  [重复] {pk}: 重复物候={dup_phase}")
        print(f"    {entries[i1][0].rsplit('_',1)[-1]} DOY{entries[i1][1]}: 置信度={conf1:.4f} {fmt_probs(p1)}")
        print(f"    {entries[i2][0].rsplit('_',1)[-1]} DOY{entries[i2][1]}: 置信度={conf2:.4f} {fmt_probs(p2)}")
        if conf1 <= conf2:
            p_sel = np.where(np.arange(7) != dup_idx, p1, 0)
            c = np.argmax(p_sel)
            print(f"    第一日置信度更低({conf1:.4f} ≤ {conf2:.4f})，改第一日: {dup_phase} → {PHASE_NAMES[c]} ({p_sel[c]:.4f})")
            phases[i1] = PHASE_NAMES[c]
        else:
            p_sel = np.where(np.arange(7) != dup_idx, p2, 0)
            c = np.argmax(p_sel)
            print(f"    第二日置信度更低({conf2:.4f} < {conf1:.4f})，改第二日: {dup_phase} → {PHASE_NAMES[c]} ({p_sel[c]:.4f})")
            phases[i2] = PHASE_NAMES[c]
        n_fix_dup += 1
        for i, (k, _, _) in enumerate(entries):
            res[k] = [res[k][0], phases[i]]
    if n_fix_dup:
        print(f"\n  [重复] 共修复 {n_fix_dup} 个点")

    # ================================================================
    # 后处理阶段2：物候空间单调性检查
    # ================================================================
    print("\n[后处理2] 检查物候空间单调性...")
    from collections import defaultdict
    WINDOW = 2
    n_fix_spatial = 0
    # 按日期分组
    date_groups = defaultdict(list)
    for k, v in res.items():
        if v[0] != 'rice':  # ★ 仅处理水稻
            continue
        parts = k.split('_', 2)
        lon, date_str = float(parts[0]), parts[-1]
        probs = ph_cache.get(k)
        if probs is None:
            continue
        date_groups[date_str].append((k, lon, v[1], probs))
    for date_str, entries in sorted(date_groups.items()):
        entries.sort(key=lambda x: x[1])  # 经度升序
        n = len(entries)
        if n < 3:
            continue
        phases = [e[2] for e in entries]
        pi = np.array([PHASE_NAMES.index(p) for p in phases])
        anomalies = []
        for i in range(n):
            left = max(0, i - WINDOW)
            right = min(n, i + WINDOW + 1)
            window = np.concatenate([pi[left:i], pi[i+1:right]])
            if len(window) == 0:
                continue
            median_phase = np.median(window)
            dev = pi[i] - median_phase
            if abs(dev) < 1:
                continue
            # 极值+趋势方向判断
            is_peak = 0 < i < n - 1 and pi[i] > pi[i-1] and pi[i] > pi[i+1]
            is_valley = 0 < i < n - 1 and pi[i] < pi[i-1] and pi[i] < pi[i+1]
            if is_peak or is_valley:
                lh = pi[left:i]
                rh = pi[i+1:right]
                if len(lh) > 0 and len(rh) > 0:
                    lm, rm = np.median(lh), np.median(rh)
                    if lm > rm and is_valley:
                        continue
                    if lm < rm and is_peak:
                        continue
            else:
                lh = pi[left:i]
                rh = pi[i+1:right]
                if len(lh) > 0 and len(rh) > 0 and np.median(lh) != np.median(rh):
                    continue
            anomalies.append(i)
        if not anomalies:
            continue
        print(f"\n  [空间] {date_str} ({n}个点) 异常 {len(anomalies)} 处:")
        # 打印物候序列
        seq_str = ' '.join(f"{phases[j]:>12}" for j in range(n))
        print(f"    {seq_str}")
        # 打印经度序列
        lon_str = ' '.join(f"{entries[j][1]:>12.4f}" for j in range(n))
        print(f"    {lon_str}")
        # 打印标记
        mark = ' '.join("     ←异常    " if j in anomalies else "              " for j in range(n))
        print(f"    {mark}")
        for i in anomalies:
            k = entries[i][0]
            probs = entries[i][3]
            cur = phases[i]
            cur_idx = pi[i]
            med = np.median(np.concatenate([pi[max(0,i-WINDOW):i], pi[i+1:min(n,i+WINDOW+1)]]))
            new_phase = PHASE_NAMES[int(round(med))]
            def fmt_probs(probs):
                parts = [f"{PHASE_NAMES[i]}={probs[i]:.4f}" for i in range(7)]
                return '[' + ', '.join(parts) + ']'
            print(f"      {entries[i][0].rsplit('_',1)[-1]} lon={entries[i][1]:.6f} {cur} → {new_phase}  邻域中位数={new_phase}  {fmt_probs(probs)}")
            res[k] = [res[k][0], new_phase]
            n_fix_spatial += 1
    if n_fix_spatial:
        print(f"\n  [空间] 共修复 {n_fix_spatial} 个点")

    total_fixes = n_fix_dup + n_fix_spatial
    if total_fixes:
        print(f"\n  [后处理] 总计修复 {total_fixes} 个点 (重复{n_fix_dup} + 空间{n_fix_spatial})")

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    with open(OUTPUT_JSON,'w',encoding='utf-8') as f: json.dump(res,f,indent=2,ensure_ascii=False)

    crop_dist = Counter(v[0] for v in res.values())
    phase_dist = Counter(v[1] for v in res.values())
    print(f"\n推理完成！")
    print(f" 总预测点数: {len(res)} / {len(df)}")
    print(f" 模型推理: {len(sp)} 个点")
    print(f" PhenoLSTM: {len(m45t14)}折集成（Delta T τ={DELTA_TAU}, 全局经度）")
    print(f" 兜底处理: {fc} 个点")
    if fm: print(f" 补全处理: {fm} 个点")
    print(f" 作物分布: {dict(crop_dist)}")
    print(f" 物候分布: {dict(phase_dist)}")

    # ================================================================
    # 自评打分：用 index.csv 的真实标签对比预测，计算官方评分口径
    # 总分 = 0.4 * 作物macro-F1 + 0.6 * 物候macro-F1
    # ================================================================
    print("\n[自评打分] 对比 index.csv 真实标签...")
    label_columns = [c for c in ['crop_type', 'phenophase_name'] if c in df.columns]
    if len(label_columns) == 2:
        def macro_f1(y_true, y_pred, labels):
            f1s = []
            for lab in labels:
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
            return float(np.mean(f1s))

        true_crop, pred_crop, true_phase, pred_phase = [], [], [], []
        match_cnt, miss_cnt = 0, 0
        for _, r in df.iterrows():
            key = make_key(r['Longitude'], r['Latitude'], r['phenophase_date'])
            if key not in res:
                miss_cnt += 1
                continue
            match_cnt += 1
            tc = str(r['crop_type']).strip().lower()
            tp_ = str(r['phenophase_name']).strip()
            pc, pp = res[key]
            true_crop.append(tc); pred_crop.append(pc)
            true_phase.append(tp_); pred_phase.append(pp)

        crop_f1 = macro_f1(true_crop, pred_crop, CROP_NAMES)
        phase_f1 = macro_f1(true_phase, pred_phase, PHASE_NAMES)
        total = 0.4 * crop_f1 + 0.6 * phase_f1

        score_report = {
            'crop_macro_f1': crop_f1,
            'pheno_macro_f1': phase_f1,
            'total_score': total,
            'matched': match_cnt,
            'missing': miss_cnt,
            'weights': {'crop': 0.4, 'pheno': 0.6},
        }
        with open(OUTPUT_DIR / "result_score.json", 'w', encoding='utf-8') as f:
            json.dump(score_report, f, indent=2, ensure_ascii=False)

        print(f"  匹配样本: {match_cnt}, 未匹配: {miss_cnt}")
        print(f"  作物 macro-F1: {crop_f1:.6f}  (权重 0.4)")
        print(f"  物候 macro-F1: {phase_f1:.6f}  (权重 0.6)")
        print(f"  =====> 总分: {total:.6f} =====")
        print(f"  评分报告已写入: {OUTPUT_DIR / 'result_score.json'}")
    else:
        print("  index.csv 无标签列(crop_type/phenophase_name)，跳过自评打分")

    print("\n=== RESULT_JSON_START ===")
    print(json.dumps(res, ensure_ascii=False))
    print("=== RESULT_JSON_END ===")

if __name__=='__main__':
    main()
