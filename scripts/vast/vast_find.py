#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, json, re, shutil, subprocess
from typing import Any

REFERENCE_CPU = "AMD Ryzen 9 9950X"
REFERENCE_SCORE = 4727
CPU_SCORES = {
    "Intel Core Ultra 9 285K": 5085,
    "Intel Core Ultra 7 270K Plus": 5068,
    "Intel Core Ultra 7 265K": 4928,
    "Intel Core Ultra 7 265KF": 4927,
    "Intel Core Ultra 9 285": 4900,
    "Intel Core Ultra 9 290K Plus": 4823,
    "Intel Core i9-14900KS": 4811,
    "Intel Core Ultra 5 250K Plus": 4780,
    "Intel Core Ultra 7 265F": 4751,
    "AMD Ryzen 9 9950X3D": 4738,
    "AMD Ryzen 9 9950X": 4727,
    "Intel Core Ultra 5 250KF Plus": 4723,
    "Intel Core i9-13900KS": 4715,
    "Intel Core Ultra 5 245K": 4715,
    "Intel Core Ultra 5 245KF": 4714,
    "AMD Ryzen 7 9850X3D": 4704,
    "AMD Ryzen 9 PRO 9965": 4690,
    "Intel Core Ultra 7 265": 4690,
    "Intel Core i9-14900K": 4689,
    "Intel Core i9-14900KF": 4688,
    "AMD Ryzen 7 9700F": 4686,
    "AMD Ryzen 9 9900X": 4673,
    "AMD Ryzen 9 9950X3D2": 4668,
    "AMD Ryzen 7 9700X": 4643,
    "AMD Ryzen 5 PRO 9655": 4640,
    "AMD Ryzen 5 PRO 9645": 4636,
    "AMD Ryzen 9 9900X3D": 4636,
    "AMD Ryzen 7 PRO 9745": 4624,
    "AMD Ryzen 9 PRO 9945": 4619,
    "AMD Ryzen 9 PRO 9965X3D": 4613,
    "AMD Ryzen 7 PRO 9755": 4609,
    "Intel Core Ultra 9 285T": 4600,
    "Intel Core i9-13900K": 4596,
    "Intel Core i9-13900KF": 4580,
    "AMD Ryzen 9 PRO 9955": 4579,
    "AMD Ryzen Threadripper 9960X": 4577,
    "AMD EPYC 4245P": 4575,
    "AMD EPYC 4465P": 4575,
    "Intel Core Ultra 3 205": 4575,
    "AMD Ryzen Threadripper PRO 9945WX": 4572,
    "AMD Ryzen 5 9600X": 4571,
    "Intel Core Ultra 5 235A": 4557,
    "AMD Ryzen Threadripper PRO 9965WX": 4554,
    "AMD Ryzen Threadripper 9980X": 4540,
    "AMD EPYC 4585PX": 4538,
    "AMD Ryzen Threadripper PRO 9995WX": 4537,
    "AMD Ryzen Threadripper 9970X": 4533,
    "AMD Ryzen Threadripper PRO 9955WX": 4530,
    "Intel Xeon 6349P": 4528,
    "Intel Core Ultra 5 230F": 4527,
    "Intel Core Ultra 5 235": 4513,
    "Intel Core i9-14900F": 4504,
    "AMD Ryzen Threadripper PRO 9985WX": 4484,
    "Intel Core i7-14700KF": 4465,
    "Intel Core i7-14700K": 4454,
    "AMD Ryzen 7 PRO 9755X3D": 4442,
    "AMD Ryzen 7 9800X3D": 4421,
    "Intel Core Ultra 5 225": 4413,
    "AMD Ryzen Threadripper PRO 9975WX": 4410,
    "AMD EPYC 4345P": 4408,
    "Intel Core Ultra 5 245": 4406,
    "Intel Core Ultra 5 225F": 4401,
    "Intel Core i9-13900F": 4398,
    "Intel Core Ultra 5 245T": 4383,
    "AMD Ryzen 5 9600": 4377,
    "Intel Core Ultra 7 265T": 4344,
    "Intel Core Ultra 5 235T": 4343,
    "Intel Core i7-13700KF": 4329,
    "Intel Core i7-13700K": 4326,
    "Intel Core i9-14900": 4323,
    "Intel Core i9-12900KS": 4323,
    "AMD EPYC 4545P": 4318,
    "Intel Xeon 6369P": 4303,
    "Intel Xeon E-2488": 4300,
    "AMD EPYC 4564P": 4292,
    "Intel Core i9-13900": 4281,
    "Intel Core i5-14600K": 4268,
    "Intel Core Ultra 5 225T": 4259,
    "AMD EPYC 9175F": 4256,
    "AMD Ryzen 5 9500F": 4254,
    "Intel Core i7-14700F": 4254,
    "Intel Core i5-14600KF": 4252,
    "AMD Ryzen 9 7950X": 4252,
    "Intel Core i7-14700": 4236,
    "Intel Xeon 6357P": 4233,
    "AMD Ryzen 9 7900X": 4226,
    "Intel Xeon 6353P": 4226,
    "Intel Xeon 6325P": 4213,
    "Intel Core i7-14790F": 4191,
    "AMD Ryzen 7 7700X": 4176,
    "AMD Ryzen 9 PRO 7945": 4175,
    "AMD Ryzen Threadripper 7970X": 4174,
    "AMD EPYC 9575F": 4173,
    "Intel Core i5-14600": 4168,
    "Intel Core i9-13900T": 4165,
    "AMD EPYC 4464P": 4146,
    "AMD Ryzen 9 7950X3D": 4144,
    "AMD Ryzen 5 7600X": 4129,
    "Intel Core i9-12900K": 4128,
    "AMD Ryzen Threadripper 7960X": 4124,
    "AMD Ryzen 9 7900": 4120,
    "Intel Core i9-12900KF": 4120,
    "AMD EPYC 4484PX": 4119,
    "AMD Ryzen 9 7900X3D": 4118,
    "AMD Ryzen 7 PRO 7745": 4117,
    "Intel Core i7-13700F": 4116,
    "Intel Core i5-13600K": 4111,
    "Intel Core i5-13600KF": 4111,
    "Intel Xeon 6337P": 4104,
    "Intel Core i7-13700": 4092,
    "AMD Ryzen Threadripper PRO 7955WX": 4086,
    "AMD Ryzen Threadripper PRO 7945WX": 4076,
    "AMD Ryzen 7 7700": 4050,
    "Intel Core i5-13600": 4047,
    "Intel Core i7-13790F": 4035,
    "AMD Ryzen Threadripper 7980X": 4024,
    "Intel Xeon 676X": 4015,
    "Intel Core i9-12900F": 4013,
    "AMD Ryzen Threadripper PRO 7965WX": 4009,
    "Intel Core i7-12700K": 4003,
    "Intel Core i9-12900": 4002,
    "AMD Ryzen Threadripper PRO 7975WX": 3989,
    "Intel Core i7-12700KF": 3980,
    "AMD Ryzen Threadripper PRO 7985WX": 3962,
    "Intel Core i5-14490F": 3958,
    "AMD Ryzen 7 PRO 8700G": 3958,
    "AMD Ryzen 5 PRO 7645": 3954,
    "Intel Core i5-14500": 3950,
    "AMD Ryzen 5 PRO 8500G": 3949,
    "AMD Ryzen 5 PRO 8600G": 3941,
    "Intel Xeon 674X": 3933,
    "AMD Ryzen 7 8700G": 3921,
    "Intel Core i5-12600K": 3917,
    "Intel Core i5-12600KF": 3916,
    "Intel Core i7-14700T": 3911,
    "AMD Ryzen 5 7600": 3907,
    "AMD Ryzen 5 PRO 8500GE": 3907,
    "AMD Ryzen 5 8500GE": 3904,
    "AMD EPYC 4124P": 3897,
    "Intel Xeon E-2478": 3895,
    "AMD Ryzen 5 8500G": 3876,
    "AMD Ryzen 5 8600G": 3876,
    "AMD Ryzen 7 8700F": 3873,
    "AMD Ryzen 5 PRO 8600GE": 3872,
    "Intel Xeon 636": 3871,
    "AMD Ryzen 7 PRO 8700GE": 3859,
    "Intel Core i5-13490F": 3861,
    "Intel Core i5-13500": 3855,
    "AMD EPYC 9655P": 3849,
    "AMD EPYC 9655": 3847,
    "Intel Core i7-12700F": 3841,
    "Intel Core i7-12700": 3841,
    "Intel Xeon E-2434": 3840,
    "Intel Xeon E-2486": 3837,
    "Intel Xeon 638": 3835,
    "AMD Ryzen Threadripper PRO 7995WX": 3831,
    "Intel Xeon E-2468": 3828,
    "AMD Ryzen 5 7500F": 3825,
    "AMD EPYC 9R45": 3812,
    "AMD EPYC 9275F": 3810,
    "Intel Core i7-13700T": 3803,
    "AMD EPYC 4584PX": 3795,
    "Intel Xeon 6315P": 3783,
    "AMD EPYC 9455 Embedded": 3782,
    "AMD EPYC 9475F": 3779,
    "Intel Xeon 654": 3778,
    "AMD Ryzen 3 8300G": 3778,
    "Intel Core i3-14100F": 3775,
    "Intel Core i5-13600T": 3769,
    "Intel Core i3-14100": 3760,
    "AMD Ryzen 7 7800X3D": 3759,
    "Intel Xeon 678X": 3758,
    "AMD EPYC 9555 Embedded": 3757,
    "AMD EPYC 9375F": 3762,
    "AMD EPYC 9355P": 3747,
    "AMD EPYC 9455P": 3745,
    "Intel Core i5-14600T": 3744,
    "Intel Xeon 696X": 3742,
    "Intel Core i5-14400": 3741,
    "Intel Core i5-14500T": 3741,
    "Intel Core i9-12900T": 3738,
    "Intel Xeon w7-2595X": 3735,
    "AMD EPYC 4244P": 3724,
    "AMD EPYC 9535": 3720,
    "Intel Xeon w9-3595X": 3717,
    "Intel Xeon 658X": 3710,
    "Intel Core i5-14400F": 3700,
    "AMD EPYC 9565": 3696,
    "Intel Core i5-12490F": 3689,
    "AMD Ryzen 5 7400F": 3686,
    "AMD Ryzen 5 8400F": 3685,
    "AMD EPYC 9135": 3672,
    "Intel Xeon w9-3575X": 3672,
    "AMD EPYC 9255": 3655,
    "Intel Core i5-12500": 3648,
    "Intel Xeon 6507P": 3643,
    "Intel Core 5 120F": 3636,
    "Intel Core i7-12700E": 3633,
    "Intel Core i5-13400F": 3628,
    "AMD EPYC 4364P": 3619,
    "AMD Ryzen 5 7600X3D": 3606,
    "Intel Xeon w5-3535X": 3602,
    "Intel Xeon E-2436": 3601,
    "Intel Xeon w5-2555X": 3600,
    "Intel Core i3-13100F": 3599,
    "Intel Xeon 634": 3595,
    "Intel Xeon w5-2565X": 3595,
    "Intel Core 5 120": 3588,
    "Intel Core i5-13400": 3584,
    "Intel Core i5-13500T": 3577,
    "Intel Xeon w5-2545": 3573,
    "Intel Core i3-12300": 3570,
    "Intel Xeon E-2456": 3568,
    "Intel Core i7-12700T": 3567,
    "Intel Xeon E-2414": 3553,
    "Intel Xeon w7-2475X": 3552,
    "Intel Xeon w7-3555": 3549,
    "Intel Xeon W-1390P": 3544,
    "Intel Core i5-12600T": 3544,
    "AMD EPYC 9B45": 3540,
    "Intel Xeon 6527P": 3539,
    "Intel Core i3-13100": 3537,
    "AMD Ryzen 7 5800XT": 3534,
    "AMD EPYC 9755": 3526,
    "AMD EPYC 4344P": 3526,
    "Intel Core i5-14400T": 3516,
    "Intel Core i9-11900KF": 3516,
    "Intel Core i9-11900K": 3501,
    "Intel Core i3-14100T": 3497,
    "AMD Ryzen 5 7500X3D": 3492,
    "Intel Core i5-12400F": 3485,
    "AMD Ryzen 9 5900XT": 3478,
    "AMD Ryzen 9 5950X": 3476,
    "Intel Core i5-13400T": 3469,
    "AMD Ryzen 5 5600XT": 3468,
    "AMD Ryzen 9 5900X": 3465,
    "Intel Core i5-12400": 3465,
    "Intel Xeon w7-2495X": 3459,
    "Intel Xeon W-1370": 3459,
    "Intel Xeon w5-2465X": 3455,
    "Intel Core i5-12500T": 3453,
    "Intel Xeon w7-3545": 3452,
    "Intel Xeon Gold 6534": 3451,
    "Intel Xeon 6333P": 3450,
    "Intel Xeon 6745P": 3450,
    "Intel Xeon w9-3495X": 3449,
    "Intel Xeon W-1350P": 3448,
    "AMD Ryzen 7 5800X": 3448,
    "AMD Ryzen 7 PRO 5845": 3442,
    "AMD Ryzen 5 PRO 5645": 3441,
    "Intel Xeon W-1370P": 3438,
    "AMD Ryzen 9 5900": 3433,
    "Intel Core i3-12100F": 3432,
    "Intel Xeon w3-2525": 3426,
    "Intel Core i9-11900F": 3421,
    "AMD EPYC 9555P": 3410,
    "Intel Xeon w7-3565X": 3407,
    "Intel Xeon W-1350": 3405,
    "Intel Xeon W-1390": 3399,
    "Intel Xeon E-2388G": 3399,
    "Intel Xeon E-2386G": 3398,
    "Intel Xeon 6973P-C": 3396,
    "AMD Ryzen 7 5700X": 3386,
    "Intel Core i7-11700K": 3385,
    "Intel Xeon w5-2455X": 3380,
    "AMD Ryzen 7 5800": 3377,
    "Intel Core i3-13100T": 3373,
    "Intel Core i9-11900": 3371,
    "AMD Ryzen 7 PRO 5755G": 3366,
    "AMD Ryzen 5 5600X": 3366,
    "Intel Core i7-11700KF": 3361,
    "Intel Xeon E-2374G": 3360,
    "AMD EPYC 9115": 3360,
    "Intel Core i3-12300T": 3359,
    "Intel Xeon 6520P": 3356,
    "AMD Ryzen 7 PRO 5755GE": 3353,
    "Intel Xeon Gold 6434H": 3353,
    "Intel Xeon w5-2445": 3354,
    "Intel Xeon w5-3425": 3350,
    "AMD Ryzen Threadripper PRO 5945WX": 3348,
    "Intel Xeon Gold 5420+": 3347,
    "Intel Xeon E-2378G": 3339,
    "Intel Xeon w9-3475X": 3336,
    "AMD Ryzen 5 5600GT": 3336,
    "Intel Core i5-11600K": 3335,
    "Intel Core i5-12400T": 3335,
    "Intel Xeon w5-3525": 3330,
    "AMD Ryzen 7 5700GE": 3328,
    "Intel Core i5-11600KF": 3326,
    "AMD Ryzen Threadripper PRO 5955WX": 3322,
    "AMD Ryzen 5 5600T": 3321,
    "AMD Ryzen Threadripper PRO 5975WX": 3321,
    "Intel Xeon E-2356G": 3320,
    "AMD Ryzen Threadripper PRO 5965WX": 3317,
}
KNOWN_REJECT=("EPYC 7K62","EPYC 7H12","EPYC 7742","EPYC 7702","EPYC 7642","EPYC 7551","EPYC 7601")
GPU_SCORE={"RTX 5090":20,"RTX 4090":10}

def normalize_name(v:str)->str:
    v=v.upper().replace("(R)","").replace("(TM)","")
    return " ".join(re.sub(r"[^A-Z0-9]+"," ",v).split())
CPU_MATCH=sorted(((normalize_name(n),n,s) for n,s in CPU_SCORES.items()),key=lambda x:len(x[0]),reverse=True)

def cpu_tier_from_score(score:int|None)->str:
    if score is None:return "?"
    if score>=4491:return "A"
    if score>=4255:return "B"
    if score>=4018:return "C"
    if score>=3782:return "D"
    if score>=3309:return "E"
    return "?"

def cpu_info(name:str):
    norm=normalize_name(name)
    if any(normalize_name(x) in norm for x in KNOWN_REJECT):return "REJECT",None,None,None
    for needle,canonical,score in CPU_MATCH:
        if needle in norm:return cpu_tier_from_score(score),score,score/REFERENCE_SCORE*100.0,canonical
    return "?",None,None,None

def fnum(v:Any,default:float=0.0)->float:
    try:return float(v)
    except (TypeError,ValueError):return default

def run_json(cmd:list[str])->Any:
    p=subprocess.run(cmd,text=True,capture_output=True)
    if p.returncode:raise RuntimeError((p.stderr or p.stdout).strip())
    raw=p.stdout.strip()
    try:return json.loads(raw)
    except json.JSONDecodeError:
        try:return ast.literal_eval(raw)
        except (ValueError,SyntaxError) as e:raise RuntimeError(f"Could not parse Vast CLI output:\n{raw}") from e

def unwrap_offers(data:Any):
    if isinstance(data,list):return [x for x in data if isinstance(x,dict)]
    if isinstance(data,dict):
        for key in ("offers","results","data"):
            value=data.get(key)
            if isinstance(value,list):return [x for x in value if isinstance(x,dict)]
            if isinstance(value,dict):return [value]
        if ("id" in data or "ask_contract_id" in data) and "gpu_name" in data:return [data]
    return []

def normalize_offer(raw:dict[str,Any]):
    cpu=str(raw.get("cpu_name") or "unknown");gpu=str(raw.get("gpu_name") or "?")
    tier,score,pct,matched=cpu_info(cpu)
    return {"id":raw.get("id") or raw.get("ask_contract_id"),"gpu":gpu,
            "num_gpus":max(1,int(fnum(raw.get("num_gpus"),1))),"gpu_score":GPU_SCORE.get(gpu,0),"cpu":cpu,"cpu_match":matched,"tier":tier,"st_score":score,"st_pct":pct,"ghz":fnum(raw.get("cpu_ghz")),"vcpus":fnum(raw.get("cpu_cores_effective")),"price":fnum(raw.get("dph_total",raw.get("dph")),9999),"dph_base":fnum(raw.get("dph_base",raw.get("dph"))),"storage_cost":fnum(raw.get("storage_cost")),"min_bid":fnum(raw.get("min_bid")),"rel":fnum(raw.get("reliability")),"pcie":raw.get("pci_gen",raw.get("pcie_gen","?")),"pcie_bw":fnum(raw.get("pcie_bw")),"inet_down":fnum(raw.get("inet_down")),"disk":fnum(raw.get("disk_space")),"duration":fnum(raw.get("duration")),"loc":raw.get("geolocation",raw.get("location","?"))}

def search_offers(*,gpus=None,max_price=None,min_reliability=0.99,min_cpus=8,min_disk=25,min_duration=1.0,allowed_tiers=None,limit=100,exact_gpus=1,min_gpus=1,max_gpus=None,min_cpus_per_gpu=None,max_price_per_gpu=None):
    gpus=gpus or ["RTX 4090","RTX 5090"];gpu_list=", ".join(json.dumps(g) for g in gpus)
    n_clause=(f"num_gpus={exact_gpus}" if exact_gpus is not None else
             f"num_gpus>={max(1,min_gpus)}" + (f" num_gpus<={max_gpus}" if max_gpus else ""))
    query=f"rentable=true {n_clause} reliability>={min_reliability} direct_port_count>=1 cpu_cores_effective>={min_cpus} disk_space>={min_disk} duration>={min_duration} gpu_name in [{gpu_list}]"
    if max_price is not None:query+=f" dph<={max_price}"
    # --type on-demand is PINNED: on-demand rentals give exclusive, non-
    # preemptible control of the GPU for the life of the instance. There is
    # deliberately no flag to switch to bid/interruptible -- a preempted
    # instance is killed mid-run, and this campaign has no mid-run resume.
    data=run_json(["vastai","search","offers",query,"--type","on-demand",
                   "--order","dph","--limit",str(limit),"--raw"])
    rows=[]
    for item in unwrap_offers(data):
        row=normalize_offer(item)
        if row["tier"]=="REJECT":continue
        if allowed_tiers is not None and row["tier"] not in allowed_tiers:continue
        # Shared host resources normalised per GPU: with one independent
        # experiment per GPU, $/GPU and cores/GPU are the comparable numbers.
        # NOT divided: single-thread score, GHz, VRAM, reliability, PCIe.
        n=max(1,int(row.get("num_gpus") or 1))
        row["num_gpus"]=n
        row["price_per_gpu"]=row["price"]/n
        row["vcpus_per_gpu"]=row["vcpus"]/n
        row["disk_per_gpu"]=row["disk"]/n
        row["inet_down_per_gpu"]=row.get("inet_down",0.0)/n
        if min_cpus_per_gpu is not None and row["vcpus_per_gpu"]<min_cpus_per_gpu:continue
        if max_price_per_gpu is not None and row["price_per_gpu"]>max_price_per_gpu:continue
        rows.append(row)
    rows.sort(key=lambda r:(-(r["st_score"] or 0),-r["gpu_score"],r["price_per_gpu"],
                            -r["vcpus_per_gpu"],-r["rel"]))
    return rows

def lookup_offer(offer_id:int):
    try:data=run_json(["vastai","search","offers",f"id={offer_id}","--limit","5","--raw"])
    except Exception:return None
    for item in unwrap_offers(data):
        row=normalize_offer(item)
        if row["id"] is not None and int(row["id"])==offer_id:return row
    return None

def print_candidate(r,index=None,total=None):
    if index is not None and total is not None:print(f"Candidate {index}/{total}")
    print(f"  Offer:        {r['id']}");print(f"  Tier:         {r['tier']}")
    print(f"  Single-core:  {r['st_score']}  ({r['st_pct']:.1f}% of 9950X)" if r["st_score"] is not None else "  Single-core:  unranked")
    print(f"  GPUs:         {r['num_gpus']}x {r['gpu']}");print(f"  CPU:          {r['cpu']}");print(f"  Eff. CPUs/GPU:{r['vcpus_per_gpu']:.1f}  ({r['vcpus']:.0f} total)");print(f"  Price/GPU:    ${r['price_per_gpu']:.3f}/h  (${r['price']:.3f}/h total, on-demand incl. storage; GPU base ${r['dph_base']:.3f}/h)");print(f"  Reliability:  {r['rel']:.4f}");print(f"  PCIe:         Gen {r['pcie']} / {r['pcie_bw']:.1f} GB/s");print(f"  Download:     {r['inet_down']:.0f} Mb/s");print(f"  Max duration: {r['duration']:.1f} days");print(f"  Location:     {r['loc']}")

def print_table(rows):
    if not rows:print("No matching offers.");return
    headers=["#","tier","ST%","offer","N","gpu","cpu","CPU/G","$/G/h","$tot/h","rel","PCIe","loc"]
    out=[]
    for i,r in enumerate(rows,1):out.append([str(i),r["tier"],f'{r["st_pct"]:.1f}' if r["st_pct"] is not None else "?",str(r["id"]),str(r["num_gpus"]),r["gpu"],r["cpu"],f'{r["vcpus_per_gpu"]:.1f}',f'{r["price_per_gpu"]:.3f}',f'{r["price"]:.3f}',f'{r["rel"]:.4f}',str(r["pcie"]),str(r["loc"])])
    widths=[max(len(headers[i]),*(len(row[i]) for row in out)) for i in range(len(headers))]
    print("A >=95%; B 90-95%; C 85-90%; D 80-85%; E 70-80% of Ryzen 9 9950X PassMark single-thread")
    print("Per-GPU columns assume ONE independent experiment per GPU. ST%, reliability, VRAM and PCIe are NOT divided.\n")
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))));print("  ".join("-"*w for w in widths))
    for row in out:print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu",action="append",default=[]);ap.add_argument("--max-price",type=float);ap.add_argument("--min-reliability",type=float,default=0.99);ap.add_argument("--min-cpus",type=float,default=8);ap.add_argument("--min-disk",type=float,default=25);ap.add_argument("--min-duration",type=float,default=1.0);ap.add_argument("--tiers",default="A,B,C,D,E,?");ap.add_argument("--limit",type=int,default=100);ap.add_argument("--top",type=int,default=30)
    ap.add_argument("--gpus",type=int,help="Require EXACTLY this many GPUs per box (default 1).")
    ap.add_argument("--min-gpus",type=int,default=1);ap.add_argument("--max-gpus",type=int)
    ap.add_argument("--min-cpus-per-gpu",type=float,default=8.0,help="CPU QUANTITY floor per GPU; tiers cover CPU QUALITY.")
    ap.add_argument("--max-price-per-gpu",type=float)
    ap.add_argument("--any-gpus",action="store_true",help="Show 1..N GPU boxes together instead of exactly one count.")
    args=ap.parse_args()
    if shutil.which("vastai") is None:raise SystemExit("vastai CLI not found")
    tiers={x.strip().upper() for x in args.tiers.split(",") if x.strip()}
    exact=None if args.any_gpus else (args.gpus if args.gpus is not None else 1)
    print_table(search_offers(gpus=args.gpu or None,max_price=args.max_price,min_reliability=args.min_reliability,
                              min_cpus=0 if exact is None else args.min_cpus,min_disk=args.min_disk,
                              min_duration=args.min_duration,allowed_tiers=tiers,limit=args.limit,
                              exact_gpus=exact,min_gpus=args.min_gpus,max_gpus=args.max_gpus,
                              min_cpus_per_gpu=args.min_cpus_per_gpu,
                              max_price_per_gpu=args.max_price_per_gpu)[:args.top])
if __name__=="__main__":main()
