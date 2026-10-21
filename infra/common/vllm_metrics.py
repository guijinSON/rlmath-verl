# find each vLLMHttpServer's listening port (by pid) and print its key Prometheus gauges
import os, re, sys, urllib.request
pids=[int(x) for x in sys.argv[1:]]
listen={}
for f in [x for x in ['/proc/net/tcp','/proc/net/tcp6'] if os.path.exists(x)]:
    for line in open(f).readlines()[1:]:
        p=line.split()
        if p[3]=='0A': listen[p[9]]=int(p[1].split(':')[1],16)
keys=['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc','vllm:gpu_cache_usage_perc','vllm:generation_tokens_total','vllm:num_preemptions_total']
for pid in pids:
    ports=set()
    for fd in os.listdir(f'/proc/{pid}/fd'):
        try: t=os.readlink(f'/proc/{pid}/fd/{fd}')
        except OSError: continue
        m=re.match(r'socket:\[(\d+)\]',t)
        if m and m.group(1) in listen: ports.add(listen[m.group(1)])
    for port in sorted(ports):
        try: txt=urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=5).read().decode()
        except Exception as e: continue
        vals={}
        for k in keys:
            s=sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith(k) and not l.startswith('#'))
            if any(l.startswith(k) for l in txt.splitlines()): vals[k.split(':')[1]]=round(s,3)
        print(pid, port, vals)
