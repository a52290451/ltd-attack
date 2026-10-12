#!/usr/bin/env python3
"""Apply collector V2.4 to the EXACT reviewed V2.3 files on Perseo.
No modification of frozen manifests, status classifier, PCAP, or historical data.
V2.4: bounded late-SPA hydration, explicit open-shadow CMP, monotonic duration.
"""
from __future__ import annotations
import argparse,hashlib,os,shutil,subprocess,sys
from pathlib import Path

ORIGINAL_SHA="ee4da3e1c6f85e7e02d656232609f02b75dc10731595d433a7a18c76c6c8e75d"
EXPECTED_V24_SHA="137c163bfed44d17d702dffb8e4c15541469ea1d5c41f3c1e96c80cc8c9c6d4c"
def replace_once(s,old,new,label):
    n=s.count(old)
    if n!=1:raise RuntimeError("STOP: anchor "+label+" occurred "+str(n)+" times")
    return s.replace(old,new,1)

def transform(s):
    s=replace_once(s,
        "const timeoutMs = Math.min(Math.max(Number(timeoutSec) || 60, 10), 90) * 1000;",
        "const timeoutMs = Math.min(Math.max(Number(timeoutSec) || 60, 10), 120) * 1000; // V2.4: bounded recovery timeout",
        "max timeout")
    s=replace_once(s,
        "const started = new Date().toISOString();\n  const record = {schema_version: 2, timestamp_start_utc: started, requested_url: url,",
        "const started = new Date().toISOString();\n  const monotonicStart = process.hrtime.bigint();\n  const record = {schema_version: 2, collector_version: 'v2.4', timestamp_start_utc: started, requested_url: url,",
        "collector version")
    s=replace_once(s,
        "    let before = await pageSnapshot(page);\n    const hardBlocked = classifyVisit",
        """    let before = await pageSnapshot(page);
    // V2.4: permitir hidratación SPA sin forzar navegación ni evadir controles.
    if (record.http_status >= 200 && record.http_status < 300 && before.visibleChars < 350) {
      try {
        await page.waitForFunction(() => (document.body?.innerText || '').trim().length >= 350,
          {polling: 250, timeout: 3500});
      } catch (_) { record.errors.push('late_content_wait_elapsed_nonfatal'); }
      before = await pageSnapshot(page);
    }
    const hardBlocked = classifyVisit""",
        "SPA hydration")
    s=replace_once(s,
        "    record.timestamp_end_utc = new Date().toISOString();",
        """    record.timestamp_end_utc = new Date().toISOString();
    record.duration_monotonic_ms = Number((process.hrtime.bigint() - monotonicStart) / 1000000n);""",
        "monotonic duration")
    s=replace_once(s,
        """    } catch (_) { /* no tocar otros botones ni superar controles de acceso */ }
  }
  return { policy, action: 'not_found' };""",
        """    } catch (_) { /* no tocar otros botones ni superar controles de acceso */ }
    // V2.4: CMPs con controles explícitos dentro de shadow DOM abierto.
    // Nunca acceder a shadow roots cerrados ni automatizar CAPTCHA / acceso.
    try {
      const handle = await frame.evaluateHandle(() => {
        const patterns = [
          /^aceptar (todas|todo|todas las cookies)$/i,
          /^permitir todas( las cookies)?$/i,
          /^accept all( cookies)?$/i,
          /^allow all( cookies)?$/i,
          /^accepter tout$/i,
        ];
        const roots = [document];
        let visited = 0;
        while (roots.length && visited < 3500) {
          const root = roots.shift();
          const nodes = root.querySelectorAll('*');
          for (const el of Array.from(nodes).slice(0, 1800)) {
            visited++;
            if (el.shadowRoot) roots.push(el.shadowRoot);
            if (!el.matches('button, [role="button"], input[type="button"], input[type="submit"]')) continue;
            const label = String(el.innerText || el.getAttribute('aria-label') || el.value || '')
              .replace(/\\s+/g, ' ').trim();
            if (!patterns.some(pattern => pattern.test(label))) continue;
            const rect = el.getBoundingClientRect();
            const style = getComputedStyle(el);
            if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
              return el;
            }
          }
        }
        return null;
      });
      try {
        const button = handle.asElement();
        if (button && await button.isIntersectingViewport()) {
          await button.click({delay: 40});
          return {policy, action: 'clicked', frame: frame.url().slice(0, 180), selector: 'open-shadow-cmp-v24'};
        }
      } finally { await handle.dispose(); }
    } catch (_) { /* CMP inaccesible; registrar sin saltar controles */ }
  }
  return { policy, action: 'not_found' };""",
        "shadow consent")
    return s
def sh(cmd):
    return subprocess.run(cmd,text=True,capture_output=True)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--app",type=Path,default=Path.home()/"ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2")
    args=parser.parse_args()
    app=args.app.expanduser().resolve()
    p=app/"capture_dom.js"
    if not p.is_file() or not (app/"run_category.sh").is_file() or not (app/"quality.js").is_file():
        raise RuntimeError("STOP: collector source or quality.js missing")
    active=sh(["docker","ps","--format","{{.Names}}"])
    if active.returncode!=0:raise RuntimeError("STOP: cannot inspect Docker collectors")
    if any(x.startswith(("ltd-c-capture-","ltd-c-recovery-","ltd-c-audit120_")) for x in active.stdout.splitlines()):
        raise RuntimeError("STOP: captures still running")
    cron=sh(["crontab","-l"])
    if cron.returncode not in (0,1):raise RuntimeError("STOP: cannot inspect Cron")
    if "LTD_C_THREE_QA" in cron.stdout or "LTD_C_QA_HOURLY" in cron.stdout:
        raise RuntimeError("STOP: pause Cron before modifying collector")
    original=p.read_text(encoding="utf-8")
    before=hashlib.sha256(original.encode()).hexdigest()
    if before==EXPECTED_V24_SHA:
        print("V24_ALREADY_APPLIED")
        return
    if before!=ORIGINAL_SHA:
        raise RuntimeError("STOP: source hash differs from the audited V2.3 ZIP; do not patch unknown file")
    new=transform(original)
    after=hashlib.sha256(new.encode()).hexdigest()
    if after!=EXPECTED_V24_SHA:
        raise RuntimeError("STOP: patch hash differs from tested V2.4 (programming error)")
    backup=p.with_name("capture_dom.js.v23.original")
    if backup.exists() and hashlib.sha256(backup.read_bytes()).hexdigest()!=ORIGINAL_SHA:
        raise RuntimeError("STOP: existing V2.3 backup conflicts")
    if not backup.exists():shutil.copy2(p,backup)
    temp=p.with_name(".capture_dom.v24.tmp")
    temp.write_text(new,encoding="utf-8")
    os.chmod(temp,p.stat().st_mode & 0o777)
    # Syntax verification runs using Node inside the existing Docker browser image.
    test=sh(["docker","run","--rm","--network","none",
             "--mount",f"type=bind,src={temp},dst=/tmp/capture_dom.js,readonly",
             "--entrypoint","node","tor_1site_noise:final","--check","/tmp/capture_dom.js"])
    if test.returncode!=0:
        temp.unlink(missing_ok=True)
        raise RuntimeError("STOP: Node syntax check failed: "+test.stderr[:350])
    temp.replace(p)
    print("V24_APPLIED",p)
    print("BACKUP_V23",backup)
    print("SHA256",after)
    print("COLLECTOR QA-ONLY: do not merge V2.3 and V2.4 observations as homogeneous samples")
if __name__=="__main__":
    try:main()
    except Exception as e:print("ERROR",e,file=sys.stderr);sys.exit(2)
