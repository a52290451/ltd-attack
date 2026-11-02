#!/usr/bin/env python3
"""Build an isolated QA V2.5 consent experiment from the verified V2.4 source.

Never edits canonical collector, A/B, baseline QA, Tor, cron or models.
Uses only ordinary bsierra permissions. No sudo.
"""
import hashlib
import shutil
from pathlib import Path

SRC=Path.home()/"ltd-ops/collector-qa-recovery-v1"
DST=Path.home()/"ltd-ops/collector-qa-recovery-v25"
BASE_SHA="a4196d17ecc2b5dc52957900a1763a6985860a8c502da5ad3851ec656bd64eeb"
TARGET_SHA="1bc06ede565bc3dc16a79900424f32e12bca4da8f3fa7a73667616bdde786060"

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 source=SRC/"capture_dom.js"
 quality=SRC/"quality.js"
 if not source.is_file() or sha(source)!=BASE_SHA or not quality.is_file():
  raise SystemExit("BLOCKED: V2.4 source missing/changed")
 original=source.read_text()
 s=original
 def change(old,new):
  nonlocal s
  if s.count(old)!=1:raise RuntimeError("AMBIGUOUS_BASELINE_PATCH "+repr(old[:85]))
  s=s.replace(old,new)
 change("          /^estoy de acuerdo$/i, /^acepto$/i,",
        "          /^estoy de acuerdo$/i, /^acepto$/i, /^aceptar$/i,")
 change("          /^estoy de acuerdo$/i,\n          /^accepter tout$/i,",
        "          /^estoy de acuerdo$/i, /^aceptar$/i,\n          /^accepter tout$/i,")
 change('          \'button, [role="button"], input[type="button"], input[type="submit"]\'\n',
        '          \'button, [role="button"], input[type="button"], input[type="submit"], a[role="button"], a[href="#"], a[href^="javascript:"], a:not([href])\'\n')
 change("            if (!patterns.some(pattern => pattern.test(label))) continue;", """            if (!patterns.some(pattern => pattern.test(label))) continue;
            if (/^aceptar$/i.test(label)) {
              let ancestor=el, cookieContext=false;
              for (let j=0; j<6 && ancestor; j++) {
                const attrs = [ancestor.id, ancestor.className, ancestor.getAttribute?.('aria-label')]
                  .map(x => typeof x === 'string' ? x : '').join(' ');
                const vicinity = String(ancestor.innerText || '').slice(0, 1000);
                if (/cookie|consent|privacidad/i.test(attrs+' '+vicinity)) {cookieContext=true; break;}
                ancestor=ancestor.parentElement || ancestor.getRootNode?.().host;
              }
              if (!cookieContext) continue;
            }""")
 change("          if (labels.some(re => re.test(label))) return el;", """          if (!labels.some(re => re.test(label))) continue;
          // Standalone ACEPTAR is only eligible inside an explicit cookie context.
          if (/^aceptar$/i.test(label)) {
            let ancestor=el, cookieContext=false;
            for (let j=0; j<6 && ancestor; j++) {
              const attrs=[ancestor.id, ancestor.className, ancestor.getAttribute?.('aria-label')]
                .map(x=>typeof x==='string'?x:'').join(' ');
              const vicinity=String(ancestor.innerText || '').slice(0,1000);
              if (/cookie|consent|privacidad/i.test(attrs+' '+vicinity)) {cookieContext=true;break;}
              ancestor=ancestor.parentElement || ancestor.getRootNode?.().host;
            }
            if (!cookieContext) continue;
          }
          return el;""")
 change("    record.consent_visible = snap.consentVisible;", """    record.consent_visible = snap.consentVisible;
    // QA: never silently accept an unresolved explicit all-cookie control.
    if (record.status === 'OK' && consentPolicy === 'accept_all' &&
        record.consent?.action === 'not_found' &&
        (record.consent.qa_control_labels || []).some(label =>
          /^(aceptar todas las cookies|aceptar)$/i.test(String(label).trim()))) {
      record.status = 'CONSENT_UNRESOLVED';
      record.errors.push('qa_explicit_consent_control_unresolved');
    }""")
 if hashlib.sha256(s.encode()).hexdigest()!=TARGET_SHA:
  raise SystemExit("BLOCKED: output SHA mismatch (unexpected source bytes)")
 DST.mkdir(parents=True,exist_ok=True,mode=0o755)
 target=DST/"capture_dom.js"
 if target.exists() and sha(target)!=TARGET_SHA:
  raise SystemExit("BLOCKED: existing V2.5 output has different bytes")
 target.write_text(s)
 shutil.copyfile(quality,DST/"quality.js")
 DST.chmod(0o755)
 for item in (target,DST/"quality.js"):item.chmod(0o644)
 print("QA25_PREPARED",target)
 print("QA25_SHA256",sha(target))
 print("QA25_POLICY standard_explicit_consent_only_no_login_or_challenge_bypass")
 print("BASELINE_COLLECTOR=UNMODIFIED")
 print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
if __name__=="__main__":main()
