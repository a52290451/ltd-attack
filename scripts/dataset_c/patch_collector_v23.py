#!/usr/bin/env python3
"""Parche seguro para el P16C V2.2 instalado en Perseo; sin tocar inventarios ni PCAP."""
from pathlib import Path
import argparse, shutil, os

p = argparse.ArgumentParser()
p.add_argument("--app", required=True, type=Path)
app = p.parse_args().app.expanduser().resolve()
jpath, spath = app/"capture_dom.js", app/"run_category.sh"
assert jpath.is_file() and spath.is_file(), "No se encontró el recolector V2.2"
js, sh = jpath.read_text(), spath.read_text()

def change(s, old, new, why):
    if new in s:
        print("YA APLICADO:",why)
        return s
    if s.count(old) != 1:
        raise RuntimeError("No se modifica nada: anclaje diferente en "+why)
    return s.replace(old,new,1)

js = change(js,
"""  '[data-testid="uc-accept-all-button"]',
];""",
"""  '[data-testid="uc-accept-all-button"]',
  '#acceptAllButton', '#accept-cookies', '.cky-btn-accept',
  '[data-cky-tag="accept-button"]', '.cmplz-accept', '#cmplz-accept',
  '#tarteaucitronAllAllowed', '#axeptio_btn_acceptAll',
  '.fc-cta-consent', '[data-action="accept-all"]',
];""","gestores de cookies")

js = change(js,
"""    try {
      const candidates = await frame.$$('button, [role="button"]');
      for (const button of candidates.slice(0, 90)) {
        const label = await button.evaluate(node => (node.innerText || node.getAttribute('aria-label') || '').trim());
        if (isAcceptAllLabel(label) && await button.isIntersectingViewport()) {
          await button.click({ delay: 40 });
          return { policy, action: 'clicked', frame: frame.url().slice(0, 180), selector: 'visible-label', label: label.slice(0, 60) };
        }
      }
    } catch (_) { /* nunca pulsar controles distintos de aceptar cookies */ }
""",
"""    // Inspección DOM de controles explícitos de consentimiento en un solo viaje CDP.
    try {
      const h = await frame.evaluateHandle(() => {
        const labels = [
          /^aceptar (todas|todo|todas las cookies)$/i,
          /^acepto (todas|todo)$/i,
          /^permitir todas( las cookies)?$/i,
          /^aceptar (y continuar|cookies)$/i,
          /^accept all( cookies)?$/i,
          /^allow all( cookies)?$/i,
          /^agree (to all|and continue)$/i,
          /^tout accepter$/i, /^accepter tout$/i, /^alle akzeptieren$/i,
        ];
        for (const el of Array.from(document.querySelectorAll(
          'button, [role="button"], input[type="button"], input[type="submit"]'
        )).slice(0, 800)) {
          const r = el.getBoundingClientRect();
          const st = getComputedStyle(el);
          if (!r.width || !r.height || st.visibility === 'hidden' || st.display === 'none') continue;
          const label = String(el.innerText || el.getAttribute('aria-label') || el.value || '')
            .replace(/\\s+/g,' ').trim();
          if (labels.some(re => re.test(label))) return el;
        }
        return null;
      });
      try {
        const b = h.asElement();
        if (b && await b.isIntersectingViewport()) {
          const label = await b.evaluate(el => String(
            el.innerText || el.getAttribute('aria-label') || el.value || ''
          ).trim().slice(0,60));
          await b.click({delay:40});
          return {policy, action:'clicked', frame:frame.url().slice(0,180),
            selector:'visible-label-v23', label};
        }
      } finally { await h.dispose(); }
    } catch (_) { /* no tocar otros botones ni superar controles de acceso */ }
""","botones visibles en iframes")

js = change(js,
"""    record.consent_visible = snap.consentVisible;
    if (record.status === 'OK' && outFile !== 'none') {""",
"""    record.consent_visible = snap.consentVisible;
    if (record.status === 'CONSENT_UNRESOLVED' && outFile !== 'none') {
      try {
        const qa = outFile.replace(/\\.html$/i, '.qa.html');
        fs.mkdirSync(path.dirname(qa), {recursive:true});
        fs.writeFileSync(qa, (await page.content()).slice(0,3000000), {mode:0o600});
        record.qa_html = qa;
      } catch (_) { record.errors.push('qa_html_unavailable'); }
    }
    if (record.status === 'OK' && outFile !== 'none') {""","HTML de QA")

sh = change(sh,
""": "${MAX_URLS:=0}" # 0=20; piloto: MAX_URLS=1/2""",
""": "${MAX_URLS:=0}" # 0=20; piloto: MAX_URLS=1/2
: "${SITE_SLOT_FILTER:=}" # slots 1..20 separados por comas""","slots para repetir")

sh = change(sh,
"""  if (( MAX_URLS > 0 && slot_index > MAX_URLS )); then break; fi
  host=""",
"""  if (( MAX_URLS > 0 && slot_index > MAX_URLS )); then break; fi
  if [[ -n "$SITE_SLOT_FILTER" && ",$SITE_SLOT_FILTER," != *",$slot_index,"* ]]; then continue; fi
  host=""","selección de slots")

sh = change(sh,
"""[[ "$MAX_URLS" =~ ^[0-9]+$ && "$MIN_PCAP_BYTES" =~ ^[0-9]+$ && "$MIN_PCAP_PACKETS" =~ ^[0-9]+$ ]] || { echo "ERROR: límites de QA inválidos" >&2; exit 3; }""",
"""[[ "$MAX_URLS" =~ ^[0-9]+$ && "$MIN_PCAP_BYTES" =~ ^[0-9]+$ && "$MIN_PCAP_PACKETS" =~ ^[0-9]+$ ]] || { echo "ERROR: límites de QA inválidos" >&2; exit 3; }
if [[ -n "$SITE_SLOT_FILTER" ]]; then
  [[ "$SITE_SLOT_FILTER" =~ ^([1-9]|1[0-9]|20)(,([1-9]|1[0-9]|20))*$ ]] ||
    { echo "ERROR: SITE_SLOT_FILTER inválido" >&2; exit 3; }
fi""","validación slots")

for path, data in [(jpath,js),(spath,sh)]:
    if path.read_text()==data: continue
    bak = path.with_name(path.name+".v22.backup")
    if not bak.exists(): shutil.copy2(path,bak)
    tmp=path.with_name(path.name+".v23.tmp")
    tmp.write_text(data)
    os.chmod(tmp,path.stat().st_mode & 0o777)
    tmp.replace(path)
    print("ACTUALIZADO:",path)
print("LISTO: node --check capture_dom.js && bash -n run_category.sh")
