import asyncio
import aiohttp
import json
import random
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42  # determinismo na detecção de idioma

# ============================================================
# CONFIGURAÇÃO — ajuste aqui para escalar
# ============================================================
TOTAL_SITES_PER_PASS   = 1000     # meta de sites pt-BR válidos
OVERSAMPLE_FACTOR      = 2.5       # busca ~25k URLs para garantir 10k
MAX_CONCURRENT_REQUESTS = 50
MAX_CONCURRENT_CDX      = 8
REQUEST_TIMEOUT         = 20
MAX_TEXT_LENGTH         = 8000
MIN_TEXT_LENGTH         = 200
MAX_PER_DOMAIN          = 3
BATCH_SIZE              = 200
CDX_API_BASE            = "https://index.commoncrawl.org"
OUTPUT_FILE = f"dataset_ptbr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

HEADERS = {
    "User-Agent": "PTBR-WebDatasetBot/1.0 (Research; contact: seu-email@exemplo.com)",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Domínios pt-BR conhecidos (usados como sementes no Common Crawl)
PT_BR_DOMAINS = [
    "globo.com", "g1.globo.com", "uol.com.br", "folha.uol.com.br",
    "estadao.com.br", "terra.com.br", "r7.com", "ig.com.br",
    "cnnbrasil.com.br", "bbc.com", "tecmundo.com.br", "canaltech.com.br",
    "techtudo.com.br", "olhardigital.com.br", "brasilescola.uol.com.br",
    "mundoeducacao.uol.com.br", "significados.com.br", "todamateria.com.br",
    "super.abril.com.br", "veja.abril.com.br", "exame.com",
    "infomoney.com.br", "valor.globo.com", "agenciabrasil.ebc.com.br",
    "gov.br", "usp.br", "unicamp.br", "ufrj.br", "ufmg.br", "unb.br",
    "tabnews.com.br", "nucleo.jor.br", "intercept.com.br",
    "cartacapital.com.br", "gazetadopovo.com.br", "correiobraziliense.com.br",
    "jornal.usp.br", "saude.gov.br", "mec.gov.br", "inep.gov.br",
    "ibge.gov.br", "ipea.gov.br", "scielo.br", "fiocruz.br",
]

# ============================================================
# ETAPA 1 — DESCOBERTA DE URLs (3 fases, prioridade pt-BR)
# ============================================================
async def get_latest_index(session):
    try:
        async with session.get(f"{CDX_API_BASE}/collinfo.json", timeout=15) as resp:
            data = await resp.json()
            return data[0]["id"]
    except Exception:
        return "CC-MAIN-2024-10"


async def cdx_query(session, index, params, semaphore):
    """Consulta única ao índice CDX, respeitando concorrência."""
    async with semaphore:
        try:
            await asyncio.sleep(random.uniform(0.1, 0.6))
            async with session.get(
                f"{CDX_API_BASE}/{index}", params=params, timeout=40
            ) as resp:
                if resp.status != 200:
                    return []
                urls = []
                for line in (await resp.text()).strip().split("\n"):
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("url"):
                            urls.append(d["url"])
                    except json.JSONDecodeError:
                        continue
                return urls
        except Exception:
            return []


async def discover_urls(session, target_count):
    """Descobre URLs priorizando conteúdo pt-BR."""
    index = await get_latest_index(session)
    print(f"[*] Índice Common Crawl: {index}")
    print(f"[*] Meta de descoberta: {target_count} URLs")

    urls = set()
    cdx_sem = asyncio.Semaphore(MAX_CONCURRENT_CDX)

    # ----- FASE 1: domínios pt-BR conhecidos (matchType=domain) -----
    print("[*] Fase 1: coletando de domínios pt-BR conhecidos...")
    phase1_tasks = [
        cdx_query(session, index, {
            "url": d, "matchType": "domain", "output": "json",
            "limit": 300, "fl": "url",
        }, cdx_sem)
        for d in PT_BR_DOMAINS
    ]
    for batch in await asyncio.gather(*phase1_tasks):
        urls.update(batch)
    print(f"[+] Fase 1: {len(urls)} URLs.")

    # ----- FASE 2: prefixos comuns dentro dos domínios pt-BR -----
    print("[*] Fase 2: expandindo por caminhos (prefix) em pt-BR...")
    path_prefixes = ["/blog", "/noticias", "/artigos", "/wiki", "/educacao",
                     "/tecnologia", "/ciencia", "/saude", "/economia", "/esportes"]
    phase2_tasks = []
    for domain in random.sample(PT_BR_DOMAINS, min(20, len(PT_BR_DOMAINS))):
        prefix = random.choice(path_prefixes)
        phase2_tasks.append(cdx_query(session, index, {
            "url": f"{domain}{prefix}", "matchType": "prefix",
            "output": "json", "limit": 150, "fl": "url",
        }, cdx_sem))
    for batch in await asyncio.gather(*phase2_tasks):
        urls.update(batch)
    print(f"[+] Total após Fase 2: {len(urls)} URLs.")

    # ----- FASE 3: amostragem aleatória (garante volume) -----
    if len(urls) < target_count:
        print("[*] Fase 3: preenchendo com amostragem aleatória ampla...")
        seeds = ["http://www.", "https://www.", "https://blog.",
                 "https://noticias.", "https://portal."]
        while len(urls) < target_count:
            prefix = random.choice(seeds) + random.choice("abcdefghijklmnopqrstuvwxyz")
            batch = await cdx_query(session, index, {
                "url": prefix, "output": "json", "limit": 100, "fl": "url",
            }, cdx_sem)
            urls.update(batch)
            if len(urls) % 1000 < 100:
                print(f"    ... {len(urls)} URLs acumuladas")

    result = list(urls)
    random.shuffle(result)  # diversidade no pipeline
    print(f"[+] Total descoberto: {len(result)} URLs.")
    return result[:target_count]


# ============================================================
# ETAPA 2 — DETECÇÃO pt-BR (4 camadas)
# ============================================================
def is_pt_br(text, html_lang, content_language, domain):
    """
    Retorna True se o conteúdo for provavelmente português (BR ou PT).
    1) Atributo <html lang="pt...">
    2) Header Content-Language
    3) TLD .br + detecção de idioma
    4) Detecção de idioma genérica
    """
    if html_lang and html_lang.lower().startswith("pt"):
        return True
    if content_language and "pt" in content_language.lower():
        return True
    if len(text) < 100:
        return False
    try:
        lang = detect(text)
    except LangDetectException:
        return False
    if domain.endswith(".br"):
        return lang == "pt"
    return lang == "pt"


# ============================================================
# ETAPA 3 — EXTRAÇÃO ASSÍNCRONA
# ============================================================
async def extract_content(session, url, semaphore, domain_counter):
    async with semaphore:
        try:
            await asyncio.sleep(random.uniform(0.3, 1.5))

            async with session.get(
                url, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                if resp.status != 200:
                    return None
                ct = resp.headers.get("Content-Type", "")
                if "text/html" not in ct:
                    return None

                html = await resp.text(errors="ignore")
                soup = BeautifulSoup(html, "lxml")

                html_lang = soup.html.get("lang") if soup.html else None
                content_language = resp.headers.get("Content-Language", "")

                for tag in soup(["script", "style", "nav", "footer",
                                 "header", "aside", "form", "iframe", "noscript"]):
                    tag.decompose()

                title = (soup.title.string.strip()
                         if soup.title and soup.title.string else "")
                text = " ".join(soup.get_text(separator=" ", strip=True).split())

                if len(text) < MIN_TEXT_LENGTH:
                    return None
                text = text[:MAX_TEXT_LENGTH]

                domain = urlparse(url).netloc
                if domain_counter.get(domain, 0) >= MAX_PER_DOMAIN:
                    return None

                if not is_pt_br(text, html_lang, content_language, domain):
                    return None

                domain_counter[domain] = domain_counter.get(domain, 0) + 1

                return {
                    "url": url,
                    "domain": domain,
                    "title": title,
                    "text": text,
                    "lang": "pt-BR",
                    "source": "common_crawl",
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "char_count": len(text),
                    "html_lang": html_lang,
                    "content_language": content_language,
                }
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return None
        except Exception:
            return None


async def process_batch(urls, batch_id, domain_counter):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [extract_content(session, u, semaphore, domain_counter) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, dict)]
    print(f"[Lote {batch_id:>3}] {len(valid)}/{len(urls)} válidos pt-BR.")
    return valid


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
async def main():
    start = time.time()
    target = TOTAL_SITES_PER_PASS
    discovery_target = int(target * OVERSAMPLE_FACTOR)

    print("=" * 60)
    print(f"🎯 Meta: {target} sites pt-BR válidos")
    print(f"🔎 URLs a descobrir: ~{discovery_target}")
    print(f"💾 Saída: {OUTPUT_FILE}")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        urls = await discover_urls(session, discovery_target)

    if not urls:
        print("[!] Nenhuma URL descoberta — verifique conexão ou índice.")
        return

    domain_counter = {}
    total_valid = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(0, len(urls), BATCH_SIZE):
            if total_valid >= target:
                print(f"[✓] Meta de {target} atingida — encerrando extração.")
                break
            batch = urls[i:i+BATCH_SIZE]
            results = await process_batch(batch, i // BATCH_SIZE + 1, domain_counter)
            for item in results:
                if total_valid >= target:
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_valid += 1
            f.flush()
            await asyncio.sleep(0.5)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"✅ Concluído em {elapsed/60:.2f} minutos")
    print(f"✅ Sites pt-BR no dataset : {total_valid}")
    print(f"✅ Domínios únicos        : {len(domain_counter)}")
    print(f"✅ Arquivo                : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())