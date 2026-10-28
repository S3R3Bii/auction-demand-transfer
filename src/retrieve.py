"""Bounded official aggregate retrieval; raw responses are not redistributed."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
ENDPOINT='https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query'
FIELDS=['record_date','cusip','security_type','security_term','original_security_term','auction_date','announcemt_date','issue_date','maturity_date','bid_to_cover_ratio','offering_amt','reopening','floating_rate','inflation_index_security','total_tendered','total_accepted','soma_tendered','soma_accepted']

def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True);records=[];rows=0;pages=1
    existing=ROOT/'data/manifest.json';old=json.loads(existing.read_text()) if existing.exists() else None
    for page in range(1,9):
        params={'fields':','.join(FIELDS),'filter':'auction_date:gte:2009-01-01,auction_date:lte:2026-09-30','sort':'auction_date,cusip','page[size]':1000,'page[number]':page}
        url=ENDPOINT+'?'+urllib.parse.urlencode(params);target=raw/f'auctions-{page}.json'
        if not target.exists():
            with urllib.request.urlopen(url,timeout=45) as response:target.write_bytes(response.read())
        blob=target.read_bytes();obj=json.loads(blob);pages=int(obj['meta']['total-pages']);assert pages<=8,'Bounded retrieval exceeded'
        assert set(obj['data'][0])==set(FIELDS);rows+=len(obj['data']);records.append(dict(file=f'data/raw/auctions-{page}.json',url=url,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),rows=len(obj['data'])))
        if page==pages:break
    assert rows==int(obj['meta']['total-count'])
    manifest=dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),publisher='U.S. Department of the Treasury, Bureau of the Fiscal Service',dataset='Treasury Securities Auctions Data',fields=FIELDS,labels=obj['meta']['labels'],types=obj['meta']['dataTypes'],formats=obj['meta']['dataFormats'],rows=rows,pages=pages,files=records,raw_redistributed=False,revision_scope='Current historical response; original publication vintages unverified')
    if old:
        assert old['files']==records,'Source snapshot changed; explicit review required'
    else:existing.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':rows,'pages':pages,'raw_files_kept_local':True,'model_results_not_inspected':True}))

if __name__=='__main__':main()
