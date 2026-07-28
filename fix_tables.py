import sys, zipfile, shutil, os
from lxml import etree

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def q(t): return f'{{{W}}}{t}'
TEXTW=9360  # US Letter, 1" margins

def fix(docx):
    tmp=docx+'.d'
    if os.path.exists(tmp): shutil.rmtree(tmp)
    with zipfile.ZipFile(docx) as z: z.extractall(tmp)
    p=os.path.join(tmp,'word','document.xml')
    tree=etree.parse(p); root=tree.getroot()
    for tbl in root.iter(q('tbl')):
        rows=tbl.findall(q('tr'))
        if not rows: continue
        ncol=max(len(r.findall(q('tc'))) for r in rows)
        # char count per column
        chars=[0]*ncol
        for r in rows:
            for i,tc in enumerate(r.findall(q('tc'))):
                if i>=ncol: break
                chars[i]+=sum(len(t.text or '') for t in tc.iter(q('t')))
        tot=sum(chars) or ncol
        raw=[max(700,int(TEXTW*c/tot)) for c in chars]
        s=sum(raw); widths=[int(w*TEXTW/s) for w in raw]
        widths[-1]+=TEXTW-sum(widths)
        # tblPr: tblW + tblLayout
        pr=tbl.find(q('tblPr'))
        if pr is None:
            pr=etree.SubElement(tbl,q('tblPr')); tbl.insert(0,pr)
        for tag in ('tblW','tblLayout'):
            e=pr.find(q(tag))
            if e is not None: pr.remove(e)
        w=etree.SubElement(pr,q('tblW')); w.set(q('type'),'dxa'); w.set(q('w'),str(TEXTW))
        lay=etree.SubElement(pr,q('tblLayout')); lay.set(q('type'),'fixed')
        # tblGrid
        g=tbl.find(q('tblGrid'))
        if g is not None: tbl.remove(g)
        g=etree.Element(q('tblGrid'))
        for wd in widths:
            gc=etree.SubElement(g,q('gridCol')); gc.set(q('w'),str(wd))
        tbl.insert(list(tbl).index(pr)+1,g)
        # per-cell tcW
        for r in rows:
            for i,tc in enumerate(r.findall(q('tc'))):
                if i>=ncol: break
                tcpr=tc.find(q('tcPr'))
                if tcpr is None:
                    tcpr=etree.Element(q('tcPr')); tc.insert(0,tcpr)
                e=tcpr.find(q('tcW'))
                if e is not None: tcpr.remove(e)
                cw=etree.SubElement(tcpr,q('tcW')); cw.set(q('type'),'dxa'); cw.set(q('w'),str(widths[i]))
    tree.write(p,xml_declaration=True,encoding='UTF-8',standalone=True)
    out="/tmp/"+os.path.basename(docx)
    if os.path.exists(out): os.remove(out)
    base=os.getcwd()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for dp,_,fs in os.walk(tmp):
            for f in fs:
                fp=os.path.join(dp,f)
                z.write(fp,os.path.relpath(fp,tmp))
    shutil.rmtree(tmp)
    print('fixed',docx)

for d in sys.argv[1:]: fix(d)
