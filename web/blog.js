/* Curated, read-only blog examples. Viewing never runs inference or saves a record. */
'use strict';
const PlateauBlog = (() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const asset = name => {
    if(!/^blog-record-[a-z0-9-]+\.json$/.test(name))throw new Error('Invalid blog asset');
    return name;
  };
  function validateCatalog(data) {
    if(data?.version!==1 || !Array.isArray(data.examples) || !data.examples.length)throw new Error('Invalid blog catalog');
    const ids=new Set();
    for(const entry of data.examples){
      if(!/^[a-z0-9-]+$/.test(entry.id) || ids.has(entry.id) || !entry.title || entry.kind!=='record')throw new Error('Invalid blog example');
      ids.add(entry.id);asset(entry.record_file);
    }
    return data.examples;
  }
  function linkedId(hash) { return new URLSearchParams(String(hash || '').replace(/^#/, '')).get('example'); }
  function shareUrl(href,id) {
    const url=new URL(href);url.hash=new URLSearchParams({example:id}).toString();return url.href;
  }
  function mount({document,window,fetch,normalize,chart,open,notify}) {
    const $=id=>document.getElementById(id);
    let entries=[],selected=null,record=null,readouts=[],generation=0,ready=false,busy=false;
    const cache=new Map();
    const message=text=>{$('blog-message').textContent=text;};
    const updateButton=()=>{
      $('blog-open').disabled=!record || !ready || busy;
      $('blog-open').title=!ready?'The workbench is connecting. The preview and download are already available.':busy?'Wait for the current experiment to finish.':'';
    };
    const pairHtml=pairs=>pairs.map(([a,b])=>`<div class="blog-pair"><p><b>A</b><span>${escape(a)}</span></p><p><b>B</b><span>${escape(b)}</span></p></div>`).join('');
    function draw() {
      const curve=readouts.find(row=>row.key===$('blog-readout').value);
      if(!curve)return;
      const metric=$('blog-metric').value || 'c';
      $('blog-visual').innerHTML=`<div class="blog-chart-label">${escape(curve.title)} · ${metric}(t)</div>${chart(curve,metric)}<p class="blog-chart-caption">Original measured samples · dashed line: ${metric}(t) = t</p>`;
      $('blog-readout-note').textContent=curve.key==='delayed_logits'
        ? 'Delayed readout: append “ is” and measure the following-token logits. Only the original country position is patched; the context is fixed to A. These are separate measurements from the main Workbench curves.'
        : 'Readout at the final input-token position. Values are loaded from the saved experiment.';
      if(curve.predictions){
        const segments=[];
        for(const p of curve.predictions){
          if(segments.at(-1)?.token===p.token)segments.at(-1).end=p.t;
          else segments.push({token:p.token,start:p.t,end:p.t});
        }
        $('blog-readout-note').textContent+=' Top-1: '+segments.map(s=>`${JSON.stringify(s.token.trim())} at t=${s.start.toFixed(2)}–${s.end.toFixed(2)}`).join('; ')+'.';
      }
    }
    async function json(path) {
      const response=await fetch(path);
      if(!response.ok)throw new Error('Could not load '+path);
      return response.json();
    }
    async function select(id,{link=true}={}) {
      const request=++generation;
      selected=entries.find(entry=>entry.id===id) || null;record=null;readouts=[];updateButton();
      $('blog-detail').classList.toggle('hidden',!selected);
      $('blog-link-field').classList.add('hidden');
      $('blog-open').classList.add('hidden');$('blog-download').classList.add('hidden');
      $('blog-readout-controls').classList.add('hidden');$('blog-readout-note').textContent='';
      $('blog-visual').innerHTML='';$('blog-prompts').innerHTML='';$('blog-metadata').textContent='';
      $('blog-select').value=selected?.id || '';
      if(!selected){
        if(link && !id){try{const url=new URL(window.location.href);url.hash='';window.history.replaceState(null,'',url.href);}catch{}}
        message(id?'This blog example was not found. Choose an example from the list.':'');return;
      }
      const entry=selected;
      if(link){try{window.history.replaceState(null,'',shareUrl(window.location.href,entry.id));}catch{}}
      $('blog-title').textContent=entry.title;
      $('blog-reference').textContent=entry.reference;
      $('blog-kind').textContent=entry.provenance==='rerun'?'Measured rerun':'Saved experiment';
      $('blog-summary').textContent=entry.summary;
      $('blog-note').textContent=entry.note || 'Original measured results. Open the saved experiment to inspect tokens, c(t), d(t), and L2 distances. Viewing needs no model download or API key.';
      message('Loading saved results…');
      try{
        const original=cache.has(entry.id)?cache.get(entry.id):await json(asset(entry.record_file));
        if(request!==generation)return;
        const view=normalize(original);
        if(original.id!==entry.source_record_id)throw new Error('Saved record does not match the catalog');
        const curve=view.curves.find(row=>row.key==='logits');
        if(!curve?.c?.every(Number.isFinite))throw new Error('This example has no recorded c(t) curve');
        cache.set(entry.id,original);record=original;
        $('blog-prompts').innerHTML=pairHtml([[view.sequence_a,view.sequence_b]]);
        $('blog-metadata').textContent=`${view.model_label || view.model} · ${view.settings.patch_layer===-1?'Embedding':'After layer '+view.settings.patch_layer} · ${view.settings.interpolation.toUpperCase()} · ${view.settings.steps} samples · ${view.settings.patch_position==='different_suffix'?'First difference → end':'Final token only'}${entry.answer?' · Shared answer: '+entry.answer:''}`;
        readouts=[...view.curves,...Object.values(original.blog_readouts || {})];
        $('blog-readout').innerHTML=readouts.map(row=>`<option value="${escape(row.key)}">${escape(row.title)}</option>`).join('');
        $('blog-readout').value=entry.preview_readout || 'logits';$('blog-metric').value='c';
        $('blog-readout-controls').classList.remove('hidden');draw();
        $('blog-open').classList.remove('hidden');updateButton();
        $('blog-download').href=asset(entry.record_file);$('blog-download').download=entry.record_file;
        $('blog-download').classList.remove('hidden');
        message('Saved results ready. No inference was run.');
      }catch(error){
        if(request===generation){record=null;updateButton();message('Could not display this example: '+error.message+'. Select it again to retry.');}
      }
    }
    $('blog-select').addEventListener('change',()=>select($('blog-select').value));
    $('blog-readout').addEventListener('change',draw);$('blog-metric').addEventListener('change',draw);
    $('blog-open').onclick=()=>{
      if(!record || !ready || busy)return;
      // Give the workbench its own copy so annotations cannot modify bundled examples.
      open(JSON.parse(JSON.stringify(record)),selected);
    };
    $('blog-copy-link').onclick=async()=>{
      if(!selected)return;
      const url=shareUrl(window.location.href,selected.id);
      try{await window.navigator.clipboard.writeText(url);notify('Blog example link copied.');}
      catch{$('blog-link').value=url;$('blog-link-field').classList.remove('hidden');$('blog-link').focus();$('blog-link').select();}
    };
    $('nav-blog').onclick=()=>{
      $('blog-examples').scrollIntoView({behavior:'smooth',block:'start'});$('blog-select').focus();
    };
    window.addEventListener('hashchange',()=>select(linkedId(window.location.hash),{link:false}));
    const loaded=(async()=>{
      try{
        entries=validateCatalog(await json('blog-examples.json'));
        $('blog-select').innerHTML='<option value="">Choose a blog example…</option>'+entries.map(entry=>`<option value="${entry.id}">${escape(entry.reference+' · '+entry.title)}</option>`).join('');
        $('blog-select').disabled=false;message('');
        const id=linkedId(window.location.hash);
        if(id)await select(id,{link:false});
      }catch(error){message('Blog examples could not load. Reload the page to retry.');}
    })();
    return {loaded,select,setReady(value){ready=value;updateButton();},setBusy(value){busy=value;updateButton();}};
  }
  return {mount,validateCatalog,linkedId,shareUrl};
})();
if(typeof module!=='undefined')module.exports=PlateauBlog;
