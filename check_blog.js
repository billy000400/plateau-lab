/* Verify real bundled results, read-only previews, deep links, and private copies. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const Blog=require('./web/blog.js'),Records=require('./web/records.js');
const {context,elements,document,run}=require('./check_dom.js').create();
const catalog=JSON.parse(fs.readFileSync('web/blog-examples.json','utf8'));
const entries=Blog.validateCatalog(catalog);
assert.equal(entries.length,6);
assert(entries.every(e=>e.kind==='record'),'every example must contain measured data');
assert.throws(()=>Blog.validateCatalog({...catalog,examples:[{...entries[0],record_file:'../data/private.json'}]}));
assert.throws(()=>Blog.validateCatalog({...catalog,examples:[entries[0],entries[0]]}));
const layers={'copying-vs-recall':10,'opposite-counting':3,'hamlet-macbeth':0};
for(const entry of entries){
  const content=fs.readFileSync('web/'+entry.record_file);
  assert.equal(crypto.createHash('sha256').update(content).digest('hex'),entry.sha256);
  const original=JSON.parse(content),view=Records.normalize(original);
  assert.equal(original.id,entry.source_record_id);
  if(entry.id in layers){
    assert.equal(original.model,'google/gemma-2-9b-it');
    assert.equal(original.settings.patch_layer,layers[entry.id]);
    assert.equal(original.settings.patch_position,'different_suffix');
    assert.equal(original.settings.steps,41);
  }else{
    assert.equal(original.settings.steps,101);
    assert.equal(original.blog_reproduction.example,entry.id);
    assert(original.model_revision);
  }
  assert.equal(original.settings.interpolation,'slerp');
  assert.deepEqual(view.curves,original.curves);
  assert.equal(view.curves.find(c=>c.key==='logits').c.length,original.settings.steps);
  for(const curve of [...view.curves,...Object.values(original.blog_readouts || {})]){
    assert.equal(curve.t.length,original.settings.steps);
    assert(curve.c.every(Number.isFinite));
    assert.equal(curve.c[0],0);assert.equal(curve.c.at(-1),1);
    assert(curve.c.every((v,i)=>!i || v>=curve.c[i-1]));
    const total=curve.step_lengths.reduce((sum,v)=>sum+v,0);
    assert(Math.abs(total-curve.total_length)<1e-8*Math.max(1,total));
  }
  if(entry.id.startsWith('japan-germany')){
    assert.equal(entry.preview_readout,'delayed_logits');
    assert.equal(original.blog_readouts.delayed_logits.forced_continuation,' is');
    assert.equal(original.blog_readouts.delayed_logits.predictions[0].token,' Tokyo');
    assert.equal(original.blog_readouts.delayed_logits.predictions.at(-1).token,' Berlin');
    assert(original.path_predictions.every(p=>p.token_matrix.steps.length===3));
    if(entry.id==='japan-germany-gpt2-block0'){
      const delayed=original.blog_readouts.delayed_logits;
      assert(Math.abs(delayed.transition_width_sample_grid-.26)<1e-6);
      assert(Math.abs(delayed.transition_width-.2613)<1e-4);
      assert.equal(delayed.predictions.find(p=>p.token===' Berlin').t,.5);
      assert(Math.abs(delayed.natural_endpoint_jsd_bits.immediate-.0014)<.00005);
      assert(Math.abs(delayed.natural_endpoint_jsd_bits.delayed-.9945)<.00005);
    }
  }
}
const location={href:'https://example.test/lab/#example=opposite-counting',hash:'#example=opposite-counting'};
const handlers={},requests=[],opened=[],copied=[];
const window={location,history:{replaceState(_,__,href){location.href=href;location.hash=new URL(href).hash;}},
  navigator:{clipboard:{async writeText(value){copied.push(value);}}},addEventListener(name,fn){handlers[name]=fn;}};
let delayed=null,fail=false;
const fetch=async path=>{
  requests.push(path);
  assert.match(path,/^blog-[a-z0-9-]+\.json$/,'previews only request static public files');
  if(fail)throw Error('Offline');
  if(delayed && path.includes('copying-vs-recall'))await delayed.promise;
  return {ok:true,json:async()=>JSON.parse(fs.readFileSync('web/'+path,'utf8'))};
};
const gallery=Blog.mount({document,window,fetch,normalize:Records.normalize,
  chart:(curve,metric)=>{context.curve=curve;context.metric=metric;return run('curveSvg(curve,false,metric)');},notify(){},
  open(record){opened.push(record);context.opened=record;run('renderResult(opened)');}});
(async()=>{
  await gallery.loaded;
  assert.equal(elements.get('blog-select').value,'opposite-counting');
  assert.match(elements.get('blog-metadata').textContent,/After layer 3/);
  assert.match(elements.get('blog-visual').innerHTML,/polyline/);
  assert(elements.get('blog-open').disabled,'preview works before the API is connected');
  elements.get('blog-open').onclick();assert.equal(opened.length,0);
  gallery.setReady(true);gallery.setBusy(true);
  assert(elements.get('blog-open').disabled);elements.get('blog-open').onclick();assert.equal(opened.length,0);
  gallery.setBusy(false);elements.get('blog-open').onclick();assert.equal(opened.length,1);
  assert.equal(run('result.settings.patch_layer'),3);
  assert.match(elements.get('l2-body').innerHTML,/Source L2/);
  // An annotated copy belongs only to the reader's collection.
  run('localLibrary=false;remoteMode=true;api=async()=>{throw Error("No API requests expected");};');
  elements.get('notes').value='My private observation';await run('save()');
  assert.equal(run('library.examples.length'),1);
  assert.equal(run('library.history.length'),0);
  assert.equal(run('library.examples[0].notes'),'My private observation');
  elements.get('blog-open').onclick();assert.equal(opened.at(-1).notes,'counting up vs. counting down');
  await elements.get('blog-copy-link').onclick();
  assert.equal(Blog.linkedId(new URL(copied[0]).hash),'opposite-counting');
  assert.equal(new URL(copied[0]).pathname,'/lab/');
  for(const id of ['copying-vs-recall','hamlet-macbeth']){
    await gallery.select(id);elements.get('blog-open').onclick();
    assert.equal(run('result.settings.patch_layer'),layers[id]);
    assert.equal(elements.get('blog-download').download,`blog-record-${id}.json`);
  }
  for(const id of ['big-in','big-large','japan-germany-gpt2-block0']){
    await gallery.select(id);
    assert(!elements.get('blog-open').classList.contains('hidden'));
    assert(!elements.get('blog-download').classList.contains('hidden'));
    assert.match(elements.get('blog-visual').innerHTML,/polyline/);
    assert(!elements.get('blog-visual').innerHTML.includes('<img'));
  }
  assert.equal(elements.get('blog-readout').value,'delayed_logits');
  assert.match(elements.get('blog-readout-note').textContent,/Tokyo.*Berlin/);
  elements.get('blog-metric').value='d';elements.get('blog-metric').handlers.change();
  assert.match(elements.get('blog-visual').innerHTML,/relative endpoint distance/);
  elements.get('blog-readout').value='logits';elements.get('blog-readout').handlers.change();
  assert.match(elements.get('blog-readout-note').textContent,/final input-token/);
  await gallery.select('missing');assert.match(elements.get('blog-message').textContent,/not found/);
  await gallery.select('');assert.equal(location.hash,'');
  // A late response must not replace a more recently selected record.
  const pending={};pending.promise=new Promise(resolve=>pending.resolve=resolve);delayed=pending;
  const other=Blog.mount({document,window,fetch,normalize:Records.normalize,chart:()=>'<svg></svg>',open(){},notify(){}});
  await other.loaded;
  const waiting=other.select('copying-vs-recall');await other.select('hamlet-macbeth');pending.resolve();await waiting;
  assert.equal(elements.get('blog-title').textContent,'Different factual associations');
  assert(!elements.get('blog-open').classList.contains('hidden'));
  delayed=null;fail=true;await other.select('copying-vs-recall');
  assert.match(elements.get('blog-message').textContent,/Could not display/);
  assert(elements.get('blog-open').disabled);
  fail=false;await other.select('copying-vs-recall');assert.match(elements.get('blog-message').textContent,/ready/);
  assert(requests.every(path=>!path.startsWith('/api/')));
  console.log('Blog checks passed: source hashes, exact curves/layers, offline preview, private copy, deep links, race and error handling.');
})().catch(error=>{console.error(error);process.exitCode=1;});
