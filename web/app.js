'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let modelLayers = {'gpt2-large':36, 'gpt2':12, 'gpt2-medium':24, 'gpt2-xl':48};
let modelCatalog = [];
const API=(window.PLATEAU_API || '').replace(/\/$/,'');
let localLibrary=true, remoteMode=false, previewAvailable=true, resultSource=null, tokenizeTimer, tokenizeRequest=0;
const browserCollections=PlateauCollections.create();
let diskLibrary={history:[],examples:[]};
let legacyEmbeddingResult = false;
const presets = [
  ['The house was big','The house was in'],
  ['The capital of France','The capital of Germany'],
  ['She was very happy','She was very sad'],
  ['The weather was hot','The weather was cold'],
  ['He opened the door','He opened the book'],
  ['I would like some coffee','After a long walk, she asked for tea'],
  ['The answer is yes','The answer is no'],
  ['The movie was good','The movie was bad'],
  ['The capital of France is Paris. The capital of Japan is','The capital of France is Paris. The capital of Germany is'],
];
let result = null, busy = false, currentJob = null, presetIndex = 3;
let library = {examples:[], history:[]}, scope = 'examples';
const selectedRecords = {examples:new Set(), history:new Set()};
let exporting = false;
let toastTimer;
function formatDate(value, full=false) {
  const iso = new Date(value).toISOString();
  return full ? iso.slice(0, 19).replace('T', ' ') + ' UTC' : iso.slice(0, 10);
}
async function api(path, body, headers={}) {
  const response = await fetch(API+path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed. Please try again.');
  return data;
}
function toast(text) { $('toast').textContent=text; $('toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),3500); }
function status(text, progress=0, error=false) { $('status').classList.remove('hidden'); $('status').classList.toggle('error',error); $('status').classList.toggle('indeterminate',progress===null); $('status-text').textContent=text; $('progress').style.width=progress===null?'30%':`${Math.max(0,Math.min(1,progress))*100}%`; }
function modelNote() {
  const model=modelCatalog.find(m=>m.id===$('model').value);
  if(!model){$('model-storage').textContent='This saved model is not currently available. Choose an available model to rerun.';return;}
  if(remoteMode){$('model-storage').textContent=model.running===false?'This model is not currently running on NDIF. Choose another model or try again later.':'Inference runs on NDIF. Model availability and precision depend on its current deployments.';return;}
  $('model-storage').textContent=(model.cached
    ? `${model.label} is saved on this computer. Future visits load it from disk without downloading again.`
    : `${model.label} downloads once${model.download_gb ? ` (about ${model.download_gb} GB)` : ''}, then stays saved on this computer.`)+
    (model.memory_note ? ' '+model.memory_note : '');
}
function showHardware(hardware) {
  $('hardware-badge').textContent=hardware.backend==='CPU'?'CPU':'GPU';
  $('hardware-badge').title=`${hardware.name} (${hardware.backend})`;
  $('hardware-status').textContent=`Compute: ${hardware.name} · ${hardware.backend} · float32 · ${hardware.selection==='auto'?'automatically selected':'manual selection'}`+
    (hardware.notes.length ? '. '+hardware.notes.join(' ') : '');
}
async function refreshHardware() {
  try { showHardware(await api('/api/hardware')); }
  catch(error){$('hardware-status').textContent='Hardware detection: '+error.message;}
}
async function refreshModels() {
  const config=await api('/api/config');
  const selected=$('model').value;
  remoteMode=!!config.remote;localLibrary=config.local_library ?? !remoteMode;
  previewAvailable=config.token_preview!==false;requiresKey=!!config.requires_key;sharedAccess=!!config.shared_access;
  modelCatalog=config.models;
  modelLayers=Object.fromEntries(modelCatalog.map(m=>[m.id,m.layers]));
  $('model').innerHTML=[...new Set(modelCatalog.map(m=>m.family))].map(family=>
    `<optgroup label="${esc(family)}">`+modelCatalog.filter(m=>m.family===family).map(m=>
      `<option value="${esc(m.id)}">${esc(m.label)}${remoteMode?(m.running===false?' · Not running':''):(m.cached?' · Saved':' · Download')}</option>`).join('')+'</optgroup>').join('');
  $('model').value=modelLayers[selected] ? selected : (config.default_model || modelCatalog[0]?.id || '');
  $('connection-settings').classList.toggle('hidden',!remoteMode);
  $('code-note').classList.toggle('hidden',!sharedAccess);
  $('key-label').textContent=sharedAccess?'NDIF API key or lab access code':'NDIF API key';
  $('ndif-key').placeholder=sharedAccess?'Your key or the lab code':'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';
  $('workspace-name').textContent=remoteMode?'Hosted workspace':'Local workspace';
  $('storage-note').textContent=localLibrary?'Saved on this computer.':'Private collections in this browser.';
  if(remoteMode){$('hardware-badge').textContent='NDIF';$('hardware-status').textContent='Compute: NDIF remote inference · model precision is recorded with each result.';}
  else refreshHardware();
  modelNote();
  keyState();
  return config;
}
function settings() { return {model:$('model').value, sequence_a:$('sequence-a').value, sequence_b:$('sequence-b').value, interpolation:$('interpolation').value, patch_layer:Number($('patch-layer').value), patch_position:$('patch-position').value, steps:Number($('steps').value), context:$('context').value}; }
function suffixMode() { return $('patch-position').value==='different_suffix'; }
function patchLabel(settings) { return settings.patch_position==='different_suffix'?'First difference → end':'Final token only'; }
function patchNote() {
  const suffix=suffixMode();
  $('context-field').classList.toggle('hidden',suffix);
  $('context').disabled=busy || suffix;
  $('input-hint').textContent=suffix
    ? 'Up to 256 tokens per sequence. Suffix mode requires equal token counts.'
    : 'Any two different sequences, up to 256 tokens each. Prefixes and lengths may differ.';
  $('patch-note').textContent=suffix
    ? 'Equal token counts required. Pair positions directly and interpolate every token from the first difference onward, including later matching tokens. The same t is used for each pair; the shared prefix stays fixed.'
    : 'Interpolate only the final token from each input. Prefixes and token counts may differ; choose A or B as the fixed context.';
}
function remember() { try{localStorage.setItem('plateau-draft-v1',JSON.stringify(settings()));}catch{} }
function layers(selected=0, savedResult=false) {
  const count=modelLayers[$('model').value];
  legacyEmbeddingResult=savedResult && selected===-1;
  $('patch-layer').max=String(count-1);
  $('patch-layer').value=String(Math.max(0,Math.min(selected,count-1)));
  $('layer-max').textContent=String(count-1);
  layerNote();
}
function layerNote() {
  const layer=Number($('patch-layer').value), count=modelLayers[$('model').value];
  $('layer-value').textContent=`After layer ${layer}${legacyEmbeddingResult?' · next run':''}`;
  $('patch-layer').setAttribute('aria-valuetext',`After layer ${layer}${legacyEmbeddingResult?', for the next run':''}`);
  $('layer-note').textContent=legacyEmbeddingResult
    ? 'This older result used embedding interpolation. The slider selects a hidden layer for the next run; the saved curves below are unchanged.'
    : `Hidden space: interpolate the selected token states after layer ${layer} (resid_post), then ${layer===count-1?'apply final normalization and the output head':`continue from layer ${layer+1}`}. Layers are numbered 0–${count-1}.`;
  patchNote();
}
function setForm(record) {
  const s=record.settings || record;
  $('sequence-a').value=record.sequence_a;
  $('sequence-b').value=record.sequence_b;
  if(!modelLayers[record.model]){
    const option=document.createElement('option');option.value=record.model;option.textContent=(record.model_label || record.model)+' · Unavailable';option.disabled=true;$('model').append(option);
    modelLayers[record.model]=Math.max(1,...(record.l2_distances?.layers || []).map(r=>r.layer+1),(s.patch_layer || 0)+1);
  }
  $('model').value=record.model;
  $('patch-position').value=s.patch_position || 'last_token';
  layers(s.patch_layer ?? 0, Boolean(record.curves));
  $('interpolation').value=s.interpolation || 'slerp';
  $('context').value=s.context || 'a';
  $('steps').value=String(s.steps || 41);
  if(!$('steps').value){const option=document.createElement('option');option.value=String(s.steps);option.textContent=String(s.steps);$('steps').append(option);$('steps').value=String(s.steps);}
  modelNote();
}
function setBusy(value) {
  busy=value;
  ['model','interpolation','patch-layer','patch-position','steps','context','sequence-a','sequence-b','swap','next-preset','run'].forEach(id=>$(id).disabled=value);
  patchNote();
  document.querySelectorAll('[data-preset]').forEach(el=>el.disabled=value);
  $('cancel').classList.toggle('hidden',!value);
  $('save').disabled=value || !result;
  $('export-result').disabled=value || !result;
  $('run').innerHTML=value?'<span>◌</span> Computing…':'<span>▶</span> Run experiment <kbd>⌘/Ctrl ↵</kbd>';
}
const emptyCharts='<div class="chart-wait"><svg viewBox="0 0 96 40" aria-hidden="true"><path d="M5 32h22c15 0 12-24 28-24h34"/></svg><div>Explore the response of the model.<small>Run to measure this metric for residual layers and logits.</small></div></div>';
function invalidate() {
  result=null;resultSource=null;tokenizeRequest++;
  $('save').disabled=true;$('export-result').disabled=true;
  $('effect-section').classList.add('hidden');scheduleTokenize();
  ['inspect','result-meta','token-details','input-tokenization','l2-section'].forEach(id=>$(id).classList.add('hidden'));
  $('export-l2').disabled=true;
  legacyEmbeddingResult=false;layerNote();
  $('status').classList.add('hidden');
  $('charts').innerHTML=emptyCharts;
  $('c-charts').innerHTML=emptyCharts;
  $('c-legacy').classList.add('hidden');
  $('method-note').textContent=suffixMode()
    ? 'Interpolate each token state from the first difference through the end, with the same t at every position. The identical prefix stays fixed; t = 0 and t = 1 reproduce natural A and B. c(t) and d(t) are measured at the final token.'
    : `Interpolate the final-token states from A and B with context ${$('context').value.toUpperCase()} held fixed. Distances use the two patched endpoints in this context, which may differ from the original prompt outputs.`;
  for(const side of ['a','b']) { $('count-'+side).textContent='Ready'; $('prediction-'+side).textContent='Run an experiment to see the continuation'; $('prediction-'+side).classList.add('muted'); }
  $('notes').value=''; $('tag').value='Unclassified';
  remember();
}
function preset(index) { if(busy)return; $('sequence-a').value=presets[index][0];$('sequence-b').value=presets[index][1];invalidate(); }
function metricMessage(curve, metric) {
  if(metric==='c' && !Object.hasOwn(curve,'c'))return 'c(t) requires rerunning this legacy experiment.';
  return curve[metric+'_undefined_reason'] || `${metric}(t) undefined`;
}
function curveSvg(curve, compact=false, metric='c') {
  const values=curve[metric];
  if(!Array.isArray(values) || values.some(v=>!Number.isFinite(v)))return `<div class="metric-unavailable">${esc(metricMessage(curve,metric))}</div>`;
  const w=252,h=190,left=35,right=235,top=20,bottom=152;
  const xx=t=>left+t*(right-left),yy=value=>bottom-value*(bottom-top);
  const points=curve.t.map((t,i)=>`${xx(t).toFixed(2)},${yy(values[i]).toFixed(2)}`).join(' ');
  const name=metric==='c'?'cumulative path progress c(t)':'relative endpoint distance d(t)';
  const label=`${curve.title}: horizontal axis is interpolation coefficient t from 0 to 1; vertical axis is ${name} from 0 to 1; ${curve.t.length} measured samples.`;
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>
    ${[0,.5,1].map(v=>`<line x1="${left}" y1="${yy(v)}" x2="${right}" y2="${yy(v)}" stroke="#eeeff2"/><text x="${left-8}" y="${yy(v)+3}" text-anchor="end" fill="#a2a5b0" font-size="8">${v.toFixed(1)}</text><text x="${xx(v)}" y="${bottom+15}" text-anchor="middle" fill="#a2a5b0" font-size="8">${v.toFixed(1)}</text>`).join('')}
    <text x="9" y="14" fill="#999caa" font-size="8">${metric}(t)</text><text x="${(left+right)/2}" y="${h-7}" text-anchor="middle" fill="#999caa" font-size="8">Interpolation coefficient t</text>
    <line x1="${left}" y1="${bottom}" x2="${right}" y2="${top}" stroke="#c2c5cf" stroke-dasharray="3 4" stroke-width="1"><title>${metric==='c'?'Uniform path progress: c(t)=t; constant accumulation per unit t, not necessarily a straight trajectory.':'d = t reference'}</title></line>
    <polyline points="${points}" fill="none" stroke="#806ace" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
    ${compact?'':curve.t.map((t,i)=>`<circle cx="${xx(t)}" cy="${yy(values[i])}" r="1.6" fill="#806ace"><title>t=${t.toFixed(3)} · ${metric}(t)=${values[i].toFixed(5)}${metric==='c'?` · cumulative L2=${formatL2(curve.cumulative_length?.[i])} · total path L2=${formatL2(curve.total_length)}`:''}</title></circle>`).join('')}
  </svg>`;
}
function renderCurves(record, metric) {
  const grid=$(metric==='c'?'c-charts':'charts');
  grid.style.setProperty('--curve-columns',record.curves.length);
  grid.innerHTML=record.curves.map((curve,i)=>`<article class="chart" data-metric="${metric}" data-readout="${esc(curve.key)}"><div class="chart-title"><strong>${esc(curve.title)}</strong><span class="caption">${curve.key==='logits'?'Output':i===0?'Early':i===record.curves.length-2?'Final':'Middle'}</span></div>${curveSvg(curve,false,metric)}${metric==='c' && Object.hasOwn(curve,'c')?`<p class="path-total" title="${esc(curve.total_length)}">Total path L2 · ${formatL2(curve.total_length)}</p>`:''}</article>`).join('');
}
function libraryPreview(record) {
  if(!record.curves)record=PlateauRecords.normalize(record);
  const curve=record.curves.find(c=>c.key==='logits'),legacy=!Object.hasOwn(curve,'c');
  return `<div class="preview-metric${legacy?' legacy-preview':''}">${legacy?'Legacy · d(t) only':'c(t) · Cumulative path progress'}</div>${curveSvg(curve,true,legacy?'d':'c')}`;
}
function setupMetricHelp() {
  const button=$('c-info'), help=$('c-definition'), group=button.closest('.metric-help');
  let pinned=false, hovered=false;
  const show=value=>{help.classList.toggle('hidden',!value);button.setAttribute('aria-expanded',String(value));};
  group.addEventListener('mouseenter',()=>{hovered=true;show(true);});
  group.addEventListener('mouseleave',()=>{hovered=false;if(!pinned && document.activeElement!==button)show(false);});
  button.addEventListener('focus',()=>show(true));
  button.addEventListener('blur',()=>{if(!pinned && !hovered)show(false);});
  button.addEventListener('click',()=>{pinned=!pinned;show(pinned);});
  button.addEventListener('keydown',event=>{if(event.key==='Escape'){pinned=false;show(false);}});
  document.addEventListener('pointerdown',event=>{if(!group.contains(event.target)){pinned=false;show(false);}});
}
function metricSummary(record, metric) {
  const stats=record.metrics?.[metric] || (metric==='d'?record.metrics:null);
  if(!stats)return `<span>Logits c(t): rerun to measure</span>`;
  if(!Number.isFinite(stats.max_abs_slope))return `<span>Logits ${metric}(t): undefined</span>`;
  return `<span>Logits max |Δ${metric}/Δt| <b>${stats.max_abs_slope.toFixed(2)}</b> @ t ≈ ${stats.peak_t.toFixed(3)}</span>`;
}
function tokenText(value) { return String(value).replace(/ /g,'␠').replace(/\n/g,'↵').replace(/\t/g,'⇥').replace(/\r/g,'␍'); }
function renderInputTokens(record) {
  $('input-tokenization').classList.remove('hidden');
  const suffix=record.settings.patch_position==='different_suffix';
  $('tokenization-note').textContent=suffix
    ? 'Highlighted: every token from the first difference through the end, including matching tokens after it. Each hidden-state pair is interpolated with the same t. Hover for token IDs.'
    : 'Highlighted: the final token whose state is interpolated. Hover over a token to see its ID.';
  record.input_tokens.forEach((tokens,i)=>{
    const side=i?'b':'a';
    const start=record.settings[i?'patch_start_b':'patch_start_a'] ?? Math.max(0,tokens.length-1);
    $('token-summary-'+side).textContent=tokens.length?`${tokens.length} tokens · interpolating ${tokens.length-start} at ${start===tokens.length-1?`position ${start}`:`positions ${start}–${tokens.length-1}`}`:'No tokens recorded';
    $('input-tokens-'+side).innerHTML=tokens.map((token,index)=>{
      const patched=index>=start, first=suffix && index===start;
      const description=`Position ${index} · token ID ${token.id}${patched?' · interpolated position':''}${first?' · first difference':''}`;
      return `<span class="input-token${patched?' final-token':''}${first?' first-difference':''}" role="listitem" title="${esc(description)}"><small>${index}</small><code>${esc(tokenText(token.text))}</code>${first?'<span class="token-target">First difference</span>':patched && index===tokens.length-1?'<span class="token-target">Final</span>':''}<span class="sr-only">${esc(description)}</span></span>`;
    }).join('');
  });
}
function formatL2(value) {
  if(!Number.isFinite(value))return '—';
  if(value===0)return '0';
  return value<0.001 || value>=100000 ? value.toExponential(3) : value.toLocaleString('en-US',{maximumFractionDigits:4});
}
function l2Svg(distances) {
  const layers=distances.layers, last=layers[layers.length-1].layer;
  const left=64,right=582,top=35,bottom=204;
  const max=Math.max(0,...layers.flatMap(row=>[row.natural_l2,row.patched_l2]).filter(Number.isFinite))*1.1 || 1;
  const x=layer=>left+layer/Math.max(last,1)*(right-left), y=value=>bottom-value/max*(bottom-top);
  const ticks=[...new Set([0,Math.round(last/4),Math.round(last/2),Math.round(last*3/4),last])];
  const label='L2 distance by Transformer layer. Original A/B and patched endpoints in the fixed context. Dashed vertical line marks the interpolation layer.';
  return `<svg viewBox="0 0 610 246" role="img" aria-label="${label}"><title>${label}</title>
    <text x="${left}" y="15" fill="#7d808d" font-size="10">L2 distance (raw)</text>
    ${[0,.25,.5,.75,1].map(f=>`<line x1="${left}" x2="${right}" y1="${y(f*max)}" y2="${y(f*max)}" stroke="#eeeff2"/><text x="${left-8}" y="${y(f*max)+3}" text-anchor="end" fill="#7d808d" font-size="9">${formatL2(f*max)}</text>`).join('')}
    ${ticks.map(layer=>`<text x="${x(layer)}" y="${bottom+17}" text-anchor="middle" fill="#7d808d" font-size="10">${layer}</text>`).join('')}
    <text x="${(left+right)/2}" y="239" text-anchor="middle" fill="#7d808d" font-size="10">Transformer layer (0-based, resid_post)</text>
    ${distances.patch_layer>=0?`<line x1="${x(distances.patch_layer)}" x2="${x(distances.patch_layer)}" y1="${top}" y2="${bottom}" stroke="#b7aacd" stroke-dasharray="3 4"/><text x="${x(distances.patch_layer)}" y="28" text-anchor="${distances.patch_layer>last/2?'end':'start'}" fill="#7762c7" font-size="9">Interpolation · layer ${distances.patch_layer}</text>`:''}
    ${[['natural_l2','#427eaa','Original A/B','5 3'],['patched_l2','#7762c7','Patched endpoints','']].map(([key,color,name,dash])=>{const available=layers.filter(row=>Number.isFinite(row[key]));return available.length?`<polyline points="${available.map(row=>`${x(row.layer)},${y(row[key])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="${dash}"/>${available.map(row=>`<circle cx="${x(row.layer)}" cy="${y(row[key])}" r="2.3" fill="${color}"><title>Layer ${row.layer} · ${name}: ${row[key]}</title></circle>`).join('')}`:'';}).join('')}
  </svg>`;
}
function renderL2(record) {
  const distances=record.l2_distances;
  $('l2-section').classList.remove('hidden');
  $('export-l2').disabled=!distances?.layers?.length;
  if(!distances?.layers?.length){
    $('l2-body').innerHTML='<p class="l2-missing">L2 distances were not saved with this older result. Run the experiment again to measure them.</p>';
    return;
  }
  const context=(record.settings.context || 'a').toUpperCase();
  const count=distances.source_token_count || 1;
  $('l2-body').innerHTML=`<div class="l2-source"><div><span>At interpolation · ${distances.patch_layer===-1?'embedding':'after layer '+distances.patch_layer}</span><strong title="${distances.source_l2}">${formatL2(distances.source_l2)}</strong><span>${count>1?`All ${count} selected tokens`:'Final token'}</span></div><p>${count>1?'‖H<sub>A</sub> − H<sub>B</sub>‖<sub>F</sub><br>L2 over all selected token vectors concatenated together.':'‖h<sub>A</sub> − h<sub>B</sub>‖₂<br>The distance between the two source vectors being interpolated.'}${count>1?`<br>Final-token source L2: <b>${formatL2(distances.source_last_token_l2)}</b>. The per-layer plot below measures only the final token.`:''}</p></div>
    ${count>1?`<details class="source-token-details"><summary>Source L2 for each of the ${count} interpolated tokens</summary><div class="source-token-distances">${(distances.source_token_l2 || []).map(row=>`<span>Position ${row.position_a===row.position_b?row.position_a:`A:${row.position_a} / B:${row.position_b}`}<b title="${row.l2}">${formatL2(row.l2)}</b></span>`).join('')}</div></details>`:''}
    <div class="l2-explanation"><p><b>Original A/B:</b> each input runs naturally, using its own final-token state at every layer.</p><p><b>Patched endpoints:</b> compare t = 0 and t = 1 in fixed context ${esc(context)}. Before interpolation, both endpoints share the same state, so their distance is zero.</p></div>
    <div class="l2-grid"><figure class="l2-figure"><div class="l2-legend"><span><i class="l2-natural"></i>Original A/B</span><span><i class="l2-patched"></i>Patched endpoints · C=${esc(context)}</span></div>${l2Svg(distances)}<figcaption>${esc(record.model_label)} · ${esc(record.backend?.remote?'NDIF inference':'local inference')} · ${esc(formatDate(record.created_at,true))}. Block outputs before final normalization. The L2 axis is not restricted to [0, 1].</figcaption></figure>
    <div class="l2-table-wrap" tabindex="0" aria-label="L2 distances for every layer"><table class="l2-table"><caption class="sr-only">Last-token residual-stream L2 distance at every layer</caption><thead><tr><th scope="col">Layer</th><th scope="col">Original A/B</th><th scope="col">Patched endpoints</th></tr></thead><tbody>${distances.layers.map(row=>`<tr class="${row.layer===distances.patch_layer?'l2-patch-row':''}"><th scope="row">${row.layer}${row.layer===distances.patch_layer?'<small>Interpolation</small>':''}</th><td title="${row.natural_l2}">${formatL2(row.natural_l2)}</td><td title="${row.patched_l2}">${formatL2(row.patched_l2)}</td></tr>`).join('')}</tbody></table></div></div>`;
}
function downloadL2() {
  if(!result?.l2_distances)return;
  const record=result, distances=record.l2_distances;
  const rows=[['id','model','sequence_a','sequence_b','patch_layer','fixed_context','position','representation','source_l2','layer','natural_l2','patched_l2','patch_position','patch_start_a','patch_start_b','patch_count','source_last_token_l2'],
    ...distances.layers.map(row=>[record.id,record.model,record.sequence_a,record.sequence_b,distances.patch_layer,record.settings.context || 'a','last_token','resid_post',distances.source_l2,row.layer,row.natural_l2,row.patched_l2,record.settings.patch_position || 'last_token',record.settings.patch_start_a ?? record.input_tokens[0].length-1,record.settings.patch_start_b ?? record.input_tokens[1].length-1,distances.source_token_count || 1,distances.source_last_token_l2 ?? distances.source_l2])];
  const csv='\uFEFF'+rows.map(row=>row.map(value=>'"'+String(value ?? '').replace(/"/g,'""')+'"').join(',')).join('\r\n')+'\r\n';
  download(new Blob([csv],{type:'text/csv;charset=utf-8'}),`plateau-l2-${record.id}.csv`);
}
function renderResult(record) {
  resultSource=record;record=PlateauRecords.normalize(record);result=record;tokenizeRequest++;setForm(record);remember();
  renderInputTokens(record);
  renderL2(record);
  const annotation=library.examples.find(r=>r.id===record.id) || record;
  $('notes').value=annotation.notes || '';
  const tag=annotation.tag || 'Unclassified';
  if(![...$('tag').options].some(o=>o.value===tag)){const option=document.createElement('option');option.value=tag;option.textContent=tag;$('tag').append(option);}
  $('tag').value=tag;
  $('save').textContent=library.examples.some(r=>r.id===record.id)?'Update example':'＋ Save example';
  record.predictions.forEach((prediction,i)=>{
    const side=i?'b':'a';
    $('count-'+side).textContent=`${record.input_tokens[i].length} tokens`;
    $('prediction-'+side).classList.remove('muted');
    const words=Array.isArray(prediction.words),pieces=words?prediction.words:prediction.tokens.slice(0,3).map(t=>tokenText(t.text));
    document.querySelectorAll('.prediction-label')[i].textContent=words?'Next 3 words · GREEDY':'Recorded continuation · GREEDY';
    $('prediction-'+side).innerHTML=pieces.map((word,n)=>`<span class="word"><small>${n+1}</small>${esc(word)}</span>`).join('')+
      `<span class="continuation">↳ ${esc(prediction.continuation)}${words && prediction.word_count<3?' · Generation ended before three words':''}${words && prediction.complete===false?' · Token limit reached; the final word may be incomplete':''}</span>`;
  });
  renderCurves(record,'c');
  renderCurves(record,'d');
  $('c-legacy').classList.toggle('hidden',record.curves.every(curve=>Object.hasOwn(curve,'c')));
  $('effect-section').classList.remove('hidden');
  $('effect-metric').innerHTML=record.effect.metrics.map(m=>`<option value="${esc(m.id)}">${esc(m.label)}</option>`).join('');
  $('effect-metric').value=record.effect.rows.some(row=>row.c_status!=='not_recorded')?'c':'relative_l2_shinkle';
  applyEffectPreset(effectPreset==='custom'?'representative':effectPreset);
  $('layer-quantity').innerHTML=layerQuantities(record).map(q=>`<option value="${esc(q.id)}">${esc(q.label)}</option>`).join('');
  renderLayerPlot();
  $('result-meta').classList.remove('hidden');
  $('result-meta').innerHTML=`<span><b>${esc(record.model_label || record.model)}</b> · ${esc(record.hardware?.name || (record.backend?.remote?'NDIF':record.device || record.backend?.device || 'Recorded'))} · ${esc(record.dtype || 'Recorded precision')}</span><span>${esc(record.settings.interpolation.toUpperCase())} @ ${record.settings.patch_layer===-1?'embedding':'after layer '+record.settings.patch_layer} · ${esc(patchLabel(record.settings))} · ${record.settings.steps} samples · fixed context ${esc((record.settings.context || "a").toUpperCase())}</span>${metricSummary(record,'c')}${metricSummary(record,'d')}<span>${Number.isFinite(record.elapsed_seconds)?record.elapsed_seconds.toFixed(1):'—'} s</span>`;
  $('inspect').classList.toggle('hidden',!record.path_predictions.length);
  $('t-slider').max=Math.max(0,record.path_predictions.length-1);
  $('t-slider').value=Math.floor(record.path_predictions.length/2);
  updateT();
  $('token-details').classList.remove('hidden');
  $('token-body').innerHTML=record.predictions.map((prediction,i)=>`<div class="token-row">Generated ${i?'B':'A'}: ${prediction.tokens.map(t=>`<span class="token-chip" title="ID ${esc(t.id)} · p=${Number.isFinite(t.probability)?(100*t.probability).toFixed(2)+'%':'not recorded'}">${esc(tokenText(t.text))}</span>`).join('')}</div>`).join('')+`<p>Generated tokens include look-ahead tokens used to confirm the third word boundary. Words follow English word boundaries; ␠ marks a space and ↵ a newline. Layers are numbered from 0; resid_post is recorded at the block output, before final normalization (LayerNorm for GPT-2/Pythia; RMSNorm for Qwen). Source: ${esc(record.backend?.remote?'NDIF remote':'local')} model inference · ${esc(formatDate(record.created_at, true))}.</p>`;
  const experiment=record.experiment;
  $('method-note').textContent=experiment
    ? (record.settings.patch_position==='different_suffix'
      ? `Interpolating ${record.settings.patch_count} token positions, ${record.settings.patch_start_a}–${record.input_tokens[0].length-1}, from the first difference through the end. The identical prefix stays fixed. Both endpoints reproduce the original A/B outputs. c(t), d(t) and per-layer L2 measure the final token. `
      : `Fixed context ${experiment.fixed_context} · A: ${experiment.source_lengths[0]} tokens; B: ${experiment.source_lengths[1]} tokens. `+
        (experiment.shared_tokenized_prefix ? 'Prefixes match, so the endpoints reproduce the original A/B outputs. ' : 'The endpoints are patched outputs within the selected context, not the natural outputs of both prompts. '))+
      `Next tokens at the patched endpoints: ${JSON.stringify(experiment.patched_endpoint_next_tokens[0])} → ${JSON.stringify(experiment.patched_endpoint_next_tokens[1])}.`
    : 'Original matching-prefix experiment: both endpoints reproduce the natural A/B outputs. Re-running uses the selected fixed context.';
  $('save').disabled=busy;$('export-result').disabled=busy;
}
function updateT() {
  if(!result)return;
  const index=Number($('t-slider').value),p=result.path_predictions[index];
  if(!p)return;
  $('t-value').textContent=`t = ${p.t.toFixed(3)}  → next token ${JSON.stringify(p.token)}`;
  $('t-slider').setAttribute('aria-valuetext',`Sample ${index+1} of ${result.path_predictions.length}, t = ${p.t.toFixed(3)}`);
  $('token-matrix').innerHTML=PlateauTokenMatrix.render(p);
  $('sample-values').innerHTML=result.curves.map(curve=>{
    const value=metric=>Number.isFinite(curve[metric]?.[index])?curve[metric][index].toFixed(5):metricMessage(curve,metric);
    return `<div><strong>${esc(curve.title)}</strong><span>c(t): ${esc(value('c'))}</span><span>d(t): ${esc(value('d'))}</span>${Object.hasOwn(curve,'c')?`<small>Cumulative L2: ${formatL2(curve.cumulative_length?.[index])} / total path L2: ${formatL2(curve.total_length)}</small>`:''}</div>`;
  }).join('');
}
async function run() {
  if(busy)return;
  if(!$('sequence-a').value.trim() || !$('sequence-b').value.trim()){status('Enter both sequences first.',0,true);return;}
  if(!modelCatalog.some(m=>m.id===$('model').value)){status('Choose an available model to rerun this saved experiment.',0,true);return;}
  if(requiresKey && !ndifKey()){status('Add your NDIF API key or lab access code in Connection settings.',0,true);showSettings(true);$('ndif-key').focus();return;}
  invalidate();setBusy(true);status('Preparing the model…');
  try { const job=await api('/api/run',settings(),authHeaders());currentJob=job.id;rememberJob(job.id);await poll(job.id); }
  catch(error){status('Experiment did not complete: '+error.message,0,true);}
  finally {currentJob=null;setBusy(false);await refreshModels().catch(()=>{});}
}
async function poll(id) {
  let failures=0;
  while(true) {
    let job;
    try {job=await api('/api/jobs/'+id);failures=0;}
    catch(error){if(++failures>=5)throw new Error('Cannot connect to the server. Reload to recover this experiment.');status('Connection interrupted. Retrying…');await new Promise(r=>setTimeout(r,1500));continue;}
    status(job.message,job.progress,job.status==='error');
    if(job.hardware && !remoteMode)showHardware(job.hardware);
    if(job.status==='done'){
      forgetJob();
      if(!localLibrary && !await browserCollections.put('runs',job.result))toast('Run kept for this session only. Export it before leaving; browser storage is unavailable.');
      await loadLibrary();renderResult(job.result);return;
    }
    if(job.status==='error' || job.status==='cancelled'){forgetJob();return;}
    await new Promise(r=>setTimeout(r,650));
  }
}
async function loadLibrary() {
  const [runs,examples]=await Promise.all([browserCollections.read('runs'),browserCollections.read('examples')]);
  diskLibrary=localLibrary?await api('/api/library'):{history:[],examples:[]};
  const combine=(browser,disk)=>{
    const merged=new Map();
    for(const record of [...browser,...disk]){try{PlateauRecords.normalize(record);merged.set(record.id,record);}catch{}}
    return [...merged.values()].sort((a,b)=>b.created_at.localeCompare(a.created_at));
  };
  library={history:combine(runs,diskLibrary.history),examples:combine(examples,diskLibrary.examples)};
  for(const group of ['examples','history']) {
    const existing=new Set(library[group].map(r=>r.id));
    for(const id of selectedRecords[group])if(!existing.has(id))selectedRecords[group].delete(id);
  }
  $('saved-count').textContent=library.examples.length;
  $('library-storage').textContent=(localLibrary?'Local runs and examples are saved on this computer. Browser imports stay in this browser.':'History, Examples, categories and notes are private to this browser. Export JSON/JSONL to back up or move them.')+
    (!browserCollections.persistent?' Browser storage is unavailable or full; new browser records are kept for this session only. Export before leaving.':'');
  renderLibrary();
}
function showView(view) {
  $('workbench').classList.toggle('hidden',view!=='work');$('library-view').classList.toggle('hidden',view==='work');
  $('nav-work').classList.toggle('active',view==='work');$('nav-library').classList.toggle('active',view!=='work');
  if(view!=='work')loadLibrary().catch(e=>toast(e.message));
}
function visibleRecords() {
  const search=$('search').value.toLowerCase();
  return library[scope].filter(r=>[r.sequence_a,r.sequence_b,r.tag,r.notes,r.model_label].join(' ').toLowerCase().includes(search));
}
function updateSelection() {
  const selected=selectedRecords[scope], visible=visibleRecords();
  const hidden=selected.size-visible.filter(r=>selected.has(r.id)).length;
  $('selection-count').textContent=`${selected.size} selected${hidden?` · ${hidden} hidden by search`:''}`;
  $('clear-selection').disabled=!selected.size;
  $('select-visible').disabled=!visible.some(r=>!selected.has(r.id));
  $('delete-history').classList.toggle('hidden',scope!=='history');
  $('delete-history').disabled=!selected.size;
  $('export-all').disabled=!library[scope].length;
  for(const id of ['export-json','export-csv','export-effects'])$(id).disabled=!selected.size || exporting;
  document.querySelectorAll('[data-record]').forEach(card=>{
    const checked=selected.has(card.dataset.record);
    card.classList.toggle('is-selected',checked);
    card.setAttribute('aria-pressed',String(checked));
    card.querySelector('.selection-mark').textContent=checked?'✓':'';
  });
}
function openRecord(id) {
  if(busy){toast('Wait for the current experiment to finish before opening another record.');return;}
  const record=library[scope].find(r=>r.id===id);
  if(!record)return;
  renderResult(record);showView('work');
  status('Saved results restored. Change the inputs or settings to run a new experiment.',1);
  window.scrollTo({top:0,behavior:'smooth'});
}
function renderLibrary() {
  const data=visibleRecords(), search=$('search').value;
  $('library-caption').textContent=`${data.length} shown / ${library[scope].length} ${scope==='examples'?'saved examples':'completed experiments'} · Click to select; double-click to open. Keyboard: Space selects, Enter opens. Exports include all selected records, even those hidden by search.`;
  $('library-grid').innerHTML=data.length?data.map(r=>`<button class="example-card" data-record="${esc(r.id)}" aria-pressed="false" aria-label="Select example: ${esc(r.sequence_a)} / ${esc(r.sequence_b)}" title="Click to select · Double-click or press Enter to open"><div class="top"><span class="card-model"><span class="selection-mark" aria-hidden="true"></span>${esc(r.model_label)}</span><span class="tag">${esc(r.tag || 'Unclassified')}</span></div><p><span class="prefix">A</span>${esc(r.sequence_a)}</p><p><span class="prefix">B</span>${esc(r.sequence_b)}</p>${libraryPreview(r)}<div class="top"><span>Logits · ${esc(r.settings.interpolation.toUpperCase())} · ${r.settings.patch_layer===-1?'Embedding':'After layer '+r.settings.patch_layer} · ${r.settings.patch_position==='different_suffix'?'Suffix':'Final token'} · C=${esc((r.settings.context || "a").toUpperCase())}</span><span>${formatDate(r.created_at)}</span></div>${r.notes?`<p class="note">${esc(r.notes)}</p>`:''}</button>`).join(''):`<div class="library-empty">${search?'No examples match your search.':scope==='examples'?'No saved examples yet.<br>Run a pair and keep the curves you want to study.':'No completed experiments yet.<br>Start your first pair in the Workbench.'}</div>`;
  document.querySelectorAll('[data-record]').forEach(card=>{
    // Keep the card DOM in place so the browser can recognize a double-click.
    card.onclick=event=>{
      if(event.detail>1)return;
      const selected=selectedRecords[scope], id=card.dataset.record;
      if(selected.has(id))selected.delete(id);else selected.add(id);
      updateSelection();
    };
    card.ondblclick=()=>{
      selectedRecords[scope].add(card.dataset.record);updateSelection();openRecord(card.dataset.record);
    };
    card.onkeydown=event=>{
      if(event.key==='Enter'){event.preventDefault();openRecord(card.dataset.record);}
    };
  });
  updateSelection();
}
async function exportSelected(format) {
  const exportScope=scope, ids=[...selectedRecords[scope]],records=library[scope].filter(r=>ids.includes(r.id));
  if(!records.length || exporting)return;
  exporting=true;updateSelection();
  try {
    if(localLibrary && format!=='effects' && records.every(r=>diskLibrary[exportScope].some(d=>d.id===r.id))){
      const response=await fetch(API+'/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scope:exportScope,format,ids})});
      if(!response.ok){const error=await response.json();throw new Error(error.error || 'Export failed.');}
      download(await response.blob(),`plateau-${exportScope}-selected.${format}`);
    }else{
      const content=format==='effects'?PlateauRecords.csv(records):format==='csv'?PlateauRecords.classicCsv(records):records.map(r=>JSON.stringify(r)).join('\n')+'\n';
      download(new Blob([content],{type:format==='jsonl'?'application/x-ndjson':'text/csv;charset=utf-8'}),`plateau-${exportScope}-selected.${format==='effects'?'all-layers.csv':format}`);
    }
    toast(`Exported ${ids.length} selected ${exportScope==='history'?'experiment':'example'}${ids.length===1?'':'s'}.`);
  } catch(error){toast('Export failed: '+error.message);}
  finally{exporting=false;updateSelection();}
}
async function save() {
  if(!result || busy)return;
  $('save').disabled=true;
  try{
    const tag=$('tag').value,notes=$('notes').value;
    let persistent=true;
    if(localLibrary && [...diskLibrary.history,...diskLibrary.examples].some(r=>r.id===result.id))await api('/api/examples',{id:result.id,tag,notes});
    else persistent=await browserCollections.put('examples',{...resultSource,tag,notes,saved_at:new Date().toISOString()});
    await loadLibrary();resultSource=library.examples.find(r=>r.id===result.id) || resultSource;$('save').textContent='Update example';
    toast(persistent?'Saved to Examples':'Saved for this session only. Export your example before leaving.');
  }
  catch(e){toast('Save failed: '+e.message);}
  finally{$('save').disabled=!result || busy;}
}
async function init() {
  setupMetricHelp();
  $('c-charts').innerHTML=emptyCharts;
  $('charts').innerHTML=emptyCharts;
  let config;
  status('Connecting to the server…',null);
  for(let attempt=0;;attempt++){
    try{config=await refreshModels();break;}
    catch(error){if(attempt>=20){status('Cannot reach the server: '+error.message,0,true);return;}await new Promise(r=>setTimeout(r,3000));}
  }
  layers();
  let draft=null;
  try {draft=JSON.parse(localStorage.getItem('plateau-draft-v1'));if(draft && modelLayers[draft.model]){draft.patch_position ??= 'different_suffix';setForm(draft);}}catch{}
  ['sequence-a','sequence-b'].forEach(id=>$(id).addEventListener('input',invalidate));
  ['interpolation','patch-position','steps','context'].forEach(id=>$(id).addEventListener('change',invalidate));
  $('patch-layer').addEventListener('input',invalidate);
  $('model').onchange=()=>{layers();modelNote();invalidate();};
  $('ndif-key').value=stored(KEY_STORE) || stored(CODE_STORE);
  $('ndif-key-remember').checked=!!(()=>{try{return localStorage.getItem(KEY_STORE) || localStorage.getItem(CODE_STORE);}catch{return false;}})();
  try{localStorage.removeItem(CODE_STORE);sessionStorage.removeItem(CODE_STORE);}catch{}
  saveKey();$('ndif-key').addEventListener('input',saveKey);$('ndif-key-remember').addEventListener('change',saveKey);
  $('settings-toggle').onclick=event=>{event.stopPropagation();showSettings($('settings-panel').classList.contains('hidden'));};
  document.addEventListener('click',event=>{if(!event.target.closest('.settings'))showSettings(false);});
  document.addEventListener('keydown',event=>{if(event.key==='Escape' && !$('settings-panel').classList.contains('hidden')){showSettings(false);$('settings-toggle').focus();}});
  $('run').onclick=run;$('save').onclick=save;$('t-slider').oninput=updateT;
  $('export-l2').onclick=downloadL2;
  $('export-result').onclick=()=>{if(resultSource)downloadJson(resultSource,`plateau-${result.id}.json`);};
  $('cancel').onclick=async()=>{if(currentJob){try{await api(`/api/jobs/${currentJob}/cancel`,{});toast('Stop requested. If a model is downloading, the current step must finish first.');}catch(e){toast(e.message);}}};
  $('swap').onclick=()=>{const a=$('sequence-a').value;$('sequence-a').value=$('sequence-b').value;$('sequence-b').value=a;invalidate();};
  document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>preset(Number(b.dataset.preset)));
  $('next-preset').onclick=()=>{preset(presetIndex);presetIndex=(presetIndex+1)%presets.length;};
  $('nav-work').onclick=()=>showView('work');$('nav-library').onclick=()=>showView('library');
  $('tab-saved').onclick=()=>{scope='examples';$('tab-saved').classList.add('selected');$('tab-history').classList.remove('selected');renderLibrary();};
  $('tab-history').onclick=()=>{scope='history';$('tab-history').classList.add('selected');$('tab-saved').classList.remove('selected');renderLibrary();};
  $('search').oninput=renderLibrary;
  $('select-visible').onclick=()=>{visibleRecords().forEach(r=>selectedRecords[scope].add(r.id));updateSelection();};
  $('clear-selection').onclick=()=>{selectedRecords[scope].clear();updateSelection();};
  $('export-json').onclick=()=>exportSelected('jsonl');$('export-csv').onclick=()=>exportSelected('csv');$('export-effects').onclick=()=>exportSelected('effects');
  $('export-all').onclick=()=>downloadJson(library[scope],`plateau-${scope}.json`);
  $('delete-history').onclick=deleteSelectedHistory;
  $('import-records').onclick=()=>$('import-files').click();
  $('import-files').onchange=()=>{importRuns([...$('import-files').files]);$('import-files').value='';};
  $('effect-metric').onchange=renderEffect;$('effect-overlay').onchange=renderEffect;$('layer-quantity').onchange=renderLayerPlot;
  $('effect-preset').onchange=()=>{if($('effect-preset').value!=='custom')applyEffectPreset($('effect-preset').value);};
  $('effect-rows').addEventListener('change',event=>{
    const key=event.target.dataset?.key;if(!key)return;
    event.target.checked?effectSelection.add(key):effectSelection.delete(key);
    effectPreset='custom';$('effect-preset').value='custom';renderEffect();
  });
  window.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter'){event.preventDefault();run();}});
  invalidate();
  try {
    await loadLibrary();
    const recover=config.active_job || rememberedJob();
    if(recover){currentJob=recover;const job=await api('/api/jobs/'+currentJob);setForm(job.request);setBusy(true);try{await poll(currentJob);}finally{currentJob=null;setBusy(false);await refreshModels().catch(()=>{});}}
    else if(library.history.length) {
      const current=settings();
      const matching=library.history.find(r=>r.model===current.model&&r.sequence_a===current.sequence_a&&r.sequence_b===current.sequence_b&&r.settings.patch_layer===current.patch_layer&&(r.settings.patch_position || 'last_token')===current.patch_position&&r.settings.steps===current.steps&&r.settings.interpolation===current.interpolation&&(r.settings.context || "a")===current.context);
      if(matching){renderResult(matching);status('Restored the previous measured results for this pair.',1);}
    }
  } catch(e){forgetJob();currentJob=null;status('Could not restore the previous workspace: '+e.message,0,true);}
}
// NDIF key: per-viewer, kept in sessionStorage, or localStorage when "Remember" is checked.
const KEY_STORE='plateau-ndif-key', CODE_STORE='plateau-lab-code';  // CODE_STORE: read once to migrate
const NDIF_KEY_FORMAT=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
let requiresKey=false, sharedAccess=false;
function stored(name) { try{return localStorage.getItem(name) || sessionStorage.getItem(name) || '';}catch{return '';} }
function ndifKey() { return $('ndif-key').value.trim(); }
function saveKey() {
  const key=ndifKey(), remember=$('ndif-key-remember').checked;
  try{ key?sessionStorage.setItem(KEY_STORE,key):sessionStorage.removeItem(KEY_STORE); }catch{}
  try{ key && remember?localStorage.setItem(KEY_STORE,key):localStorage.removeItem(KEY_STORE); }catch{}
  keyState();
}
function keyState() { $('settings-toggle').classList.toggle('needs-attention',requiresKey && !ndifKey()); }
// One field: an NDIF key has a fixed format; anything else is sent as the lab access code.
function authHeaders() {
  const value=ndifKey();
  if(!value) return {};
  return NDIF_KEY_FORMAT.test(value) || !sharedAccess ? {'X-NDIF-Key':value} : {'X-Access-Code':value};
}
function showSettings(open) { $('settings-panel').classList.toggle('hidden',!open); $('settings-toggle').setAttribute('aria-expanded',String(open)); }
function scheduleTokenize() {
  clearTimeout(tokenizeTimer);
  if(previewAvailable)tokenizeTimer=setTimeout(tokenizeInputs,200);
}
async function tokenizeInputs() {
  const request=++tokenizeRequest,s=settings();
  try{
    const preview=await api('/api/tokenize',{model:s.model,sequence_a:s.sequence_a,sequence_b:s.sequence_b,patch_position:s.patch_position});
    if(request!==tokenizeRequest || result || busy)return;
    renderInputTokens({input_tokens:preview.tokens,settings:{...s,patch_start_a:preview.patch_starts?.[0] ?? preview.tokens[0].length,patch_start_b:preview.patch_starts?.[1] ?? preview.tokens[1].length}});
    preview.tokens.forEach((tokens,i)=>{$('count-'+(i?'b':'a')).textContent=tokens.length+' tokens';$('count-'+(i?'b':'a')).classList.toggle('over-limit',tokens.length>preview.max_tokens);});
    if(s.patch_position==='different_suffix' && preview.tokens[0].length!==preview.tokens[1].length)$('tokenization-note').textContent='Suffix interpolation requires equal token counts. Edit the inputs or choose Final token only.';
  }catch(error){if(request===tokenizeRequest && !result)$('tokenization-note').textContent='Token preview unavailable: '+error.message;}
}
const JOB_KEY='plateau-active-job';
function rememberJob(id){try{sessionStorage.setItem(JOB_KEY,id);}catch{}}
function rememberedJob(){try{return sessionStorage.getItem(JOB_KEY);}catch{return null;}}
function forgetJob(){try{sessionStorage.removeItem(JOB_KEY);}catch{}}
function download(blob,name) {
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),60000);
}
function downloadJson(data,name){download(new Blob([JSON.stringify(data)],{type:'application/json'}),name);}
async function importRuns(files) {
  let added=0,skipped=0,persistent=true;
  const target=scope==='examples'?'examples':'runs';
  for(const file of files){
    try{
      for(const candidate of PlateauRecords.parse(await file.text())){
        try{PlateauRecords.normalize(candidate);persistent=(await browserCollections.put(target,candidate)) && persistent;added++;}
        catch{skipped++;}
      }
    }catch{skipped++;}
  }
  await loadLibrary();
  toast('Imported '+added+' records into '+(target==='runs'?'History':'Examples')+(skipped?' · skipped '+skipped+' invalid records':'')+(!persistent?'. Session only; export before leaving.':'.'));
}
async function deleteSelectedHistory() {
  if(scope!=='history' || !selectedRecords.history.size)return;
  const ids=[...selectedRecords.history];
  if(!confirm('Delete '+ids.length+' selected History records? Saved Examples are kept.'))return;
  try{
    const diskIds=ids.filter(id=>diskLibrary.history.some(r=>r.id===id));
    if(localLibrary && diskIds.length)await api('/api/history/delete',{ids:diskIds});
    await browserCollections.remove('runs',ids);
    await loadLibrary();toast('Selected History records deleted. Saved Examples are kept.');
  }catch(error){await loadLibrary().catch(()=>{});toast('Could not complete deletion: '+error.message);}
}
init();
