/* All-layer plots from the hosted Explorer, used by the unified workbench. */
'use strict';
const definedMetric=(row,metric)=>PlateauRecords.valid(row?.values?.[metric]);
const RAMP=['#a594e8','#7f6ad6','#5a44b8','#3b2789'], LOGITS_COLOR='#427eaa';
let effectSelection=new Set(), effectPreset='representative';
// Categorical slots in fixed order (reference palette); tokens past the eighth fold into "Other".
const TOKEN_COLORS=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'], OTHER_COLOR='#9a9ca6';
// Runs of equal argmax token along t; boundaries sit halfway between the samples where it changes.
function tokenSegments(record) {
  const t=record.effect.t, preds=record.path_predictions, order=new Map(), segments=[];
  preds.forEach((p,i)=>{
    if(!order.has(p.token_id)) order.set(p.token_id,order.size);
    const start=i===0?t[0]:(t[i-1]+t[i])/2;
    if(segments.length && segments[segments.length-1].id===p.token_id) return;
    if(segments.length) segments[segments.length-1].end=start;
    segments.push({id:p.token_id,token:p.token,start,end:t[t.length-1]});
  });
  const color=id=>order.get(id)<TOKEN_COLORS.length?TOKEN_COLORS[order.get(id)]:OTHER_COLOR;
  return {segments:segments.map(s=>({...s,color:color(s.id)})),
    tokens:[...order.keys()].map(id=>({id,token:preds.find(p=>p.token_id===id).token,color:color(id),other:order.get(id)>=TOKEN_COLORS.length}))};
}
function hexMix(a,b,f) { const p=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)); const [x,y]=[p(a),p(b)]; return '#'+x.map((v,i)=>Math.round(v+(y[i]-v)*f).toString(16).padStart(2,'0')).join(''); }
// Color follows the layer's depth (never its rank in the selection): light = shallow, dark = deep.
function effectColor(row, nLayers) {
  if(row.key==='logits') return LOGITS_COLOR;
  const f=nLayers>1?row.layer/(nLayers-1):1, x=f*(RAMP.length-1), i=Math.min(RAMP.length-2,Math.floor(x));
  return hexMix(RAMP[i],RAMP[i+1],x-i);
}
function effectPresetKeys(record, preset) {
  const rows=record.effect.rows.filter(row=>row.defined), layers=rows.filter(row=>row.key!=='logits');
  if(preset==='last') return layers.slice(-1).map(row=>row.key);
  if(preset==='logits') return ['logits'];
  if(preset==='all') return rows.map(row=>row.key);
  if(preset==='none') return [];
  if(preset==='even10') return [...new Set(Array.from({length:Math.min(10,layers.length)},(_,i)=>layers[Math.round(i*(layers.length-1)/Math.max(1,Math.min(10,layers.length)-1))].key))];
  return [...(record.settings.representative_layers || []).map(String),'logits'].filter(key=>rows.some(row=>row.key===key));
}
function applyEffectPreset(preset) {
  effectPreset=preset; $('effect-preset').value=preset;
  effectSelection=new Set(effectPresetKeys(result,preset));
  renderEffect();
}
// Across-layers figure: endpoint L2 or a metric's plateau score per recorded block.
function layerQuantities(record) {
  return [{id:'total_length',label:'Total sampled path L2',value:row=>row.total_length,reference:null},{id:'endpoint_l2',label:'Endpoint L2',value:row=>row.endpoint_l2,reference:null},
    ...record.effect.metrics.filter(m=>m.plateau_score).map(m=>({id:'plateau:'+m.id,label:`Plateau score Δt · ${m.label}`,
      value:row=>row.plateau_score[m.id],reference:{value:0.8,label:m.id==='c'?'c = t':'d = t'}}))];
}
function renderLayerPlot() {
  if(!result)return;
  const quantity=layerQuantities(result).find(q=>q.id===$('layer-quantity').value) || layerQuantities(result)[0];
  const rows=result.effect.rows.filter(row=>row.key!=='logits');
  const points=rows.map(row=>({row,v:quantity.value(row)}));
  const values=points.map(p=>p.v).filter(v=>v!=null && Number.isFinite(v));
  const w=960,h=300,left=62,right=w-18,top=24,bottom=h-44;
  if(!rows.length){$('layer-plot').textContent='No layer readouts were saved in this result.';return;}
  const first=rows[0].layer,last=rows[rows.length-1].layer;
  let lo=Math.min(0,...values),hi=Math.max(...values,quantity.reference?quantity.reference.value:-Infinity)*1.08;
  if(!Number.isFinite(hi) || hi<=lo) hi=lo+1;
  const x=layer=>left+(last===first?.5:(layer-first)/(last-first))*(right-left), y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const step=Math.max(1,Math.ceil((last-first+1)/16)), ticks=rows.filter(r=>(r.layer-first)%step===0 || r.layer===last).map(r=>r.layer);
  // Line segments break at undefined layers.
  const segments=[];let current=[];
  points.forEach(p=>{if(p.v==null || !Number.isFinite(p.v)){if(current.length)segments.push(current);current=[];}else current.push(p);});
  if(current.length)segments.push(current);
  const patch=result.settings.patch_layer, label=`${quantity.label} for layers ${first} to ${last}.`;
  $('layer-plot').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>
    ${[0,.25,.5,.75,1].map(f=>{const v=lo+f*(hi-lo);return `<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#eeeff2"/><text x="${left-9}" y="${y(v)+3.5}" text-anchor="end" fill="#8a8d99" font-size="11">${formatL2(+v.toPrecision(3))}</text>`;}).join('')}
    ${ticks.map(layer=>`<text x="${x(layer)}" y="${bottom+19}" text-anchor="middle" fill="#8a8d99" font-size="11">${layer}</text>`).join('')}
    <text x="${(left+right)/2}" y="${h-8}" text-anchor="middle" fill="#7d808d" font-size="11">Layer (block output, 0-based)</text>
    <text x="${left}" y="13" fill="#7d808d" font-size="11">${esc(quantity.label)}</text>
    <line x1="${x(patch)}" x2="${x(patch)}" y1="${top}" y2="${bottom}" stroke="#b7aacd" stroke-dasharray="3 4"/><text x="${x(patch)+5}" y="${top+9}" fill="#7762c7" font-size="10">Patch · layer ${patch}</text>
    ${quantity.reference?`<line x1="${left}" x2="${right}" y1="${y(quantity.reference.value)}" y2="${y(quantity.reference.value)}" stroke="#c2c5cf" stroke-dasharray="4 5"/><text x="${right}" y="${y(quantity.reference.value)-5}" text-anchor="end" fill="#9a9ca6" font-size="10">${quantity.reference.label} (${quantity.reference.value})</text>`:''}
    ${segments.map(seg=>`<polyline points="${seg.map(p=>`${x(p.row.layer).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')}" fill="none" stroke="#7762c7" stroke-width="2" stroke-linejoin="round"/>`).join('')}
    ${points.filter(p=>p.v!=null && Number.isFinite(p.v)).map(p=>`<circle cx="${x(p.row.layer)}" cy="${y(p.v)}" r="4" fill="#7762c7" stroke="#fff" stroke-width="2"/>`).join('')}
    <line id="layer-crosshair" y1="${top}" y2="${bottom}" stroke="#9a9ca6" visibility="hidden"/>
    <rect id="layer-hit" x="${left-10}" y="${top}" width="${right-left+20}" height="${bottom-top}" fill="transparent"/>
  </svg><div id="layer-tooltip" class="effect-tooltip hidden" role="status"></div>`;
  const svg=$('layer-plot').querySelector('svg'), tip=$('layer-tooltip'), cross=$('layer-crosshair');
  $('layer-hit').addEventListener('pointermove',event=>{
    const box=svg.getBoundingClientRect(), vx=(event.clientX-box.left)/box.width*w;
    const p=points.reduce((best,q)=>Math.abs(x(q.row.layer)-vx)<Math.abs(x(best.row.layer)-vx)?q:best,points[0]);
    cross.setAttribute('x1',x(p.row.layer));cross.setAttribute('x2',x(p.row.layer));cross.setAttribute('visibility','visible');
    const head=document.createElement('div'), line=document.createElement('div'), val=document.createElement('b'), name=document.createElement('span');
    head.className='effect-tooltip-head'; head.textContent=p.row.label+(p.row.patched?' · patched':'');
    val.textContent=p.v==null?'undefined':formatL2(p.v); name.textContent=quantity.label;
    line.append(val,name); tip.replaceChildren(head,line); tip.classList.remove('hidden');
    const px=x(p.row.layer)/w*box.width;
    tip.style.left=`${px>box.width/2?px-tip.offsetWidth-12:px+12}px`; tip.style.top='12px';
  });
  $('layer-hit').addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');tip.classList.add('hidden');});
}
function renderEffect() {
  if(!result)return;
  const effect=result.effect, metric=$('effect-metric').value, info=effect.metrics.find(m=>m.id===metric);
  const nLayers=result.l2_distances.layers.length, t=effect.t;
  const shown=effect.rows.filter(row=>definedMetric(row,metric) && effectSelection.has(row.key));
  // Plot
  const overlay=$('effect-overlay').value==='next_token' && result.path_predictions.length===effect.t.length?tokenSegments(result):null;
  const w=720,h=overlay?458:430,left=52,right=w-18,top=overlay?46:18,bottom=h-46;
  // The metric's range is always shown and extended to fit values outside it (e.g. overshoot).
  const values=shown.flatMap(r=>r.values[metric]), [r0,r1]=info.range || [Infinity,-Infinity];
  let lo=Math.min(r0,...values), hi=Math.max(r1,...values);
  if(!Number.isFinite(lo) || !Number.isFinite(hi)){lo=0;hi=1;} if(hi===lo){hi=lo+1;}
  const x=v=>left+v*(right-left), y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const ticks=[0,.25,.5,.75,1];
  const label=`${info.label} against interpolation coefficient t for ${shown.length} selected output${shown.length===1?'':'s'}.`;
  $('effect-plot').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>
    ${ticks.map(f=>`<line x1="${left}" x2="${right}" y1="${y(lo+f*(hi-lo))}" y2="${y(lo+f*(hi-lo))}" stroke="#eeeff2"/><text x="${left-9}" y="${y(lo+f*(hi-lo))+3.5}" text-anchor="end" fill="#8a8d99" font-size="11">${+(lo+f*(hi-lo)).toFixed(2)}</text><text x="${x(f)}" y="${bottom+19}" text-anchor="middle" fill="#8a8d99" font-size="11">${f}</text>`).join('')}
    ${overlay?overlay.segments.map(s=>`<rect x="${x(s.start)}" y="${top}" width="${Math.max(0,x(s.end)-x(s.start))}" height="${bottom-top}" fill="${s.color}" fill-opacity=".07"/>`).join('')+
      overlay.segments.map(s=>{const sw=x(s.end)-x(s.start), text=tokenText(s.token), fits=text.length*6.6+10<sw;
        return `<g><title>${esc(`Next token ${JSON.stringify(s.token)} for t ≈ ${s.start.toFixed(3)}–${s.end.toFixed(3)}`)}</title><rect x="${x(s.start)+1}" y="${top-26}" width="${Math.max(0,sw-2)}" height="20" rx="3" fill="${s.color}" fill-opacity=".16"/><rect x="${x(s.start)+1}" y="${top-26}" width="${Math.max(0,sw-2)}" height="3" rx="1.5" fill="${s.color}"/>${fits?`<text x="${(x(s.start)+x(s.end))/2}" y="${top-11}" text-anchor="middle" fill="#3d3f4a" font-size="11" font-family="SFMono-Regular,Consolas,monospace">${esc(text)}</text>`:''}</g>`;}).join(''):''}
    <text x="${(left+right)/2}" y="${h-8}" text-anchor="middle" fill="#7d808d" font-size="11">Interpolation coefficient t</text>
    <text x="${left}" y="11" fill="#7d808d" font-size="11">${esc(info.label)}${overlay?' · strip: next-token prediction':''}</text>
    <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#c2c5cf" stroke-dasharray="4 5"/>
    ${lo<0?`<line x1="${left}" x2="${right}" y1="${y(0)}" y2="${y(0)}" stroke="#d6d8df"/>`:''}${hi>1?`<line x1="${left}" x2="${right}" y1="${y(1)}" y2="${y(1)}" stroke="#d6d8df"/>`:''}
    ${shown.map(row=>`<polyline points="${row.values[metric].map((v,i)=>`${x(t[i]).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${effectColor(row,nLayers)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`).join('')}
    <line id="effect-crosshair" y1="${top}" y2="${bottom}" stroke="#9a9ca6" stroke-width="1" visibility="hidden"/>
    <rect id="effect-hit" x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="transparent"/>
  </svg><div id="effect-tooltip" class="effect-tooltip hidden" role="status"></div>`;
  $('effect-empty').classList.toggle('hidden',shown.length>0);
  $('effect-legend').innerHTML=shown.map(row=>`<span><i style="background:${effectColor(row,nLayers)}"></i>${esc(row.label)}</span>`).join('')+
    (overlay?`<span class="legend-break">Next token:</span>`+overlay.tokens.filter(tok=>!tok.other).map(tok=>`<span><i class="token-swatch" style="background:${tok.color}"></i><code>${esc(tokenText(tok.token))}</code></span>`).join('')+
      (overlay.tokens.some(tok=>tok.other)?`<span><i class="token-swatch" style="background:${OTHER_COLOR}"></i>Other (${overlay.tokens.filter(tok=>tok.other).length})</span>`:''):'');
  // Crosshair + tooltip: snap to the nearest sample, list every shown series there.
  const svg=$('effect-plot').querySelector('svg'), tip=$('effect-tooltip'), cross=$('effect-crosshair');
  const move=event=>{
    if(!shown.length)return;
    const box=svg.getBoundingClientRect(), vx=(event.clientX-box.left)/box.width*w;
    const i=t.reduce((best,v,j)=>Math.abs(x(v)-vx)<Math.abs(x(t[best])-vx)?j:best,0);
    cross.setAttribute('x1',x(t[i]));cross.setAttribute('x2',x(t[i]));cross.setAttribute('visibility','visible');
    tip.replaceChildren();
    const head=document.createElement('div');head.className='effect-tooltip-head';head.textContent=`t = ${t[i].toFixed(3)}`;tip.append(head);
    if(overlay){const next=document.createElement('div'), code=document.createElement('code');next.className='effect-tooltip-head';code.textContent=tokenText(result.path_predictions[i].token);next.append('Next token ',code);tip.append(next);}
    [...shown].sort((a,b)=>b.values[metric][i]-a.values[metric][i]).forEach(row=>{
      const line=document.createElement('div'), sw=document.createElement('i'), val=document.createElement('b'), name=document.createElement('span');
      sw.style.background=effectColor(row,nLayers); val.textContent=row.values[metric][i].toFixed(4); name.textContent=row.label+(metric==='c'?` · cumulative L2 ${formatL2(row.cumulative_length?.[i])} / ${formatL2(row.total_length)}`:'');
      line.append(sw,val,name); tip.append(line);
    });
    tip.classList.remove('hidden');
    const px=x(t[i])/w*box.width;
    tip.style.left=`${px>box.width/2?px-tip.offsetWidth-12:px+12}px`; tip.style.top='12px';
  };
  $('effect-hit').addEventListener('pointermove',move);
  $('effect-hit').addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');tip.classList.add('hidden');});
  // List
  $('effect-score-head').textContent=info.plateau_score?'Plateau score Δt':'Plateau score';
  $('effect-rows').innerHTML=effect.rows.map(row=>{
    const score=row.plateau_score[metric], checked=effectSelection.has(row.key), available=definedMetric(row,metric);
    const reason=metric==='c'?row.c_undefined_reason:row.d_undefined_reason;
    return `<tr class="${available?'':'undefined'}${checked?' selected':''}"><td><input type="checkbox" data-key="${esc(row.key)}" aria-label="Show ${esc(row.label)}"${checked?' checked':''}${available?'':' disabled'}></td>
      <th scope="row"><i class="swatch" style="background:${effectColor(row,nLayers)}"></i>${esc(row.label)}${row.patched?'<small>Patched</small>':''}</th>
      <td title="${row.endpoint_l2}">${formatL2(row.endpoint_l2)}</td>
      <td>${!available?esc(reason || 'undefined'):score==null?'—':score.toFixed(3)}</td></tr>`;
  }).join('');
}
