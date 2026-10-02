/* Shared, side-effect-free migration and export helpers. Never infer c from d. */
'use strict';
const PlateauRecords = (() => {
  const C={id:'c',label:'Cumulative path progress c(t)',range:[0,1],plateau_score:true};
  const D={id:'relative_l2_shinkle',label:'Relative endpoint distance d(t)',range:[0,1],plateau_score:true};
  const valid = values => Array.isArray(values) && values.length>0 && values.every(Number.isFinite);
  function width(t,values) {
    if(!valid(values))return null;
    const crossing=level=>{const i=values.findIndex(v=>v>=level);return i<0?null:i===0?t[0]:t[i-1]+(level-values[i-1])/(values[i]-values[i-1])*(t[i]-t[i-1]);};
    const a=crossing(.1),b=crossing(.9);return a==null || b==null?null:b-a;
  }
  function normalize(input) {
    if(!input || ![1,2,3,4,5,6,7].includes(input.schema_version ?? 1) || !input.id ||
       typeof input.model!=='string' || !Number.isFinite(Date.parse(input.created_at)) ||
       !input.settings || typeof input.sequence_a!=='string' || typeof input.sequence_b!=='string')
      throw new Error('Unsupported or incomplete Plateau Lab result');
    const r=JSON.parse(JSON.stringify(input));
    const number=(value,name,integer=false)=>{
      if(value!=null && (!Number.isFinite(value) || (integer && !Number.isInteger(value))))throw new Error('Invalid '+name);
    };
    for(const key of ['patch_layer','steps','patch_count','patch_start_a','patch_start_b'])number(r.settings[key],key,true);
    if(r.settings.interpolation!=null && !['linear','slerp'].includes(r.settings.interpolation))throw new Error('Invalid interpolation');
    for(const curve of r.curves || [])number(curve.total_length,'total path length');
    if(r.l2_distances){
      for(const key of ['patch_layer','source_token_count'])number(r.l2_distances[key],key,true);
      for(const key of ['source_l2','source_last_token_l2'])number(r.l2_distances[key],key);
      for(const row of r.l2_distances.layers || []){
        number(row.layer,'layer',true);number(row.natural_l2,'natural L2');number(row.patched_l2,'patched L2');
      }
      for(const row of r.l2_distances.source_token_l2 || []){
        number(row.position_a,'position A',true);number(row.position_b,'position B',true);number(row.l2,'source token L2');
      }
    }
    r.input_tokens ??= [[],[]]; r.predictions ??= [{},{}];
    r.predictions=r.predictions.map(p=>({...p,tokens:p.tokens || [],continuation:p.continuation || ''}));
    r.metrics ??= {}; r.path_predictions ??= [];
    if(!r.effect) {
      if(!Array.isArray(r.curves) || !r.curves.length)throw new Error('No measured curves');
      const t=r.curves[0].t;
      if(!valid(t) || t.length<2)throw new Error('Invalid sample grid');
      const rows=r.curves.map(curve=>{
        if(curve.t.length!==t.length || curve.t.some((v,i)=>v!==t[i]))throw new Error('Mismatched sample grids');
        const values={c:Array.isArray(curve.c)?curve.c:null,relative_l2_shinkle:curve.d};
        const endpoint=r.l2_distances?.layers?.find(layer=>String(layer.layer)===curve.key)?.patched_l2 ?? null;
        return {...curve,layer:curve.key==='logits'?null:Number(curve.key),label:curve.title,
          patched:curve.key===String(r.settings.patch_layer),endpoint_l2:curve.endpoint_l2 ?? endpoint,
          values,plateau_score:Object.fromEntries(Object.entries(values).map(([key,v])=>[key,width(t,v)]))};
      });
      r.effect={t,metrics:[C,D],rows};
    }
    if(!valid(r.effect.t) || r.effect.t.length<2 || !Array.isArray(r.effect.rows) || !r.effect.rows.length ||
       !Array.isArray(r.effect.metrics) || !r.effect.metrics.length)throw new Error('Invalid effect table');
    if(!r.effect.metrics.some(m=>m.id==='c'))r.effect.metrics.unshift(C);
    r.effect.rows.forEach(row=>{
      if(!row.values || typeof row.key!=='string')throw new Error('Invalid readout');
      number(row.layer,'readout layer',true);number(row.endpoint_l2,'endpoint L2');number(row.total_length,'total path length');
      row.defined_metrics={}; row.plateau_score ??= {};
      r.effect.metrics.forEach(m=>{
        const values=row.values[m.id];
        if(values!=null && (!Array.isArray(values) || values.length!==r.effect.t.length))throw new Error('Invalid metric length');
        row.defined_metrics[m.id]=valid(values);
        if(!row.defined_metrics[m.id])row.values[m.id]=null;
        row.plateau_score[m.id] ??= width(r.effect.t,row.values[m.id]);
      });
      row.defined=Object.values(row.defined_metrics).some(Boolean);
      row.c_status ??= row.values.c?'ok':'not_recorded';
      row.c_undefined_reason ??= row.c_status==='not_recorded'?'c(t) was not recorded; rerun to measure it.':null;
    });
    r.settings.representative_layers ??= (r.curves || r.effect.rows).filter(row=>row.key!=='logits').map(row=>Number(row.key));
    r.l2_distances ??= {layers:r.effect.rows.filter(row=>row.key!=='logits').map(row=>({layer:row.layer,patched_l2:row.endpoint_l2}))};
    // Reconstruct only the d arrays actually recorded in schemas 5/6, never c.
    r.curves ??= r.effect.rows.filter(row=>row.key==='logits' || r.settings.representative_layers.includes(Number(row.key)))
      .map(row=>({key:row.key,title:row.label,t:r.effect.t,d:row.values.relative_l2_shinkle || r.effect.t.map(()=>null),
        ...(row.values.c?{c:row.values.c,step_lengths:row.step_lengths,cumulative_length:row.cumulative_length,total_length:row.total_length}:{}),
        c_status:row.c_status,d_status:row.d_status,c_undefined_reason:row.c_undefined_reason,d_undefined_reason:row.d_undefined_reason}));
    return r;
  }
  function parse(text) {
    try {const data=JSON.parse(text); return Array.isArray(data)?data:[data];}
    catch {return text.split(/\r?\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));}
  }
  function csv(records) {
    const columns=['id','model','sequence_a','sequence_b','continuation_a','continuation_b','tag','notes','schema_version',
      'patch_layer','patch_position','context','interpolation','generation','curve','t','c','d',
      'relative_l2_janiak','relative_l2_projected','step_length','cumulative_length','total_length',
      'c_status','d_status','c_undefined_reason','d_undefined_reason','metric_definitions'];
    const quote=v=>'"'+String(v ?? '').replace(/"/g,'""')+'"';
    const lines=[columns];
    records.map(normalize).forEach(r=>r.effect.rows.forEach(row=>r.effect.t.forEach((t,i)=>{
      const source=(r.curves || []).find(c=>c.key===row.key) || row;
      const data={...r,...r.settings,curve:row.key,t,continuation_a:r.predictions[0].continuation,
        continuation_b:r.predictions[1].continuation,c:row.values.c?.[i],d:row.values.relative_l2_shinkle?.[i],
        relative_l2_janiak:row.values.relative_l2_janiak?.[i],relative_l2_projected:row.values.relative_l2_projected?.[i],
        step_length:source.step_lengths?.[i],cumulative_length:source.cumulative_length?.[i],total_length:source.total_length,
        c_status:source.c_status || row.c_status,d_status:source.d_status || row.d_status || (row.values.relative_l2_shinkle?'ok':'undefined'),
        c_undefined_reason:source.c_undefined_reason || row.c_undefined_reason,d_undefined_reason:source.d_undefined_reason || row.d_undefined_reason,
        metric_definitions:r.metric_definitions?JSON.stringify(r.metric_definitions):''};
      lines.push(columns.map(c=>data[c]));
    })));
    return '\ufeff'+lines.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  }
  // Same representative-curve columns as the original local CSV export.
  function classicCsv(records) {
    const columns=['id','model','sequence_a','sequence_b','continuation_a','continuation_b','tag','notes',
      'patch_layer','interpolation','curve','t','d','fixed_context','endpoint_reference','device','hardware_name',
      'dtype','batch_size','prediction_cache','patch_position','patch_start_a','patch_start_b','patch_count',
      'interpolation_unit','measurement_position','schema_version','c','step_length','cumulative_length','total_length',
      'c_status','d_status','c_undefined_reason','d_undefined_reason','metric_definitions'];
    const quote=value=>'"'+String(value===true?'True':value===false?'False':value ?? '').replace(/"/g,'""')+'"';
    const rows=[columns];
    for(const input of records){
      const r=normalize(input),s=r.settings;
      for(const curve of r.curves)curve.t.forEach((t,i)=>{
        const data={...r,...s,curve:curve.title,t,d:curve.d?.[i],c:curve.c?.[i],
          continuation_a:r.predictions[0].continuation,continuation_b:r.predictions[1].continuation,
          fixed_context:s.context || 'a',endpoint_reference:s.endpoint_reference || 'natural_matching_prefix',
          device:r.device || r.backend?.device || '',hardware_name:r.hardware?.name || '',prediction_cache:s.prediction_cache ?? false,
          patch_position:s.patch_position || 'last_token',patch_start_a:s.patch_start_a ?? r.input_tokens[0].length-1,
          patch_start_b:s.patch_start_b ?? r.input_tokens[1].length-1,patch_count:s.patch_count ?? 1,
          interpolation_unit:s.interpolation_unit || 'per_token_shared_t',measurement_position:s.measurement_position || 'last_token',
          schema_version:r.schema_version ?? 1,step_length:curve.step_lengths?.[i],cumulative_length:curve.cumulative_length?.[i],
          total_length:curve.total_length,c_status:curve.c_status || 'not_recorded',d_status:curve.d_status || 'ok',
          c_undefined_reason:curve.c_undefined_reason,d_undefined_reason:curve.d_undefined_reason,
          metric_definitions:r.metric_definitions?JSON.stringify(r.metric_definitions):''};
        rows.push(columns.map(key=>data[key]));
      });
    }
    return '\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  }
  return {normalize,parse,csv,classicCsv,valid};
})();
if(typeof module!=='undefined')module.exports=PlateauRecords;
