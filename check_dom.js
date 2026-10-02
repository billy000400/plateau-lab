// DOM/storage harness for the real workbench functions; not visual browser QA.
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {IDBFactory}=require('fake-indexeddb');
function create() {
  const elements=new Map(),html=fs.readFileSync('web/index.html','utf8');
  function element(id,tag='div') {
    const classes=new Set(),attrs={},handlers={},added=[];
    let markup='',value='';
    return {id,tag,attrs,handlers,dataset:{},checked:false,disabled:false,textContent:'',style:{setProperty:(k,v)=>{attrs[k]=v;}},
      get innerHTML(){return markup;},set innerHTML(v){
        markup=v;added.length=0;
        for(const match of String(v).matchAll(/<([a-z]+)\b[^>]*\bid="([^"]+)"[^>]*>/g))elements.set(match[2],element(match[2],match[1]));
      },
      get options(){return [...markup.matchAll(/<option(?: value="([^"]*)")?[^>]*>([^<]*)/g)].map(m=>({value:m[1] ?? m[2]})).concat(added);},
      get value(){return value || (tag==='select'?this.options[0]?.value || '':'');},set value(v){value=String(v);},
      classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c),toggle:(c,on)=>{on?classes.add(c):classes.delete(c);}},
      setAttribute:(k,v)=>{attrs[k]=v;},addEventListener:(name,fn)=>{handlers[name]=fn;},
      querySelector:()=>element('svg'),getBoundingClientRect:()=>({left:0,width:720}),
      append(...nodes){added.push(...nodes);},appendChild(node){added.push(node);},replaceChildren(){},remove(){},click(){this.onclick?.({detail:1});},focus(){document.activeElement=this;},
      closest:()=>elements.get('help-group'),contains:target=>target===elements.get('c-info') || target===elements.get('c-definition')};
  }
  for(const m of html.matchAll(/<([a-z]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)){
    assert(!elements.has(m[2]),'duplicate id: '+m[2]);const el=element(m[2],m[1]);elements.set(m[2],el);
    for(const cls of (m[0].match(/class="([^"]*)"/)?.[1] || '').split(' '))if(cls)el.classList.add(cls);
  }
  for(const m of html.matchAll(/<(select|textarea)\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/\1>/g)){
    if(m[1]==='select')elements.get(m[2]).innerHTML=m[3];else elements.get(m[2]).value=m[3];
  }
  elements.set('help-group',element('help-group'));
  const labels=[element('prediction-label-a'),element('prediction-label-b')];
  const cards=new Map();
  const document={activeElement:null,handlers:{},body:element('body'),
    getElementById:id=>{assert(elements.has(id),'missing element: '+id);return elements.get(id);},
    addEventListener(name,fn){this.handlers[name]=fn;},createElement:tag=>element(tag,tag),
    querySelectorAll(selector){
      if(selector==='.prediction-label')return labels;
      if(selector==='[data-record]')return [...elements.get('library-grid').innerHTML.matchAll(/data-record="([^"]+)"/g)].map(m=>{
        if(!cards.has(m[1])){const card=element(m[1]);card.dataset.record=m[1];cards.set(m[1],card);}return cards.get(m[1]);
      });
      return [];
    }};
  const storage=()=>{const data=new Map();return {getItem:key=>data.get(key) ?? null,setItem:(key,v)=>data.set(key,String(v)),removeItem:key=>data.delete(key)};};
  const context=vm.createContext({document,console,Blob,URL,window:{addEventListener(){},scrollTo(){}},
    indexedDB:new IDBFactory(),localStorage:storage(),sessionStorage:storage(),
    setTimeout:()=>1,clearTimeout(){},confirm:()=>true,
    fetch:async()=>{throw Error('Unexpected network request');},
    PlateauRecords:require('./web/records.js'),PlateauCollections:require('./web/collections.js'),PlateauTokenMatrix:require('./web/token-matrix.js')});
  // The module's default factory belongs to Node; explicitly use this isolated DB.
  context.PlateauCollections={create:()=>require('./web/collections.js').create(context.indexedDB)};
  vm.runInContext(fs.readFileSync('web/effects.js','utf8'),context);
  vm.runInContext(fs.readFileSync('web/app.js','utf8').replace(/\ninit\(\);\s*$/,''),context);
  return {context,elements,html,document,run:code=>vm.runInContext(code,context)};
}
module.exports={create};
