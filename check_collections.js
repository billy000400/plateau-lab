// Real IndexedDB API semantics via fake-indexeddb; no browser/user data is touched.
'use strict';
const assert=require('node:assert/strict');
const {IDBFactory}=require('fake-indexeddb');
const {create}=require('./web/collections.js');
(async()=>{
  const factory=new IDBFactory();
  // Seed the exact version-one store used by the old hosted Explorer.
  await new Promise((resolve,reject)=>{
    const request=factory.open('plateau-lab',1);
    request.onupgradeneeded=()=>request.result.createObjectStore('runs',{keyPath:'id'}).put({id:'old',notes:'keep me'});
    request.onerror=()=>reject(request.error);request.onsuccess=()=>{request.result.close();resolve();};
  });
  const collections=create(factory);
  assert.deepEqual(await collections.read('runs'),[{id:'old',notes:'keep me'}]);
  assert.deepEqual(await collections.read('examples'),[]);
  assert(await collections.put('examples',{id:'old',tag:'Plateau',notes:'first'}));
  assert(await collections.put('examples',{id:'old',tag:'Smooth transition',notes:'updated'}));
  assert(await collections.put('runs',{id:'new',token_matrix:{steps:[1,2,3]}}));
  const reloaded=create(factory);
  assert.equal((await reloaded.read('runs')).length,2);
  assert.deepEqual(await reloaded.read('examples'),[{id:'old',tag:'Smooth transition',notes:'updated'}]);
  await reloaded.remove('runs',['old']);
  assert.equal((await reloaded.read('runs')).length,1);
  assert.equal((await reloaded.read('examples')).length,1);
  const privateWindow=create(undefined);
  assert.equal(await privateWindow.put('runs',{id:'session',notes:'export me'}),false);
  assert.equal(privateWindow.persistent,false);
  assert.deepEqual(await privateWindow.read('runs'),[{id:'session',notes:'export me'}]);
  // A failed write must remain exportable even if subsequent reads work.
  const quotaFactory={open(){throw new Error('Quota exceeded');}};
  const full=create(quotaFactory);
  assert.equal(await full.put('examples',{id:'quota',notes:'unsaved'}),false);
  assert.equal((await full.read('examples'))[0].notes,'unsaved');
  assert.equal(full.persistent,false);
  console.log('PASS: v1→v2 migration, persistent notes/Examples, reload, history-only deletion and storage failure fallback.');
})().catch(error=>{console.error(error);process.exitCode=1;});
