/* Private browser collections. Version 2 adds Examples without changing old runs. */
'use strict';
const PlateauCollections = (() => {
  function create(factory=globalThis.indexedDB) {
    const snapshots={runs:new Map(),examples:new Map()}, pending={runs:new Map(),examples:new Map()};
    let persistent=true;
    function database() {
      return new Promise((resolve,reject)=>{
        if(!factory){reject(new Error('Browser storage is unavailable'));return;}
        const request=factory.open('plateau-lab',2);
        let blocked=false;
        request.onupgradeneeded=()=>{
          for(const name of ['runs','examples'])if(!request.result.objectStoreNames.contains(name))request.result.createObjectStore(name,{keyPath:'id'});
        };
        request.onblocked=()=>{blocked=true;reject(new Error('Close other Plateau Lab tabs to upgrade browser storage'));};
        request.onsuccess=()=>{
          const db=request.result;
          db.onversionchange=()=>db.close();
          if(blocked)db.close();else resolve(db);
        };
        request.onerror=()=>reject(request.error);
      });
    }
    async function transaction(name,mode,action) {
      const db=await database();
      return new Promise((resolve,reject)=>{
        const tx=db.transaction(name,mode);
        let request;
        tx.oncomplete=()=>{db.close();resolve(request?.result);};
        tx.onerror=tx.onabort=()=>{db.close();reject(tx.error || new Error('Browser storage transaction failed'));};
        try {request=action(tx.objectStore(name));}catch(error){tx.abort();db.close();reject(error);}
      });
    }
    return {
      get persistent(){return persistent && !pending.runs.size && !pending.examples.size;},
      async read(name) {
        try {snapshots[name]=new Map((await transaction(name,'readonly',store=>store.getAll())).map(r=>[r.id,r]));persistent=true;}
        catch {persistent=false;}
        return [...new Map([...snapshots[name],...pending[name]]).values()];
      },
      async put(name,record) {
        const copy=JSON.parse(JSON.stringify(record));
        try {await transaction(name,'readwrite',store=>store.put(copy));snapshots[name].set(copy.id,copy);pending[name].delete(copy.id);persistent=true;return true;}
        catch {pending[name].set(copy.id,copy);persistent=false;return false;}
      },
      async remove(name,ids) {
        await transaction(name,'readwrite',store=>{for(const id of ids)store.delete(id);});
        for(const id of ids){snapshots[name].delete(id);pending[name].delete(id);}
      },
    };
  }
  return {create};
})();
if(typeof module!=='undefined')module.exports=PlateauCollections;
