"""Lossless, disk-backed append-only telemetry; no simulation equations change.

Lazy immutable dicts keep audits from expanding the entire archive at once.
Pickle archives contain compressed records, not references to working databases.
"""
from collections.abc import Mapping,Sequence
from pathlib import Path
import pickle
import sqlite3
import zlib


class LazyRecord(dict):
    def __init__(self,owner,index):
        # A nonempty dict subclass also uses items() in the JSON C encoder.
        super().__init__({'_lazy_record':True});self.owner,self.index=owner,index
    def payload(self):return self.owner.load(self.index)
    def __getitem__(self,key):
        header=self.owner.headers[self.index]
        return header[key] if key in header else self.payload()[key]
    def __iter__(self):return iter(self.payload())
    def __len__(self):return len(self.payload())
    def __contains__(self,key):return key in self.payload()
    def get(self,key,default=None):
        header=self.owner.headers[self.index]
        return header.get(key,default) if key in header else self.payload().get(key,default)
    def keys(self):return self.payload().keys()
    def items(self):return self.payload().items()
    def values(self):return self.payload().values()
    def copy(self):return dict(self.payload())
    def __reduce_ex__(self,protocol):return dict,(self.payload(),)


class MemoryEvents(Sequence):
    def __init__(self,initial=()):
        self.headers=[];self.blobs=[];self.last_index=None;self.last_value=None
        self.extend(initial)
    def __len__(self):return len(self.headers)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self))) ]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return LazyRecord(self,index)
    def __iter__(self):return (LazyRecord(self,i) for i in range(len(self)))
    def append(self,value):
        if isinstance(value,tuple) and len(value)==2 and isinstance(value[1],bytes):header,blob=value
        else:
            if isinstance(value,LazyRecord):value=value.copy()
            header={k:value[k] for k in ('event','event_id','time','schema') if k in value}
            blob=zlib.compress(pickle.dumps(value,protocol=5),1)
        self.headers.append(header);self.put(blob)
    def extend(self,values):
        for value in values:self.append(value)
    def put(self,blob):self.blobs.append(blob)
    def encoded(self):return zip(self.headers,self.blobs)
    def load(self,index):
        if self.last_index!=index:
            self.last_value=pickle.loads(zlib.decompress(self.blobs[index]));self.last_index=index
        return self.last_value
    def __reduce_ex__(self,protocol):return MemoryEvents,(),None,iter(self.encoded()),None


class DiskEvents(MemoryEvents):
    def __init__(self,path,initial=()):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('Telemetry working database already exists: '+str(path))
        self.path=path;self.connection=sqlite3.connect(path)
        self.connection.execute('PRAGMA journal_mode=WAL');self.connection.execute('PRAGMA synchronous=NORMAL')
        self.connection.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        super().__init__(initial)
    def put(self,blob):
        self.connection.execute('INSERT INTO records(id,payload) VALUES (?,?)',(len(self.headers)-1,blob))
        if len(self.headers)%256==0:self.connection.commit()
    def encoded(self):
        self.connection.commit()
        return ((header,row[0]) for header,row in zip(self.headers,self.connection.execute('SELECT payload FROM records ORDER BY id')))
    def load(self,index):
        if self.last_index!=index:
            blob=self.connection.execute('SELECT payload FROM records WHERE id=?',(index,)).fetchone()[0]
            self.last_value=pickle.loads(zlib.decompress(blob));self.last_index=index
        return self.last_value


class EventIndex(Mapping):
    def __init__(self,events):self.events=events
    def __getitem__(self,key):
        index=int(key.removeprefix('rdt:'));value=self.events[index]
        if value['event_id']!=key:raise KeyError(key)
        return value
    def __setitem__(self,key,value):
        if self[key]['event_id']!=value['event_id']:raise ValueError('Event index differs from sequence')
    def __iter__(self):return (e['event_id'] for e in self.events)
    def __len__(self):return len(self.events)


def install(driver):
    import panama_uav.redesign as redesign
    if getattr(redesign,'_compact_telemetry_installed',False):return
    original=redesign.experimental_class
    def compact_class(condition):
        parent=original(condition)
        class CompactRemote(parent):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                folder=Path(self.parameters.failure_directory).parent/'compact_working'
                self.remote_events=DiskEvents(folder/'remote.sqlite',self.remote_events)
                self.events_by_id=EventIndex(self.remote_events)
                self.oracle_events=DiskEvents(folder/'oracle.sqlite',self.oracle_events)
                self.transport.log=DiskEvents(folder/'transport.sqlite',self.transport.log)
        return CompactRemote
    redesign.experimental_class=compact_class;redesign._compact_telemetry_installed=True
