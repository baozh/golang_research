// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// Package syncmap provides a concurrent map implementation.
// It is a prototype for a proposed addition to the sync package
// in the standard library.

// (https://golang.org/issue/18177)
// https://go.googlesource.com/sync/+/master/syncmap/map.go

//设计思路（copy-on-write）：
//使用了atomic.Value, atomic.Pointer来对struct, pointer做原子的load, store操作。
//有一个read map, 一个dirty map。
//写操作：如果read map中含有这个key slot，则更新它。
//       否则，如果dirty map中含有这个key slot，则更新它。
//       否则，创建一个新的entry，插入到dirty map中。
//读操作：先从read map中查找（不需要加锁），如果找不到，再从dirty map(需要加锁)中找。
//删除：不会实际删除key-value slot，只是将entry.value.p 置为nil。
//遍历：如果dirty map不为空，则将它合并到 read map中。再遍历read map。
//更新read map的时机：在读操作中，记录read map miss的次数，如果超过len(dirty map)，则将dirty map的数据导到read map中。

package syncmap

import (
	"sync"
	"sync/atomic"
	"unsafe"
)

// Map is a concurrent map with amortized-constant-time loads, stores, and deletes.
// It is safe for multiple goroutines to call a Map's methods concurrently.
//
// The zero Map is valid and empty.
//
// A Map must not be copied after first use.
type Map struct {
	mu sync.Mutex

	// read contains the portion of the map's contents that are safe for
	// concurrent access (with or without mu held).
	//
	// The read field itself is always safe to load, but must only be stored with
	// mu held.
	//
	// Entries stored in read may be updated concurrently without mu,
	// but updating a previously-expunged entry requires that the entry be copied to the dirty
	// map and unexpunged with mu held. 见store()中的注释说明.
	read atomic.Value // readOnly (read map)，查询、更新一个已存在的key slot时不需要加锁, 全量复制时需要加锁
	//这里利用了atomic.Value的原子操作，read的readOnly struct中子元素的读取也是原子的。

	// dirty contains the portion of the map's contents that require mu to be
	// held. To ensure that the dirty map can be promoted to the read map quickly,
	// it also includes all of the non-expunged entries in the read map.
	//
	// Expunged entries are not stored in the dirty map. An expunged entry in the
	// clean map must be unexpunged and added to the dirty map before a new value
	// can be stored to it. 见store()中的注释说明.
	//
	// If the dirty map is nil, the next write to the map will initialize it by
	// making a shallow copy of the clean map, omitting stale entries.
	//dirty map包含有read map所有的元素。但是，在初始化dirty map时，没拷贝read map中nil的kv对。见store()中的dirtyLocked().
	dirty map[interface{}]*entry //dirty map 用来写，访问时需要加mu锁。它包含了read map的所有元素和刚插入的元素.

	// misses counts the number of loads since the read map was last updated that
	// needed to lock mu to determine whether the key was present.
	//
	// Once enough misses have occurred to cover the cost of copying the dirty
	// map, the dirty map will be promoted to the read map (in the unamended
	// state) and the next store to the map will make a new dirty copy.
	misses int //查询时 命中dirty map的次数。如果读的次数超过一定次数，会将dirty map合并到read map中.
}

// readOnly is an immutable struct stored atomically in the Map.read field.
type readOnly struct {
	m       map[interface{}]*entry
	amended bool // true if the dirty map contains some key not in m. 只要dirty map不为空，amended就为true
}

// expunged is an arbitrary pointer that marks entries which have been deleted
// from the dirty map.
//指示 此kv slot已删除，且dirty map中不含有此key slot, 而此时read map含有此kv slot.
var expunged = unsafe.Pointer(new(interface{}))

// An entry is a slot in the map corresponding to a particular key.
type entry struct {
	// p points to the interface{} value stored for the entry.
	//
	// If p == nil, the entry has been deleted and m.dirty == nil.
	//
	// If p == expunged, the entry has been deleted, m.dirty != nil, and the entry
	// is missing from m.dirty.
	//
	// Otherwise, the entry is valid and recorded in m.read.m[key] and, if m.dirty
	// != nil, in m.dirty[key].
	//
	// An entry can be deleted by atomic replacement with nil: when m.dirty is
	// next created, it will atomically replace nil with expunged and leave
	// m.dirty[key] unset.
	//
	// An entry's associated value can be updated by atomic replacement, provided
	// p != expunged. If p == expunged, an entry's associated value can be updated
	// only after first setting m.dirty[key] = e so that lookups using the dirty
	// map find the entry.
	p unsafe.Pointer // *interface{}  更新时用原子操作.
}

func newEntry(i interface{}) *entry {
	return &entry{p: unsafe.Pointer(&i)}
}

// Load returns the value stored in the map for a key, or nil if no
// value is present.
// The ok result indicates whether value was found in the map.
func (m *Map) Load(key interface{}) (value interface{}, ok bool) {
	read, _ := m.read.Load().(readOnly)
	e, ok := read.m[key]
	if !ok && read.amended {
		m.mu.Lock()
		// Avoid reporting a spurious miss if m.dirty got promoted while we were
		// blocked on m.mu. (If further loads of the same key will not miss, it's
		// not worth copying the dirty map for this key.)
		// 这里又查了一次，是因为：在锁区间中 可能会将dirty map加载到read map中，所以需要再读一次。
		read, _ = m.read.Load().(readOnly)
		e, ok = read.m[key]
		if !ok && read.amended {
			e, ok = m.dirty[key]
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map. 不管从dirty map中有没有找到此key/value slot，都视为一个read map miss
			m.missLocked()
		}
		m.mu.Unlock()
	}
	if !ok {
		return nil, false
	}
	return e.load()
}
func (e *entry) load() (value interface{}, ok bool) {
	p := atomic.LoadPointer(&e.p)
	if p == nil || p == expunged {
		return nil, false
	}
	return *(*interface{})(p), true //p是*interface{}型
}

// Store sets the value for a key.
func (m *Map) Store(key, value interface{}) {
	//要求：dirty map包含有read map所有的元素。但是，在初始化dirty map时没拷贝nil的kv对。所以，
	//在初始化dirty map时，将read map中的 nil kv置为expunged。之后store(k,v)时，如果发现read map中的key slot的value为expunged，则还需要设置dirty map中的值。
	//分几种情况：
	//1. read map有此key，则更新它的value。
	//            更新value时，若value不是expunged，则更新。
	//                        若value是expunged，则 将它置为nil，并将entry也保存到dirty map中。
	//2. read map中没有此key，而dirty map中有此key。则更新e中的值。
	//3. read map, dirty map都没有此key。则：（若dirty map为空）用read map 初始化dirty map, 然后 将新的entry保存到dirty map中。

	//如果read map中已存在这个key，则更新它
	//如果tryStore()返回false，说明之前初始化了dirty map中，read map中的nil e.p 会被置为expunged
	read, _ :=
		m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok && e.tryStore(&value) {
		return
	}

	m.mu.Lock()
	read, _ = m.read.Load().(readOnly) //再查一次
	if e, ok := read.m[key]; ok {
		if e.unexpungeLocked() {
			// The entry was previously expunged, which implies that there is a
			// non-nil dirty map and this entry is not in it.
			m.dirty[key] = e //设置了 同一个entry地址
		}
		e.storeLocked(&value) //set it
	} else if e, ok := m.dirty[key]; ok {
		e.storeLocked(&value) //set it
	} else { //如果read map, dirty map中都没查到，则将它插入到dirty map中。
		if !read.amended {
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			m.dirtyLocked()                                  //用read map初始化 dirty map
			m.read.Store(readOnly{m: read.m, amended: true}) //更新read map.amended
		}
		m.dirty[key] = newEntry(value) //存储在dirty map
	}
	m.mu.Unlock()
}

// tryStore stores a value if the entry has not been expunged.
//
// If the entry is expunged, tryStore returns false and leaves the entry
// unchanged.
// 设置e.p为i
func (e *entry) tryStore(i *interface{}) bool {
	p := atomic.LoadPointer(&e.p)
	if p == expunged {
		return false
	}
	for {
		if atomic.CompareAndSwapPointer(&e.p, p, unsafe.Pointer(i)) {
			return true
		}
		p = atomic.LoadPointer(&e.p)
		if p == expunged {
			return false
		}
	}
}

// unexpungeLocked ensures that the entry is not marked as expunged.
//
// If the entry was previously expunged, it must be added to the dirty map
// before m.mu is unlocked.
// 判断e是否置为expunged标记, 如果是，则置为nil
func (e *entry) unexpungeLocked() (wasExpunged bool) {
	return atomic.CompareAndSwapPointer(&e.p, expunged, nil)
}

// storeLocked unconditionally stores a value to the entry.
//
// The entry must be known not to be expunged.
// 设置e.p为i
func (e *entry) storeLocked(i *interface{}) {
	atomic.StorePointer(&e.p, unsafe.Pointer(i))
}

// LoadOrStore returns the existing value for the key if present.
// Otherwise, it stores and returns the given value.
// The loaded result is true if the value was loaded, false if stored.
// 如果key存在，则返回 它的value, true
// 如果key不存在，则设置value, 返回新的value, false.
func (m *Map) LoadOrStore(key, value interface{}) (actual interface{}, loaded bool) {
	// Avoid locking if it's a clean hit.
	read, _ := m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok {
		//如果已存在key slot，则读取value。（如果value是nil，则设置更新它。否则，返回它的value。）
		actual, loaded, ok := e.tryLoadOrStore(value)
		if ok {
			return actual, loaded
		}
	}
	m.mu.Lock()
	read, _ = m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok { //如果read map已存在key slot，(若value是expunged)要拷贝一份到dirty map中。
		if e.unexpungeLocked() {
			m.dirty[key] = e
		}
		actual, loaded, _ = e.tryLoadOrStore(value)
	} else if e, ok := m.dirty[key]; ok {
		actual, loaded, _ = e.tryLoadOrStore(value)
		m.missLocked()
	} else { //如果read map, dirty map中都没查到，则将它插入到dirty map中。
		if !read.amended { //如果dirty map为空，则要用read map来初始化它.
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			m.dirtyLocked()
			m.read.Store(readOnly{m: read.m, amended: true})
		}
		m.dirty[key] = newEntry(value)
		actual, loaded = value, false
	}
	m.mu.Unlock()
	return actual, loaded
}

// tryLoadOrStore atomically loads or stores a value if the entry is not
// expunged.
//
// If the entry is expunged, tryLoadOrStore leaves the entry unchanged and
// returns with ok==false.
func (e *entry) tryLoadOrStore(i interface{}) (actual interface{}, loaded, ok bool) {
	p := atomic.LoadPointer(&e.p)
	if p == expunged {
		return nil, false, false
	}
	if p != nil {
		return *(*interface{})(p), true, true
	}
	// Copy the interface after the first load to make this method more amenable
	// to escape analysis: if we hit the "load" path or the entry is expunged, we
	// shouldn't bother heap-allocating.
	ic := i
	for {
		if atomic.CompareAndSwapPointer(&e.p, nil, unsafe.Pointer(&ic)) {
			return i, false, true
		}
		p = atomic.LoadPointer(&e.p)
		if p == expunged {
			return nil, false, false
		}
		if p != nil {
			return *(*interface{})(p), true, true
		}
	}
}

// Delete deletes the value for a key.
func (m *Map) Delete(key interface{}) {
	read, _ := m.read.Load().(readOnly)
	e, ok := read.m[key]
	if !ok && read.amended {
		m.mu.Lock()
		read, _ = m.read.Load().(readOnly)
		e, ok = read.m[key]
		if !ok && read.amended {
			delete(m.dirty, key)
		}
		m.mu.Unlock()
	}
	if ok {
		e.delete() //将e.p置为nil
	}
}
func (e *entry) delete() (hadValue bool) {
	for {
		p := atomic.LoadPointer(&e.p)
		if p == nil || p == expunged {
			return false
		}
		if atomic.CompareAndSwapPointer(&e.p, p, nil) {
			return true
		}
	}
}

// Range calls f sequentially for each key and value present in the map.
// If f returns false, range stops the iteration.
//
// Range does not necessarily correspond to any consistent snapshot of the Map's
// contents: no key will be visited more than once, but if the value for any key
// is stored or deleted concurrently, Range may reflect any mapping for that key
// from any point during the Range call.
//
// Range may be O(N) with the number of elements in the map even if f returns
// false after a constant number of calls.
func (m *Map) Range(f func(key, value interface{}) bool) {
	// We need to be able to iterate over all of the keys that were already
	// present at the start of the call to Range.
	// If read.amended is false, then read.m satisfies that property without
	// requiring us to hold m.mu for a long time.
	read, _ := m.read.Load().(readOnly)
	if read.amended {
		// m.dirty contains keys not in read.m. Fortunately, Range is already O(N)
		// (assuming the caller does not break out early), so a call to Range
		// amortizes an entire copy of the map: we can promote the dirty copy
		// immediately!
		m.mu.Lock()
		read, _ = m.read.Load().(readOnly)
		if read.amended {
			// 将dirty map中的元素 合并到read map
			read = readOnly{m: m.dirty}
			m.read.Store(read)
			m.dirty = nil
			m.misses = 0
		}
		m.mu.Unlock()
	}
	//遍历read map, 这样就不需要加锁了。
	for k, e := range read.m {
		v, ok := e.load()
		if !ok { //value被删除
			continue
		}
		if !f(k, v) {
			break
		}
	}
}
func (m *Map) missLocked() {
	m.misses++
	if m.misses < len(m.dirty) {
		return
	}
	m.read.Store(readOnly{m: m.dirty})
	m.dirty = nil
	m.misses = 0
}
func (m *Map) dirtyLocked() {
	if m.dirty != nil {
		return
	}
	read, _ := m.read.Load().(readOnly)
	m.dirty = make(map[interface{}]*entry, len(read.m))
	for k, e := range read.m {
		if !e.tryExpungeLocked() {
			m.dirty[k] = e
		}
	}
}

//将e.nil置为e.expunged，并返回e是否已删除
func (e *entry) tryExpungeLocked() (isExpunged bool) {
	p := atomic.LoadPointer(&e.p)
	for p == nil {
		//p 为nil说明 这个k/v被删除了
		if atomic.CompareAndSwapPointer(&e.p, nil, expunged) {
			return true
		}
		p = atomic.LoadPointer(&e.p)
	}
	return p == expunged
}
