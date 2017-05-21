// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// This file contains the implementation of Go's map type.
//
// A map is just a hash table. The data is arranged
// into an array of buckets. Each bucket contains up to
// 8 key/value pairs. The low-order bits of the hash are
// used to select a bucket. Each bucket contains a few
// high-order bits of each hash to distinguish the entries
// within a single bucket.
//
// If more than 8 keys hash to a bucket, we chain on
// extra buckets.
//
// When the hashtable grows, we allocate a new array
// of buckets twice as big. Buckets are incrementally
// copied from the old bucket array to the new bucket array.
//
// Map iterators walk through the array of buckets and
// return the keys in walk order (bucket #, then overflow
// chain order, then bucket index).  To maintain iteration
// semantics, we never move keys within their bucket (if
// we did, keys might be returned 0 or 2 times).  When
// growing the table, iterators remain iterating through the
// old table and must check the new table if the bucket
// they are iterating through has been moved ("evacuated")
// to the new table.

// Picking loadFactor: too large and we have lots of overflow
// buckets, too small and we waste a lot of space. I wrote
// a simple program to check some stats for different loads:
// (64-bit, 8 byte keys and values)
//  loadFactor    %overflow  bytes/entry     hitprobe    missprobe
//        4.00         2.13        20.77         3.00         4.00
//        4.50         4.05        17.30         3.25         4.50
//        5.00         6.85        14.77         3.50         5.00
//        5.50        10.55        12.94         3.75         5.50
//        6.00        15.27        11.67         4.00         6.00
//        6.50        20.90        10.79         4.25         6.50
//        7.00        27.14        10.15         4.50         7.00
//        7.50        34.03         9.73         4.75         7.50
//        8.00        41.10         9.40         5.00         8.00
//
// %overflow   = percentage of buckets which have an overflow bucket
// bytes/entry = overhead bytes used per key/value pair
// hitprobe    = # of entries to check when looking up a present key
// missprobe   = # of entries to check when looking up an absent key
//
// Keep in mind this data is for maximally loaded tables, i.e. just
// before the table grows. Typical tables will be somewhat less loaded.

import (
	"github.com/baozh/golang_research/runtime/internal/atomic"
	"github.com/baozh/golang_research/runtime/internal/sys"
	"github.com/baozh/golang_research/unsafe"
)

const (
	// Maximum number of key/value pairs a bucket can hold.
	bucketCntBits = 3
	bucketCnt     = 1 << bucketCntBits

	// Maximum average load of a bucket that triggers growth.
	loadFactor = 6.5

	// Maximum key or value size to keep inline (instead of mallocing per element).
	// Must fit in a uint8.
	// Fast versions cannot handle big values - the cutoff size for
	// fast versions in ../../cmd/internal/gc/walk.go must be at most this value.
	maxKeySize   = 128
	maxValueSize = 128

	// data offset should be the size of the bmap struct, but needs to be
	// aligned correctly. For amd64p32 this means 64-bit alignment
	// even though pointers are 32 bit.
	// unsafe.Offsetof() 获取一个struct对齐后的大小，可以学习一下！
	dataOffset = unsafe.Offsetof(struct {
		b bmap
		v int64
	}{}.v)

	// Possible tophash values. We reserve a few possibilities for special marks.
	// Each bucket (including its overflow buckets, if any) will have either all or none of its
	// entries in the evacuated* states (except during the evacuate() method, which only happens
	// during map writes and thus no one else can observe the map during that time).
	empty          = 0 // cell is empty   //表示这个bucket entry为空。（一个bucket有8个bucket entry）
	evacuatedEmpty = 1 // cell is empty, bucket is evacuated.
	evacuatedX     = 2 // key/value is valid.  Entry has been evacuated to first half of larger table.
	evacuatedY     = 3 // same as above, but evacuated to second half of larger table.
	minTopHash     = 4 // minimum tophash for a normal filled cell.
	//bmap->tophash是1~3 表示 此bucket entry已经迁移了，从新的bucket中取数据。

	// flags   这些来标识 并发的读（遍历），写操作       在hmap->flags中
	iterator    = 1 // there may be an iterator using buckets
	oldIterator = 2 // there may be an iterator using oldbuckets
	hashWriting = 4 // a goroutine is writing to the map

	// sentinel bucket ID for iterator checks
	noCheck = 1<<(8*sys.PtrSize) - 1
)

// A header for a Go map.
type hmap struct {
	// Note: the format of the Hmap is encoded in ../../cmd/internal/gc/reflect.go and
	// ../reflect/type.go. Don't change this structure without also changing that code!
	count int    //已存储的key/value对个数    // # live cells == size of map.  Must be first (used by len() builtin)
	flags uint8  //指标hashmap的一些状态（是否有协程在写，是否在遍历buckets,oldbuckets）
	B     uint8  // log_2 of # of buckets (can hold up to loadFactor * 2^B items)   2^B个bucket 注：一个bucket可以存8个key/value对
	hash0 uint32 // hash seed

	buckets    unsafe.Pointer // array of 2^B Buckets. may be nil if count==0.
	oldbuckets unsafe.Pointer // previous bucket array of half the size, non-nil only when growing
	nevacuate  uintptr        // progress counter for evacuation (buckets less than this have been evacuated)  代表下一次待迁移的oldbucket index，值域:0 ~ oldbuckets_size -1

	// If both key and value do not contain pointers and are inline, then we mark bucket
	// type as containing no pointers. This avoids scanning such maps.
	// However, bmap.overflow is a pointer. In order to keep overflow buckets
	// alive, we store pointers to all overflow buckets in hmap.overflow.
	// Overflow is used only if key and value do not contain pointers.  //why? overflow[]为什么不能存 含有指针的k/v对？
	// overflow[0] contains overflow buckets for hmap.buckets.
	// overflow[1] contains overflow buckets for hmap.oldbuckets.
	//第一个指针 是为了 减少hmap结构体的大小
	//第二个指针 是为了 能存储 hiter中slice的指针
	// The first indirection allows us to reduce static size of hmap.
	// The second indirection allows to store a pointer to the slice in hiter.
	overflow *[2]*[]*bmap //指针—> 一个数组，数组元素*[]*bmp类型，存储所有 冲突的buckets.
}

// A bucket for a Go map.
// bucket分配的内存是平坦的，前面是bmap struct，后面是key/value数据
type bmap struct {
	tophash [bucketCnt]uint8 //由于 tophash在1~3 表示 此bucket已经迁移了，用作标志。所以 tophash = 真正的hash + 4
	// Followed by bucketCnt keys and then bucketCnt values.
	// NOTE: packing all the keys together and then all the values together makes the
	// code a bit more complicated than alternating key/value/key/value/... but it allows
	// us to eliminate padding which would be needed for, e.g., map[int64]int8.
	// Followed by an overflow pointer.

	//key/vlue data

	//overflow pointer
}

// A hash iteration structure.
// If you modify hiter, also change cmd/internal/gc/reflect.go to indicate
// the layout of this structure.
type hiter struct {
	key         unsafe.Pointer // Must be in first position.  Write nil to indicate iteration end (see cmd/internal/gc/range.go).
	value       unsafe.Pointer // Must be in second position (see cmd/internal/gc/range.go).
	t           *maptype       // 记录key,value的元类型信息
	h           *hmap
	buckets     unsafe.Pointer // bucket ptr at hash_iter initialization time
	bptr        *bmap          // current bucket
	overflow    [2]*[]*bmap    // keeps overflow buckets alive   没用到啊
	startBucket uintptr        // bucket iteration started at
	offset      uint8          // bucket内部遍历key/value对的offset   intra-bucket offset to start from during iteration (should be big enough to hold bucketCnt-1)
	wrapped     bool           // already wrapped around from end of bucket array to beginning    标识是否 遍历了一遍所有bucket
	B           uint8
	i           uint8   //下一次要遍历的bucket中的 kv index
	bucket      uintptr //下一次要遍历的bucket index
	checkBucket uintptr
}

//是否 已经迁移好了
func evacuated(b *bmap) bool {
	h := b.tophash[0]
	return h > empty && h < minTopHash
}

func (b *bmap) overflow(t *maptype) *bmap {
	return *(**bmap)(add(unsafe.Pointer(b), uintptr(t.bucketsize)-sys.PtrSize))
}

func (h *hmap) setoverflow(t *maptype, b, ovf *bmap) {
	//如果maptype是存储key/value内容的，而不是指针，则将它加入到 h.overflow链表中。
	if t.bucket.kind&kindNoPointers != 0 {
		h.createOverflow()
		*h.overflow[0] = append(*h.overflow[0], ovf)
	}
	*(**bmap)(add(unsafe.Pointer(b), uintptr(t.bucketsize)-sys.PtrSize)) = ovf
}

func (h *hmap) createOverflow() {
	if h.overflow == nil {
		h.overflow = new([2]*[]*bmap)
	}
	if h.overflow[0] == nil {
		h.overflow[0] = new([]*bmap)
	}
}

// makemap implements a Go map creation make(map[k]v, hint)
// If the compiler has determined that the map or the first bucket
// can be created on the stack, h and/or bucket may be non-nil.
// If h != nil, the map can be created directly in h.
// If bucket != nil, bucket can be used as the first bucket.
// hit指 存key/value对元素的个数
func makemap(t *maptype, hint int64, h *hmap, bucket unsafe.Pointer) *hmap {
	if sz := unsafe.Sizeof(hmap{}); sz > 48 || sz != t.hmap.size {
		println("runtime: sizeof(hmap) =", sz, ", t.hmap.size =", t.hmap.size)
		throw("bad hmap size")
	}

	if hint < 0 || int64(int32(hint)) != hint {
		panic(plainError("makemap: size out of range"))
		// TODO: make hint an int, then none of this nonsense
	}

	if !ismapkey(t.key) {
		throw("runtime.makemap: unsupported map key type")
	}

	// check compiler's and reflect's math
	if t.key.size > maxKeySize && (!t.indirectkey || t.keysize != uint8(sys.PtrSize)) ||
		t.key.size <= maxKeySize && (t.indirectkey || t.keysize != uint8(t.key.size)) {
		throw("key size wrong")
	}
	if t.elem.size > maxValueSize && (!t.indirectvalue || t.valuesize != uint8(sys.PtrSize)) ||
		t.elem.size <= maxValueSize && (t.indirectvalue || t.valuesize != uint8(t.elem.size)) {
		throw("value size wrong")
	}

	// invariants we depend on. We should probably check these at compile time
	// somewhere, but for now we'll do it here.
	if t.key.align > bucketCnt {
		throw("key align too big")
	}
	if t.elem.align > bucketCnt {
		throw("value align too big")
	}
	if t.key.size%uintptr(t.key.align) != 0 {
		throw("key size not a multiple of key align")
	}
	if t.elem.size%uintptr(t.elem.align) != 0 {
		throw("value size not a multiple of value align")
	}
	if bucketCnt < 8 {
		throw("bucketsize too small for proper alignment")
	}
	if dataOffset%uintptr(t.key.align) != 0 {
		throw("need padding in bucket (key)")
	}
	if dataOffset%uintptr(t.elem.align) != 0 {
		throw("need padding in bucket (value)")
	}

	// find size parameter which will hold the requested # of elements
	B := uint8(0)
	for ; hint > bucketCnt && float32(hint) > loadFactor*float32(uintptr(1)<<B); B++ {
	}

	// allocate initial hash table
	// if B == 0, the buckets field is allocated lazily later (in mapassign)
	// If hint is large zeroing this memory could take a while.
	buckets := bucket
	if B != 0 {
		buckets = newarray(t.bucket, 1<<B)
	}

	// initialize Hmap
	if h == nil {
		h = (*hmap)(newobject(t.hmap))
	}
	h.count = 0
	h.B = B
	h.flags = 0
	h.hash0 = fastrand1()
	h.buckets = buckets
	h.oldbuckets = nil
	h.nevacuate = 0

	return h
}

// mapaccess1 returns a pointer to h[key].  Never returns nil, instead
// it will return a reference to the zero object for the value type if
// the key is not in the map.
// NOTE: The returned pointer may keep the whole map live, so don't
// hold onto it for very long.
func mapaccess1(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		pc := funcPC(mapaccess1)
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0])
	}

	//是否有协程在写hashmap
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize))) //取hash的低B位 => 第几个bucket         //b是 指向 符合条件的首个bucket
	if c := h.oldbuckets; c != nil {
		oldb := (*bmap)(add(c, (hash&(m>>1))*uintptr(t.bucketsize))) //取hash的低B-1位(因为oldbuckets的桶数比buckets少一倍，所以少取一位) => 第几个bucket
		if !evacuated(oldb) {                                        //why???
			b = oldb
		}
	}
	top := uint8(hash >> (sys.PtrSize*8 - 8)) // 取hash高8位
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize)) //这里为什么要加dataOffset? 因为bucket分配的内存是平坦的，前面是bmap，后面是key/value数据
			if t.indirectkey {
				k = *((*unsafe.Pointer)(k))
			}
			if alg.equal(key, k) {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.valuesize))
				if t.indirectvalue {
					v = *((*unsafe.Pointer)(v))
				}
				return v
			}
		}
		b = b.overflow(t) //遍历下一个bucket
		if b == nil {
			return unsafe.Pointer(&zeroVal[0])
		}
	}
}

//这个和mapaccess1() 没区别，多返回了一个bool参数.
func mapaccess2(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		pc := funcPC(mapaccess2)
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	if h == nil || h.count == 0 {
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + (hash&m)*uintptr(t.bucketsize))) //取hash的低B位 => 第几个bucket         //b是 指向 符合条件的首个bucket
	if c := h.oldbuckets; c != nil {
		oldb := (*bmap)(unsafe.Pointer(uintptr(c) + (hash&(m>>1))*uintptr(t.bucketsize))) //取hash的低B-1位 => 第几个bucket
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (sys.PtrSize*8 - 8)) // 取hash高8位
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey {
				k = *((*unsafe.Pointer)(k))
			}
			if alg.equal(key, k) {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.valuesize))
				if t.indirectvalue {
					v = *((*unsafe.Pointer)(v))
				}
				return v, true
			}
		}
		b = b.overflow(t)
		if b == nil {
			return unsafe.Pointer(&zeroVal[0]), false
		}
	}
}

// returns both key and value. Used by map iterator
func mapaccessK(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, unsafe.Pointer) {
	if h == nil || h.count == 0 {
		return nil, nil
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))
	m := uintptr(1)<<h.B - 1
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		oldb := (*bmap)(unsafe.Pointer(uintptr(c) + (hash&(m>>1))*uintptr(t.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := uint8(hash >> (sys.PtrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey {
				k = *((*unsafe.Pointer)(k))
			}
			if alg.equal(key, k) {
				v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.valuesize))
				if t.indirectvalue {
					v = *((*unsafe.Pointer)(v))
				}
				return k, v
			}
		}
		b = b.overflow(t)
		if b == nil {
			return nil, nil
		}
	}
}

func mapaccess1_fat(t *maptype, h *hmap, key, zero unsafe.Pointer) unsafe.Pointer {
	v := mapaccess1(t, h, key)
	if v == unsafe.Pointer(&zeroVal[0]) {
		return zero
	}
	return v
}

func mapaccess2_fat(t *maptype, h *hmap, key, zero unsafe.Pointer) (unsafe.Pointer, bool) {
	v := mapaccess1(t, h, key)
	if v == unsafe.Pointer(&zeroVal[0]) {
		return zero, false
	}
	return v, true
}

//set(key, value)
func mapassign1(t *maptype, h *hmap, key unsafe.Pointer, val unsafe.Pointer) {
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		pc := funcPC(mapassign1)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
		raceReadObjectPC(t.elem, val, callerpc, pc)
	}
	if msanenabled {
		msanread(key, t.key.size)
		msanread(val, t.elem.size)
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	h.flags |= hashWriting //置标志

	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))

	if h.buckets == nil {
		h.buckets = newarray(t.bucket, 1) //分配一个bucket
	}

again:
	bucket := hash & (uintptr(1)<<h.B - 1) //取hash的低B位 => 第几个bucket
	if h.oldbuckets != nil {
		growWork(t, h, bucket) //将oldbucket中的元素移动中 新map中的bucket中. //注：这里需要先迁移oldbucket，再set(key, value)。
	}
	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(t.bucketsize)))
	top := uint8(hash >> (sys.PtrSize*8 - 8)) // 取hash高8位
	if top < minTopHash {
		top += minTopHash
	}

	var inserti *uint8
	var insertk unsafe.Pointer
	var insertv unsafe.Pointer
	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				//如果这个bucket entry为空，则可以将key/value插入到这里。
				if b.tophash[i] == empty && inserti == nil {
					inserti = &b.tophash[i]
					insertk = add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
					insertv = add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.valuesize))
				}
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			k2 := k
			if t.indirectkey {
				k2 = *((*unsafe.Pointer)(k2))
			}
			if !alg.equal(key, k2) {
				continue
			}

			//如果hashtable中已存在这个key/value对，则 更新它.
			// already have a mapping for key. Update it.
			if t.needkeyupdate {
				typedmemmove(t.key, k2, key)
			}
			v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.valuesize))
			v2 := v
			if t.indirectvalue {
				v2 = *((*unsafe.Pointer)(v2))
			}
			typedmemmove(t.elem, v2, val)
			goto done
		}
		ovf := b.overflow(t) //下一个bucket
		if ovf == nil {
			break
		}
		b = ovf
	}

	// did not find mapping for key. Allocate new cell & add entry.
	if float32(h.count) >= loadFactor*float32((uintptr(1)<<h.B)) && h.count >= bucketCnt {
		hashGrow(t, h) //扩容
		goto again     // Growing the table invalidates everything, so try again
	}

	if inserti == nil {
		// all current buckets are full, allocate a new one.
		newb := (*bmap)(newobject(t.bucket)) //分配一个bucket空间
		h.setoverflow(t, b, newb)            //加入链表中
		inserti = &newb.tophash[0]
		insertk = add(unsafe.Pointer(newb), dataOffset)
		insertv = add(insertk, bucketCnt*uintptr(t.keysize))
	}

	// store new key/value at insert position
	if t.indirectkey {
		kmem := newobject(t.key)
		*(*unsafe.Pointer)(insertk) = kmem
		insertk = kmem
	}
	if t.indirectvalue {
		vmem := newobject(t.elem)
		*(*unsafe.Pointer)(insertv) = vmem
		insertv = vmem
	}

	//存入key/value
	typedmemmove(t.key, insertk, key)
	typedmemmove(t.elem, insertv, val)
	*inserti = top
	h.count++

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

//delete(key)
func mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		pc := funcPC(mapdelete)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	if h == nil || h.count == 0 {
		return
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	h.flags |= hashWriting //置标志

	alg := t.key.alg
	hash := alg.hash(key, uintptr(h.hash0))
	bucket := hash & (uintptr(1)<<h.B - 1)
	if h.oldbuckets != nil {
		growWork(t, h, bucket) //注：这里需要先迁移oldbucket，再删除。
	}

	b := (*bmap)(unsafe.Pointer(uintptr(h.buckets) + bucket*uintptr(t.bucketsize)))
	top := uint8(hash >> (sys.PtrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}

	for {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				continue
			}

			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			k2 := k
			if t.indirectkey {
				k2 = *((*unsafe.Pointer)(k2))
			}
			if !alg.equal(key, k2) {
				continue
			}

			//释放key, value空间
			memclr(k, uintptr(t.keysize))
			v := unsafe.Pointer(uintptr(unsafe.Pointer(b)) + dataOffset + bucketCnt*uintptr(t.keysize) + i*uintptr(t.valuesize))
			memclr(v, uintptr(t.valuesize))
			b.tophash[i] = empty //置标志
			h.count--
			goto done
		}
		b = b.overflow(t) //下一个bucket
		if b == nil {
			goto done
		}
	}

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting //置标志
}

//初始化it
func mapiterinit(t *maptype, h *hmap, it *hiter) {
	// Clear pointer fields so garbage collector does not complain.
	it.key = nil
	it.value = nil
	it.t = nil
	it.h = nil
	it.buckets = nil
	it.bptr = nil
	it.overflow[0] = nil
	it.overflow[1] = nil

	if raceenabled && h != nil {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapiterinit))
	}

	if h == nil || h.count == 0 {
		it.key = nil
		it.value = nil
		return
	}

	if unsafe.Sizeof(hiter{})/sys.PtrSize != 12 {
		throw("hash_iter size incorrect") // see ../../cmd/internal/gc/reflect.go
	}
	it.t = t
	it.h = h

	// grab snapshot of bucket state
	it.B = h.B
	it.buckets = h.buckets
	if t.bucket.kind&kindNoPointers != 0 {
		// Allocate the current slice and remember pointers to both current and old.
		// This preserves all relevant overflow buckets alive even if
		// the table grows and/or overflow buckets are added to the table
		// while we are iterating.
		h.createOverflow()
		it.overflow = *h.overflow //暂存overflow的值到it中.
	}

	// decide where to start
	r := uintptr(fastrand1())
	if h.B > 31-bucketCntBits {
		r += uintptr(fastrand1()) << 31
	}
	it.startBucket = r & (uintptr(1)<<h.B - 1)    //取r的低B位 作为 开始遍历的第一个bucket index
	it.offset = uint8(r >> h.B & (bucketCnt - 1)) //注: >>和& 是同一优先级的操作符，从左到右运算   //r右移B位，再取它的低3位

	// iterator state
	it.bucket = it.startBucket
	it.wrapped = false
	it.bptr = nil

	// Remember we have an iterator.
	// Can run concurrently with another hash_iter_init().
	if old := h.flags; old&(iterator|oldIterator) != iterator|oldIterator {
		atomic.Or8(&h.flags, iterator|oldIterator) //置 标志
	}

	mapiternext(it)
}

func mapiternext(it *hiter) {
	h := it.h
	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&it))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapiternext))
	}
	t := it.t
	bucket := it.bucket
	b := it.bptr
	i := it.i
	checkBucket := it.checkBucket
	alg := t.key.alg

next:
	if b == nil { //表示 刚开始遍历，或者 刚遍历完一个bucket及其overflow链表中的bucket
		//如果遍历完了，则退出
		if bucket == it.startBucket && it.wrapped {
			// end of iteration
			it.key = nil
			it.value = nil //遍历完了，为什么不 置h.flags标志呢？（bzh: 不置标志，也不影响读、写、迁移数据，flag olditerator只是用来在迁移完成后 判断是否要立即释放到GC）
			return
		}

		//设置b
		if h.oldbuckets != nil && it.B == h.B { //注：it.B == h.B 表明在开始遍历前 就已经扩容了，此时在迁移过程中.
			// Iterator was started in the middle of a grow, and the grow isn't done yet.
			// If the bucket we're looking at hasn't been filled in yet (i.e. the old
			// bucket hasn't been evacuated) then we need to iterate through the old
			// bucket and only return the ones that will be migrated to this bucket.
			// 如果扩容导致的迁移还没完成，则需要 从h.oldbucket获取b地址
			oldbucket := bucket & (uintptr(1)<<(it.B-1) - 1) //取bucket的 低B-1位
			b = (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
			if !evacuated(b) {
				checkBucket = bucket
			} else {
				b = (*bmap)(add(it.buckets, bucket*uintptr(t.bucketsize)))
				checkBucket = noCheck
			}
		} else { //若开始遍历后，扩容才开始（插入元素导致的扩容），还在迁移过程中。 => 此时 it.buckets 存储的是 扩容后的oldbuckets中的内容。
			b = (*bmap)(add(it.buckets, bucket*uintptr(t.bucketsize)))
			checkBucket = noCheck
		}

		bucket++
		if bucket == uintptr(1)<<it.B { //遍历到最大地址的bucket了，要再从bucket index 0遍历。
			bucket = 0
			it.wrapped = true
		}
		i = 0
	}

	for ; i < bucketCnt; i++ {
		offi := (i + it.offset) & (bucketCnt - 1) //offi: 在bucket中的 key/value index
		k := add(unsafe.Pointer(b), dataOffset+uintptr(offi)*uintptr(t.keysize))
		v := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+uintptr(offi)*uintptr(t.valuesize))
		if b.tophash[offi] != empty && b.tophash[offi] != evacuatedEmpty { //如果此key/value对 不为空
			if checkBucket != noCheck { // b在oldbuckets中, 需要做一些校验。
				// Special case: iterator was started during a grow and the
				// grow is not done yet. We're working on a bucket whose
				// oldbucket has not been evacuated yet. Or at least, it wasn't
				// evacuated when we started the bucket. So we're iterating
				// through the oldbucket, skipping any keys that will go
				// to the other new bucket (each oldbucket expands to two
				// buckets during a grow).
				k2 := k
				if t.indirectkey {
					k2 = *((*unsafe.Pointer)(k2))
				}
				if t.reflexivekey || alg.equal(k2, k2) {
					// If the item in the oldbucket is not destined for
					// the current new bucket in the iteration, skip it.
					// 如果 这个oldbucket 要迁移的key/value对 的目标迁移newBucket index(即hash&(uintptr(1)<<it.B-1)) 不是当前正在遍历的bucket，则跳过。
					// why??? 感觉这种情况不会发生。
					hash := alg.hash(k2, uintptr(h.hash0))
					if hash&(uintptr(1)<<it.B-1) != checkBucket {
						continue
					}
				} else {
					// Hash isn't repeatable if k != k (NaNs).  We need a
					// repeatable and randomish choice of which direction
					// to send NaNs during evacuation. We'll use the low
					// bit of tophash to decide which way NaNs go.
					// NOTE: this case is why we need two evacuate tophash
					// values, evacuatedX and evacuatedY, that differ in
					// their low bit.
					// 根据 b.tophash[offi]的最低位决定是否要skil. 这个没看懂？感觉这种情况不会发生。
					if checkBucket>>(it.B-1) != uintptr(b.tophash[offi]&1) {
						continue
					}
				}
			}

			if b.tophash[offi] != evacuatedX && b.tophash[offi] != evacuatedY {
				// （如果it.buckets是buckets）new bucket b 已是正常迁移好的bucket(tophash > minTopHash)
				// (如果it.buckets是oldbuckets) oldbucket b是还未迁移的bucket，所以可以直接读取。
				// this is the golden data, we can return it.
				if t.indirectkey {
					k = *((*unsafe.Pointer)(k))
				}
				it.key = k
				if t.indirectvalue {
					v = *((*unsafe.Pointer)(v))
				}
				it.value = v
			} else {
				// The hash table has grown since the iterator was started.
				// The golden data for this key is now somewhere else.
				// (如果it.buckets是oldbuckets) oldbucket b是已迁移好的bucket ，则 此时需要从 新的hashmap中 查询kv对。
				k2 := k
				if t.indirectkey {
					k2 = *((*unsafe.Pointer)(k2))
				}
				if t.reflexivekey || alg.equal(k2, k2) {
					// Check the current hash table for the data.
					// This code handles the case where the key
					// has been deleted, updated, or deleted and reinserted.
					// NOTE: we need to regrab the key as it has potentially been
					// updated to an equal() but not identical key (e.g. +0.0 vs -0.0).
					//从 新的hashmap中 查询kv对
					rk, rv := mapaccessK(t, h, k2)
					if rk == nil {
						continue // key has been deleted
					}
					it.key = rk
					it.value = rv
				} else {
					// if key!=key then the entry can't be deleted or
					// updated, so we can just return it. That's lucky for
					// us because when key!=key we can't look it up
					// successfully in the current table.
					// 如果 equal函数 不能判等，则直接从 oldbucket 读 老数据。
					it.key = k2
					if t.indirectvalue {
						v = *((*unsafe.Pointer)(v))
					}
					it.value = v
				}
			}

			it.bucket = bucket
			if it.bptr != b { // avoid unnecessary write barrier; see issue 14921
				it.bptr = b
			}
			it.i = i + 1
			it.checkBucket = checkBucket
			return
		}
	}

	//如果当前b中 取不到 非空的kv，则走到 下一个overflow bucket
	b = b.overflow(t)
	i = 0
	goto next
}

//将hash桶 增加一倍
func hashGrow(t *maptype, h *hmap) {
	if h.oldbuckets != nil {
		throw("evacuation not done in time")
	}
	oldbuckets := h.buckets
	newbuckets := newarray(t.bucket, 1<<(h.B+1)) //2倍原来的空间
	flags := h.flags &^ (iterator | oldIterator)
	if h.flags&iterator != 0 {
		flags |= oldIterator //置 遍历标志
	}
	// commit the grow (atomic wrt gc)
	h.B++
	h.flags = flags
	h.oldbuckets = oldbuckets
	h.buckets = newbuckets
	h.nevacuate = 0

	if h.overflow != nil {
		// Promote current overflow buckets to the old generation.
		if h.overflow[1] != nil {
			throw("overflow is not nil")
		}
		h.overflow[1] = h.overflow[0]
		h.overflow[0] = nil
	}

	// the actual copying of the hash table data is done incrementally
	// by growWork() and evacuate().
}

func growWork(t *maptype, h *hmap, bucket uintptr) {
	noldbuckets := uintptr(1) << (h.B - 1)

	//这里会 迁移两次，每次迁移一个bucket桶链表
	// make sure we evacuate the oldbucket corresponding
	// to the bucket we're about to use
	evacuate(t, h, bucket&(noldbuckets-1))

	// evacuate one more oldbucket to make progress on growing
	if h.oldbuckets != nil {
		evacuate(t, h, h.nevacuate)
	}
}

// 将oldbucket中  第oldbucket个bucket中的元素（也包括它后续的链表）移动到 新map的bucket中
func evacuate(t *maptype, h *hmap, oldbucket uintptr) {
	b := (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
	newbit := uintptr(1) << (h.B - 1)
	alg := t.key.alg
	if !evacuated(b) {
		//可以考虑：在迁移过后，将老的bucket放到一个池里，以备以后使用。
		// TODO: reuse overflow buckets instead of using new ones, if there
		// is no iterator using the old buckets.  (If !oldIterator.)

		x := (*bmap)(add(h.buckets, oldbucket*uintptr(t.bucketsize)))          //要迁移到h.buckets的前半部分的bucket地址
		y := (*bmap)(add(h.buckets, (oldbucket+newbit)*uintptr(t.bucketsize))) //要迁移到h.buckets的后半部分的bucket地址
		xi := 0
		yi := 0
		xk := add(unsafe.Pointer(x), dataOffset)
		yk := add(unsafe.Pointer(y), dataOffset)
		xv := add(xk, bucketCnt*uintptr(t.keysize))
		yv := add(yk, bucketCnt*uintptr(t.keysize))
		for ; b != nil; b = b.overflow(t) {
			k := add(unsafe.Pointer(b), dataOffset)
			v := add(k, bucketCnt*uintptr(t.keysize))
			for i := 0; i < bucketCnt; i, k, v = i+1, add(k, uintptr(t.keysize)), add(v, uintptr(t.valuesize)) {
				top := b.tophash[i] //hash的高8位不变，所以直接拷贝oldbucket b.tophash即可.
				if top == empty {
					b.tophash[i] = evacuatedEmpty
					continue
				}
				if top < minTopHash {
					throw("bad map state")
				}
				k2 := k
				if t.indirectkey {
					k2 = *((*unsafe.Pointer)(k2))
				}
				// Compute hash to make our evacuation decision (whether we need
				// to send this key/value to bucket x or bucket y).
				hash := alg.hash(k2, uintptr(h.hash0))
				// 当map new buckets正在被遍历
				if h.flags&iterator != 0 {
					if !t.reflexivekey && !alg.equal(k2, k2) {
						// If key != key (NaNs), then the hash could be (and probably
						// will be) entirely different from the old hash. Moreover,
						// it isn't reproducible. Reproducibility is required in the
						// presence of iterators, as our evacuation decision must
						// match whatever decision the iterator made.
						// Fortunately, we have the freedom to send these keys either
						// way. Also, tophash is meaningless for these kinds of keys.
						// We let the low bit of tophash drive the evacuation decision.
						// We recompute a new random tophash for the next level so
						// these keys will get evenly distributed across all buckets
						// after multiple grows.
						// 如果key != key(NaNs), 则 重新计算top。
						// 根据top的最低位 决定 将key/value存储到 前部分、还是后部分的buckets中。
						if (top & 1) != 0 {
							hash |= newbit
						} else {
							hash &^= newbit
						}
						top = uint8(hash >> (sys.PtrSize*8 - 8)) // 取高8位
						if top < minTopHash {
							top += minTopHash
						}
					}
				}
				if (hash & newbit) == 0 { //如果hash低B位(倒数第B位)是0，则存储 到x bucket    （由于hash的低B位 决定 存到第几个bucket，所以 hash的倒数第B位 决定 存到前部分、还是后部分的buckets中）
					//这里为什么先设置标志值，再迁移数据。应该是迁移好了再置标志呀？遍历的时候不会出错吗？
					//不会，这是表示在迁移过程中。如果遍历的时候 在迁移，则会重新从新的hashmap查询kv地址
					b.tophash[i] = evacuatedX
					if xi == bucketCnt {
						//如果这个bucket存储满了，则分配一个新的bucket，然后链接在后面.
						newx := (*bmap)(newobject(t.bucket))
						h.setoverflow(t, x, newx)
						x = newx
						xi = 0
						xk = add(unsafe.Pointer(x), dataOffset)
						xv = add(xk, bucketCnt*uintptr(t.keysize))
					}
					//拷贝tophash, key, value
					x.tophash[xi] = top
					if t.indirectkey {
						*(*unsafe.Pointer)(xk) = k2 // copy pointer
					} else {
						typedmemmove(t.key, xk, k) // copy value
					}
					if t.indirectvalue {
						*(*unsafe.Pointer)(xv) = *(*unsafe.Pointer)(v)
					} else {
						typedmemmove(t.elem, xv, v)
					}
					xi++
					xk = add(xk, uintptr(t.keysize))
					xv = add(xv, uintptr(t.valuesize))
				} else { //如果hash低B位是1，则存储 到y bucket
					b.tophash[i] = evacuatedY
					if yi == bucketCnt {
						//如果这个bucket存储满了，则分配一个新的bucket，然后链接在后面.
						newy := (*bmap)(newobject(t.bucket))
						h.setoverflow(t, y, newy)
						y = newy
						yi = 0
						yk = add(unsafe.Pointer(y), dataOffset)
						yv = add(yk, bucketCnt*uintptr(t.keysize))
					}
					//拷贝tophash, key, value
					y.tophash[yi] = top
					if t.indirectkey {
						*(*unsafe.Pointer)(yk) = k2
					} else {
						typedmemmove(t.key, yk, k)
					}
					if t.indirectvalue {
						*(*unsafe.Pointer)(yv) = *(*unsafe.Pointer)(v)
					} else {
						typedmemmove(t.elem, yv, v)
					}
					yi++
					yk = add(yk, uintptr(t.keysize))
					yv = add(yv, uintptr(t.valuesize))
				}
			}
		}
		// Unlink the overflow buckets & clear key/value to help GC.
		if h.flags&oldIterator == 0 {
			b = (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
			memclr(add(unsafe.Pointer(b), dataOffset), uintptr(t.bucketsize)-dataOffset) //主动释放空间
			//为什么tophash[]空间没释放？还有链表之后的其它bucket也没释放。
			//如果 正在链表老的bucket，那它什么时候 释放？
			//answer: GC会扫描，应该会自动释放.
		}
	}

	// Advance evacuation mark
	if oldbucket == h.nevacuate {
		h.nevacuate = oldbucket + 1 //下次 迁移下一个bucket
		if oldbucket+1 == newbit {  // newbit == # of oldbuckets   //迁移完成了
			// Growing is all done. Free old main bucket array.
			h.oldbuckets = nil
			// Can discard old overflow buckets as well.
			// If they are still referenced by an iterator,
			// then the iterator holds a pointers to the slice.
			if h.overflow != nil {
				h.overflow[1] = nil
			}
		}
	}
}

func ismapkey(t *_type) bool {
	return t.alg.hash != nil
}

// Reflect stubs. Called from ../reflect/asm_*.s

//go:linkname reflect_makemap reflect.makemap
func reflect_makemap(t *maptype) *hmap {
	return makemap(t, 0, nil, nil)
}

//go:linkname reflect_mapaccess reflect.mapaccess
func reflect_mapaccess(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	val, ok := mapaccess2(t, h, key)
	if !ok {
		// reflect wants nil for a missing element
		val = nil
	}
	return val
}

//go:linkname reflect_mapassign reflect.mapassign
func reflect_mapassign(t *maptype, h *hmap, key unsafe.Pointer, val unsafe.Pointer) {
	mapassign1(t, h, key, val)
}

//go:linkname reflect_mapdelete reflect.mapdelete
func reflect_mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	mapdelete(t, h, key)
}

//go:linkname reflect_mapiterinit reflect.mapiterinit
func reflect_mapiterinit(t *maptype, h *hmap) *hiter {
	it := new(hiter)
	mapiterinit(t, h, it)
	return it
}

//go:linkname reflect_mapiternext reflect.mapiternext
func reflect_mapiternext(it *hiter) {
	mapiternext(it)
}

//go:linkname reflect_mapiterkey reflect.mapiterkey
func reflect_mapiterkey(it *hiter) unsafe.Pointer {
	return it.key
}

//go:linkname reflect_maplen reflect.maplen
func reflect_maplen(h *hmap) int {
	if h == nil {
		return 0
	}
	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&h))
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(reflect_maplen))
	}
	return h.count
}

//go:linkname reflect_ismapkey reflect.ismapkey
func reflect_ismapkey(t *_type) bool {
	return ismapkey(t)
}

const maxZero = 1024 // must match value in ../cmd/compile/internal/gc/walk.go
var zeroVal [maxZero]byte
