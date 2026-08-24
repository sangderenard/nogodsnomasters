(module ;; step
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (memory (export "memory") 1)
  (func (export "run") (param $count i32) (param $phase i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    i32.const 0
    local.set $i
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $phase
      f64.load
      local.set $v0
      ;; cos via baked lookup table (see the .wasm)
      local.get $v0
      local.set $v1
      f64.const 0.5
      local.get $v1
      f64.mul
      local.set $v2
      f64.const 0.5
      local.get $v2
      f64.add
      local.set $v3
      local.get $v0
      f64.const 2.094395102393195
      f64.sub
      local.set $v4
      ;; cos via baked lookup table (see the .wasm)
      local.get $v4
      local.set $v5
      f64.const 0.5
      local.get $v5
      f64.mul
      local.set $v6
      f64.const 0.5
      local.get $v6
      f64.add
      local.set $v7
      local.get $v0
      f64.const 4.18879020478639
      f64.sub
      local.set $v8
      ;; cos via baked lookup table (see the .wasm)
      local.get $v8
      local.set $v9
      f64.const 0.5
      local.get $v9
      f64.mul
      local.set $v10
      f64.const 0.5
      local.get $v10
      f64.add
      local.set $v11
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v0
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v3
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v7
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v11
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)
