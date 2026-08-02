(module ;; render
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (memory (export "memory") 1)
  (func (export "run") (param $count i32) (param $unit_x i32) (param $gain i32) (param $unit_y i32) (param $out0 i32)
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
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $unit_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $gain
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $unit_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $v0
      local.get $v1
      f64.mul
      local.set $v3
      f64.const 1.0
      local.get $v1
      f64.sub
      local.set $v4
      local.get $v2
      local.get $v4
      f64.mul
      local.set $v5
      local.get $v3
      local.get $v5
      f64.add
      local.set $v6
      local.get $v6
      f64.const 255.0
      f64.mul
      local.set $v7
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v7
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
