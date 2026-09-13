(module
  (memory (export "memory") 1)
  (func (export "abstract_ui_beam_dynamics_step") (param $io i32)
    (local $t0 f64)
    (local $t1 f64)
    (local $t2 f64)
    (local $t3 f64)
    (local $t4 f64)
    (local $t5 f64)
    (local $t6 f64)
    (local $t7 f64)
    (local $t8 f64)
    (local $t9 f64)
    (local $t10 f64)
    (local $t11 f64)
    (local $t12 f64)
    (local $t13 f64)
    (local $t14 f64)
    (local $t15 f64)
    (local $t16 f64)
    (local $t17 f64)
    (local $t18 f64)
    (local $t19 f64)
    (local $t20 f64)
    (local $t21 f64)
    (local $t22 f64)
    (local $t23 f64)
    (local $t24 f64)
    (local $t25 f64)
    (local $t26 f64)
    (local $t27 f64)
    (local $t28 f64)
    (local $t29 f64)
    (local $t30 f64)
    (local $t31 f64)
    (local $t32 f64)
    (local $t33 f64)
    (local $t34 f64)
    (local $t35 f64)
    (local $t36 f64)
    (local $t37 f64)
    (local $t38 f64)
    (local $t39 f64)
    (local $t40 f64)
    (local $t41 f64)
    (local $t42 f64)
    (local $t43 f64)
    (local $t44 f64)
    (local $t45 f64)
    (local $t46 f64)
    (local $t47 f64)
    (local $t48 f64)
    (local $t49 f64)
    (local $t50 f64)
    (local $t51 f64)
    (local $t52 f64)
    (local $t53 f64)
    (local $t54 f64)
    (local $t55 f64)
    (local $t56 f64)
    (local $t57 f64)
    (local $t58 f64)
    (local $t59 f64)
    (local $t60 f64)
    (local $t61 f64)
    (local $t62 f64)
    (local $t63 f64)
    (local $t64 f64)
    (local $t65 f64)
    (local $t66 f64)
    (local $t67 f64)
    (local $t68 f64)
    (local $t69 f64)
    (local $t70 f64)
    (local $t71 f64)
    (local $t72 f64)
    (local $t73 f64)
    (local $t74 f64)
    (local $t75 f64)
    (local $t76 f64)
    (local $t77 f64)
    (local $t78 f64)
    (local $t79 f64)
    (local $t80 f64)
    (local $t81 f64)
    (local $t82 f64)
    (local $t83 f64)
    (local $t84 f64)
    (local $t85 f64)
    (local $t86 f64)
    (local $t87 f64)
    (local $t88 f64)
    (local $t89 f64)
    (local $t90 f64)
    (local $t91 f64)
    (local $t92 f64)
    (local $t93 f64)
    (local $t94 f64)
    (local $t95 f64)
    (local $t96 f64)
    (local $t97 f64)
    (local $t98 f64)
    (local $t99 f64)
    (local $t100 f64)
    (local $t101 f64)
    (local $t102 f64)
    (local $t103 f64)
    (local $t104 f64)
    (local $t105 f64)
    local.get $io i32.const 0 i32.add f64.load local.set $t6
    local.get $io i32.const 8 i32.add f64.load local.set $t74
    local.get $io i32.const 16 i32.add f64.load local.set $t10
    local.get $io i32.const 24 i32.add f64.load local.set $t1
    local.get $io i32.const 32 i32.add f64.load local.set $t28
    local.get $io i32.const 40 i32.add f64.load local.set $t4
    local.get $io i32.const 48 i32.add f64.load local.set $t0
    local.get $io i32.const 56 i32.add f64.load local.set $t2
    local.get $io i32.const 64 i32.add f64.load local.set $t12
    local.get $io i32.const 72 i32.add f64.load local.set $t56
    local.get $io i32.const 80 i32.add f64.load local.set $t16
    local.get $io i32.const 88 i32.add f64.load local.set $t50
    f64.const 0x1.0000000000000p+0 local.set $t105
    f64.const 0x1.12e0be826d695p-30 local.set $t3
    f64.const 0x1.0000000000000p-2 local.set $t5
    f64.const 0x1.921fb54442d18p+1 local.set $t8
    f64.const 0x1.0000000000000p+1 local.set $t13
    f64.const -0x1.0000000000000p+0 local.set $t15
    f64.const -0x1.8b9877433214cp+1 local.set $t29
    f64.const 0x1.19799812dea11p-40 local.set $t33
    f64.const 0x1.0000000000000p+2 local.set $t34
    f64.const 0x1.999999999999ap-5 local.set $t49
    f64.const 0x1.9db22d0e56042p-1 local.set $t52
    f64.const 0x1.ed82fd75e2046p+0 local.set $t54
    f64.const -0x1.3a92a30553261p-10 local.set $t55
    f64.const 0x1.1e04c05921038p+0 local.set $t60
    f64.const -0x1.a36e2eb1c432dp-12 local.set $t61
    f64.const -0x1.1126666666666p+8 local.set $t65
    f64.const 0x1.f400000000000p+8 local.set $t67
    f64.const -0x1.c20cc7a5d9935p+1 local.set $t73
    f64.const 0x1.0000000000000p-1 local.set $t78
    f64.const -0x1.0000000000000p-1 local.set $t84
    f64.const 0x1.6083126e978d5p+0 local.set $t94
    f64.const 0x1.c20cc7a5d9935p-1 local.set $t100
    local.get $t5 local.get $t6 f64.mul local.set $t7
    local.get $t5 local.get $t8 f64.mul local.set $t9
    local.get $t12 local.get $t12 f64.mul local.set $t14
    local.get $t15 local.get $t16 f64.mul local.set $t17
    local.get $t29 local.get $t8 f64.mul local.set $t30
    local.get $t4 local.get $t4 f64.mul local.get $t4 f64.mul local.get $t4 f64.mul local.set $t35
    local.get $t8 local.get $t10 f64.mul local.set $t36
    local.get $t12 local.get $t12 f64.mul local.get $t12 f64.mul local.get $t12 f64.mul local.set $t43
    local.get $t49 local.get $t50 f64.mul local.set $t51
    local.get $t52 local.get $t50 f64.mul local.set $t53
    local.get $t55 local.get $t56 f64.mul local.set $t57
    local.get $t61 local.get $t56 f64.mul local.set $t62
    local.get $t65 local.get $t56 f64.add local.set $t66
    local.get $t73 local.get $t74 f64.mul local.set $t75
    local.get $t8 f64.sqrt local.set $t79
    local.get $t105 local.get $t4 f64.div local.set $t93
    local.get $t94 local.get $t0 f64.mul local.set $t95
    local.get $t94 local.get $t1 f64.mul local.set $t96
    f64.const 0x1.0000000000000p+0 local.get $t8 f64.sqrt f64.div local.set $t101
    local.get $t9 local.get $t10 f64.mul local.set $t11
    local.get $t12 local.get $t17 f64.add local.set $t18
    local.get $t30 local.get $t4 f64.mul local.set $t31
    local.get $t54 local.get $t57 f64.add local.set $t58
    local.get $t60 local.get $t62 f64.add local.set $t63
    local.get $t66 local.get $t67 f64.le f64.convert_i32_u local.set $t68
    local.get $t75 local.get $t4 f64.mul local.set $t76
    local.get $t100 local.get $t101 f64.mul local.set $t102
    local.get $t18 local.get $t18 f64.mul local.set $t19
    local.get $t31 local.get $t0 f64.mul local.set $t32
    local.get $t18 local.get $t18 f64.mul local.get $t18 f64.mul local.get $t18 f64.mul local.set $t44
    local.get $t53 local.get $t58 f64.mul local.set $t59
    local.get $t50 local.get $t63 f64.mul local.set $t64
    local.get $t76 local.get $t2 f64.mul local.set $t77
    local.get $t15 local.get $t19 f64.mul local.set $t20
    local.get $t15 local.get $t44 f64.mul local.set $t45
    local.get $t64 local.get $t59 local.get $t68 f64.const 0 f64.ne select local.set $t69
    local.get $t77 local.get $t79 f64.mul local.set $t80
    local.get $t14 local.get $t20 f64.add local.set $t21
    local.get $t43 local.get $t45 f64.add local.set $t46
    local.get $t51 local.get $t69 f64.max local.set $t70
    local.get $t11 local.get $t21 f64.mul local.set $t22
    local.get $t36 local.get $t21 f64.mul local.set $t37
    local.get $t46 local.get $t70 f64.mul local.set $t81
    local.get $t7 local.get $t22 f64.add local.set $t23
    local.get $t6 local.get $t37 f64.add local.set $t38
    local.get $t81 f64.sqrt local.set $t82
    local.get $t4 local.get $t23 f64.mul local.set $t24
    local.get $t35 local.get $t38 f64.mul local.set $t39
    local.get $t80 local.get $t82 f64.mul local.set $t83
    local.get $t102 local.get $t82 f64.mul local.set $t103
    local.get $t3 local.get $t24 f64.max local.set $t25
    local.get $t33 local.get $t39 f64.max local.set $t40
    local.get $t105 local.get $t25 f64.div local.set $t26
    local.get $t105 local.get $t40 f64.div local.set $t41
    f64.const 0x1.0000000000000p+0 local.get $t40 f64.sqrt f64.div local.set $t85
    local.get $t1 local.get $t26 f64.mul local.set $t27
    local.get $t32 local.get $t41 f64.mul local.set $t42
    local.get $t83 local.get $t85 f64.mul local.set $t86
    local.get $t103 local.get $t85 f64.mul local.set $t104
    local.get $t42 local.get $t46 f64.mul local.set $t47
    local.get $t86 local.get $t23 f64.mul local.set $t87
    local.get $t47 local.get $t23 f64.mul local.set $t48
    local.get $t48 local.get $t70 f64.mul local.set $t71
    local.get $t28 local.get $t71 f64.add local.set $t72
    local.get $t72 local.get $t87 f64.add local.set $t88
    local.get $t27 local.get $t88 f64.mul local.set $t89
    local.get $t2 local.get $t89 f64.add local.set $t90
    local.get $t1 local.get $t90 f64.mul local.set $t91
    local.get $t96 local.get $t90 f64.mul local.set $t97
    local.get $t0 local.get $t91 f64.add local.set $t92
    local.get $t95 local.get $t97 f64.add local.set $t98
    local.get $t93 local.get $t98 f64.mul local.set $t99
    local.get $io i32.const 96 i32.add local.get $t92 f64.store
    local.get $io i32.const 104 i32.add local.get $t90 f64.store
    local.get $io i32.const 112 i32.add local.get $t92 f64.store
    local.get $io i32.const 120 i32.add local.get $t99 f64.store
    local.get $io i32.const 128 i32.add local.get $t104 f64.store
    local.get $io i32.const 136 i32.add local.get $t70 f64.store
  )
)
