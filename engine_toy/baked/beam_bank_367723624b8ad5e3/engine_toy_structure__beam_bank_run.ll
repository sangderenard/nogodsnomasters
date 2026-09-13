source_filename = "turing.ssa-llvm.engine_toy_structure__beam_bank_run"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.maxnum.f64(double, double)
declare double @llvm.pow.f64(double, double)

define internal void @__ssa_engine_toy_structure__beam_bank_run__planned_region_0(ptr %arg.0, ptr %arg.1, ptr %arg.2, ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %extents) {
entry:
  %value.29 = alloca double, i64 1, align 8
  %value.49 = alloca i64, i64 1, align 8
  %value.51 = alloca double, i64 1, align 8
  %value.53 = alloca i64, i64 1, align 8
  %value.55 = alloca double, i64 1, align 8
  %value.57 = alloca i64, i64 1, align 8
  %value.59 = alloca i64, i64 1, align 8
  %value.61 = alloca double, i64 1, align 8
  %value.63 = alloca i64, i64 1, align 8
  %value.65 = alloca i64, i64 1, align 8
  %value.84 = alloca double, i64 1, align 8
  %value.28 = alloca double, i64 1, align 8
  %value.30 = alloca double, i64 1, align 8
  %value.31 = alloca double, i64 1, align 8
  %value.32 = alloca double, i64 1, align 8
  %value.33 = alloca double, i64 1, align 8
  %value.34 = alloca double, i64 1, align 8
  %value.38 = alloca double, i64 1, align 8
  %value.39 = alloca double, i64 1, align 8
  %value.42 = alloca double, i64 1, align 8
  %value.43 = alloca double, i64 1, align 8
  %value.44 = alloca double, i64 1, align 8
  %value.45 = alloca double, i64 1, align 8
  %value.46 = alloca double, i64 1, align 8
  %value.47 = alloca double, i64 1, align 8
  %value.52 = alloca double, i64 1, align 8
  %value.54 = alloca double, i64 1, align 8
  %value.56 = alloca double, i64 1, align 8
  %value.60 = alloca double, i64 1, align 8
  %value.62 = alloca double, i64 1, align 8
  %value.64 = alloca double, i64 1, align 8
  %value.66 = alloca double, i64 1, align 8
  %value.67 = alloca double, i64 1, align 8
  %value.70 = alloca double, i64 1, align 8
  %value.91 = alloca double, i64 1, align 8
  %value.94 = alloca double, i64 1, align 8
  store double 0x400921FB54442D18, ptr %value.29, align 8
  store i64 20, ptr %value.49, align 8
  store double 0x40A5D24CCCCCCCCC, ptr %value.51, align 8
  store i64 2500, ptr %value.53, align 8
  store double 0x4071126666666666, ptr %value.55, align 8
  store i64 500, ptr %value.57, align 8
  store i64 3, ptr %value.59, align 8
  store double 0x40B2D37333333333, ptr %value.61, align 8
  store i64 101, ptr %value.63, align 8
  store i64 312500, ptr %value.65, align 8
  store double 0x40036B3356D1EED7, ptr %value.84, align 8
  %load.11.219.0 = load i32, ptr %arg.1, align 4
  %address.11.219 = getelementptr double, ptr %arg.0, i32 %load.11.219.0
  %pinned.load.12.17 = load double, ptr %address.11.219, align 8
  store double %pinned.load.12.17, ptr %out.0, align 8
  %address.13.220 = getelementptr double, ptr %arg.2, i32 %load.11.219.0
  %pinned.load.14.18 = load double, ptr %address.13.220, align 8
  store double %pinned.load.14.18, ptr %out.1, align 8
  %address.15.221 = getelementptr double, ptr %arg.3, i32 %load.11.219.0
  %pinned.load.16.19 = load double, ptr %address.15.221, align 8
  store double %pinned.load.16.19, ptr %out.2, align 8
  %address.17.222 = getelementptr double, ptr %arg.4, i32 %load.11.219.0
  %pinned.load.18.20 = load double, ptr %address.17.222, align 8
  store double %pinned.load.18.20, ptr %out.3, align 8
  %address.19.223 = getelementptr double, ptr %arg.5, i32 %load.11.219.0
  %pinned.load.20.21 = load double, ptr %address.19.223, align 8
  store double %pinned.load.20.21, ptr %out.4, align 8
  %address.21.224 = getelementptr double, ptr %arg.6, i32 %load.11.219.0
  %pinned.load.22.22 = load double, ptr %address.21.224, align 8
  store double %pinned.load.22.22, ptr %out.5, align 8
  %address.23.225 = getelementptr double, ptr %arg.7, i32 %load.11.219.0
  %pinned.load.24.23 = load double, ptr %address.23.225, align 8
  store double %pinned.load.24.23, ptr %out.6, align 8
  %address.25.226 = getelementptr double, ptr %arg.8, i32 %load.11.219.0
  %pinned.load.26.24 = load double, ptr %address.25.226, align 8
  store double %pinned.load.26.24, ptr %out.7, align 8
  %address.27.227 = getelementptr double, ptr %arg.9, i32 %load.11.219.0
  %pinned.load.28.25 = load double, ptr %address.27.227, align 8
  store double %pinned.load.28.25, ptr %out.8, align 8
  %address.29.228 = getelementptr double, ptr %arg.10, i32 %load.11.219.0
  %pinned.load.30.26 = load double, ptr %address.29.228, align 8
  store double %pinned.load.30.26, ptr %out.9, align 8
  %address.31.229 = getelementptr double, ptr %arg.11, i32 %load.11.219.0
  %pinned.load.32.27 = load double, ptr %address.31.229, align 8
  store double %pinned.load.32.27, ptr %out.10, align 8
  %load.33.28.0 = load double, ptr %out.1, align 8
  %load.33.28.1 = load double, ptr %out.2, align 8
  %scalar.33.28 = fsub double %load.33.28.0, %load.33.28.1
  store double %scalar.33.28, ptr %value.28, align 8
  %load.34.30.1 = load double, ptr %out.3, align 8
  %scalar.34.30 = fmul double 0x400921FB54442D18, %load.34.30.1
  store double %scalar.34.30, ptr %value.30, align 8
  %scalar.35.31 = fmul double %load.33.28.0, %load.33.28.0
  store double %scalar.35.31, ptr %value.31, align 8
  %scalar.36.32 = fmul double %scalar.33.28, %scalar.33.28
  store double %scalar.36.32, ptr %value.32, align 8
  %scalar.37.33 = fsub double %scalar.35.31, %scalar.36.32
  store double %scalar.37.33, ptr %value.33, align 8
  %scalar.38.34 = fmul double %scalar.34.30, %scalar.37.33
  store double %scalar.38.34, ptr %value.34, align 8
  %load.39.35.0 = load double, ptr %out.7, align 8
  %scalar.39.35 = fadd double %load.39.35.0, %scalar.38.34
  store double %scalar.39.35, ptr %out.11, align 8
  %load.40.38.0 = load double, ptr %out.0, align 8
  %scalar.40.38 = fmul double %load.40.38.0, %load.40.38.0
  store double %scalar.40.38, ptr %value.38, align 8
  %scalar.41.39 = fmul double %scalar.40.38, %load.40.38.0
  store double %scalar.41.39, ptr %value.39, align 8
  %scalar.42.40 = fmul double %scalar.41.39, %load.40.38.0
  store double %scalar.42.40, ptr %out.12, align 8
  %scalar.43.42 = fmul double %load.33.28.0, %load.33.28.0
  store double %scalar.43.42, ptr %value.42, align 8
  %scalar.44.43 = fmul double %scalar.43.42, %load.33.28.0
  store double %scalar.44.43, ptr %value.43, align 8
  %scalar.45.44 = fmul double %scalar.44.43, %load.33.28.0
  store double %scalar.45.44, ptr %value.44, align 8
  %scalar.46.45 = fmul double %scalar.33.28, %scalar.33.28
  store double %scalar.46.45, ptr %value.45, align 8
  %scalar.47.46 = fmul double %scalar.46.45, %scalar.33.28
  store double %scalar.47.46, ptr %value.46, align 8
  %scalar.48.47 = fmul double %scalar.47.46, %scalar.33.28
  store double %scalar.48.47, ptr %value.47, align 8
  %scalar.49.48 = fsub double %scalar.45.44, %scalar.48.47
  store double %scalar.49.48, ptr %out.13, align 8
  %load.50.50.0 = load double, ptr %out.4, align 8
  %convert.50.50.1 = sitofp i64 20 to double
  %scalar.50.50 = fdiv double %load.50.50.0, %convert.50.50.1
  store double %scalar.50.50, ptr %out.14, align 8
  %load.51.52.0 = load double, ptr %out.5, align 8
  %scalar.51.52 = fsub double %load.51.52.0, 0x40A5D24CCCCCCCCC
  store double %scalar.51.52, ptr %value.52, align 8
  %convert.52.54.1 = sitofp i64 2500 to double
  %scalar.52.54 = fdiv double %load.50.50.0, %convert.52.54.1
  store double %scalar.52.54, ptr %value.54, align 8
  %scalar.53.56 = fsub double %load.51.52.0, 0x4071126666666666
  store double %scalar.53.56, ptr %value.56, align 8
  %convert.54.58.1 = sitofp i64 500 to double
  %scalar.54.58 = fcmp ole double %scalar.53.56, %convert.54.58.1
  store i1 %scalar.54.58, ptr %out.15, align 1
  %convert.55.60.0 = sitofp i64 3 to double
  %scalar.55.60 = fmul double %convert.55.60.0, %load.51.52.0
  store double %scalar.55.60, ptr %value.60, align 8
  %scalar.56.62 = fsub double %scalar.55.60, 0x40B2D37333333333
  store double %scalar.56.62, ptr %value.62, align 8
  %convert.57.64.0 = sitofp i64 101 to double
  %scalar.57.64 = fmul double %convert.57.64.0, %load.50.50.0
  store double %scalar.57.64, ptr %value.64, align 8
  %convert.58.66.1 = sitofp i64 312500 to double
  %scalar.58.66 = fdiv double %scalar.57.64, %convert.58.66.1
  store double %scalar.58.66, ptr %value.66, align 8
  %scalar.59.67 = fneg double %scalar.51.52
  store double %scalar.59.67, ptr %value.67, align 8
  %scalar.60.68 = fmul double %scalar.59.67, %scalar.52.54
  store double %scalar.60.68, ptr %out.16, align 8
  %scalar.61.70 = fneg double %scalar.58.66
  store double %scalar.61.70, ptr %value.70, align 8
  %scalar.62.71 = fmul double %scalar.61.70, %scalar.56.62
  store double %scalar.62.71, ptr %out.17, align 8
  %load.63.80.0 = load double, ptr %out.6, align 8
  %load.63.80.1 = load double, ptr %out.10, align 8
  %scalar.63.80 = fmul double %load.63.80.0, %load.63.80.1
  store double %scalar.63.80, ptr %out.18, align 8
  br label %ew.head.64.85
ew.head.64.85:
  %ew.i.64.85 = phi i32 [ 0, %entry ], [ %ew.next.64.85, %ew.body.64.85 ]
  %ew.more.64.85 = icmp slt i32 %ew.i.64.85, 1
  br i1 %ew.more.64.85, label %ew.body.64.85, label %ew.done.64.85
ew.body.64.85:
  %ew.op.64.85.0 = load double, ptr %value.84, align 8
  %ew.op.64.85.1 = load double, ptr %out.9, align 8
  %ew.val.64.85 = fmul double %ew.op.64.85.0, %ew.op.64.85.1
  %ew.dst.64.85 = getelementptr double, ptr %out.19, i32 %ew.i.64.85
  store double %ew.val.64.85, ptr %ew.dst.64.85, align 8
  %ew.next.64.85 = add i32 %ew.i.64.85, 1
  br label %ew.head.64.85
ew.done.64.85:
  %scalar.65.91 = fneg double %scalar.51.52
  store double %scalar.65.91, ptr %value.91, align 8
  %scalar.66.92 = fmul double %scalar.65.91, %scalar.52.54
  store double %scalar.66.92, ptr %out.20, align 8
  %scalar.67.94 = fneg double %scalar.58.66
  store double %scalar.67.94, ptr %value.94, align 8
  %scalar.68.95 = fmul double %scalar.67.94, %scalar.56.62
  store double %scalar.68.95, ptr %out.21, align 8
  ret void
}

define internal void @__ssa_engine_toy_structure__beam_bank_run__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr %arg.16, ptr %extents) {
entry:
  %value.87 = alloca i64, i64 1, align 8
  %value.100 = alloca i64, i64 1, align 8
  %value.102 = alloca i64, i64 1, align 8
  %value.104 = alloca i64, i64 1, align 8
  %value.106 = alloca i64, i64 1, align 8
  %value.108 = alloca i64, i64 1, align 8
  %value.110 = alloca i64, i64 1, align 8
  %value.37 = alloca double, i64 1, align 8
  %value.79 = alloca double, i64 1, align 8
  %value.78 = alloca double, i64 1, align 8
  %value.81 = alloca double, i64 1, align 8
  %value.77 = alloca double, i64 1, align 8
  %value.86 = alloca double, i64 1, align 8
  %value.90 = alloca double, i64 1, align 8
  %value.99 = alloca double, i64 1, align 8
  %value.36 = alloca double, i64 1, align 8
  %value.41 = alloca double, i64 1, align 8
  %value.88 = alloca double, i64 1, align 8
  %value.89 = alloca double, i64 1, align 8
  %value.101 = alloca i64, i64 1, align 8
  %value.103 = alloca i64, i64 1, align 8
  %value.105 = alloca i64, i64 1, align 8
  %value.107 = alloca i64, i64 1, align 8
  %value.109 = alloca i64, i64 1, align 8
  %value.111 = alloca i64, i64 1, align 8
  %value.114 = alloca double, i64 1, align 8
  %value.115 = alloca double, i64 1, align 8
  %value.116 = alloca double, i64 1, align 8
  %value.117 = alloca double, i64 1, align 8
  %value.118 = alloca double, i64 1, align 8
  %value.119 = alloca double, i64 1, align 8
  %value.120 = alloca double, i64 1, align 8
  %value.121 = alloca double, i64 1, align 8
  %value.122 = alloca double, i64 1, align 8
  %value.123 = alloca double, i64 1, align 8
  %value.124 = alloca double, i64 1, align 8
  %value.125 = alloca double, i64 1, align 8
  %value.126 = alloca double, i64 1, align 8
  %value.127 = alloca double, i64 1, align 8
  %value.128 = alloca double, i64 1, align 8
  %value.129 = alloca double, i64 1, align 8
  %value.130 = alloca double, i64 1, align 8
  %value.131 = alloca double, i64 1, align 8
  %value.132 = alloca double, i64 1, align 8
  %value.133 = alloca double, i64 1, align 8
  %value.134 = alloca double, i64 1, align 8
  %value.135 = alloca double, i64 1, align 8
  %value.136 = alloca double, i64 1, align 8
  %value.137 = alloca double, i64 1, align 8
  %value.138 = alloca double, i64 1, align 8
  %value.139 = alloca double, i64 1, align 8
  %value.140 = alloca double, i64 1, align 8
  %value.141 = alloca double, i64 1, align 8
  %value.142 = alloca double, i64 1, align 8
  store i64 4, ptr %value.87, align 8
  store i64 3, ptr %value.100, align 8
  store i64 0, ptr %value.102, align 8
  store i64 3, ptr %value.104, align 8
  store i64 1, ptr %value.106, align 8
  store i64 3, ptr %value.108, align 8
  store i64 2, ptr %value.110, align 8
  store double 0x3D719799812DEA11, ptr %value.37, align 8
  store double 0x3FE0000000000000, ptr %value.79, align 8
  store double 0x3FEC20CC7A5D9935, ptr %value.78, align 8
  store double 0x3FE0000000000000, ptr %value.81, align 8
  store double 0x3FFC5BF891B4EF6A, ptr %value.77, align 8
  store double 0x3E112E0BE826D695, ptr %value.86, align 8
  store double 0x3FF6083126E978D5, ptr %value.90, align 8
  store double 0x3FE0000000000000, ptr %value.99, align 8
  %load.15.36.0 = load double, ptr %arg.0, align 8
  %load.15.36.1 = load double, ptr %arg.1, align 8
  %scalar.15.36 = fmul double %load.15.36.0, %load.15.36.1
  store double %scalar.15.36, ptr %value.36, align 8
  %load.16.41.0 = load double, ptr %arg.2, align 8
  %scalar.16.41 = fmul double %load.16.41.0, %load.15.36.1
  store double %scalar.16.41, ptr %value.41, align 8
  %convert.17.88.1 = sitofp i64 4 to double
  %scalar.17.88 = fdiv double %scalar.15.36, %convert.17.88.1
  store double %scalar.17.88, ptr %value.88, align 8
  %load.18.89.0 = load double, ptr %arg.3, align 8
  %scalar.18.89 = fneg double %load.18.89.0
  store double %scalar.18.89, ptr %value.89, align 8
  %load.19.101.0 = load i32, ptr %arg.4, align 4
  %convert.19.101.1 = trunc i64 3 to i32
  %scalar.19.101 = mul i32 %load.19.101.0, %convert.19.101.1
  %declared.19.101 = sext i32 %scalar.19.101 to i64
  store i64 %declared.19.101, ptr %value.101, align 8
  %scalar.20.103 = add i64 %declared.19.101, 0
  store i64 %scalar.20.103, ptr %value.103, align 8
  %convert.21.105.1 = trunc i64 3 to i32
  %scalar.21.105 = mul i32 %load.19.101.0, %convert.21.105.1
  %declared.21.105 = sext i32 %scalar.21.105 to i64
  store i64 %declared.21.105, ptr %value.105, align 8
  %scalar.22.107 = add i64 %declared.21.105, 1
  store i64 %scalar.22.107, ptr %value.107, align 8
  %convert.23.109.1 = trunc i64 3 to i32
  %scalar.23.109 = mul i32 %load.19.101.0, %convert.23.109.1
  %declared.23.109 = sext i32 %scalar.23.109 to i64
  store i64 %declared.23.109, ptr %value.109, align 8
  %scalar.24.111 = add i64 %declared.23.109, 2
  store i64 %scalar.24.111, ptr %value.111, align 8
  %scalar.25.114 = call double @llvm.maxnum.f64(double 0x3D719799812DEA11, double %scalar.16.41)
  store double %scalar.25.114, ptr %value.114, align 8
  %load.26.115.0 = load double, ptr %arg.5, align 8
  %load.26.115.1 = load double, ptr %arg.6, align 8
  %scalar.26.115 = call double @llvm.maxnum.f64(double %load.26.115.0, double %load.26.115.1)
  store double %scalar.26.115, ptr %value.115, align 8
  %load.27.116.0 = load double, ptr %arg.7, align 8
  %scalar.27.116 = fmul double %load.27.116.0, %scalar.26.115
  store double %scalar.27.116, ptr %value.116, align 8
  %scalar.28.117 = call double @llvm.pow.f64(double %scalar.25.114, double 0x3FE0000000000000)
  store double %scalar.28.117, ptr %value.117, align 8
  %scalar.29.118 = fdiv double 0x3FEC20CC7A5D9935, %scalar.28.117
  store double %scalar.29.118, ptr %value.118, align 8
  %scalar.30.119 = call double @llvm.pow.f64(double %scalar.27.116, double 0x3FE0000000000000)
  store double %scalar.30.119, ptr %value.119, align 8
  %load.31.120.0 = load double, ptr %arg.8, align 8
  %scalar.31.120 = fmul double %load.31.120.0, %scalar.30.119
  store double %scalar.31.120, ptr %value.120, align 8
  br label %ew.head.32.121
ew.head.32.121:
  %ew.i.32.121 = phi i32 [ 0, %entry ], [ %ew.next.32.121, %ew.body.32.121 ]
  %ew.more.32.121 = icmp slt i32 %ew.i.32.121, 1
  br i1 %ew.more.32.121, label %ew.body.32.121, label %ew.done.32.121
ew.body.32.121:
  %ew.op.32.121.0 = load double, ptr %value.120, align 8
  %ew.op.32.121.1 = load double, ptr %value.77, align 8
  %ew.val.32.121 = fmul double %ew.op.32.121.0, %ew.op.32.121.1
  %ew.dst.32.121 = getelementptr double, ptr %value.121, i32 %ew.i.32.121
  store double %ew.val.32.121, ptr %ew.dst.32.121, align 8
  %ew.next.32.121 = add i32 %ew.i.32.121, 1
  br label %ew.head.32.121
ew.done.32.121:
  br label %ew.head.33.122
ew.head.33.122:
  %ew.i.33.122 = phi i32 [ 0, %ew.done.32.121 ], [ %ew.next.33.122, %ew.body.33.122 ]
  %ew.more.33.122 = icmp slt i32 %ew.i.33.122, 1
  br i1 %ew.more.33.122, label %ew.body.33.122, label %ew.done.33.122
ew.body.33.122:
  %ew.op.33.122.0.addr = getelementptr double, ptr %value.121, i32 %ew.i.33.122
  %ew.op.33.122.0 = load double, ptr %ew.op.33.122.0.addr, align 8
  %ew.op.33.122.1 = load double, ptr %value.118, align 8
  %ew.val.33.122 = fmul double %ew.op.33.122.0, %ew.op.33.122.1
  %ew.dst.33.122 = getelementptr double, ptr %value.122, i32 %ew.i.33.122
  store double %ew.val.33.122, ptr %ew.dst.33.122, align 8
  %ew.next.33.122 = add i32 %ew.i.33.122, 1
  br label %ew.head.33.122
ew.done.33.122:
  br label %ew.head.34.123
ew.head.34.123:
  %ew.i.34.123 = phi i32 [ 0, %ew.done.33.122 ], [ %ew.next.34.123, %ew.body.34.123 ]
  %ew.more.34.123 = icmp slt i32 %ew.i.34.123, 1
  br i1 %ew.more.34.123, label %ew.body.34.123, label %ew.done.34.123
ew.body.34.123:
  %ew.op.34.123.0.addr = getelementptr double, ptr %value.122, i32 %ew.i.34.123
  %ew.op.34.123.0 = load double, ptr %ew.op.34.123.0.addr, align 8
  %ew.op.34.123.1 = load double, ptr %value.36, align 8
  %ew.val.34.123 = fmul double %ew.op.34.123.0, %ew.op.34.123.1
  %ew.dst.34.123 = getelementptr double, ptr %value.123, i32 %ew.i.34.123
  store double %ew.val.34.123, ptr %ew.dst.34.123, align 8
  %ew.next.34.123 = add i32 %ew.i.34.123, 1
  br label %ew.head.34.123
ew.done.34.123:
  br label %ew.head.35.124
ew.head.35.124:
  %ew.i.35.124 = phi i32 [ 0, %ew.done.34.123 ], [ %ew.next.35.124, %ew.body.35.124 ]
  %ew.more.35.124 = icmp slt i32 %ew.i.35.124, 1
  br i1 %ew.more.35.124, label %ew.body.35.124, label %ew.done.35.124
ew.body.35.124:
  %ew.op.35.124.0.addr = getelementptr double, ptr %value.123, i32 %ew.i.35.124
  %ew.op.35.124.0 = load double, ptr %ew.op.35.124.0.addr, align 8
  %ew.op.35.124.1 = load double, ptr %arg.9, align 8
  %ew.val.35.124 = fsub double %ew.op.35.124.0, %ew.op.35.124.1
  %ew.dst.35.124 = getelementptr double, ptr %value.124, i32 %ew.i.35.124
  store double %ew.val.35.124, ptr %ew.dst.35.124, align 8
  %ew.next.35.124 = add i32 %ew.i.35.124, 1
  br label %ew.head.35.124
ew.done.35.124:
  br label %ew.head.36.125
ew.head.36.125:
  %ew.i.36.125 = phi i32 [ 0, %ew.done.35.124 ], [ %ew.next.36.125, %ew.body.36.125 ]
  %ew.more.36.125 = icmp slt i32 %ew.i.36.125, 1
  br i1 %ew.more.36.125, label %ew.body.36.125, label %ew.done.36.125
ew.body.36.125:
  %ew.op.36.125.0.addr = getelementptr double, ptr %arg.10, i32 %ew.i.36.125
  %ew.op.36.125.0 = load double, ptr %ew.op.36.125.0.addr, align 8
  %ew.op.36.125.1 = load double, ptr %value.116, align 8
  %ew.val.36.125 = fmul double %ew.op.36.125.0, %ew.op.36.125.1
  %ew.dst.36.125 = getelementptr double, ptr %value.125, i32 %ew.i.36.125
  store double %ew.val.36.125, ptr %ew.dst.36.125, align 8
  %ew.next.36.125 = add i32 %ew.i.36.125, 1
  br label %ew.head.36.125
ew.done.36.125:
  br label %ew.head.37.126
ew.head.37.126:
  %ew.i.37.126 = phi i32 [ 0, %ew.done.36.125 ], [ %ew.next.37.126, %ew.body.37.126 ]
  %ew.more.37.126 = icmp slt i32 %ew.i.37.126, 1
  br i1 %ew.more.37.126, label %ew.body.37.126, label %ew.done.37.126
ew.body.37.126:
  %ew.op.37.126.0.addr = getelementptr double, ptr %value.125, i32 %ew.i.37.126
  %ew.op.37.126.0 = load double, ptr %ew.op.37.126.0.addr, align 8
  %ew.op.37.126.1 = load double, ptr %value.36, align 8
  %ew.val.37.126 = fmul double %ew.op.37.126.0, %ew.op.37.126.1
  %ew.dst.37.126 = getelementptr double, ptr %value.126, i32 %ew.i.37.126
  store double %ew.val.37.126, ptr %ew.dst.37.126, align 8
  %ew.next.37.126 = add i32 %ew.i.37.126, 1
  br label %ew.head.37.126
ew.done.37.126:
  %load.38.127.0 = load double, ptr %value.126, align 8
  %scalar.38.127 = fdiv double %load.38.127.0, %scalar.25.114
  store double %scalar.38.127, ptr %value.127, align 8
  br label %ew.head.39.128
ew.head.39.128:
  %ew.i.39.128 = phi i32 [ 0, %ew.done.37.126 ], [ %ew.next.39.128, %ew.body.39.128 ]
  %ew.more.39.128 = icmp slt i32 %ew.i.39.128, 1
  br i1 %ew.more.39.128, label %ew.body.39.128, label %ew.done.39.128
ew.body.39.128:
  %ew.op.39.128.0.addr = getelementptr double, ptr %value.124, i32 %ew.i.39.128
  %ew.op.39.128.0 = load double, ptr %ew.op.39.128.0.addr, align 8
  %ew.op.39.128.1 = load double, ptr %value.127, align 8
  %ew.val.39.128 = fadd double %ew.op.39.128.0, %ew.op.39.128.1
  %ew.dst.39.128 = getelementptr double, ptr %value.128, i32 %ew.i.39.128
  store double %ew.val.39.128, ptr %ew.dst.39.128, align 8
  %ew.next.39.128 = add i32 %ew.i.39.128, 1
  br label %ew.head.39.128
ew.done.39.128:
  br label %ew.head.40.129
ew.head.40.129:
  %ew.i.40.129 = phi i32 [ 0, %ew.done.39.128 ], [ %ew.next.40.129, %ew.body.40.129 ]
  %ew.more.40.129 = icmp slt i32 %ew.i.40.129, 1
  br i1 %ew.more.40.129, label %ew.body.40.129, label %ew.done.40.129
ew.body.40.129:
  %ew.op.40.129.0 = load double, ptr %arg.3, align 8
  %ew.op.40.129.1.addr = getelementptr double, ptr %value.128, i32 %ew.i.40.129
  %ew.op.40.129.1 = load double, ptr %ew.op.40.129.1.addr, align 8
  %ew.val.40.129 = fmul double %ew.op.40.129.0, %ew.op.40.129.1
  %ew.dst.40.129 = getelementptr double, ptr %value.129, i32 %ew.i.40.129
  store double %ew.val.40.129, ptr %ew.dst.40.129, align 8
  %ew.next.40.129 = add i32 %ew.i.40.129, 1
  br label %ew.head.40.129
ew.done.40.129:
  %scalar.41.130 = call double @llvm.maxnum.f64(double 0x3E112E0BE826D695, double %scalar.17.88)
  store double %scalar.41.130, ptr %value.130, align 8
  %load.42.131.0 = load double, ptr %value.129, align 8
  %scalar.42.131 = fdiv double %load.42.131.0, %scalar.41.130
  store double %scalar.42.131, ptr %value.131, align 8
  %load.43.132.1 = load double, ptr %arg.11, align 8
  %scalar.43.132 = fsub double %scalar.42.131, %load.43.132.1
  store double %scalar.43.132, ptr %value.132, align 8
  %scalar.44.133 = fmul double %scalar.18.89, %scalar.43.132
  store double %scalar.44.133, ptr %value.133, align 8
  %load.45.134.1 = load double, ptr %arg.12, align 8
  %scalar.45.134 = fadd double %scalar.44.133, %load.45.134.1
  store double %scalar.45.134, ptr %value.134, align 8
  %scalar.46.135 = fneg double %scalar.43.132
  store double %scalar.46.135, ptr %value.135, align 8
  %scalar.47.136 = fmul double 0x3FF6083126E978D5, %scalar.45.134
  store double %scalar.47.136, ptr %value.136, align 8
  %scalar.48.137 = fdiv double %scalar.47.136, %load.15.36.0
  store double %scalar.48.137, ptr %value.137, align 8
  %load.49.138.1 = load double, ptr %arg.13, align 8
  %scalar.49.138 = call double @llvm.maxnum.f64(double %load.26.115.0, double %load.49.138.1)
  store double %scalar.49.138, ptr %value.138, align 8
  %scalar.50.139 = fmul double %load.27.116.0, %scalar.49.138
  store double %scalar.50.139, ptr %value.139, align 8
  %scalar.51.140 = call double @llvm.pow.f64(double %scalar.50.139, double 0x3FE0000000000000)
  store double %scalar.51.140, ptr %value.140, align 8
  %scalar.52.141 = fmul double %scalar.29.118, %scalar.51.140
  store double %scalar.52.141, ptr %value.141, align 8
  %scalar.53.142 = fdiv double %scalar.52.141, 0x3FFC5BF891B4EF6A
  store double %scalar.53.142, ptr %value.142, align 8
  %address.54.230 = getelementptr double, ptr %arg.14, i32 %load.19.101.0
  store double %scalar.45.134, ptr %address.54.230, align 8
  %load.56.231.0 = load i32, ptr %arg.4, align 4
  %address.56.231 = getelementptr double, ptr %arg.15, i32 %load.56.231.0
  %load.store.57.v = load double, ptr %value.135, align 8
  store double %load.store.57.v, ptr %address.56.231, align 8
  %load.58.232.0 = load i64, ptr %value.103, align 8
  %convert.58.232.0 = trunc i64 %load.58.232.0 to i32
  %address.58.232 = getelementptr double, ptr %arg.16, i32 %convert.58.232.0
  %load.store.59.v = load double, ptr %value.134, align 8
  store double %load.store.59.v, ptr %address.58.232, align 8
  %load.60.233.0 = load i64, ptr %value.107, align 8
  %convert.60.233.0 = trunc i64 %load.60.233.0 to i32
  %address.60.233 = getelementptr double, ptr %arg.16, i32 %convert.60.233.0
  %load.store.61.v = load double, ptr %value.137, align 8
  store double %load.store.61.v, ptr %address.60.233, align 8
  %load.62.234.0 = load i64, ptr %value.111, align 8
  %convert.62.234.0 = trunc i64 %load.62.234.0 to i32
  %address.62.234 = getelementptr double, ptr %arg.16, i32 %convert.62.234.0
  %load.store.63.v = load double, ptr %value.142, align 8
  store double %load.store.63.v, ptr %address.62.234, align 8
  ret void
}

define internal void @__ssa_engine_toy_structure__beam_bank_run(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr %arg.13, ptr %out.0, ptr %extents) {
entry:
  %value.160 = alloca i64, i64 1, align 8
  %value.161 = alloca i64, i64 1, align 8
  %value.175 = alloca i64, i64 1, align 8
  %value.177 = alloca i64, i64 1, align 8
  %value.179 = alloca i64, i64 1, align 8
  %value.181 = alloca i64, i64 1, align 8
  %value.183 = alloca i64, i64 1, align 8
  %value.185 = alloca i64, i64 1, align 8
  %value.187 = alloca i64, i64 1, align 8
  %value.189 = alloca i64, i64 1, align 8
  %value.191 = alloca i64, i64 1, align 8
  %value.193 = alloca i64, i64 1, align 8
  %value.195 = alloca i64, i64 1, align 8
  %value.197 = alloca i64, i64 1, align 8
  %value.199 = alloca i64, i64 1, align 8
  %value.201 = alloca i64, i64 1, align 8
  %value.203 = alloca i64, i64 1, align 8
  %value.205 = alloca i64, i64 1, align 8
  %value.207 = alloca i64, i64 1, align 8
  %value.209 = alloca i64, i64 1, align 8
  %value.211 = alloca i64, i64 1, align 8
  %value.213 = alloca i64, i64 1, align 8
  %value.215 = alloca i64, i64 1, align 8
  %value.217 = alloca i64, i64 1, align 8
  %value.24 = alloca double, i64 1, align 8
  %value.23 = alloca double, i64 1, align 8
  %value.20 = alloca double, i64 1, align 8
  %value.25 = alloca double, i64 1, align 8
  %value.17 = alloca double, i64 1, align 8
  %value.18 = alloca double, i64 1, align 8
  %value.22 = alloca double, i64 1, align 8
  %value.19 = alloca double, i64 1, align 8
  %value.21 = alloca double, i64 1, align 8
  %value.172 = alloca i64, i64 1, align 8
  %value.242 = alloca double, i64 1, align 8
  %value.241 = alloca double, i64 1, align 8
  %value.238 = alloca double, i64 1, align 8
  %value.243 = alloca double, i64 1, align 8
  %value.235 = alloca double, i64 1, align 8
  %value.236 = alloca double, i64 1, align 8
  %value.240 = alloca double, i64 1, align 8
  %value.237 = alloca double, i64 1, align 8
  %value.239 = alloca double, i64 1, align 8
  %value.173 = alloca i1, i64 1, align 8
  %value.26 = alloca double, i64 1, align 8
  %value.27 = alloca double, i64 1, align 8
  %value.35 = alloca double, i64 1, align 8
  %value.40 = alloca double, i64 1, align 8
  %value.48 = alloca double, i64 1, align 8
  %value.50 = alloca double, i64 1, align 8
  %value.58 = alloca i1, i64 1, align 8
  %value.68 = alloca double, i64 1, align 8
  %value.71 = alloca double, i64 1, align 8
  %value.80 = alloca double, i64 1, align 8
  %value.85 = alloca double, i64 1, align 8
  %value.92 = alloca double, i64 1, align 8
  %value.95 = alloca double, i64 1, align 8
  %value.74 = alloca double, i64 1, align 8
  %value.98 = alloca double, i64 1, align 8
  store i64 0, ptr %value.160, align 8
  store i64 1, ptr %value.161, align 8
  store i64 0, ptr %value.175, align 8
  store i64 1, ptr %value.177, align 8
  store i64 2, ptr %value.179, align 8
  store i64 3, ptr %value.181, align 8
  store i64 4, ptr %value.183, align 8
  store i64 5, ptr %value.185, align 8
  store i64 6, ptr %value.187, align 8
  store i64 7, ptr %value.189, align 8
  store i64 8, ptr %value.191, align 8
  store i64 9, ptr %value.193, align 8
  store i64 10, ptr %value.195, align 8
  store i64 11, ptr %value.197, align 8
  store i64 12, ptr %value.199, align 8
  store i64 13, ptr %value.201, align 8
  store i64 14, ptr %value.203, align 8
  store i64 15, ptr %value.205, align 8
  store i64 16, ptr %value.207, align 8
  store i64 17, ptr %value.209, align 8
  store i64 18, ptr %value.211, align 8
  store i64 19, ptr %value.213, align 8
  store i64 20, ptr %value.215, align 8
  store i64 21, ptr %value.217, align 8
  %load.cast.24.24 = load double, ptr %arg.9, align 8
  store double %load.cast.24.24, ptr %value.24, align 8
  %load.cast.25.23 = load double, ptr %arg.8, align 8
  store double %load.cast.25.23, ptr %value.23, align 8
  %load.cast.26.20 = load double, ptr %arg.5, align 8
  store double %load.cast.26.20, ptr %value.20, align 8
  %load.cast.27.25 = load double, ptr %arg.10, align 8
  store double %load.cast.27.25, ptr %value.25, align 8
  %load.cast.28.17 = load double, ptr %arg.2, align 8
  store double %load.cast.28.17, ptr %value.17, align 8
  %load.cast.29.18 = load double, ptr %arg.3, align 8
  store double %load.cast.29.18, ptr %value.18, align 8
  %load.cast.30.22 = load double, ptr %arg.7, align 8
  store double %load.cast.30.22, ptr %value.22, align 8
  %load.cast.31.19 = load double, ptr %arg.4, align 8
  store double %load.cast.31.19, ptr %value.19, align 8
  %load.cast.32.21 = load double, ptr %arg.6, align 8
  store double %load.cast.32.21, ptr %value.21, align 8
  br label %loop_header
loop_header:
  %phi.171 = phi ptr [ %value.160, %entry ], [ %value.172, %loop_latch ]
  %phi.162 = phi ptr [ %arg.9, %entry ], [ %value.242, %loop_latch ]
  %phi.163 = phi ptr [ %arg.8, %entry ], [ %value.241, %loop_latch ]
  %phi.164 = phi ptr [ %arg.5, %entry ], [ %value.238, %loop_latch ]
  %phi.165 = phi ptr [ %arg.10, %entry ], [ %value.243, %loop_latch ]
  %phi.166 = phi ptr [ %arg.2, %entry ], [ %value.235, %loop_latch ]
  %phi.167 = phi ptr [ %arg.3, %entry ], [ %value.236, %loop_latch ]
  %phi.168 = phi ptr [ %arg.7, %entry ], [ %value.240, %loop_latch ]
  %phi.169 = phi ptr [ %arg.4, %entry ], [ %value.237, %loop_latch ]
  %phi.170 = phi ptr [ %arg.6, %entry ], [ %value.239, %loop_latch ]
  %load.44.173.0 = load i32, ptr %phi.171, align 4
  %load.44.173.1 = load i32, ptr %arg.0, align 4
  %scalar.44.173 = icmp slt i32 %load.44.173.0, %load.44.173.1
  store i1 %scalar.44.173, ptr %value.173, align 1
  br i1 %scalar.44.173, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_engine_toy_structure__beam_bank_run__planned_region_0(ptr %phi.166, ptr %phi.171, ptr %phi.167, ptr %phi.169, ptr %phi.164, ptr %phi.170, ptr %phi.168, ptr %phi.163, ptr %phi.162, ptr %phi.165, ptr %arg.11, ptr %arg.12, ptr %value.235, ptr %value.236, ptr %value.237, ptr %value.238, ptr %value.239, ptr %value.240, ptr %value.241, ptr %value.242, ptr %value.243, ptr %value.26, ptr %value.27, ptr %value.35, ptr %value.40, ptr %value.48, ptr %value.50, ptr %value.58, ptr %value.68, ptr %value.71, ptr %value.80, ptr %value.85, ptr %value.92, ptr %value.95, ptr %extents)
  call void @__ssa_engine_toy_structure__beam_bank_run__planned_region_1(ptr %value.235, ptr %value.35, ptr %value.40, ptr %arg.1, ptr %phi.171, ptr %value.50, ptr %value.74, ptr %value.48, ptr %value.80, ptr %value.243, ptr %value.85, ptr %value.27, ptr %value.26, ptr %value.98, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %extents)
  br label %loop_latch
loop_latch:
  %load.93.172.0 = load i32, ptr %phi.171, align 4
  %load.93.172.1 = load i64, ptr %value.161, align 8
  %convert.93.172.1 = trunc i64 %load.93.172.1 to i32
  %scalar.93.172 = add i32 %load.93.172.0, %convert.93.172.1
  %declared.93.172 = sext i32 %scalar.93.172 to i64
  store i64 %declared.93.172, ptr %value.172, align 8
  br label %loop_header
loop_exit:
  %phi.148 = phi ptr [ %phi.162, %loop_header ]
  %phi.149 = phi ptr [ %phi.163, %loop_header ]
  %phi.150 = phi ptr [ %phi.164, %loop_header ]
  %phi.151 = phi ptr [ %phi.165, %loop_header ]
  %phi.152 = phi ptr [ %phi.166, %loop_header ]
  %phi.154 = phi ptr [ %phi.167, %loop_header ]
  %phi.157 = phi ptr [ %phi.168, %loop_header ]
  %phi.158 = phi ptr [ %phi.169, %loop_header ]
  %phi.159 = phi ptr [ %phi.170, %loop_header ]
  %return.load.0.82 = load double, ptr %arg.13, align 8
  store double %return.load.0.82, ptr %out.0, align 8
  ret void
}

define void @engine_toy_structure__beam_bank_run(ptr %buffers, ptr %extents) {
entry:
  %public.addr.0 = getelementptr ptr, ptr %buffers, i64 0
  %public.0 = load ptr, ptr %public.addr.0, align 8
  %public.addr.1 = getelementptr ptr, ptr %buffers, i64 1
  %public.1 = load ptr, ptr %public.addr.1, align 8
  %public.addr.2 = getelementptr ptr, ptr %buffers, i64 2
  %public.2 = load ptr, ptr %public.addr.2, align 8
  %public.addr.3 = getelementptr ptr, ptr %buffers, i64 3
  %public.3 = load ptr, ptr %public.addr.3, align 8
  %public.addr.4 = getelementptr ptr, ptr %buffers, i64 4
  %public.4 = load ptr, ptr %public.addr.4, align 8
  %public.addr.5 = getelementptr ptr, ptr %buffers, i64 5
  %public.5 = load ptr, ptr %public.addr.5, align 8
  %public.addr.6 = getelementptr ptr, ptr %buffers, i64 6
  %public.6 = load ptr, ptr %public.addr.6, align 8
  %public.addr.7 = getelementptr ptr, ptr %buffers, i64 7
  %public.7 = load ptr, ptr %public.addr.7, align 8
  %public.addr.8 = getelementptr ptr, ptr %buffers, i64 8
  %public.8 = load ptr, ptr %public.addr.8, align 8
  %public.addr.9 = getelementptr ptr, ptr %buffers, i64 9
  %public.9 = load ptr, ptr %public.addr.9, align 8
  %public.addr.10 = getelementptr ptr, ptr %buffers, i64 10
  %public.10 = load ptr, ptr %public.addr.10, align 8
  %public.addr.11 = getelementptr ptr, ptr %buffers, i64 11
  %public.11 = load ptr, ptr %public.addr.11, align 8
  %public.addr.12 = getelementptr ptr, ptr %buffers, i64 12
  %public.12 = load ptr, ptr %public.addr.12, align 8
  %public.addr.13 = getelementptr ptr, ptr %buffers, i64 13
  %public.13 = load ptr, ptr %public.addr.13, align 8
  call void @__ssa_engine_toy_structure__beam_bank_run(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.13, ptr %extents)
  ret void
}
