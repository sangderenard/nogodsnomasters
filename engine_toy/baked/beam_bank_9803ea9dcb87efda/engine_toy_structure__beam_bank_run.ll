source_filename = "turing.ssa-llvm.engine_toy_structure__beam_bank_run"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fabs.f64(double)
declare double @llvm.maxnum.f64(double, double)
declare double @llvm.pow.f64(double, double)

define internal void @__ssa_engine_toy_structure__beam_bank_run__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr %arg.13, ptr %extents) {
entry:
  %value.28 = alloca double, i64 1, align 8
  %value.48 = alloca i64, i64 1, align 8
  %value.50 = alloca double, i64 1, align 8
  %value.52 = alloca i64, i64 1, align 8
  %value.57 = alloca double, i64 1, align 8
  %value.56 = alloca i64, i64 1, align 8
  %value.61 = alloca double, i64 1, align 8
  %value.60 = alloca i64, i64 1, align 8
  %value.65 = alloca i64, i64 1, align 8
  %value.67 = alloca double, i64 1, align 8
  %value.69 = alloca i64, i64 1, align 8
  %value.71 = alloca i64, i64 1, align 8
  %value.89 = alloca double, i64 1, align 8
  %value.92 = alloca i64, i64 1, align 8
  %value.104 = alloca i64, i64 1, align 8
  %value.106 = alloca i64, i64 1, align 8
  %value.108 = alloca i64, i64 1, align 8
  %value.110 = alloca i64, i64 1, align 8
  %value.112 = alloca i64, i64 1, align 8
  %value.114 = alloca i64, i64 1, align 8
  %value.36 = alloca double, i64 1, align 8
  %value.64 = alloca double, i64 1, align 8
  %value.55 = alloca double, i64 1, align 8
  %value.54 = alloca double, i64 1, align 8
  %value.84 = alloca double, i64 1, align 8
  %value.83 = alloca double, i64 1, align 8
  %value.86 = alloca double, i64 1, align 8
  %value.82 = alloca double, i64 1, align 8
  %value.91 = alloca double, i64 1, align 8
  %value.95 = alloca double, i64 1, align 8
  %value.103 = alloca double, i64 1, align 8
  %value.16 = alloca double, i64 1, align 8
  %value.17 = alloca double, i64 1, align 8
  %value.18 = alloca double, i64 1, align 8
  %value.19 = alloca double, i64 1, align 8
  %value.20 = alloca double, i64 1, align 8
  %value.21 = alloca double, i64 1, align 8
  %value.22 = alloca double, i64 1, align 8
  %value.23 = alloca double, i64 1, align 8
  %value.24 = alloca double, i64 1, align 8
  %value.25 = alloca double, i64 1, align 8
  %value.26 = alloca double, i64 1, align 8
  %value.27 = alloca double, i64 1, align 8
  %value.29 = alloca double, i64 1, align 8
  %value.30 = alloca double, i64 1, align 8
  %value.31 = alloca double, i64 1, align 8
  %value.32 = alloca double, i64 1, align 8
  %value.33 = alloca double, i64 1, align 8
  %value.34 = alloca double, i64 1, align 8
  %value.35 = alloca double, i64 1, align 8
  %value.37 = alloca double, i64 1, align 8
  %value.38 = alloca double, i64 1, align 8
  %value.39 = alloca double, i64 1, align 8
  %value.40 = alloca double, i64 1, align 8
  %value.41 = alloca double, i64 1, align 8
  %value.42 = alloca double, i64 1, align 8
  %value.43 = alloca double, i64 1, align 8
  %value.44 = alloca double, i64 1, align 8
  %value.45 = alloca double, i64 1, align 8
  %value.46 = alloca double, i64 1, align 8
  %value.47 = alloca double, i64 1, align 8
  %value.49 = alloca double, i64 1, align 8
  %value.51 = alloca double, i64 1, align 8
  %value.53 = alloca double, i64 1, align 8
  %value.58 = alloca double, i64 1, align 8
  %value.59 = alloca double, i64 1, align 8
  %value.62 = alloca double, i64 1, align 8
  %value.63 = alloca double, i64 1, align 8
  %value.66 = alloca double, i64 1, align 8
  %value.68 = alloca double, i64 1, align 8
  %value.70 = alloca double, i64 1, align 8
  %value.72 = alloca double, i64 1, align 8
  %value.73 = alloca double, i64 1, align 8
  %value.74 = alloca double, i64 1, align 8
  %value.75 = alloca double, i64 1, align 8
  %value.76 = alloca double, i64 1, align 8
  %value.77 = alloca double, i64 1, align 8
  %value.78 = alloca double, i64 1, align 8
  %value.79 = alloca double, i64 1, align 8
  %value.85 = alloca double, i64 1, align 8
  %value.90 = alloca double, i64 1, align 8
  %value.93 = alloca double, i64 1, align 8
  %value.94 = alloca double, i64 1, align 8
  %value.96 = alloca double, i64 1, align 8
  %value.97 = alloca double, i64 1, align 8
  %value.98 = alloca double, i64 1, align 8
  %value.99 = alloca double, i64 1, align 8
  %value.100 = alloca double, i64 1, align 8
  %value.101 = alloca double, i64 1, align 8
  %value.102 = alloca double, i64 1, align 8
  %value.105 = alloca i64, i64 1, align 8
  %value.107 = alloca i64, i64 1, align 8
  %value.109 = alloca i64, i64 1, align 8
  %value.111 = alloca i64, i64 1, align 8
  %value.113 = alloca i64, i64 1, align 8
  %value.115 = alloca i64, i64 1, align 8
  %value.118 = alloca double, i64 1, align 8
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
  %value.143 = alloca double, i64 1, align 8
  %value.144 = alloca double, i64 1, align 8
  %value.145 = alloca double, i64 1, align 8
  %value.146 = alloca double, i64 1, align 8
  %value.147 = alloca double, i64 1, align 8
  %value.148 = alloca double, i64 1, align 8
  %value.149 = alloca double, i64 1, align 8
  %value.150 = alloca double, i64 1, align 8
  %value.151 = alloca double, i64 1, align 8
  %value.152 = alloca double, i64 1, align 8
  %value.153 = alloca double, i64 1, align 8
  %value.154 = alloca double, i64 1, align 8
  %value.155 = alloca double, i64 1, align 8
  %value.156 = alloca double, i64 1, align 8
  store double 0x400921FB54442D18, ptr %value.28, align 8
  store i64 20, ptr %value.48, align 8
  store double 0x40A5D24CCCCCCCCC, ptr %value.50, align 8
  store i64 2500, ptr %value.52, align 8
  store double 0x4071126666666666, ptr %value.57, align 8
  store i64 500, ptr %value.56, align 8
  store double 0x4071126666666666, ptr %value.61, align 8
  store i64 500, ptr %value.60, align 8
  store i64 3, ptr %value.65, align 8
  store double 0x40B2D37333333333, ptr %value.67, align 8
  store i64 101, ptr %value.69, align 8
  store i64 312500, ptr %value.71, align 8
  store double 0x40036B3356D1EED7, ptr %value.89, align 8
  store i64 4, ptr %value.92, align 8
  store i64 3, ptr %value.104, align 8
  store i64 0, ptr %value.106, align 8
  store i64 3, ptr %value.108, align 8
  store i64 1, ptr %value.110, align 8
  store i64 3, ptr %value.112, align 8
  store i64 2, ptr %value.114, align 8
  store double 0x3D719799812DEA11, ptr %value.36, align 8
  store double 0x3D719799812DEA11, ptr %value.64, align 8
  store double 0x3FF0000000000000, ptr %value.55, align 8
  store double 0x3FE0000000000000, ptr %value.54, align 8
  store double 0x3FE0000000000000, ptr %value.84, align 8
  store double 0x3FEC20CC7A5D9935, ptr %value.83, align 8
  store double 0x3FE0000000000000, ptr %value.86, align 8
  store double 0x3FFC5BF891B4EF6A, ptr %value.82, align 8
  store double 0x3E112E0BE826D695, ptr %value.91, align 8
  store double 0x3FF6083126E978D5, ptr %value.95, align 8
  store double 0x3FE0000000000000, ptr %value.103, align 8
  %load.31.170.0 = load i32, ptr %arg.1, align 4
  %address.31.170 = getelementptr double, ptr %arg.0, i32 %load.31.170.0
  %pinned.load.32.16 = load double, ptr %address.31.170, align 8
  store double %pinned.load.32.16, ptr %value.16, align 8
  %address.33.171 = getelementptr double, ptr %arg.2, i32 %load.31.170.0
  %pinned.load.34.17 = load double, ptr %address.33.171, align 8
  store double %pinned.load.34.17, ptr %value.17, align 8
  %address.35.172 = getelementptr double, ptr %arg.3, i32 %load.31.170.0
  %pinned.load.36.18 = load double, ptr %address.35.172, align 8
  store double %pinned.load.36.18, ptr %value.18, align 8
  %address.37.173 = getelementptr double, ptr %arg.4, i32 %load.31.170.0
  %pinned.load.38.19 = load double, ptr %address.37.173, align 8
  store double %pinned.load.38.19, ptr %value.19, align 8
  %address.39.174 = getelementptr double, ptr %arg.5, i32 %load.31.170.0
  %pinned.load.40.20 = load double, ptr %address.39.174, align 8
  store double %pinned.load.40.20, ptr %value.20, align 8
  %address.41.175 = getelementptr double, ptr %arg.6, i32 %load.31.170.0
  %pinned.load.42.21 = load double, ptr %address.41.175, align 8
  store double %pinned.load.42.21, ptr %value.21, align 8
  %address.43.176 = getelementptr double, ptr %arg.7, i32 %load.31.170.0
  %pinned.load.44.22 = load double, ptr %address.43.176, align 8
  store double %pinned.load.44.22, ptr %value.22, align 8
  %address.45.177 = getelementptr double, ptr %arg.8, i32 %load.31.170.0
  %pinned.load.46.23 = load double, ptr %address.45.177, align 8
  store double %pinned.load.46.23, ptr %value.23, align 8
  %address.47.178 = getelementptr double, ptr %arg.9, i32 %load.31.170.0
  %pinned.load.48.24 = load double, ptr %address.47.178, align 8
  store double %pinned.load.48.24, ptr %value.24, align 8
  %address.49.179 = getelementptr double, ptr %arg.10, i32 %load.31.170.0
  %pinned.load.50.25 = load double, ptr %address.49.179, align 8
  store double %pinned.load.50.25, ptr %value.25, align 8
  %address.51.180 = getelementptr double, ptr %arg.11, i32 %load.31.170.0
  %pinned.load.52.26 = load double, ptr %address.51.180, align 8
  store double %pinned.load.52.26, ptr %value.26, align 8
  %load.53.27.0 = load double, ptr %value.17, align 8
  %load.53.27.1 = load double, ptr %value.18, align 8
  %scalar.53.27 = fsub double %load.53.27.0, %load.53.27.1
  store double %scalar.53.27, ptr %value.27, align 8
  %load.54.29.1 = load double, ptr %value.19, align 8
  %scalar.54.29 = fmul double 0x400921FB54442D18, %load.54.29.1
  store double %scalar.54.29, ptr %value.29, align 8
  %scalar.55.30 = fmul double %load.53.27.0, %load.53.27.0
  store double %scalar.55.30, ptr %value.30, align 8
  %scalar.56.31 = fmul double %scalar.53.27, %scalar.53.27
  store double %scalar.56.31, ptr %value.31, align 8
  %scalar.57.32 = fsub double %scalar.55.30, %scalar.56.31
  store double %scalar.57.32, ptr %value.32, align 8
  %scalar.58.33 = fmul double %scalar.54.29, %scalar.57.32
  store double %scalar.58.33, ptr %value.33, align 8
  %load.59.34.0 = load double, ptr %value.23, align 8
  %scalar.59.34 = fadd double %load.59.34.0, %scalar.58.33
  store double %scalar.59.34, ptr %value.34, align 8
  %load.60.35.0 = load double, ptr %value.16, align 8
  %scalar.60.35 = fmul double %load.60.35.0, %scalar.59.34
  store double %scalar.60.35, ptr %value.35, align 8
  %scalar.61.37 = fmul double %load.60.35.0, %load.60.35.0
  store double %scalar.61.37, ptr %value.37, align 8
  %scalar.62.38 = fmul double %scalar.61.37, %load.60.35.0
  store double %scalar.62.38, ptr %value.38, align 8
  %scalar.63.39 = fmul double %scalar.62.38, %load.60.35.0
  store double %scalar.63.39, ptr %value.39, align 8
  %scalar.64.40 = fmul double %scalar.63.39, %scalar.59.34
  store double %scalar.64.40, ptr %value.40, align 8
  %scalar.65.41 = fmul double %load.53.27.0, %load.53.27.0
  store double %scalar.65.41, ptr %value.41, align 8
  %scalar.66.42 = fmul double %scalar.65.41, %load.53.27.0
  store double %scalar.66.42, ptr %value.42, align 8
  %scalar.67.43 = fmul double %scalar.66.42, %load.53.27.0
  store double %scalar.67.43, ptr %value.43, align 8
  %scalar.68.44 = fmul double %scalar.53.27, %scalar.53.27
  store double %scalar.68.44, ptr %value.44, align 8
  %scalar.69.45 = fmul double %scalar.68.44, %scalar.53.27
  store double %scalar.69.45, ptr %value.45, align 8
  %scalar.70.46 = fmul double %scalar.69.45, %scalar.53.27
  store double %scalar.70.46, ptr %value.46, align 8
  %scalar.71.47 = fsub double %scalar.67.43, %scalar.70.46
  store double %scalar.71.47, ptr %value.47, align 8
  %load.72.49.0 = load double, ptr %value.20, align 8
  %convert.72.49.1 = sitofp i64 20 to double
  %scalar.72.49 = fdiv double %load.72.49.0, %convert.72.49.1
  store double %scalar.72.49, ptr %value.49, align 8
  %load.73.51.0 = load double, ptr %value.21, align 8
  %scalar.73.51 = fsub double %load.73.51.0, 0x40A5D24CCCCCCCCC
  store double %scalar.73.51, ptr %value.51, align 8
  %convert.74.53.1 = sitofp i64 2500 to double
  %scalar.74.53 = fdiv double %load.72.49.0, %convert.74.53.1
  store double %scalar.74.53, ptr %value.53, align 8
  %scalar.75.58 = fsub double %load.73.51.0, 0x4071126666666666
  store double %scalar.75.58, ptr %value.58, align 8
  %convert.76.59.0 = sitofp i64 500 to double
  %scalar.76.59 = fsub double %convert.76.59.0, %scalar.75.58
  store double %scalar.76.59, ptr %value.59, align 8
  %scalar.77.62 = fsub double %load.73.51.0, 0x4071126666666666
  store double %scalar.77.62, ptr %value.62, align 8
  %convert.78.63.0 = sitofp i64 500 to double
  %scalar.78.63 = fsub double %convert.78.63.0, %scalar.77.62
  store double %scalar.78.63, ptr %value.63, align 8
  %convert.79.66.0 = sitofp i64 3 to double
  %scalar.79.66 = fmul double %convert.79.66.0, %load.73.51.0
  store double %scalar.79.66, ptr %value.66, align 8
  %scalar.80.68 = fsub double %scalar.79.66, 0x40B2D37333333333
  store double %scalar.80.68, ptr %value.68, align 8
  %convert.81.70.0 = sitofp i64 101 to double
  %scalar.81.70 = fmul double %convert.81.70.0, %load.72.49.0
  store double %scalar.81.70, ptr %value.70, align 8
  %convert.82.72.1 = sitofp i64 312500 to double
  %scalar.82.72 = fdiv double %scalar.81.70, %convert.82.72.1
  store double %scalar.82.72, ptr %value.72, align 8
  %scalar.83.73 = fneg double %scalar.82.72
  store double %scalar.83.73, ptr %value.73, align 8
  %scalar.84.74 = fmul double %scalar.83.73, %scalar.80.68
  store double %scalar.84.74, ptr %value.74, align 8
  %scalar.85.75 = fneg double %scalar.73.51
  store double %scalar.85.75, ptr %value.75, align 8
  %scalar.86.76 = fmul double %scalar.85.75, %scalar.74.53
  store double %scalar.86.76, ptr %value.76, align 8
  %scalar.87.77 = fneg double %scalar.82.72
  store double %scalar.87.77, ptr %value.77, align 8
  %scalar.88.78 = fmul double %scalar.87.77, %scalar.80.68
  store double %scalar.88.78, ptr %value.78, align 8
  %scalar.89.79 = fsub double %scalar.86.76, %scalar.88.78
  store double %scalar.89.79, ptr %value.79, align 8
  %load.90.85.0 = load double, ptr %value.22, align 8
  %load.90.85.1 = load double, ptr %value.26, align 8
  %scalar.90.85 = fmul double %load.90.85.0, %load.90.85.1
  store double %scalar.90.85, ptr %value.85, align 8
  br label %ew.head.91.90
ew.head.91.90:
  %ew.i.91.90 = phi i32 [ 0, %entry ], [ %ew.next.91.90, %ew.body.91.90 ]
  %ew.more.91.90 = icmp slt i32 %ew.i.91.90, 1
  br i1 %ew.more.91.90, label %ew.body.91.90, label %ew.done.91.90
ew.body.91.90:
  %ew.op.91.90.0 = load double, ptr %value.89, align 8
  %ew.op.91.90.1 = load double, ptr %value.25, align 8
  %ew.val.91.90 = fmul double %ew.op.91.90.0, %ew.op.91.90.1
  %ew.dst.91.90 = getelementptr double, ptr %value.90, i32 %ew.i.91.90
  store double %ew.val.91.90, ptr %ew.dst.91.90, align 8
  %ew.next.91.90 = add i32 %ew.i.91.90, 1
  br label %ew.head.91.90
ew.done.91.90:
  %convert.92.93.1 = sitofp i64 4 to double
  %scalar.92.93 = fdiv double %scalar.60.35, %convert.92.93.1
  store double %scalar.92.93, ptr %value.93, align 8
  %load.93.94.0 = load double, ptr %arg.12, align 8
  %scalar.93.94 = fneg double %load.93.94.0
  store double %scalar.93.94, ptr %value.94, align 8
  %scalar.94.96 = fneg double %scalar.82.72
  store double %scalar.94.96, ptr %value.96, align 8
  %scalar.95.97 = fmul double %scalar.94.96, %scalar.80.68
  store double %scalar.95.97, ptr %value.97, align 8
  %scalar.96.98 = fneg double %scalar.73.51
  store double %scalar.96.98, ptr %value.98, align 8
  %scalar.97.99 = fmul double %scalar.96.98, %scalar.74.53
  store double %scalar.97.99, ptr %value.99, align 8
  %scalar.98.100 = fneg double %scalar.82.72
  store double %scalar.98.100, ptr %value.100, align 8
  %scalar.99.101 = fmul double %scalar.98.100, %scalar.80.68
  store double %scalar.99.101, ptr %value.101, align 8
  %scalar.100.102 = fsub double %scalar.97.99, %scalar.99.101
  store double %scalar.100.102, ptr %value.102, align 8
  %convert.101.105.1 = trunc i64 3 to i32
  %scalar.101.105 = mul i32 %load.31.170.0, %convert.101.105.1
  %declared.101.105 = sext i32 %scalar.101.105 to i64
  store i64 %declared.101.105, ptr %value.105, align 8
  %scalar.102.107 = add i64 %declared.101.105, 0
  store i64 %scalar.102.107, ptr %value.107, align 8
  %convert.103.109.1 = trunc i64 3 to i32
  %scalar.103.109 = mul i32 %load.31.170.0, %convert.103.109.1
  %declared.103.109 = sext i32 %scalar.103.109 to i64
  store i64 %declared.103.109, ptr %value.109, align 8
  %scalar.104.111 = add i64 %declared.103.109, 1
  store i64 %scalar.104.111, ptr %value.111, align 8
  %convert.105.113.1 = trunc i64 3 to i32
  %scalar.105.113 = mul i32 %load.31.170.0, %convert.105.113.1
  %declared.105.113 = sext i32 %scalar.105.113 to i64
  store i64 %declared.105.113, ptr %value.113, align 8
  %scalar.106.115 = add i64 %declared.105.113, 2
  store i64 %scalar.106.115, ptr %value.115, align 8
  %scalar.107.118 = call double @llvm.fabs.f64(double %scalar.78.63)
  store double %scalar.107.118, ptr %value.118, align 8
  %scalar.108.120 = call double @llvm.maxnum.f64(double 0x3D719799812DEA11, double %scalar.64.40)
  store double %scalar.108.120, ptr %value.120, align 8
  %scalar.109.121 = call double @llvm.maxnum.f64(double %scalar.107.118, double 0x3D719799812DEA11)
  store double %scalar.109.121, ptr %value.121, align 8
  %scalar.110.122 = fdiv double %scalar.76.59, %scalar.109.121
  store double %scalar.110.122, ptr %value.122, align 8
  %scalar.111.123 = fadd double 0x3FF0000000000000, %scalar.110.122
  store double %scalar.111.123, ptr %value.123, align 8
  %scalar.112.124 = fmul double 0x3FE0000000000000, %scalar.111.123
  store double %scalar.112.124, ptr %value.124, align 8
  %scalar.113.125 = fmul double %scalar.89.79, %scalar.112.124
  store double %scalar.113.125, ptr %value.125, align 8
  %scalar.114.126 = fadd double %scalar.84.74, %scalar.113.125
  store double %scalar.114.126, ptr %value.126, align 8
  %scalar.115.127 = call double @llvm.maxnum.f64(double %scalar.72.49, double %scalar.114.126)
  store double %scalar.115.127, ptr %value.127, align 8
  %scalar.116.128 = fmul double %scalar.71.47, %scalar.115.127
  store double %scalar.116.128, ptr %value.128, align 8
  %scalar.117.129 = call double @llvm.pow.f64(double %scalar.108.120, double 0x3FE0000000000000)
  store double %scalar.117.129, ptr %value.129, align 8
  %scalar.118.130 = fdiv double 0x3FEC20CC7A5D9935, %scalar.117.129
  store double %scalar.118.130, ptr %value.130, align 8
  %scalar.119.131 = call double @llvm.pow.f64(double %scalar.116.128, double 0x3FE0000000000000)
  store double %scalar.119.131, ptr %value.131, align 8
  %scalar.120.132 = fmul double %scalar.90.85, %scalar.119.131
  store double %scalar.120.132, ptr %value.132, align 8
  br label %ew.head.121.133
ew.head.121.133:
  %ew.i.121.133 = phi i32 [ 0, %ew.done.91.90 ], [ %ew.next.121.133, %ew.body.121.133 ]
  %ew.more.121.133 = icmp slt i32 %ew.i.121.133, 1
  br i1 %ew.more.121.133, label %ew.body.121.133, label %ew.done.121.133
ew.body.121.133:
  %ew.op.121.133.0 = load double, ptr %value.132, align 8
  %ew.op.121.133.1 = load double, ptr %value.82, align 8
  %ew.val.121.133 = fmul double %ew.op.121.133.0, %ew.op.121.133.1
  %ew.dst.121.133 = getelementptr double, ptr %value.133, i32 %ew.i.121.133
  store double %ew.val.121.133, ptr %ew.dst.121.133, align 8
  %ew.next.121.133 = add i32 %ew.i.121.133, 1
  br label %ew.head.121.133
ew.done.121.133:
  br label %ew.head.122.134
ew.head.122.134:
  %ew.i.122.134 = phi i32 [ 0, %ew.done.121.133 ], [ %ew.next.122.134, %ew.body.122.134 ]
  %ew.more.122.134 = icmp slt i32 %ew.i.122.134, 1
  br i1 %ew.more.122.134, label %ew.body.122.134, label %ew.done.122.134
ew.body.122.134:
  %ew.op.122.134.0.addr = getelementptr double, ptr %value.133, i32 %ew.i.122.134
  %ew.op.122.134.0 = load double, ptr %ew.op.122.134.0.addr, align 8
  %ew.op.122.134.1 = load double, ptr %value.130, align 8
  %ew.val.122.134 = fmul double %ew.op.122.134.0, %ew.op.122.134.1
  %ew.dst.122.134 = getelementptr double, ptr %value.134, i32 %ew.i.122.134
  store double %ew.val.122.134, ptr %ew.dst.122.134, align 8
  %ew.next.122.134 = add i32 %ew.i.122.134, 1
  br label %ew.head.122.134
ew.done.122.134:
  br label %ew.head.123.135
ew.head.123.135:
  %ew.i.123.135 = phi i32 [ 0, %ew.done.122.134 ], [ %ew.next.123.135, %ew.body.123.135 ]
  %ew.more.123.135 = icmp slt i32 %ew.i.123.135, 1
  br i1 %ew.more.123.135, label %ew.body.123.135, label %ew.done.123.135
ew.body.123.135:
  %ew.op.123.135.0.addr = getelementptr double, ptr %value.134, i32 %ew.i.123.135
  %ew.op.123.135.0 = load double, ptr %ew.op.123.135.0.addr, align 8
  %ew.op.123.135.1 = load double, ptr %value.35, align 8
  %ew.val.123.135 = fmul double %ew.op.123.135.0, %ew.op.123.135.1
  %ew.dst.123.135 = getelementptr double, ptr %value.135, i32 %ew.i.123.135
  store double %ew.val.123.135, ptr %ew.dst.123.135, align 8
  %ew.next.123.135 = add i32 %ew.i.123.135, 1
  br label %ew.head.123.135
ew.done.123.135:
  br label %ew.head.124.136
ew.head.124.136:
  %ew.i.124.136 = phi i32 [ 0, %ew.done.123.135 ], [ %ew.next.124.136, %ew.body.124.136 ]
  %ew.more.124.136 = icmp slt i32 %ew.i.124.136, 1
  br i1 %ew.more.124.136, label %ew.body.124.136, label %ew.done.124.136
ew.body.124.136:
  %ew.op.124.136.0.addr = getelementptr double, ptr %value.135, i32 %ew.i.124.136
  %ew.op.124.136.0 = load double, ptr %ew.op.124.136.0.addr, align 8
  %ew.op.124.136.1 = load double, ptr %value.24, align 8
  %ew.val.124.136 = fsub double %ew.op.124.136.0, %ew.op.124.136.1
  %ew.dst.124.136 = getelementptr double, ptr %value.136, i32 %ew.i.124.136
  store double %ew.val.124.136, ptr %ew.dst.124.136, align 8
  %ew.next.124.136 = add i32 %ew.i.124.136, 1
  br label %ew.head.124.136
ew.done.124.136:
  br label %ew.head.125.137
ew.head.125.137:
  %ew.i.125.137 = phi i32 [ 0, %ew.done.124.136 ], [ %ew.next.125.137, %ew.body.125.137 ]
  %ew.more.125.137 = icmp slt i32 %ew.i.125.137, 1
  br i1 %ew.more.125.137, label %ew.body.125.137, label %ew.done.125.137
ew.body.125.137:
  %ew.op.125.137.0.addr = getelementptr double, ptr %value.90, i32 %ew.i.125.137
  %ew.op.125.137.0 = load double, ptr %ew.op.125.137.0.addr, align 8
  %ew.op.125.137.1 = load double, ptr %value.128, align 8
  %ew.val.125.137 = fmul double %ew.op.125.137.0, %ew.op.125.137.1
  %ew.dst.125.137 = getelementptr double, ptr %value.137, i32 %ew.i.125.137
  store double %ew.val.125.137, ptr %ew.dst.125.137, align 8
  %ew.next.125.137 = add i32 %ew.i.125.137, 1
  br label %ew.head.125.137
ew.done.125.137:
  br label %ew.head.126.138
ew.head.126.138:
  %ew.i.126.138 = phi i32 [ 0, %ew.done.125.137 ], [ %ew.next.126.138, %ew.body.126.138 ]
  %ew.more.126.138 = icmp slt i32 %ew.i.126.138, 1
  br i1 %ew.more.126.138, label %ew.body.126.138, label %ew.done.126.138
ew.body.126.138:
  %ew.op.126.138.0.addr = getelementptr double, ptr %value.137, i32 %ew.i.126.138
  %ew.op.126.138.0 = load double, ptr %ew.op.126.138.0.addr, align 8
  %ew.op.126.138.1 = load double, ptr %value.35, align 8
  %ew.val.126.138 = fmul double %ew.op.126.138.0, %ew.op.126.138.1
  %ew.dst.126.138 = getelementptr double, ptr %value.138, i32 %ew.i.126.138
  store double %ew.val.126.138, ptr %ew.dst.126.138, align 8
  %ew.next.126.138 = add i32 %ew.i.126.138, 1
  br label %ew.head.126.138
ew.done.126.138:
  %load.127.139.0 = load double, ptr %value.138, align 8
  %scalar.127.139 = fdiv double %load.127.139.0, %scalar.108.120
  store double %scalar.127.139, ptr %value.139, align 8
  br label %ew.head.128.140
ew.head.128.140:
  %ew.i.128.140 = phi i32 [ 0, %ew.done.126.138 ], [ %ew.next.128.140, %ew.body.128.140 ]
  %ew.more.128.140 = icmp slt i32 %ew.i.128.140, 1
  br i1 %ew.more.128.140, label %ew.body.128.140, label %ew.done.128.140
ew.body.128.140:
  %ew.op.128.140.0.addr = getelementptr double, ptr %value.136, i32 %ew.i.128.140
  %ew.op.128.140.0 = load double, ptr %ew.op.128.140.0.addr, align 8
  %ew.op.128.140.1 = load double, ptr %value.139, align 8
  %ew.val.128.140 = fadd double %ew.op.128.140.0, %ew.op.128.140.1
  %ew.dst.128.140 = getelementptr double, ptr %value.140, i32 %ew.i.128.140
  store double %ew.val.128.140, ptr %ew.dst.128.140, align 8
  %ew.next.128.140 = add i32 %ew.i.128.140, 1
  br label %ew.head.128.140
ew.done.128.140:
  br label %ew.head.129.141
ew.head.129.141:
  %ew.i.129.141 = phi i32 [ 0, %ew.done.128.140 ], [ %ew.next.129.141, %ew.body.129.141 ]
  %ew.more.129.141 = icmp slt i32 %ew.i.129.141, 1
  br i1 %ew.more.129.141, label %ew.body.129.141, label %ew.done.129.141
ew.body.129.141:
  %ew.op.129.141.0 = load double, ptr %arg.12, align 8
  %ew.op.129.141.1.addr = getelementptr double, ptr %value.140, i32 %ew.i.129.141
  %ew.op.129.141.1 = load double, ptr %ew.op.129.141.1.addr, align 8
  %ew.val.129.141 = fmul double %ew.op.129.141.0, %ew.op.129.141.1
  %ew.dst.129.141 = getelementptr double, ptr %value.141, i32 %ew.i.129.141
  store double %ew.val.129.141, ptr %ew.dst.129.141, align 8
  %ew.next.129.141 = add i32 %ew.i.129.141, 1
  br label %ew.head.129.141
ew.done.129.141:
  %scalar.130.142 = call double @llvm.maxnum.f64(double 0x3E112E0BE826D695, double %scalar.92.93)
  store double %scalar.130.142, ptr %value.142, align 8
  %load.131.143.0 = load double, ptr %value.141, align 8
  %scalar.131.143 = fdiv double %load.131.143.0, %scalar.130.142
  store double %scalar.131.143, ptr %value.143, align 8
  %scalar.132.144 = fsub double %scalar.131.143, %load.90.85.1
  store double %scalar.132.144, ptr %value.144, align 8
  %scalar.133.145 = fmul double %scalar.93.94, %scalar.132.144
  store double %scalar.133.145, ptr %value.145, align 8
  %load.134.146.1 = load double, ptr %value.25, align 8
  %scalar.134.146 = fadd double %scalar.133.145, %load.134.146.1
  store double %scalar.134.146, ptr %value.146, align 8
  %scalar.135.147 = fneg double %scalar.132.144
  store double %scalar.135.147, ptr %value.147, align 8
  %scalar.136.148 = fmul double 0x3FF6083126E978D5, %scalar.134.146
  store double %scalar.136.148, ptr %value.148, align 8
  %scalar.137.149 = fdiv double %scalar.136.148, %load.60.35.0
  store double %scalar.137.149, ptr %value.149, align 8
  %scalar.138.150 = fmul double %scalar.100.102, %scalar.112.124
  store double %scalar.138.150, ptr %value.150, align 8
  %scalar.139.151 = fadd double %scalar.95.97, %scalar.138.150
  store double %scalar.139.151, ptr %value.151, align 8
  %scalar.140.152 = call double @llvm.maxnum.f64(double %scalar.72.49, double %scalar.139.151)
  store double %scalar.140.152, ptr %value.152, align 8
  %scalar.141.153 = fmul double %scalar.71.47, %scalar.140.152
  store double %scalar.141.153, ptr %value.153, align 8
  %scalar.142.154 = call double @llvm.pow.f64(double %scalar.141.153, double 0x3FE0000000000000)
  store double %scalar.142.154, ptr %value.154, align 8
  %scalar.143.155 = fmul double %scalar.118.130, %scalar.142.154
  store double %scalar.143.155, ptr %value.155, align 8
  %scalar.144.156 = fdiv double %scalar.143.155, 0x3FFC5BF891B4EF6A
  store double %scalar.144.156, ptr %value.156, align 8
  %address.145.181 = getelementptr double, ptr %arg.10, i32 %load.31.170.0
  store double %scalar.134.146, ptr %address.145.181, align 8
  %load.147.182.0 = load i32, ptr %arg.1, align 4
  %address.147.182 = getelementptr double, ptr %arg.11, i32 %load.147.182.0
  %load.store.148.v = load double, ptr %value.147, align 8
  store double %load.store.148.v, ptr %address.147.182, align 8
  %load.149.183.0 = load i64, ptr %value.107, align 8
  %convert.149.183.0 = trunc i64 %load.149.183.0 to i32
  %address.149.183 = getelementptr double, ptr %arg.13, i32 %convert.149.183.0
  %load.store.150.v = load double, ptr %value.146, align 8
  store double %load.store.150.v, ptr %address.149.183, align 8
  %load.151.184.0 = load i64, ptr %value.111, align 8
  %convert.151.184.0 = trunc i64 %load.151.184.0 to i32
  %address.151.184 = getelementptr double, ptr %arg.13, i32 %convert.151.184.0
  %load.store.152.v = load double, ptr %value.149, align 8
  store double %load.store.152.v, ptr %address.151.184, align 8
  %load.153.185.0 = load i64, ptr %value.115, align 8
  %convert.153.185.0 = trunc i64 %load.153.185.0 to i32
  %address.153.185 = getelementptr double, ptr %arg.13, i32 %convert.153.185.0
  %load.store.154.v = load double, ptr %value.156, align 8
  store double %load.store.154.v, ptr %address.153.185, align 8
  ret void
}

define internal void @__ssa_engine_toy_structure__beam_bank_run(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr %arg.13, ptr %out.0, ptr %extents) {
entry:
  %value.165 = alloca i64, i64 1, align 8
  %value.166 = alloca i64, i64 1, align 8
  %value.168 = alloca i64, i64 1, align 8
  %value.169 = alloca i1, i64 1, align 8
  store i64 0, ptr %value.165, align 8
  store i64 1, ptr %value.166, align 8
  br label %loop_header
loop_header:
  %phi.167 = phi ptr [ %value.165, %entry ], [ %value.168, %loop_latch ]
  %load.4.169.0 = load i32, ptr %phi.167, align 4
  %load.4.169.1 = load i32, ptr %arg.0, align 4
  %scalar.4.169 = icmp slt i32 %load.4.169.0, %load.4.169.1
  store i1 %scalar.4.169, ptr %value.169, align 1
  br i1 %scalar.4.169, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_engine_toy_structure__beam_bank_run__planned_region_0(ptr %arg.2, ptr %phi.167, ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.1, ptr %arg.13, ptr %extents)
  br label %loop_latch
loop_latch:
  %load.8.168.0 = load i32, ptr %phi.167, align 4
  %load.8.168.1 = load i64, ptr %value.166, align 8
  %convert.8.168.1 = trunc i64 %load.8.168.1 to i32
  %scalar.8.168 = add i32 %load.8.168.0, %convert.8.168.1
  %declared.8.168 = sext i32 %scalar.8.168 to i64
  store i64 %declared.8.168, ptr %value.168, align 8
  br label %loop_header
loop_exit:
  %return.load.0.23 = load double, ptr %arg.13, align 8
  store double %return.load.0.23, ptr %out.0, align 8
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
