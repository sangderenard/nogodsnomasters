source_filename = "turing.ssa-llvm.engine_toy_ballistics__ballistics_run"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.maxnum.f64(double, double)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.pow.f64(double, double)

define internal void @__ssa_engine_toy_ballistics__ballistics_run__planned_region_0(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr %arg.22, ptr %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr %arg.29, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %extents) {
entry:
  %value.431 = alloca double, i64 1, align 8
  %value.38 = alloca i64, i64 1, align 8
  %value.43 = alloca i64, i64 1, align 8
  %value.45 = alloca double, i64 1, align 8
  %value.48 = alloca double, i64 1, align 8
  %value.60 = alloca double, i64 1, align 8
  %value.66 = alloca double, i64 1, align 8
  %value.73 = alloca i64, i64 1, align 8
  %value.76 = alloca i64, i64 1, align 8
  %value.78 = alloca i64, i64 1, align 8
  %value.80 = alloca i64, i64 1, align 8
  %value.89 = alloca i64, i64 1, align 8
  %value.94 = alloca double, i64 1, align 8
  %value.101 = alloca i64, i64 1, align 8
  %value.105 = alloca double, i64 1, align 8
  %value.108 = alloca i64, i64 1, align 8
  %value.118 = alloca i64, i64 1, align 8
  %value.120 = alloca i64, i64 1, align 8
  %value.127 = alloca double, i64 1, align 8
  %value.129 = alloca double, i64 1, align 8
  %value.133 = alloca double, i64 1, align 8
  %value.135 = alloca double, i64 1, align 8
  %value.137 = alloca double, i64 1, align 8
  %value.143 = alloca double, i64 1, align 8
  %value.146 = alloca double, i64 1, align 8
  %value.151 = alloca i64, i64 1, align 8
  %value.156 = alloca i64, i64 1, align 8
  %value.158 = alloca i64, i64 1, align 8
  %value.165 = alloca double, i64 1, align 8
  %value.172 = alloca double, i64 1, align 8
  %value.171 = alloca double, i64 1, align 8
  %value.180 = alloca double, i64 1, align 8
  %value.51 = alloca i64, i64 1, align 8
  %value.98 = alloca double, i64 1, align 8
  %value.99 = alloca double, i64 1, align 8
  %value.100 = alloca double, i64 1, align 8
  %value.107 = alloca double, i64 1, align 8
  %value.110 = alloca double, i64 1, align 8
  %value.150 = alloca double, i64 1, align 8
  %value.176 = alloca i64, i64 1, align 8
  %value.50 = alloca i64, i64 1, align 8
  %value.53 = alloca i64, i64 1, align 8
  %value.52 = alloca i64, i64 1, align 8
  %value.54 = alloca i64, i64 1, align 8
  %value.56 = alloca i64, i64 1, align 8
  %value.68 = alloca i64, i64 1, align 8
  %value.72 = alloca i64, i64 1, align 8
  %value.82 = alloca double, i64 1, align 8
  %value.83 = alloca double, i64 1, align 8
  %value.97 = alloca i64, i64 1, align 8
  %value.104 = alloca double, i64 1, align 8
  %value.91 = alloca double, i64 1, align 8
  %value.112 = alloca i64, i64 1, align 8
  %value.84 = alloca i64, i64 1, align 8
  %value.113 = alloca double, i64 1, align 8
  %value.114 = alloca double, i64 1, align 8
  %value.117 = alloca i64, i64 1, align 8
  %value.122 = alloca double, i64 1, align 8
  %value.123 = alloca double, i64 1, align 8
  %value.132 = alloca double, i64 1, align 8
  %value.140 = alloca double, i64 1, align 8
  %value.139 = alloca i64, i64 1, align 8
  %value.141 = alloca double, i64 1, align 8
  %value.142 = alloca double, i64 1, align 8
  %value.145 = alloca double, i64 1, align 8
  %value.148 = alloca double, i64 1, align 8
  %value.149 = alloca double, i64 1, align 8
  %value.164 = alloca double, i64 1, align 8
  %value.178 = alloca double, i64 1, align 8
  %value.177 = alloca i64, i64 1, align 8
  %value.179 = alloca i64, i64 1, align 8
  %value.36 = alloca double, i64 1, align 8
  %value.37 = alloca double, i64 1, align 8
  %value.39 = alloca double, i64 1, align 8
  %value.42 = alloca double, i64 1, align 8
  %value.44 = alloca double, i64 1, align 8
  %value.46 = alloca double, i64 1, align 8
  %value.47 = alloca double, i64 1, align 8
  %value.49 = alloca double, i64 1, align 8
  %value.55 = alloca double, i64 1, align 8
  %value.57 = alloca double, i64 1, align 8
  %value.58 = alloca double, i64 1, align 8
  %value.59 = alloca double, i64 1, align 8
  %value.61 = alloca double, i64 1, align 8
  %value.62 = alloca double, i64 1, align 8
  %value.63 = alloca double, i64 1, align 8
  %value.64 = alloca double, i64 1, align 8
  %value.65 = alloca double, i64 1, align 8
  %value.67 = alloca double, i64 1, align 8
  %value.69 = alloca double, i64 1, align 8
  %value.70 = alloca double, i64 1, align 8
  %value.71 = alloca double, i64 1, align 8
  %value.74 = alloca double, i64 1, align 8
  %value.75 = alloca double, i64 1, align 8
  %value.77 = alloca double, i64 1, align 8
  %value.79 = alloca double, i64 1, align 8
  %value.81 = alloca double, i64 1, align 8
  %value.85 = alloca double, i64 1, align 8
  %value.86 = alloca double, i64 1, align 8
  %value.87 = alloca double, i64 1, align 8
  %value.90 = alloca double, i64 1, align 8
  %value.95 = alloca double, i64 1, align 8
  %value.96 = alloca double, i64 1, align 8
  %value.102 = alloca double, i64 1, align 8
  %value.103 = alloca double, i64 1, align 8
  %value.106 = alloca double, i64 1, align 8
  %value.109 = alloca double, i64 1, align 8
  %value.111 = alloca double, i64 1, align 8
  %value.115 = alloca double, i64 1, align 8
  %value.116 = alloca double, i64 1, align 8
  %value.119 = alloca double, i64 1, align 8
  %value.121 = alloca double, i64 1, align 8
  %value.126 = alloca double, i64 1, align 8
  %value.128 = alloca double, i64 1, align 8
  %value.130 = alloca double, i64 1, align 8
  %value.131 = alloca double, i64 1, align 8
  %value.134 = alloca double, i64 1, align 8
  %value.136 = alloca double, i64 1, align 8
  %value.138 = alloca double, i64 1, align 8
  %value.144 = alloca double, i64 1, align 8
  %value.147 = alloca double, i64 1, align 8
  %value.152 = alloca double, i64 1, align 8
  %value.153 = alloca double, i64 1, align 8
  %value.157 = alloca double, i64 1, align 8
  %value.159 = alloca double, i64 1, align 8
  %value.166 = alloca double, i64 1, align 8
  %value.173 = alloca double, i64 1, align 8
  %value.174 = alloca double, i64 1, align 8
  %value.175 = alloca double, i64 1, align 8
  %value.181 = alloca double, i64 1, align 8
  %value.196 = alloca double, i64 1, align 8
  %value.197 = alloca double, i64 1, align 8
  %value.198 = alloca double, i64 1, align 8
  %value.199 = alloca double, i64 1, align 8
  %value.200 = alloca double, i64 1, align 8
  %value.201 = alloca double, i64 1, align 8
  %value.202 = alloca double, i64 1, align 8
  %value.203 = alloca double, i64 1, align 8
  %value.204 = alloca double, i64 1, align 8
  %value.205 = alloca double, i64 1, align 8
  %value.206 = alloca double, i64 1, align 8
  %value.207 = alloca double, i64 1, align 8
  %value.209 = alloca double, i64 1, align 8
  %value.210 = alloca double, i64 1, align 8
  %value.211 = alloca double, i64 1, align 8
  %value.212 = alloca double, i64 1, align 8
  %value.213 = alloca double, i64 1, align 8
  %value.214 = alloca double, i64 1, align 8
  %value.215 = alloca double, i64 1, align 8
  %value.217 = alloca double, i64 1, align 8
  %value.218 = alloca double, i64 1, align 8
  %value.219 = alloca double, i64 1, align 8
  %value.220 = alloca double, i64 1, align 8
  %value.221 = alloca double, i64 1, align 8
  %value.222 = alloca double, i64 1, align 8
  %value.223 = alloca double, i64 1, align 8
  %value.224 = alloca double, i64 1, align 8
  %value.225 = alloca double, i64 1, align 8
  %value.226 = alloca double, i64 1, align 8
  %value.227 = alloca double, i64 1, align 8
  %value.228 = alloca double, i64 1, align 8
  %value.229 = alloca double, i64 1, align 8
  %value.230 = alloca double, i64 1, align 8
  %value.231 = alloca double, i64 1, align 8
  %value.232 = alloca double, i64 1, align 8
  %value.233 = alloca double, i64 1, align 8
  %value.234 = alloca double, i64 1, align 8
  %value.235 = alloca double, i64 1, align 8
  %value.236 = alloca double, i64 1, align 8
  %value.237 = alloca double, i64 1, align 8
  %value.238 = alloca double, i64 1, align 8
  %value.239 = alloca double, i64 1, align 8
  %value.240 = alloca double, i64 1, align 8
  %value.241 = alloca double, i64 1, align 8
  %value.242 = alloca double, i64 1, align 8
  %value.243 = alloca double, i64 1, align 8
  %value.244 = alloca double, i64 1, align 8
  %value.245 = alloca double, i64 1, align 8
  %value.246 = alloca double, i64 1, align 8
  %value.247 = alloca double, i64 1, align 8
  %value.248 = alloca double, i64 1, align 8
  %value.249 = alloca double, i64 1, align 8
  %value.250 = alloca double, i64 1, align 8
  %value.251 = alloca double, i64 1, align 8
  %value.252 = alloca double, i64 1, align 8
  %value.253 = alloca double, i64 1, align 8
  %value.254 = alloca double, i64 1, align 8
  %value.255 = alloca double, i64 1, align 8
  %value.256 = alloca double, i64 1, align 8
  %value.257 = alloca double, i64 1, align 8
  %value.258 = alloca double, i64 1, align 8
  %value.259 = alloca double, i64 1, align 8
  %value.260 = alloca double, i64 1, align 8
  %value.261 = alloca double, i64 1, align 8
  %value.262 = alloca double, i64 1, align 8
  %value.263 = alloca double, i64 1, align 8
  %value.264 = alloca double, i64 1, align 8
  %value.265 = alloca double, i64 1, align 8
  %value.266 = alloca double, i64 1, align 8
  %value.267 = alloca double, i64 1, align 8
  %value.268 = alloca double, i64 1, align 8
  %value.269 = alloca double, i64 1, align 8
  %value.270 = alloca double, i64 1, align 8
  %value.271 = alloca double, i64 1, align 8
  %value.272 = alloca double, i64 1, align 8
  %value.273 = alloca double, i64 1, align 8
  %value.274 = alloca double, i64 1, align 8
  %value.275 = alloca double, i64 1, align 8
  %value.276 = alloca double, i64 1, align 8
  %value.277 = alloca double, i64 1, align 8
  %value.278 = alloca double, i64 1, align 8
  %value.279 = alloca double, i64 1, align 8
  %value.280 = alloca double, i64 1, align 8
  %value.281 = alloca double, i64 1, align 8
  %value.282 = alloca double, i64 1, align 8
  %value.283 = alloca double, i64 1, align 8
  %value.284 = alloca double, i64 1, align 8
  %value.286 = alloca double, i64 1, align 8
  %value.287 = alloca double, i64 1, align 8
  %value.288 = alloca double, i64 1, align 8
  %value.289 = alloca double, i64 1, align 8
  %value.290 = alloca double, i64 1, align 8
  %value.291 = alloca double, i64 1, align 8
  %value.292 = alloca double, i64 1, align 8
  %value.293 = alloca double, i64 1, align 8
  %value.294 = alloca double, i64 1, align 8
  %value.295 = alloca double, i64 1, align 8
  %value.296 = alloca double, i64 1, align 8
  %value.297 = alloca double, i64 1, align 8
  %value.298 = alloca double, i64 1, align 8
  %value.299 = alloca double, i64 1, align 8
  %value.300 = alloca double, i64 1, align 8
  %value.301 = alloca double, i64 1, align 8
  %value.302 = alloca double, i64 1, align 8
  %value.303 = alloca double, i64 1, align 8
  %value.304 = alloca double, i64 1, align 8
  %value.305 = alloca double, i64 1, align 8
  %value.306 = alloca double, i64 1, align 8
  %value.307 = alloca double, i64 1, align 8
  %value.308 = alloca double, i64 1, align 8
  %value.309 = alloca double, i64 1, align 8
  %value.311 = alloca double, i64 1, align 8
  %value.312 = alloca double, i64 1, align 8
  %value.313 = alloca double, i64 1, align 8
  %value.314 = alloca double, i64 1, align 8
  %value.315 = alloca double, i64 1, align 8
  %value.316 = alloca double, i64 1, align 8
  %value.317 = alloca double, i64 1, align 8
  %value.319 = alloca double, i64 1, align 8
  %value.321 = alloca double, i64 1, align 8
  %value.324 = alloca double, i64 1, align 8
  %value.325 = alloca double, i64 1, align 8
  %value.326 = alloca double, i64 1, align 8
  %value.329 = alloca double, i64 1, align 8
  %value.330 = alloca double, i64 1, align 8
  %value.332 = alloca double, i64 1, align 8
  %value.333 = alloca double, i64 1, align 8
  %value.335 = alloca double, i64 1, align 8
  %value.336 = alloca double, i64 1, align 8
  %value.338 = alloca double, i64 1, align 8
  %value.339 = alloca double, i64 1, align 8
  %value.341 = alloca double, i64 1, align 8
  %value.342 = alloca double, i64 1, align 8
  %value.344 = alloca double, i64 1, align 8
  %value.345 = alloca double, i64 1, align 8
  store double 0x3FF0000000000000, ptr %value.431, align 8
  store i64 3, ptr %value.38, align 8
  store i64 1, ptr %value.43, align 8
  store double 0x3FE0000000000000, ptr %value.45, align 8
  store double 0x3FE0000000000000, ptr %value.48, align 8
  store double 0x3FC5555555555555, ptr %value.60, align 8
  store double 0x3FA999999999999A, ptr %value.66, align 8
  store i64 3, ptr %value.73, align 8
  store i64 2, ptr %value.76, align 8
  store i64 2, ptr %value.78, align 8
  store i64 3, ptr %value.80, align 8
  store i64 -750, ptr %value.89, align 8
  store double 0x40788B2F704A9409, ptr %value.94, align 8
  store i64 3, ptr %value.101, align 8
  store double 0x42399C82CC000000, ptr %value.105, align 8
  store i64 2, ptr %value.108, align 8
  store i64 3, ptr %value.118, align 8
  store i64 1, ptr %value.120, align 8
  store double 0x3FD999999999999A, ptr %value.127, align 8
  store double 0x3FD999999999999A, ptr %value.129, align 8
  store double 0x3FD999999999999A, ptr %value.133, align 8
  store double 0x3D719799812DEA11, ptr %value.135, align 8
  store double 0x3FD999999999999A, ptr %value.137, align 8
  store double 0x3F35D867C3ECE2A5, ptr %value.143, align 8
  store double 0x3FEFFFFDE7210BE9, ptr %value.146, align 8
  store i64 1, ptr %value.151, align 8
  store i64 2, ptr %value.156, align 8
  store i64 1, ptr %value.158, align 8
  store double 0x3FB999999999999A, ptr %value.165, align 8
  store double 0x3FD999999999999A, ptr %value.172, align 8
  store double 0x3FA003A45C94E17B, ptr %value.171, align 8
  store double 0x3FF0000000000000, ptr %value.180, align 8
  store i64 0, ptr %value.51, align 8
  store double 0x3E112E0BE826D695, ptr %value.98, align 8
  store double 0x3F847AE147AE147B, ptr %value.99, align 8
  store double 0x3F847AE147AE147B, ptr %value.100, align 8
  store double 0x3EB0C6F7A0B5ED8D, ptr %value.107, align 8
  store double 0x3D719799812DEA11, ptr %value.110, align 8
  store double 0x3E112E0BE826D695, ptr %value.150, align 8
  store i64 0, ptr %value.176, align 8
  store i64 1, ptr %value.50, align 8
  store i64 1, ptr %value.53, align 8
  store i64 1, ptr %value.52, align 8
  store i64 0, ptr %value.54, align 8
  store i64 0, ptr %value.56, align 8
  store i64 1, ptr %value.68, align 8
  store i64 0, ptr %value.72, align 8
  store double 0x3D719799812DEA11, ptr %value.82, align 8
  store double 0x3FE0000000000000, ptr %value.83, align 8
  store i64 0, ptr %value.97, align 8
  store double 0x41ADCD6500000000, ptr %value.104, align 8
  store double 0x400921FB54442D18, ptr %value.91, align 8
  store i64 500, ptr %value.112, align 8
  store i64 0, ptr %value.84, align 8
  store double 0x3D719799812DEA11, ptr %value.113, align 8
  store double 0x3FE0000000000000, ptr %value.114, align 8
  store i64 2, ptr %value.117, align 8
  store double 0x3F35D867C3ECE2A5, ptr %value.122, align 8
  store double 0x3E112E0BE826D695, ptr %value.123, align 8
  store double 0x3FE999999999999A, ptr %value.132, align 8
  store double 0x4079000000000000, ptr %value.140, align 8
  store i64 0, ptr %value.139, align 8
  store double 0x3FE999999999999A, ptr %value.141, align 8
  store double 0x3FF0000000000000, ptr %value.142, align 8
  store double 0x3FE999999999999A, ptr %value.145, align 8
  store double 0x3D719799812DEA11, ptr %value.148, align 8
  store double 0x3FE0000000000000, ptr %value.149, align 8
  store double 0x3F8211F35EF83318, ptr %value.164, align 8
  store double 0x4072526666666666, ptr %value.178, align 8
  store i64 0, ptr %value.177, align 8
  store i64 2, ptr %value.179, align 8
  %load.71.36.1 = load double, ptr %arg.0, align 8
  %scalar.71.36 = fdiv double 0x3FF0000000000000, %load.71.36.1
  store double %scalar.71.36, ptr %value.36, align 8
  %load.72.37.0 = load double, ptr %arg.1, align 8
  %scalar.72.37 = fmul double %load.72.37.0, %scalar.71.36
  store double %scalar.72.37, ptr %value.37, align 8
  %convert.73.39.1 = sitofp i64 3 to double
  %scalar.73.39 = fadd double %scalar.72.37, %convert.73.39.1
  store double %scalar.73.39, ptr %value.39, align 8
  %scalar.74.42 = fdiv double 0x3FF0000000000000, %scalar.73.39
  store double %scalar.74.42, ptr %value.42, align 8
  %load.75.44.0 = load double, ptr %arg.2, align 8
  %convert.75.44.1 = sitofp i64 1 to double
  %scalar.75.44 = fsub double %load.75.44.0, %convert.75.44.1
  store double %scalar.75.44, ptr %value.44, align 8
  %load.76.46.1 = load double, ptr %arg.3, align 8
  %scalar.76.46 = fmul double 0x3FE0000000000000, %load.76.46.1
  store double %scalar.76.46, ptr %value.46, align 8
  br label %ew.head.77.47
ew.head.77.47:
  %ew.i.77.47 = phi i32 [ 0, %entry ], [ %ew.next.77.47, %ew.body.77.47 ]
  %ew.more.77.47 = icmp slt i32 %ew.i.77.47, 1
  br i1 %ew.more.77.47, label %ew.body.77.47, label %ew.done.77.47
ew.body.77.47:
  %ew.op.77.47.0 = load double, ptr %arg.4, align 8
  %ew.op.77.47.1 = load double, ptr %arg.4, align 8
  %ew.val.77.47 = fmul double %ew.op.77.47.0, %ew.op.77.47.1
  %ew.dst.77.47 = getelementptr double, ptr %value.47, i32 %ew.i.77.47
  store double %ew.val.77.47, ptr %ew.dst.77.47, align 8
  %ew.next.77.47 = add i32 %ew.i.77.47, 1
  br label %ew.head.77.47
ew.done.77.47:
  %scalar.78.49 = fmul double 0x3FE0000000000000, %load.71.36.1
  store double %scalar.78.49, ptr %value.49, align 8
  %load.79.55.0 = load double, ptr %arg.5, align 8
  %scalar.79.55 = fneg double %load.79.55.0
  store double %scalar.79.55, ptr %value.55, align 8
  br label %ew.head.80.57
ew.head.80.57:
  %ew.i.80.57 = phi i32 [ 0, %ew.done.77.47 ], [ %ew.next.80.57, %ew.body.80.57 ]
  %ew.more.80.57 = icmp slt i32 %ew.i.80.57, 1
  br i1 %ew.more.80.57, label %ew.body.80.57, label %ew.done.80.57
ew.body.80.57:
  %ew.op.80.57.0 = load double, ptr %arg.6, align 8
  %ew.op.80.57.1 = load double, ptr %arg.6, align 8
  %ew.val.80.57 = fmul double %ew.op.80.57.0, %ew.op.80.57.1
  %ew.dst.80.57 = getelementptr double, ptr %value.57, i32 %ew.i.80.57
  store double %ew.val.80.57, ptr %ew.dst.80.57, align 8
  %ew.next.80.57 = add i32 %ew.i.80.57, 1
  br label %ew.head.80.57
ew.done.80.57:
  br label %ew.head.81.58
ew.head.81.58:
  %ew.i.81.58 = phi i32 [ 0, %ew.done.80.57 ], [ %ew.next.81.58, %ew.body.81.58 ]
  %ew.more.81.58 = icmp slt i32 %ew.i.81.58, 1
  br i1 %ew.more.81.58, label %ew.body.81.58, label %ew.done.81.58
ew.body.81.58:
  %ew.op.81.58.0.addr = getelementptr double, ptr %value.57, i32 %ew.i.81.58
  %ew.op.81.58.0 = load double, ptr %ew.op.81.58.0.addr, align 8
  %ew.op.81.58.1 = load double, ptr %value.46, align 8
  %ew.val.81.58 = fmul double %ew.op.81.58.0, %ew.op.81.58.1
  %ew.dst.81.58 = getelementptr double, ptr %value.58, i32 %ew.i.81.58
  store double %ew.val.81.58, ptr %ew.dst.81.58, align 8
  %ew.next.81.58 = add i32 %ew.i.81.58, 1
  br label %ew.head.81.58
ew.done.81.58:
  br label %ew.head.82.59
ew.head.82.59:
  %ew.i.82.59 = phi i32 [ 0, %ew.done.81.58 ], [ %ew.next.82.59, %ew.body.82.59 ]
  %ew.more.82.59 = icmp slt i32 %ew.i.82.59, 1
  br i1 %ew.more.82.59, label %ew.body.82.59, label %ew.done.82.59
ew.body.82.59:
  %ew.op.82.59.0.addr = getelementptr double, ptr %value.58, i32 %ew.i.82.59
  %ew.op.82.59.0 = load double, ptr %ew.op.82.59.0.addr, align 8
  %ew.op.82.59.1 = load double, ptr %arg.7, align 8
  %ew.val.82.59 = fadd double %ew.op.82.59.0, %ew.op.82.59.1
  %ew.dst.82.59 = getelementptr double, ptr %value.59, i32 %ew.i.82.59
  store double %ew.val.82.59, ptr %ew.dst.82.59, align 8
  %ew.next.82.59 = add i32 %ew.i.82.59, 1
  br label %ew.head.82.59
ew.done.82.59:
  %scalar.83.61 = fmul double 0x3FC5555555555555, %load.72.37.0
  store double %scalar.83.61, ptr %value.61, align 8
  %scalar.84.62 = fadd double %scalar.83.61, %scalar.78.49
  store double %scalar.84.62, ptr %value.62, align 8
  br label %ew.head.85.63
ew.head.85.63:
  %ew.i.85.63 = phi i32 [ 0, %ew.done.82.59 ], [ %ew.next.85.63, %ew.body.85.63 ]
  %ew.more.85.63 = icmp slt i32 %ew.i.85.63, 1
  br i1 %ew.more.85.63, label %ew.body.85.63, label %ew.done.85.63
ew.body.85.63:
  %ew.op.85.63.0.addr = getelementptr double, ptr %value.47, i32 %ew.i.85.63
  %ew.op.85.63.0 = load double, ptr %ew.op.85.63.0.addr, align 8
  %ew.op.85.63.1 = load double, ptr %value.62, align 8
  %ew.val.85.63 = fmul double %ew.op.85.63.0, %ew.op.85.63.1
  %ew.dst.85.63 = getelementptr double, ptr %value.63, i32 %ew.i.85.63
  store double %ew.val.85.63, ptr %ew.dst.85.63, align 8
  %ew.next.85.63 = add i32 %ew.i.85.63, 1
  br label %ew.head.85.63
ew.done.85.63:
  br label %ew.head.86.64
ew.head.86.64:
  %ew.i.86.64 = phi i32 [ 0, %ew.done.85.63 ], [ %ew.next.86.64, %ew.body.86.64 ]
  %ew.more.86.64 = icmp slt i32 %ew.i.86.64, 1
  br i1 %ew.more.86.64, label %ew.body.86.64, label %ew.done.86.64
ew.body.86.64:
  %ew.op.86.64.0.addr = getelementptr double, ptr %value.59, i32 %ew.i.86.64
  %ew.op.86.64.0 = load double, ptr %ew.op.86.64.0.addr, align 8
  %ew.op.86.64.1.addr = getelementptr double, ptr %value.63, i32 %ew.i.86.64
  %ew.op.86.64.1 = load double, ptr %ew.op.86.64.1.addr, align 8
  %ew.val.86.64 = fadd double %ew.op.86.64.0, %ew.op.86.64.1
  %ew.dst.86.64 = getelementptr double, ptr %value.64, i32 %ew.i.86.64
  store double %ew.val.86.64, ptr %ew.dst.86.64, align 8
  %ew.next.86.64 = add i32 %ew.i.86.64, 1
  br label %ew.head.86.64
ew.done.86.64:
  br label %ew.head.87.65
ew.head.87.65:
  %ew.i.87.65 = phi i32 [ 0, %ew.done.86.64 ], [ %ew.next.87.65, %ew.body.87.65 ]
  %ew.more.87.65 = icmp slt i32 %ew.i.87.65, 1
  br i1 %ew.more.87.65, label %ew.body.87.65, label %ew.done.87.65
ew.body.87.65:
  %ew.op.87.65.0 = load double, ptr %value.44, align 8
  %ew.op.87.65.1.addr = getelementptr double, ptr %value.64, i32 %ew.i.87.65
  %ew.op.87.65.1 = load double, ptr %ew.op.87.65.1.addr, align 8
  %ew.val.87.65 = fmul double %ew.op.87.65.0, %ew.op.87.65.1
  %ew.dst.87.65 = getelementptr double, ptr %value.65, i32 %ew.i.87.65
  store double %ew.val.87.65, ptr %ew.dst.87.65, align 8
  %ew.next.87.65 = add i32 %ew.i.87.65, 1
  br label %ew.head.87.65
ew.done.87.65:
  %load.88.67.1 = load double, ptr %arg.8, align 8
  %scalar.88.67 = fmul double 0x3FA999999999999A, %load.88.67.1
  store double %scalar.88.67, ptr %value.67, align 8
  %load.89.69.1 = load double, ptr %arg.9, align 8
  %scalar.89.69 = fdiv double %load.72.37.0, %load.89.69.1
  store double %scalar.89.69, ptr %value.69, align 8
  %load.90.70.0 = load double, ptr %arg.10, align 8
  %load.90.70.1 = load double, ptr %arg.11, align 8
  %scalar.90.70 = fmul double %load.90.70.0, %load.90.70.1
  store double %scalar.90.70, ptr %value.70, align 8
  %scalar.91.71 = fadd double %scalar.90.70, %load.88.67.1
  store double %scalar.91.71, ptr %value.71, align 8
  %convert.92.74.0 = sitofp i64 3 to double
  %scalar.92.74 = fmul double %convert.92.74.0, %load.90.70.0
  store double %scalar.92.74, ptr %value.74, align 8
  %scalar.93.75 = fmul double %scalar.92.74, %scalar.74.42
  store double %scalar.93.75, ptr %value.75, align 8
  %convert.94.77.1 = sitofp i64 2 to double
  %scalar.94.77 = fadd double %scalar.72.37, %convert.94.77.1
  store double %scalar.94.77, ptr %value.77, align 8
  %convert.95.79.0 = sitofp i64 2 to double
  %scalar.95.79 = fmul double %convert.95.79.0, %load.76.46.1
  store double %scalar.95.79, ptr %value.79, align 8
  %convert.96.81.0 = sitofp i64 3 to double
  %scalar.96.81 = fmul double %convert.96.81.0, %scalar.74.42
  store double %scalar.96.81, ptr %value.81, align 8
  %scalar.97.85 = fneg double %load.90.70.0
  store double %scalar.97.85, ptr %value.85, align 8
  %load.98.86.1 = load double, ptr %arg.12, align 8
  %scalar.98.86 = fmul double %scalar.97.85, %load.98.86.1
  store double %scalar.98.86, ptr %value.86, align 8
  %scalar.99.87 = fmul double %scalar.98.86, %scalar.71.36
  store double %scalar.99.87, ptr %value.87, align 8
  %convert.100.90.0 = sitofp i64 -750 to double
  %scalar.100.90 = fmul double %convert.100.90.0, %scalar.74.42
  store double %scalar.100.90, ptr %value.90, align 8
  br label %ew.head.101.95
ew.head.101.95:
  %ew.i.101.95 = phi i32 [ 0, %ew.done.87.65 ], [ %ew.next.101.95, %ew.body.101.95 ]
  %ew.more.101.95 = icmp slt i32 %ew.i.101.95, 1
  br i1 %ew.more.101.95, label %ew.body.101.95, label %ew.done.101.95
ew.body.101.95:
  %ew.op.101.95.0 = load double, ptr %value.94, align 8
  %ew.op.101.95.1 = load double, ptr %arg.10, align 8
  %ew.val.101.95 = fmul double %ew.op.101.95.0, %ew.op.101.95.1
  %ew.dst.101.95 = getelementptr double, ptr %value.95, i32 %ew.i.101.95
  store double %ew.val.101.95, ptr %ew.dst.101.95, align 8
  %ew.next.101.95 = add i32 %ew.i.101.95, 1
  br label %ew.head.101.95
ew.done.101.95:
  br label %ew.head.102.96
ew.head.102.96:
  %ew.i.102.96 = phi i32 [ 0, %ew.done.101.95 ], [ %ew.next.102.96, %ew.body.102.96 ]
  %ew.more.102.96 = icmp slt i32 %ew.i.102.96, 1
  br i1 %ew.more.102.96, label %ew.body.102.96, label %ew.done.102.96
ew.body.102.96:
  %ew.op.102.96.0.addr = getelementptr double, ptr %value.95, i32 %ew.i.102.96
  %ew.op.102.96.0 = load double, ptr %ew.op.102.96.0.addr, align 8
  %ew.op.102.96.1 = load double, ptr %arg.0, align 8
  %ew.val.102.96 = fmul double %ew.op.102.96.0, %ew.op.102.96.1
  %ew.dst.102.96 = getelementptr double, ptr %value.96, i32 %ew.i.102.96
  store double %ew.val.102.96, ptr %ew.dst.102.96, align 8
  %ew.next.102.96 = add i32 %ew.i.102.96, 1
  br label %ew.head.102.96
ew.done.102.96:
  %load.103.102.0 = load double, ptr %arg.13, align 8
  %scalar.103.102 = fmul double %load.103.102.0, %load.103.102.0
  store double %scalar.103.102, ptr %value.102, align 8
  %convert.104.103.0 = sitofp i64 3 to double
  %scalar.104.103 = fmul double %convert.104.103.0, %scalar.103.102
  store double %scalar.104.103, ptr %value.103, align 8
  %load.105.106.1 = load double, ptr %arg.14, align 8
  %scalar.105.106 = fmul double 0x42399C82CC000000, %load.105.106.1
  store double %scalar.105.106, ptr %value.106, align 8
  %convert.106.109.1 = sitofp i64 2 to double
  %scalar.106.109 = fdiv double %load.103.102.0, %convert.106.109.1
  store double %scalar.106.109, ptr %value.109, align 8
  %load.107.111.0 = load double, ptr %arg.15, align 8
  %scalar.107.111 = fneg double %load.107.111.0
  store double %scalar.107.111, ptr %value.111, align 8
  %load.108.115.1 = load double, ptr %arg.16, align 8
  %scalar.108.115 = fmul double %load.98.86.1, %load.108.115.1
  store double %scalar.108.115, ptr %value.115, align 8
  %load.109.116.0 = load double, ptr %arg.17, align 8
  %scalar.109.116 = fneg double %load.109.116.0
  store double %scalar.109.116, ptr %value.116, align 8
  %convert.110.119.1 = sitofp i64 3 to double
  %scalar.110.119 = fdiv double %scalar.72.37, %convert.110.119.1
  store double %scalar.110.119, ptr %value.119, align 8
  %convert.111.121.1 = sitofp i64 1 to double
  %scalar.111.121 = fadd double %scalar.110.119, %convert.111.121.1
  store double %scalar.111.121, ptr %value.121, align 8
  %scalar.112.126 = fdiv double 0x3FF0000000000000, %scalar.75.44
  store double %scalar.112.126, ptr %value.126, align 8
  %scalar.113.128 = call double @llvm.pow.f64(double %load.75.44.0, double 0x3FD999999999999A)
  store double %scalar.113.128, ptr %value.128, align 8
  %load.114.130.0 = load double, ptr %arg.18, align 8
  %scalar.114.130 = call double @llvm.pow.f64(double %load.114.130.0, double 0x3FD999999999999A)
  store double %scalar.114.130, ptr %value.130, align 8
  %scalar.115.131 = fmul double %scalar.113.128, %scalar.114.130
  store double %scalar.115.131, ptr %value.131, align 8
  %scalar.116.134 = call double @llvm.pow.f64(double %scalar.112.126, double 0x3FD999999999999A)
  store double %scalar.116.134, ptr %value.134, align 8
  br label %ew.head.117.136
ew.head.117.136:
  %ew.i.117.136 = phi i32 [ 0, %ew.done.102.96 ], [ %ew.next.117.136, %ew.body.117.136 ]
  %ew.more.117.136 = icmp slt i32 %ew.i.117.136, 1
  br i1 %ew.more.117.136, label %ew.body.117.136, label %ew.done.117.136
ew.body.117.136:
  %ew.op.117.136.0.addr = getelementptr double, ptr %value.47, i32 %ew.i.117.136
  %ew.op.117.136.0 = load double, ptr %ew.op.117.136.0.addr, align 8
  %ew.op.117.136.1 = load double, ptr %value.135, align 8
  %ew.val.117.136 = fadd double %ew.op.117.136.0, %ew.op.117.136.1
  %ew.dst.117.136 = getelementptr double, ptr %value.136, i32 %ew.i.117.136
  store double %ew.val.117.136, ptr %ew.dst.117.136, align 8
  %ew.next.117.136 = add i32 %ew.i.117.136, 1
  br label %ew.head.117.136
ew.done.117.136:
  br label %ew.head.118.138
ew.head.118.138:
  %ew.i.118.138 = phi i32 [ 0, %ew.done.117.136 ], [ %ew.next.118.138, %ew.body.118.138 ]
  %ew.more.118.138 = icmp slt i32 %ew.i.118.138, 1
  br i1 %ew.more.118.138, label %ew.body.118.138, label %ew.done.118.138
ew.body.118.138:
  %ew.op.118.138.0.addr = getelementptr double, ptr %value.136, i32 %ew.i.118.138
  %ew.op.118.138.0 = load double, ptr %ew.op.118.138.0.addr, align 8
  %ew.op.118.138.1 = load double, ptr %value.137, align 8
  %ew.val.118.138 = call double @llvm.pow.f64(double %ew.op.118.138.0, double %ew.op.118.138.1)
  %ew.dst.118.138 = getelementptr double, ptr %value.138, i32 %ew.i.118.138
  store double %ew.val.118.138, ptr %ew.dst.118.138, align 8
  %ew.next.118.138 = add i32 %ew.i.118.138, 1
  br label %ew.head.118.138
ew.done.118.138:
  %scalar.119.144 = fmul double 0x3F35D867C3ECE2A5, %load.114.130.0
  store double %scalar.119.144, ptr %value.144, align 8
  %load.120.147.1 = load double, ptr %arg.19, align 8
  %scalar.120.147 = fmul double 0x3FEFFFFDE7210BE9, %load.120.147.1
  store double %scalar.120.147, ptr %value.147, align 8
  %load.121.152.0 = load double, ptr %arg.20, align 8
  %convert.121.152.1 = sitofp i64 1 to double
  %scalar.121.152 = fadd double %load.121.152.0, %convert.121.152.1
  store double %scalar.121.152, ptr %value.152, align 8
  %load.122.153.0 = load double, ptr %arg.21, align 8
  %scalar.122.153 = fmul double %load.122.153.0, %scalar.121.152
  store double %scalar.122.153, ptr %value.153, align 8
  %convert.123.157.1 = sitofp i64 2 to double
  %scalar.123.157 = fdiv double %scalar.72.37, %convert.123.157.1
  store double %scalar.123.157, ptr %value.157, align 8
  %convert.124.159.1 = sitofp i64 1 to double
  %scalar.124.159 = fadd double %scalar.123.157, %convert.124.159.1
  store double %scalar.124.159, ptr %value.159, align 8
  %scalar.125.166 = call double @llvm.pow.f64(double %load.90.70.0, double 0x3FB999999999999A)
  store double %scalar.125.166, ptr %value.166, align 8
  %scalar.126.173 = call double @llvm.pow.f64(double %load.90.70.0, double 0x3FD999999999999A)
  store double %scalar.126.173, ptr %value.173, align 8
  br label %ew.head.127.174
ew.head.127.174:
  %ew.i.127.174 = phi i32 [ 0, %ew.done.118.138 ], [ %ew.next.127.174, %ew.body.127.174 ]
  %ew.more.127.174 = icmp slt i32 %ew.i.127.174, 1
  br i1 %ew.more.127.174, label %ew.body.127.174, label %ew.done.127.174
ew.body.127.174:
  %ew.op.127.174.0 = load double, ptr %value.171, align 8
  %ew.op.127.174.1 = load double, ptr %value.173, align 8
  %ew.val.127.174 = fmul double %ew.op.127.174.0, %ew.op.127.174.1
  %ew.dst.127.174 = getelementptr double, ptr %value.174, i32 %ew.i.127.174
  store double %ew.val.127.174, ptr %ew.dst.127.174, align 8
  %ew.next.127.174 = add i32 %ew.i.127.174, 1
  br label %ew.head.127.174
ew.done.127.174:
  br label %ew.head.128.175
ew.head.128.175:
  %ew.i.128.175 = phi i32 [ 0, %ew.done.127.174 ], [ %ew.next.128.175, %ew.body.128.175 ]
  %ew.more.128.175 = icmp slt i32 %ew.i.128.175, 1
  br i1 %ew.more.128.175, label %ew.body.128.175, label %ew.done.128.175
ew.body.128.175:
  %ew.op.128.175.0.addr = getelementptr double, ptr %value.174, i32 %ew.i.128.175
  %ew.op.128.175.0 = load double, ptr %ew.op.128.175.0.addr, align 8
  %ew.op.128.175.1 = load double, ptr %arg.12, align 8
  %ew.val.128.175 = fmul double %ew.op.128.175.0, %ew.op.128.175.1
  %ew.dst.128.175 = getelementptr double, ptr %value.175, i32 %ew.i.128.175
  store double %ew.val.128.175, ptr %ew.dst.128.175, align 8
  %ew.next.128.175 = add i32 %ew.i.128.175, 1
  br label %ew.head.128.175
ew.done.128.175:
  br label %ew.head.129.181
ew.head.129.181:
  %ew.i.129.181 = phi i32 [ 0, %ew.done.128.175 ], [ %ew.next.129.181, %ew.body.129.181 ]
  %ew.more.129.181 = icmp slt i32 %ew.i.129.181, 1
  br i1 %ew.more.129.181, label %ew.body.129.181, label %ew.done.129.181
ew.body.129.181:
  %ew.op.129.181.0 = load double, ptr %value.180, align 8
  %ew.op.129.181.1 = load double, ptr %arg.22, align 8
  %ew.val.129.181 = fsub double %ew.op.129.181.0, %ew.op.129.181.1
  %ew.dst.129.181 = getelementptr double, ptr %value.181, i32 %ew.i.129.181
  store double %ew.val.129.181, ptr %ew.dst.129.181, align 8
  %ew.next.129.181 = add i32 %ew.i.129.181, 1
  br label %ew.head.129.181
ew.done.129.181:
  %convert.130.196.0 = sitofp i64 0 to double
  %load.130.196.1 = load double, ptr %arg.23, align 8
  %scalar.130.196 = call double @llvm.maxnum.f64(double %convert.130.196.0, double %load.130.196.1)
  store double %scalar.130.196, ptr %value.196, align 8
  %scalar.131.197 = call double @llvm.maxnum.f64(double 0x3E112E0BE826D695, double %load.71.36.1)
  store double %scalar.131.197, ptr %value.197, align 8
  %load.132.198.1 = load double, ptr %arg.24, align 8
  %scalar.132.198 = call double @llvm.maxnum.f64(double 0x3F847AE147AE147B, double %load.132.198.1)
  store double %scalar.132.198, ptr %value.198, align 8
  %scalar.133.199 = call double @llvm.maxnum.f64(double 0x3F847AE147AE147B, double %load.132.198.1)
  store double %scalar.133.199, ptr %value.199, align 8
  %scalar.134.200 = fmul double %scalar.132.198, %scalar.133.199
  store double %scalar.134.200, ptr %value.200, align 8
  %scalar.135.201 = fmul double %scalar.131.197, %scalar.134.200
  store double %scalar.135.201, ptr %value.201, align 8
  %scalar.136.202 = call double @llvm.maxnum.f64(double 0x3EB0C6F7A0B5ED8D, double %scalar.106.109)
  store double %scalar.136.202, ptr %value.202, align 8
  %scalar.137.203 = fdiv double %scalar.105.106, %scalar.136.202
  store double %scalar.137.203, ptr %value.203, align 8
  %scalar.138.204 = call double @llvm.maxnum.f64(double 0x3D719799812DEA11, double %load.90.70.0)
  store double %scalar.138.204, ptr %value.204, align 8
  %scalar.139.205 = call double @llvm.maxnum.f64(double 0x3E112E0BE826D695, double %scalar.122.153)
  store double %scalar.139.205, ptr %value.205, align 8
  %scalar.140.206 = fdiv double 0x3FF0000000000000, %scalar.139.205
  store double %scalar.140.206, ptr %value.206, align 8
  %convert.141.207.0 = sitofp i64 0 to double
  %scalar.141.207 = call double @llvm.maxnum.f64(double %convert.141.207.0, double %load.90.70.1)
  store double %scalar.141.207, ptr %value.207, align 8
  %convert.142.209.0 = sitofp i64 1 to double
  %scalar.142.209 = call double @llvm.minnum.f64(double %convert.142.209.0, double %scalar.130.196)
  store double %scalar.142.209, ptr %value.209, align 8
  %scalar.143.210 = fmul double %load.122.153.0, %scalar.142.209
  store double %scalar.143.210, ptr %value.210, align 8
  %scalar.144.211 = fmul double %load.121.152.0, %scalar.142.209
  store double %scalar.144.211, ptr %value.211, align 8
  %convert.145.212.1 = sitofp i64 1 to double
  %scalar.145.212 = fadd double %scalar.144.211, %convert.145.212.1
  store double %scalar.145.212, ptr %value.212, align 8
  %scalar.146.213 = fmul double %scalar.143.210, %scalar.145.212
  store double %scalar.146.213, ptr %value.213, align 8
  %convert.147.214.0 = sitofp i64 1 to double
  %scalar.147.214 = call double @llvm.minnum.f64(double %convert.147.214.0, double %scalar.146.213)
  store double %scalar.147.214, ptr %value.214, align 8
  %scalar.148.215 = fmul double %load.72.37.0, %scalar.147.214
  store double %scalar.148.215, ptr %value.215, align 8
  %load.149.216.1 = load double, ptr %arg.25, align 8
  %scalar.149.216 = fadd double %scalar.148.215, %load.149.216.1
  store double %scalar.149.216, ptr %out.0, align 8
  %scalar.150.217 = fadd double %scalar.79.55, %scalar.149.216
  store double %scalar.150.217, ptr %value.217, align 8
  %convert.151.218.0 = sitofp i64 0 to double
  %scalar.151.218 = call double @llvm.maxnum.f64(double %convert.151.218.0, double %scalar.150.217)
  store double %scalar.151.218, ptr %value.218, align 8
  %scalar.152.219 = fmul double %load.114.130.0, %scalar.151.218
  store double %scalar.152.219, ptr %value.219, align 8
  br label %ew.head.153.220
ew.head.153.220:
  %ew.i.153.220 = phi i32 [ 0, %ew.done.129.181 ], [ %ew.next.153.220, %ew.body.153.220 ]
  %ew.more.153.220 = icmp slt i32 %ew.i.153.220, 1
  br i1 %ew.more.153.220, label %ew.body.153.220, label %ew.done.153.220
ew.body.153.220:
  %ew.op.153.220.0 = load double, ptr %value.219, align 8
  %ew.op.153.220.1.addr = getelementptr double, ptr %value.65, i32 %ew.i.153.220
  %ew.op.153.220.1 = load double, ptr %ew.op.153.220.1.addr, align 8
  %ew.val.153.220 = fsub double %ew.op.153.220.0, %ew.op.153.220.1
  %ew.dst.153.220 = getelementptr double, ptr %value.220, i32 %ew.i.153.220
  store double %ew.val.153.220, ptr %ew.dst.153.220, align 8
  %ew.next.153.220 = add i32 %ew.i.153.220, 1
  br label %ew.head.153.220
ew.done.153.220:
  %convert.154.221.0 = sitofp i64 0 to double
  %load.154.221.1 = load double, ptr %value.220, align 8
  %scalar.154.221 = call double @llvm.maxnum.f64(double %convert.154.221.0, double %load.154.221.1)
  store double %scalar.154.221, ptr %value.221, align 8
  %convert.155.222.1 = sitofp i64 1 to double
  %scalar.155.222 = fsub double %scalar.147.214, %convert.155.222.1
  store double %scalar.155.222, ptr %value.222, align 8
  %load.156.223.0 = load double, ptr %arg.26, align 8
  %scalar.156.223 = fmul double %load.156.223.0, %scalar.151.218
  store double %scalar.156.223, ptr %value.223, align 8
  %scalar.157.224 = fsub double %scalar.91.71, %scalar.156.223
  store double %scalar.157.224, ptr %value.224, align 8
  %scalar.158.225 = fmul double %scalar.155.222, %scalar.89.69
  store double %scalar.158.225, ptr %value.225, align 8
  %scalar.159.226 = fadd double %scalar.158.225, %scalar.157.224
  store double %scalar.159.226, ptr %value.226, align 8
  %scalar.160.227 = call double @llvm.maxnum.f64(double %scalar.88.67, double %scalar.159.226)
  store double %scalar.160.227, ptr %value.227, align 8
  %scalar.161.228 = fdiv double %scalar.154.221, %scalar.160.227
  store double %scalar.161.228, ptr %value.228, align 8
  %scalar.162.229 = fmul double %load.98.86.1, %scalar.161.228
  store double %scalar.162.229, ptr %value.229, align 8
  %scalar.163.230 = fmul double %scalar.93.75, %scalar.162.229
  store double %scalar.163.230, ptr %value.230, align 8
  %scalar.164.231 = fmul double %scalar.163.230, %scalar.94.77
  store double %scalar.164.231, ptr %value.231, align 8
  %scalar.165.232 = fdiv double %scalar.164.231, %scalar.95.79
  store double %scalar.165.232, ptr %value.232, align 8
  %load.166.233.1 = load double, ptr %arg.6, align 8
  %scalar.166.233 = fadd double %scalar.165.232, %load.166.233.1
  store double %scalar.166.233, ptr %value.233, align 8
  %convert.167.234.0 = sitofp i64 0 to double
  %scalar.167.234 = call double @llvm.maxnum.f64(double %convert.167.234.0, double %scalar.166.233)
  store double %scalar.167.234, ptr %value.234, align 8
  %scalar.168.235 = fmul double %scalar.155.222, %scalar.89.69
  store double %scalar.168.235, ptr %value.235, align 8
  %scalar.169.236 = fadd double %scalar.168.235, %scalar.157.224
  store double %scalar.169.236, ptr %value.236, align 8
  %scalar.170.237 = call double @llvm.maxnum.f64(double %scalar.88.67, double %scalar.169.236)
  store double %scalar.170.237, ptr %value.237, align 8
  %scalar.171.238 = fdiv double %scalar.154.221, %scalar.170.237
  store double %scalar.171.238, ptr %value.238, align 8
  %scalar.172.239 = fmul double %scalar.96.81, %scalar.171.238
  store double %scalar.172.239, ptr %value.239, align 8
  %scalar.173.240 = fsub double %load.107.111.0, %scalar.172.239
  store double %scalar.173.240, ptr %value.240, align 8
  %scalar.174.241 = fmul double %scalar.173.240, %scalar.173.240
  store double %scalar.174.241, ptr %value.241, align 8
  %scalar.175.242 = fadd double %scalar.174.241, 0x3D719799812DEA11
  store double %scalar.175.242, ptr %value.242, align 8
  %scalar.176.243 = call double @llvm.pow.f64(double %scalar.175.242, double 0x3FE0000000000000)
  store double %scalar.176.243, ptr %value.243, align 8
  %scalar.177.244 = fmul double %scalar.100.90, %scalar.171.238
  store double %scalar.177.244, ptr %value.244, align 8
  %scalar.178.245 = fneg double %scalar.173.240
  store double %scalar.178.245, ptr %value.245, align 8
  %convert.179.246.0 = sitofp i64 0 to double
  %scalar.179.246 = call double @llvm.maxnum.f64(double %convert.179.246.0, double %scalar.178.245)
  store double %scalar.179.246, ptr %value.246, align 8
  br label %ew.head.180.247
ew.head.180.247:
  %ew.i.180.247 = phi i32 [ 0, %ew.done.153.220 ], [ %ew.next.180.247, %ew.body.180.247 ]
  %ew.more.180.247 = icmp slt i32 %ew.i.180.247, 1
  br i1 %ew.more.180.247, label %ew.body.180.247, label %ew.done.180.247
ew.body.180.247:
  %ew.op.180.247.0.addr = getelementptr double, ptr %value.96, i32 %ew.i.180.247
  %ew.op.180.247.0 = load double, ptr %ew.op.180.247.0.addr, align 8
  %ew.op.180.247.1 = load double, ptr %value.246, align 8
  %ew.val.180.247 = fmul double %ew.op.180.247.0, %ew.op.180.247.1
  %ew.dst.180.247 = getelementptr double, ptr %value.247, i32 %ew.i.180.247
  store double %ew.val.180.247, ptr %ew.dst.180.247, align 8
  %ew.next.180.247 = add i32 %ew.i.180.247, 1
  br label %ew.head.180.247
ew.done.180.247:
  %load.181.248.0 = load double, ptr %value.247, align 8
  %scalar.181.248 = fdiv double %load.181.248.0, %scalar.135.201
  store double %scalar.181.248, ptr %value.248, align 8
  %scalar.182.249 = call double @llvm.minnum.f64(double 0x41ADCD6500000000, double %scalar.137.203)
  store double %scalar.182.249, ptr %value.249, align 8
  %scalar.183.250 = fmul double %scalar.104.103, %scalar.182.249
  store double %scalar.183.250, ptr %value.250, align 8
  %scalar.184.251 = fadd double %scalar.181.248, %scalar.183.250
  store double %scalar.184.251, ptr %value.251, align 8
  %scalar.185.252 = fmul double 0x400921FB54442D18, %scalar.184.251
  store double %scalar.185.252, ptr %value.252, align 8
  %scalar.186.253 = fdiv double %scalar.185.252, %scalar.138.204
  store double %scalar.186.253, ptr %value.253, align 8
  %scalar.187.254 = fadd double %scalar.177.244, %scalar.186.253
  store double %scalar.187.254, ptr %value.254, align 8
  %scalar.188.255 = fmul double %scalar.99.87, %scalar.187.254
  store double %scalar.188.255, ptr %value.255, align 8
  %scalar.189.256 = fmul double %scalar.161.228, %scalar.96.81
  store double %scalar.189.256, ptr %value.256, align 8
  %scalar.190.257 = fadd double %scalar.107.111, %scalar.189.256
  store double %scalar.190.257, ptr %value.257, align 8
  %scalar.191.258 = fadd double %scalar.190.257, %scalar.176.243
  store double %scalar.191.258, ptr %value.258, align 8
  %scalar.192.259 = fmul double %scalar.188.255, %scalar.191.258
  store double %scalar.192.259, ptr %value.259, align 8
  %convert.193.260.0 = sitofp i64 500 to double
  %scalar.193.260 = fmul double %convert.193.260.0, %scalar.176.243
  store double %scalar.193.260, ptr %value.260, align 8
  %scalar.194.261 = fdiv double %scalar.192.259, %scalar.193.260
  store double %scalar.194.261, ptr %value.261, align 8
  %load.195.262.1 = load double, ptr %arg.4, align 8
  %scalar.195.262 = fadd double %scalar.194.261, %load.195.262.1
  store double %scalar.195.262, ptr %value.262, align 8
  %convert.196.263.0 = sitofp i64 0 to double
  %scalar.196.263 = call double @llvm.maxnum.f64(double %convert.196.263.0, double %scalar.195.262)
  store double %scalar.196.263, ptr %value.263, align 8
  %scalar.197.264 = fadd double %scalar.167.234, %scalar.196.263
  store double %scalar.197.264, ptr %value.264, align 8
  %scalar.198.265 = fmul double %load.98.86.1, %scalar.197.264
  store double %scalar.198.265, ptr %value.265, align 8
  %scalar.199.266 = fadd double %scalar.198.265, %load.90.70.1
  store double %scalar.199.266, ptr %value.266, align 8
  %scalar.200.267 = call double @llvm.minnum.f64(double %load.120.147.1, double %scalar.199.266)
  store double %scalar.200.267, ptr %value.267, align 8
  %scalar.201.268 = fneg double %scalar.200.267
  store double %scalar.201.268, ptr %value.268, align 8
  %scalar.202.269 = fadd double %load.109.116.0, %scalar.201.268
  store double %scalar.202.269, ptr %value.269, align 8
  %scalar.203.270 = fadd double %load.109.116.0, %scalar.201.268
  store double %scalar.203.270, ptr %value.270, align 8
  %scalar.204.271 = fmul double %scalar.202.269, %scalar.203.270
  store double %scalar.204.271, ptr %value.271, align 8
  %scalar.205.272 = fadd double %scalar.204.271, 0x3D719799812DEA11
  store double %scalar.205.272, ptr %value.272, align 8
  %scalar.206.273 = call double @llvm.pow.f64(double %scalar.205.272, double 0x3FE0000000000000)
  store double %scalar.206.273, ptr %value.273, align 8
  %scalar.207.274 = fmul double %scalar.108.115, %scalar.151.218
  store double %scalar.207.274, ptr %value.274, align 8
  %scalar.208.275 = fadd double %scalar.109.116, %scalar.200.267
  store double %scalar.208.275, ptr %value.275, align 8
  %scalar.209.276 = fadd double %scalar.208.275, %scalar.206.273
  store double %scalar.209.276, ptr %value.276, align 8
  %scalar.210.277 = fmul double %scalar.207.274, %scalar.209.276
  store double %scalar.210.277, ptr %value.277, align 8
  %convert.211.278.0 = sitofp i64 2 to double
  %scalar.211.278 = fmul double %convert.211.278.0, %scalar.206.273
  store double %scalar.211.278, ptr %value.278, align 8
  %scalar.212.279 = fdiv double %scalar.210.277, %scalar.211.278
  store double %scalar.212.279, ptr %value.279, align 8
  %scalar.213.280 = fadd double %scalar.212.279, %load.79.55.0
  store double %scalar.213.280, ptr %value.280, align 8
  %scalar.214.281 = call double @llvm.minnum.f64(double %scalar.149.216, double %scalar.213.280)
  store double %scalar.214.281, ptr %value.281, align 8
  %scalar.215.282 = fdiv double %scalar.161.228, %scalar.111.121
  store double %scalar.215.282, ptr %value.282, align 8
  %scalar.216.283 = fmul double 0x3F35D867C3ECE2A5, %scalar.152.219
  store double %scalar.216.283, ptr %value.283, align 8
  %scalar.217.284 = call double @llvm.maxnum.f64(double 0x3E112E0BE826D695, double %scalar.216.283)
  store double %scalar.217.284, ptr %value.284, align 8
  %scalar.218.285 = fdiv double %scalar.154.221, %scalar.217.284
  store double %scalar.218.285, ptr %out.1, align 8
  %scalar.219.286 = call double @llvm.pow.f64(double %scalar.154.221, double 0x3FE999999999999A)
  store double %scalar.219.286, ptr %value.286, align 8
  %scalar.220.287 = fmul double %scalar.115.131, %scalar.219.286
  store double %scalar.220.287, ptr %value.287, align 8
  %scalar.221.288 = fmul double %scalar.220.287, %scalar.116.134
  store double %scalar.221.288, ptr %value.288, align 8
  br label %ew.head.222.289
ew.head.222.289:
  %ew.i.222.289 = phi i32 [ 0, %ew.done.180.247 ], [ %ew.next.222.289, %ew.body.222.289 ]
  %ew.more.222.289 = icmp slt i32 %ew.i.222.289, 1
  br i1 %ew.more.222.289, label %ew.body.222.289, label %ew.done.222.289
ew.body.222.289:
  %ew.op.222.289.0 = load double, ptr %value.288, align 8
  %ew.op.222.289.1.addr = getelementptr double, ptr %value.138, i32 %ew.i.222.289
  %ew.op.222.289.1 = load double, ptr %ew.op.222.289.1.addr, align 8
  %ew.val.222.289 = fmul double %ew.op.222.289.0, %ew.op.222.289.1
  %ew.dst.222.289 = getelementptr double, ptr %value.289, i32 %ew.i.222.289
  store double %ew.val.222.289, ptr %ew.dst.222.289, align 8
  %ew.next.222.289 = add i32 %ew.i.222.289, 1
  br label %ew.head.222.289
ew.done.222.289:
  %scalar.223.290 = fsub double %scalar.218.285, 0x4079000000000000
  store double %scalar.223.290, ptr %value.290, align 8
  %convert.224.291.0 = sitofp i64 0 to double
  %scalar.224.291 = call double @llvm.maxnum.f64(double %convert.224.291.0, double %scalar.223.290)
  store double %scalar.224.291, ptr %value.291, align 8
  br label %ew.head.225.292
ew.head.225.292:
  %ew.i.225.292 = phi i32 [ 0, %ew.done.222.289 ], [ %ew.next.225.292, %ew.body.225.292 ]
  %ew.more.225.292 = icmp slt i32 %ew.i.225.292, 1
  br i1 %ew.more.225.292, label %ew.body.225.292, label %ew.done.225.292
ew.body.225.292:
  %ew.op.225.292.0.addr = getelementptr double, ptr %value.289, i32 %ew.i.225.292
  %ew.op.225.292.0 = load double, ptr %ew.op.225.292.0.addr, align 8
  %ew.op.225.292.1 = load double, ptr %value.291, align 8
  %ew.val.225.292 = fmul double %ew.op.225.292.0, %ew.op.225.292.1
  %ew.dst.225.292 = getelementptr double, ptr %value.292, i32 %ew.i.225.292
  store double %ew.val.225.292, ptr %ew.dst.225.292, align 8
  %ew.next.225.292 = add i32 %ew.i.225.292, 1
  br label %ew.head.225.292
ew.done.225.292:
  %scalar.226.293 = call double @llvm.pow.f64(double %scalar.160.227, double 0x3FE999999999999A)
  store double %scalar.226.293, ptr %value.293, align 8
  %scalar.227.294 = fmul double %scalar.119.144, %scalar.218.285
  store double %scalar.227.294, ptr %value.294, align 8
  %scalar.228.295 = call double @llvm.maxnum.f64(double 0x3FF0000000000000, double %scalar.227.294)
  store double %scalar.228.295, ptr %value.295, align 8
  %scalar.229.296 = call double @llvm.pow.f64(double %scalar.228.295, double 0x3FE999999999999A)
  store double %scalar.229.296, ptr %value.296, align 8
  %scalar.230.297 = fmul double %scalar.226.293, %scalar.229.296
  store double %scalar.230.297, ptr %value.297, align 8
  %load.231.298.0 = load double, ptr %value.292, align 8
  %scalar.231.298 = fdiv double %load.231.298.0, %scalar.230.297
  store double %scalar.231.298, ptr %value.298, align 8
  %scalar.232.299 = fadd double %scalar.201.268, %scalar.120.147
  store double %scalar.232.299, ptr %value.299, align 8
  %scalar.233.300 = fadd double %scalar.201.268, %scalar.120.147
  store double %scalar.233.300, ptr %value.300, align 8
  %scalar.234.301 = fmul double %scalar.232.299, %scalar.233.300
  store double %scalar.234.301, ptr %value.301, align 8
  %scalar.235.302 = fadd double %scalar.234.301, 0x3D719799812DEA11
  store double %scalar.235.302, ptr %value.302, align 8
  %scalar.236.303 = call double @llvm.pow.f64(double %scalar.235.302, double 0x3FE0000000000000)
  store double %scalar.236.303, ptr %value.303, align 8
  %load.237.304.0 = load double, ptr %arg.27, align 8
  %scalar.237.304 = fmul double %load.237.304.0, %scalar.162.229
  store double %scalar.237.304, ptr %value.304, align 8
  %load.238.305.1 = load double, ptr %arg.28, align 8
  %scalar.238.305 = fdiv double %scalar.237.304, %load.238.305.1
  store double %scalar.238.305, ptr %value.305, align 8
  %scalar.239.306 = fadd double %scalar.238.305, %load.130.196.1
  store double %scalar.239.306, ptr %value.306, align 8
  %scalar.240.307 = call double @llvm.minnum.f64(double %scalar.239.306, double %scalar.140.206)
  store double %scalar.240.307, ptr %value.307, align 8
  %scalar.241.308 = fmul double %scalar.215.282, %scalar.124.159
  store double %scalar.241.308, ptr %value.308, align 8
  br label %ew.head.242.309
ew.head.242.309:
  %ew.i.242.309 = phi i32 [ 0, %ew.done.225.292 ], [ %ew.next.242.309, %ew.body.242.309 ]
  %ew.more.242.309 = icmp slt i32 %ew.i.242.309, 1
  br i1 %ew.more.242.309, label %ew.body.242.309, label %ew.done.242.309
ew.body.242.309:
  %ew.op.242.309.0 = load double, ptr %value.164, align 8
  %ew.op.242.309.1 = load double, ptr %value.298, align 8
  %ew.val.242.309 = fmul double %ew.op.242.309.0, %ew.op.242.309.1
  %ew.dst.242.309 = getelementptr double, ptr %value.309, i32 %ew.i.242.309
  store double %ew.val.242.309, ptr %ew.dst.242.309, align 8
  %ew.next.242.309 = add i32 %ew.i.242.309, 1
  br label %ew.head.242.309
ew.done.242.309:
  %load.243.310.0 = load double, ptr %value.309, align 8
  %scalar.243.310 = fdiv double %load.243.310.0, %scalar.125.166
  store double %scalar.243.310, ptr %out.2, align 8
  br label %ew.head.244.311
ew.head.244.311:
  %ew.i.244.311 = phi i32 [ 0, %ew.done.242.309 ], [ %ew.next.244.311, %ew.body.244.311 ]
  %ew.more.244.311 = icmp slt i32 %ew.i.244.311, 1
  br i1 %ew.more.244.311, label %ew.body.244.311, label %ew.done.244.311
ew.body.244.311:
  %ew.op.244.311.0.addr = getelementptr double, ptr %value.175, i32 %ew.i.244.311
  %ew.op.244.311.0 = load double, ptr %ew.op.244.311.0.addr, align 8
  %ew.op.244.311.1 = load double, ptr %value.298, align 8
  %ew.val.244.311 = fmul double %ew.op.244.311.0, %ew.op.244.311.1
  %ew.dst.244.311 = getelementptr double, ptr %value.311, i32 %ew.i.244.311
  store double %ew.val.244.311, ptr %ew.dst.244.311, align 8
  %ew.next.244.311 = add i32 %ew.i.244.311, 1
  br label %ew.head.244.311
ew.done.244.311:
  br label %ew.head.245.312
ew.head.245.312:
  %ew.i.245.312 = phi i32 [ 0, %ew.done.244.311 ], [ %ew.next.245.312, %ew.body.245.312 ]
  %ew.more.245.312 = icmp slt i32 %ew.i.245.312, 1
  br i1 %ew.more.245.312, label %ew.body.245.312, label %ew.done.245.312
ew.body.245.312:
  %ew.op.245.312.0.addr = getelementptr double, ptr %value.311, i32 %ew.i.245.312
  %ew.op.245.312.0 = load double, ptr %ew.op.245.312.0.addr, align 8
  %ew.op.245.312.1 = load double, ptr %value.207, align 8
  %ew.val.245.312 = fmul double %ew.op.245.312.0, %ew.op.245.312.1
  %ew.dst.245.312 = getelementptr double, ptr %value.312, i32 %ew.i.245.312
  store double %ew.val.245.312, ptr %ew.dst.245.312, align 8
  %ew.next.245.312 = add i32 %ew.i.245.312, 1
  br label %ew.head.245.312
ew.done.245.312:
  br label %ew.head.246.313
ew.head.246.313:
  %ew.i.246.313 = phi i32 [ 0, %ew.done.245.312 ], [ %ew.next.246.313, %ew.body.246.313 ]
  %ew.more.246.313 = icmp slt i32 %ew.i.246.313, 1
  br i1 %ew.more.246.313, label %ew.body.246.313, label %ew.done.246.313
ew.body.246.313:
  %ew.op.246.313.0.addr = getelementptr double, ptr %value.312, i32 %ew.i.246.313
  %ew.op.246.313.0 = load double, ptr %ew.op.246.313.0.addr, align 8
  %ew.op.246.313.1 = load double, ptr %arg.7, align 8
  %ew.val.246.313 = fadd double %ew.op.246.313.0, %ew.op.246.313.1
  %ew.dst.246.313 = getelementptr double, ptr %value.313, i32 %ew.i.246.313
  store double %ew.val.246.313, ptr %ew.dst.246.313, align 8
  %ew.next.246.313 = add i32 %ew.i.246.313, 1
  br label %ew.head.246.313
ew.done.246.313:
  %scalar.247.314 = fmul double %load.75.44.0, %scalar.216.283
  store double %scalar.247.314, ptr %value.314, align 8
  %scalar.248.315 = fmul double %scalar.247.314, %scalar.112.126
  store double %scalar.248.315, ptr %value.315, align 8
  %scalar.249.316 = fsub double %scalar.218.285, 0x4072526666666666
  store double %scalar.249.316, ptr %value.316, align 8
  %convert.250.317.0 = sitofp i64 0 to double
  %scalar.250.317 = call double @llvm.maxnum.f64(double %convert.250.317.0, double %scalar.249.316)
  store double %scalar.250.317, ptr %value.317, align 8
  %scalar.251.318 = fmul double %scalar.248.315, %scalar.250.317
  store double %scalar.251.318, ptr %out.3, align 8
  %scalar.252.319 = fmul double %scalar.196.263, %scalar.196.263
  store double %scalar.252.319, ptr %value.319, align 8
  %scalar.253.320 = fmul double %scalar.252.319, %scalar.78.49
  store double %scalar.253.320, ptr %out.4, align 8
  %scalar.254.321 = fmul double %scalar.167.234, %scalar.167.234
  store double %scalar.254.321, ptr %value.321, align 8
  %scalar.255.322 = fmul double %scalar.254.321, %scalar.76.46
  store double %scalar.255.322, ptr %out.5, align 8
  %scalar.256.323 = fmul double %load.114.130.0, %scalar.214.281
  store double %scalar.256.323, ptr %out.6, align 8
  %scalar.257.324 = fsub double %scalar.200.267, %scalar.120.147
  store double %scalar.257.324, ptr %value.324, align 8
  %scalar.258.325 = fadd double %scalar.257.324, %scalar.236.303
  store double %scalar.258.325, ptr %value.325, align 8
  %convert.259.326.0 = sitofp i64 2 to double
  %scalar.259.326 = fmul double %convert.259.326.0, %scalar.236.303
  store double %scalar.259.326, ptr %value.326, align 8
  %scalar.260.327 = fdiv double %scalar.258.325, %scalar.259.326
  store double %scalar.260.327, ptr %out.7, align 8
  %load.261.328.0 = load double, ptr %arg.29, align 8
  %scalar.261.328 = call double @llvm.maxnum.f64(double %load.261.328.0, double %scalar.241.308)
  store double %scalar.261.328, ptr %out.8, align 8
  %scalar.262.329 = fsub double %scalar.200.267, %load.90.70.1
  store double %scalar.262.329, ptr %value.329, align 8
  br label %ew.head.263.330
ew.head.263.330:
  %ew.i.263.330 = phi i32 [ 0, %ew.done.246.313 ], [ %ew.next.263.330, %ew.body.263.330 ]
  %ew.more.263.330 = icmp slt i32 %ew.i.263.330, 1
  br i1 %ew.more.263.330, label %ew.body.263.330, label %ew.done.263.330
ew.body.263.330:
  %ew.op.263.330.0.addr = getelementptr double, ptr %value.181, i32 %ew.i.263.330
  %ew.op.263.330.0 = load double, ptr %ew.op.263.330.0.addr, align 8
  %ew.op.263.330.1 = load double, ptr %value.329, align 8
  %ew.val.263.330 = fmul double %ew.op.263.330.0, %ew.op.263.330.1
  %ew.dst.263.330 = getelementptr double, ptr %value.330, i32 %ew.i.263.330
  store double %ew.val.263.330, ptr %ew.dst.263.330, align 8
  %ew.next.263.330 = add i32 %ew.i.263.330, 1
  br label %ew.head.263.330
ew.done.263.330:
  br label %ew.head.264.331
ew.head.264.331:
  %ew.i.264.331 = phi i32 [ 0, %ew.done.263.330 ], [ %ew.next.264.331, %ew.body.264.331 ]
  %ew.more.264.331 = icmp slt i32 %ew.i.264.331, 1
  br i1 %ew.more.264.331, label %ew.body.264.331, label %ew.done.264.331
ew.body.264.331:
  %ew.op.264.331.0 = load double, ptr %arg.11, align 8
  %ew.op.264.331.1.addr = getelementptr double, ptr %value.330, i32 %ew.i.264.331
  %ew.op.264.331.1 = load double, ptr %ew.op.264.331.1.addr, align 8
  %ew.val.264.331 = fadd double %ew.op.264.331.0, %ew.op.264.331.1
  %ew.dst.264.331 = getelementptr double, ptr %out.9, i32 %ew.i.264.331
  store double %ew.val.264.331, ptr %ew.dst.264.331, align 8
  %ew.next.264.331 = add i32 %ew.i.264.331, 1
  br label %ew.head.264.331
ew.done.264.331:
  %scalar.265.332 = fsub double %scalar.196.263, %load.195.262.1
  store double %scalar.265.332, ptr %value.332, align 8
  br label %ew.head.266.333
ew.head.266.333:
  %ew.i.266.333 = phi i32 [ 0, %ew.done.264.331 ], [ %ew.next.266.333, %ew.body.266.333 ]
  %ew.more.266.333 = icmp slt i32 %ew.i.266.333, 1
  br i1 %ew.more.266.333, label %ew.body.266.333, label %ew.done.266.333
ew.body.266.333:
  %ew.op.266.333.0.addr = getelementptr double, ptr %value.181, i32 %ew.i.266.333
  %ew.op.266.333.0 = load double, ptr %ew.op.266.333.0.addr, align 8
  %ew.op.266.333.1 = load double, ptr %value.332, align 8
  %ew.val.266.333 = fmul double %ew.op.266.333.0, %ew.op.266.333.1
  %ew.dst.266.333 = getelementptr double, ptr %value.333, i32 %ew.i.266.333
  store double %ew.val.266.333, ptr %ew.dst.266.333, align 8
  %ew.next.266.333 = add i32 %ew.i.266.333, 1
  br label %ew.head.266.333
ew.done.266.333:
  br label %ew.head.267.334
ew.head.267.334:
  %ew.i.267.334 = phi i32 [ 0, %ew.done.266.333 ], [ %ew.next.267.334, %ew.body.267.334 ]
  %ew.more.267.334 = icmp slt i32 %ew.i.267.334, 1
  br i1 %ew.more.267.334, label %ew.body.267.334, label %ew.done.267.334
ew.body.267.334:
  %ew.op.267.334.0 = load double, ptr %arg.4, align 8
  %ew.op.267.334.1.addr = getelementptr double, ptr %value.333, i32 %ew.i.267.334
  %ew.op.267.334.1 = load double, ptr %ew.op.267.334.1.addr, align 8
  %ew.val.267.334 = fadd double %ew.op.267.334.0, %ew.op.267.334.1
  %ew.dst.267.334 = getelementptr double, ptr %out.10, i32 %ew.i.267.334
  store double %ew.val.267.334, ptr %ew.dst.267.334, align 8
  %ew.next.267.334 = add i32 %ew.i.267.334, 1
  br label %ew.head.267.334
ew.done.267.334:
  %scalar.268.335 = fsub double %scalar.240.307, %load.130.196.1
  store double %scalar.268.335, ptr %value.335, align 8
  br label %ew.head.269.336
ew.head.269.336:
  %ew.i.269.336 = phi i32 [ 0, %ew.done.267.334 ], [ %ew.next.269.336, %ew.body.269.336 ]
  %ew.more.269.336 = icmp slt i32 %ew.i.269.336, 1
  br i1 %ew.more.269.336, label %ew.body.269.336, label %ew.done.269.336
ew.body.269.336:
  %ew.op.269.336.0.addr = getelementptr double, ptr %value.181, i32 %ew.i.269.336
  %ew.op.269.336.0 = load double, ptr %ew.op.269.336.0.addr, align 8
  %ew.op.269.336.1 = load double, ptr %value.335, align 8
  %ew.val.269.336 = fmul double %ew.op.269.336.0, %ew.op.269.336.1
  %ew.dst.269.336 = getelementptr double, ptr %value.336, i32 %ew.i.269.336
  store double %ew.val.269.336, ptr %ew.dst.269.336, align 8
  %ew.next.269.336 = add i32 %ew.i.269.336, 1
  br label %ew.head.269.336
ew.done.269.336:
  br label %ew.head.270.337
ew.head.270.337:
  %ew.i.270.337 = phi i32 [ 0, %ew.done.269.336 ], [ %ew.next.270.337, %ew.body.270.337 ]
  %ew.more.270.337 = icmp slt i32 %ew.i.270.337, 1
  br i1 %ew.more.270.337, label %ew.body.270.337, label %ew.done.270.337
ew.body.270.337:
  %ew.op.270.337.0 = load double, ptr %arg.23, align 8
  %ew.op.270.337.1.addr = getelementptr double, ptr %value.336, i32 %ew.i.270.337
  %ew.op.270.337.1 = load double, ptr %ew.op.270.337.1.addr, align 8
  %ew.val.270.337 = fadd double %ew.op.270.337.0, %ew.op.270.337.1
  %ew.dst.270.337 = getelementptr double, ptr %out.11, i32 %ew.i.270.337
  store double %ew.val.270.337, ptr %ew.dst.270.337, align 8
  %ew.next.270.337 = add i32 %ew.i.270.337, 1
  br label %ew.head.270.337
ew.done.270.337:
  %scalar.271.338 = fsub double %scalar.167.234, %load.166.233.1
  store double %scalar.271.338, ptr %value.338, align 8
  br label %ew.head.272.339
ew.head.272.339:
  %ew.i.272.339 = phi i32 [ 0, %ew.done.270.337 ], [ %ew.next.272.339, %ew.body.272.339 ]
  %ew.more.272.339 = icmp slt i32 %ew.i.272.339, 1
  br i1 %ew.more.272.339, label %ew.body.272.339, label %ew.done.272.339
ew.body.272.339:
  %ew.op.272.339.0.addr = getelementptr double, ptr %value.181, i32 %ew.i.272.339
  %ew.op.272.339.0 = load double, ptr %ew.op.272.339.0.addr, align 8
  %ew.op.272.339.1 = load double, ptr %value.338, align 8
  %ew.val.272.339 = fmul double %ew.op.272.339.0, %ew.op.272.339.1
  %ew.dst.272.339 = getelementptr double, ptr %value.339, i32 %ew.i.272.339
  store double %ew.val.272.339, ptr %ew.dst.272.339, align 8
  %ew.next.272.339 = add i32 %ew.i.272.339, 1
  br label %ew.head.272.339
ew.done.272.339:
  br label %ew.head.273.340
ew.head.273.340:
  %ew.i.273.340 = phi i32 [ 0, %ew.done.272.339 ], [ %ew.next.273.340, %ew.body.273.340 ]
  %ew.more.273.340 = icmp slt i32 %ew.i.273.340, 1
  br i1 %ew.more.273.340, label %ew.body.273.340, label %ew.done.273.340
ew.body.273.340:
  %ew.op.273.340.0 = load double, ptr %arg.6, align 8
  %ew.op.273.340.1.addr = getelementptr double, ptr %value.339, i32 %ew.i.273.340
  %ew.op.273.340.1 = load double, ptr %ew.op.273.340.1.addr, align 8
  %ew.val.273.340 = fadd double %ew.op.273.340.0, %ew.op.273.340.1
  %ew.dst.273.340 = getelementptr double, ptr %out.12, i32 %ew.i.273.340
  store double %ew.val.273.340, ptr %ew.dst.273.340, align 8
  %ew.next.273.340 = add i32 %ew.i.273.340, 1
  br label %ew.head.273.340
ew.done.273.340:
  %scalar.274.341 = fsub double %scalar.214.281, %load.79.55.0
  store double %scalar.274.341, ptr %value.341, align 8
  br label %ew.head.275.342
ew.head.275.342:
  %ew.i.275.342 = phi i32 [ 0, %ew.done.273.340 ], [ %ew.next.275.342, %ew.body.275.342 ]
  %ew.more.275.342 = icmp slt i32 %ew.i.275.342, 1
  br i1 %ew.more.275.342, label %ew.body.275.342, label %ew.done.275.342
ew.body.275.342:
  %ew.op.275.342.0.addr = getelementptr double, ptr %value.181, i32 %ew.i.275.342
  %ew.op.275.342.0 = load double, ptr %ew.op.275.342.0.addr, align 8
  %ew.op.275.342.1 = load double, ptr %value.341, align 8
  %ew.val.275.342 = fmul double %ew.op.275.342.0, %ew.op.275.342.1
  %ew.dst.275.342 = getelementptr double, ptr %value.342, i32 %ew.i.275.342
  store double %ew.val.275.342, ptr %ew.dst.275.342, align 8
  %ew.next.275.342 = add i32 %ew.i.275.342, 1
  br label %ew.head.275.342
ew.done.275.342:
  br label %ew.head.276.343
ew.head.276.343:
  %ew.i.276.343 = phi i32 [ 0, %ew.done.275.342 ], [ %ew.next.276.343, %ew.body.276.343 ]
  %ew.more.276.343 = icmp slt i32 %ew.i.276.343, 1
  br i1 %ew.more.276.343, label %ew.body.276.343, label %ew.done.276.343
ew.body.276.343:
  %ew.op.276.343.0 = load double, ptr %arg.5, align 8
  %ew.op.276.343.1.addr = getelementptr double, ptr %value.342, i32 %ew.i.276.343
  %ew.op.276.343.1 = load double, ptr %ew.op.276.343.1.addr, align 8
  %ew.val.276.343 = fadd double %ew.op.276.343.0, %ew.op.276.343.1
  %ew.dst.276.343 = getelementptr double, ptr %out.13, i32 %ew.i.276.343
  store double %ew.val.276.343, ptr %ew.dst.276.343, align 8
  %ew.next.276.343 = add i32 %ew.i.276.343, 1
  br label %ew.head.276.343
ew.done.276.343:
  br label %ew.head.277.344
ew.head.277.344:
  %ew.i.277.344 = phi i32 [ 0, %ew.done.276.343 ], [ %ew.next.277.344, %ew.body.277.344 ]
  %ew.more.277.344 = icmp slt i32 %ew.i.277.344, 1
  br i1 %ew.more.277.344, label %ew.body.277.344, label %ew.done.277.344
ew.body.277.344:
  %ew.op.277.344.0.addr = getelementptr double, ptr %value.313, i32 %ew.i.277.344
  %ew.op.277.344.0 = load double, ptr %ew.op.277.344.0.addr, align 8
  %ew.op.277.344.1 = load double, ptr %arg.7, align 8
  %ew.val.277.344 = fsub double %ew.op.277.344.0, %ew.op.277.344.1
  %ew.dst.277.344 = getelementptr double, ptr %value.344, i32 %ew.i.277.344
  store double %ew.val.277.344, ptr %ew.dst.277.344, align 8
  %ew.next.277.344 = add i32 %ew.i.277.344, 1
  br label %ew.head.277.344
ew.done.277.344:
  br label %ew.head.278.345
ew.head.278.345:
  %ew.i.278.345 = phi i32 [ 0, %ew.done.277.344 ], [ %ew.next.278.345, %ew.body.278.345 ]
  %ew.more.278.345 = icmp slt i32 %ew.i.278.345, 1
  br i1 %ew.more.278.345, label %ew.body.278.345, label %ew.done.278.345
ew.body.278.345:
  %ew.op.278.345.0.addr = getelementptr double, ptr %value.181, i32 %ew.i.278.345
  %ew.op.278.345.0 = load double, ptr %ew.op.278.345.0.addr, align 8
  %ew.op.278.345.1.addr = getelementptr double, ptr %value.344, i32 %ew.i.278.345
  %ew.op.278.345.1 = load double, ptr %ew.op.278.345.1.addr, align 8
  %ew.val.278.345 = fmul double %ew.op.278.345.0, %ew.op.278.345.1
  %ew.dst.278.345 = getelementptr double, ptr %value.345, i32 %ew.i.278.345
  store double %ew.val.278.345, ptr %ew.dst.278.345, align 8
  %ew.next.278.345 = add i32 %ew.i.278.345, 1
  br label %ew.head.278.345
ew.done.278.345:
  br label %ew.head.279.346
ew.head.279.346:
  %ew.i.279.346 = phi i32 [ 0, %ew.done.278.345 ], [ %ew.next.279.346, %ew.body.279.346 ]
  %ew.more.279.346 = icmp slt i32 %ew.i.279.346, 1
  br i1 %ew.more.279.346, label %ew.body.279.346, label %ew.done.279.346
ew.body.279.346:
  %ew.op.279.346.0 = load double, ptr %arg.7, align 8
  %ew.op.279.346.1.addr = getelementptr double, ptr %value.345, i32 %ew.i.279.346
  %ew.op.279.346.1 = load double, ptr %ew.op.279.346.1.addr, align 8
  %ew.val.279.346 = fadd double %ew.op.279.346.0, %ew.op.279.346.1
  %ew.dst.279.346 = getelementptr double, ptr %out.14, i32 %ew.i.279.346
  store double %ew.val.279.346, ptr %ew.dst.279.346, align 8
  %ew.next.279.346 = add i32 %ew.i.279.346, 1
  br label %ew.head.279.346
ew.done.279.346:
  ret void
}

define internal void @__ssa_engine_toy_ballistics__ballistics_run__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2, ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12) {
entry:
  %value.182 = alloca i64, i64 1, align 8
  %value.183 = alloca i64, i64 1, align 8
  %value.184 = alloca i64, i64 1, align 8
  %value.185 = alloca i64, i64 1, align 8
  %value.186 = alloca i64, i64 1, align 8
  %value.187 = alloca i64, i64 1, align 8
  %value.188 = alloca i64, i64 1, align 8
  %value.189 = alloca i64, i64 1, align 8
  %value.190 = alloca i64, i64 1, align 8
  %value.191 = alloca i64, i64 1, align 8
  %value.192 = alloca i64, i64 1, align 8
  %value.193 = alloca i64, i64 1, align 8
  store i64 0, ptr %value.182, align 8
  store i64 1, ptr %value.183, align 8
  store i64 2, ptr %value.184, align 8
  store i64 3, ptr %value.185, align 8
  store i64 4, ptr %value.186, align 8
  store i64 5, ptr %value.187, align 8
  store i64 6, ptr %value.188, align 8
  store i64 7, ptr %value.189, align 8
  store i64 8, ptr %value.190, align 8
  store i64 9, ptr %value.191, align 8
  store i64 10, ptr %value.192, align 8
  store i64 11, ptr %value.193, align 8
  %convert.12.419.0 = trunc i64 0 to i32
  %address.12.419 = getelementptr double, ptr %arg.0, i32 %convert.12.419.0
  %load.store.13.v = load double, ptr %arg.1, align 8
  store double %load.store.13.v, ptr %address.12.419, align 8
  %load.14.420.0 = load i64, ptr %value.183, align 8
  %convert.14.420.0 = trunc i64 %load.14.420.0 to i32
  %address.14.420 = getelementptr double, ptr %arg.0, i32 %convert.14.420.0
  %load.store.15.v = load double, ptr %arg.2, align 8
  store double %load.store.15.v, ptr %address.14.420, align 8
  %load.16.421.0 = load i64, ptr %value.184, align 8
  %convert.16.421.0 = trunc i64 %load.16.421.0 to i32
  %address.16.421 = getelementptr double, ptr %arg.0, i32 %convert.16.421.0
  %load.store.17.v = load double, ptr %arg.3, align 8
  store double %load.store.17.v, ptr %address.16.421, align 8
  %load.18.422.0 = load i64, ptr %value.185, align 8
  %convert.18.422.0 = trunc i64 %load.18.422.0 to i32
  %address.18.422 = getelementptr double, ptr %arg.0, i32 %convert.18.422.0
  %load.store.19.v = load double, ptr %arg.4, align 8
  store double %load.store.19.v, ptr %address.18.422, align 8
  %load.20.423.0 = load i64, ptr %value.186, align 8
  %convert.20.423.0 = trunc i64 %load.20.423.0 to i32
  %address.20.423 = getelementptr double, ptr %arg.0, i32 %convert.20.423.0
  %load.store.21.v = load double, ptr %arg.5, align 8
  store double %load.store.21.v, ptr %address.20.423, align 8
  %load.22.424.0 = load i64, ptr %value.187, align 8
  %convert.22.424.0 = trunc i64 %load.22.424.0 to i32
  %address.22.424 = getelementptr double, ptr %arg.0, i32 %convert.22.424.0
  %load.store.23.v = load double, ptr %arg.6, align 8
  store double %load.store.23.v, ptr %address.22.424, align 8
  %load.24.425.0 = load i64, ptr %value.188, align 8
  %convert.24.425.0 = trunc i64 %load.24.425.0 to i32
  %address.24.425 = getelementptr double, ptr %arg.0, i32 %convert.24.425.0
  %load.store.25.v = load double, ptr %arg.7, align 8
  store double %load.store.25.v, ptr %address.24.425, align 8
  %load.26.426.0 = load i64, ptr %value.189, align 8
  %convert.26.426.0 = trunc i64 %load.26.426.0 to i32
  %address.26.426 = getelementptr double, ptr %arg.0, i32 %convert.26.426.0
  %load.store.27.v = load double, ptr %arg.8, align 8
  store double %load.store.27.v, ptr %address.26.426, align 8
  %load.28.427.0 = load i64, ptr %value.190, align 8
  %convert.28.427.0 = trunc i64 %load.28.427.0 to i32
  %address.28.427 = getelementptr double, ptr %arg.0, i32 %convert.28.427.0
  %load.store.29.v = load double, ptr %arg.9, align 8
  store double %load.store.29.v, ptr %address.28.427, align 8
  %load.30.428.0 = load i64, ptr %value.191, align 8
  %convert.30.428.0 = trunc i64 %load.30.428.0 to i32
  %address.30.428 = getelementptr double, ptr %arg.0, i32 %convert.30.428.0
  %load.store.31.v = load double, ptr %arg.10, align 8
  store double %load.store.31.v, ptr %address.30.428, align 8
  %load.32.429.0 = load i64, ptr %value.192, align 8
  %convert.32.429.0 = trunc i64 %load.32.429.0 to i32
  %address.32.429 = getelementptr double, ptr %arg.0, i32 %convert.32.429.0
  %load.store.33.v = load double, ptr %arg.11, align 8
  store double %load.store.33.v, ptr %address.32.429, align 8
  %load.34.430.0 = load i64, ptr %value.193, align 8
  %convert.34.430.0 = trunc i64 %load.34.430.0 to i32
  %address.34.430 = getelementptr double, ptr %arg.0, i32 %convert.34.430.0
  %load.store.35.v = load double, ptr %arg.12, align 8
  store double %load.store.35.v, ptr %address.34.430, align 8
  ret void
}

define internal void @__ssa_engine_toy_ballistics__ballistics_run(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr %arg.23, ptr %out.0, ptr %extents) {
entry:
  %value.368 = alloca i64, i64 1, align 8
  %value.369 = alloca i64, i64 1, align 8
  %value.370 = alloca double, i64 1, align 8
  %value.371 = alloca double, i64 1, align 8
  %value.372 = alloca double, i64 1, align 8
  %value.373 = alloca double, i64 1, align 8
  %value.374 = alloca double, i64 1, align 8
  %value.375 = alloca double, i64 1, align 8
  %value.376 = alloca double, i64 1, align 8
  %value.377 = alloca double, i64 1, align 8
  %value.389 = alloca i64, i64 1, align 8
  %value.391 = alloca i64, i64 1, align 8
  %value.393 = alloca i64, i64 1, align 8
  %value.395 = alloca i64, i64 1, align 8
  %value.397 = alloca i64, i64 1, align 8
  %value.399 = alloca i64, i64 1, align 8
  %value.401 = alloca i64, i64 1, align 8
  %value.403 = alloca i64, i64 1, align 8
  %value.405 = alloca i64, i64 1, align 8
  %value.407 = alloca i64, i64 1, align 8
  %value.409 = alloca i64, i64 1, align 8
  %value.411 = alloca i64, i64 1, align 8
  %value.413 = alloca i64, i64 1, align 8
  %value.415 = alloca i64, i64 1, align 8
  %value.417 = alloca i64, i64 1, align 8
  %value.337 = alloca double, i64 1, align 8
  %value.343 = alloca double, i64 1, align 8
  %value.328 = alloca double, i64 1, align 8
  %value.340 = alloca double, i64 1, align 8
  %value.331 = alloca double, i64 1, align 8
  %value.334 = alloca double, i64 1, align 8
  %value.346 = alloca double, i64 1, align 8
  %value.386 = alloca i64, i64 1, align 8
  %value.435 = alloca double, i64 1, align 8
  %value.437 = alloca double, i64 1, align 8
  %value.432 = alloca double, i64 1, align 8
  %value.436 = alloca double, i64 1, align 8
  %value.433 = alloca double, i64 1, align 8
  %value.434 = alloca double, i64 1, align 8
  %value.438 = alloca double, i64 1, align 8
  %value.387 = alloca i1, i64 1, align 8
  %value.216 = alloca double, i64 1, align 8
  %value.285 = alloca double, i64 1, align 8
  %value.310 = alloca double, i64 1, align 8
  %value.318 = alloca double, i64 1, align 8
  %value.320 = alloca double, i64 1, align 8
  %value.322 = alloca double, i64 1, align 8
  %value.323 = alloca double, i64 1, align 8
  %value.327 = alloca double, i64 1, align 8
  store i64 0, ptr %value.368, align 8
  store i64 1, ptr %value.369, align 8
  store double 0x0000000000000000, ptr %value.370, align 8
  store double 0x0000000000000000, ptr %value.371, align 8
  store double 0x0000000000000000, ptr %value.372, align 8
  store double 0x0000000000000000, ptr %value.373, align 8
  store double 0x0000000000000000, ptr %value.374, align 8
  store double 0x0000000000000000, ptr %value.375, align 8
  store double 0x0000000000000000, ptr %value.376, align 8
  store double 0x0000000000000000, ptr %value.377, align 8
  store i64 0, ptr %value.389, align 8
  store i64 1, ptr %value.391, align 8
  store i64 2, ptr %value.393, align 8
  store i64 3, ptr %value.395, align 8
  store i64 4, ptr %value.397, align 8
  store i64 5, ptr %value.399, align 8
  store i64 6, ptr %value.401, align 8
  store i64 7, ptr %value.403, align 8
  store i64 8, ptr %value.405, align 8
  store i64 9, ptr %value.407, align 8
  store i64 10, ptr %value.409, align 8
  store i64 11, ptr %value.411, align 8
  store i64 12, ptr %value.413, align 8
  store i64 13, ptr %value.415, align 8
  store i64 14, ptr %value.417, align 8
  store double 0x0000000000000000, ptr %value.337, align 8
  store double 0x0000000000000000, ptr %value.343, align 8
  store double 0x0000000000000000, ptr %value.328, align 8
  store double 0x0000000000000000, ptr %value.340, align 8
  store double 0x0000000000000000, ptr %value.331, align 8
  store double 0x0000000000000000, ptr %value.334, align 8
  store double 0x0000000000000000, ptr %value.346, align 8
  br label %loop_header
loop_header:
  %phi.385 = phi ptr [ %value.368, %entry ], [ %value.386, %loop_latch ]
  %phi.378 = phi ptr [ %value.370, %entry ], [ %value.435, %loop_latch ]
  %phi.379 = phi ptr [ %value.372, %entry ], [ %value.437, %loop_latch ]
  %phi.380 = phi ptr [ %value.373, %entry ], [ %value.432, %loop_latch ]
  %phi.381 = phi ptr [ %value.374, %entry ], [ %value.436, %loop_latch ]
  %phi.382 = phi ptr [ %value.375, %entry ], [ %value.433, %loop_latch ]
  %phi.383 = phi ptr [ %value.376, %entry ], [ %value.434, %loop_latch ]
  %phi.384 = phi ptr [ %value.377, %entry ], [ %value.438, %loop_latch ]
  %load.41.387.0 = load i32, ptr %phi.385, align 4
  %load.41.387.1 = load i32, ptr %arg.0, align 4
  %scalar.41.387 = icmp slt i32 %load.41.387.0, %load.41.387.1
  store i1 %scalar.41.387, ptr %value.387, align 1
  br i1 %scalar.41.387, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_engine_toy_ballistics__ballistics_run__planned_region_0(ptr %arg.4, ptr %arg.5, ptr %arg.9, ptr %arg.20, ptr %phi.383, ptr %phi.379, ptr %phi.381, ptr %phi.384, ptr %arg.3, ptr %arg.8, ptr %arg.2, ptr %phi.382, ptr %arg.1, ptr %arg.18, ptr %arg.17, ptr %arg.14, ptr %arg.22, ptr %arg.21, ptr %arg.6, ptr %arg.15, ptr %arg.12, ptr %arg.11, ptr %value.371, ptr %phi.378, ptr %arg.16, ptr %arg.19, ptr %arg.7, ptr %arg.10, ptr %arg.13, ptr %phi.380, ptr %value.216, ptr %value.285, ptr %value.310, ptr %value.318, ptr %value.320, ptr %value.322, ptr %value.323, ptr %value.327, ptr %value.432, ptr %value.433, ptr %value.434, ptr %value.435, ptr %value.436, ptr %value.437, ptr %value.438, ptr %extents)
  br label %loop_latch
loop_latch:
  %load.75.386.0 = load i32, ptr %phi.385, align 4
  %load.75.386.1 = load i64, ptr %value.369, align 8
  %convert.75.386.1 = trunc i64 %load.75.386.1 to i32
  %scalar.75.386 = add i32 %load.75.386.0, %convert.75.386.1
  %declared.75.386 = sext i32 %scalar.75.386 to i64
  store i64 %declared.75.386, ptr %value.386, align 8
  br label %loop_header
loop_exit:
  %phi.360 = phi ptr [ %phi.378, %loop_header ]
  %phi.362 = phi ptr [ %phi.379, %loop_header ]
  %phi.363 = phi ptr [ %phi.380, %loop_header ]
  %phi.364 = phi ptr [ %phi.381, %loop_header ]
  %phi.365 = phi ptr [ %phi.382, %loop_header ]
  %phi.366 = phi ptr [ %phi.383, %loop_header ]
  %phi.367 = phi ptr [ %phi.384, %loop_header ]
  call void @__ssa_engine_toy_ballistics__ballistics_run__planned_region_1(ptr %arg.23, ptr %phi.365, ptr %phi.366, ptr %phi.360, ptr %phi.364, ptr %phi.362, ptr %phi.367, ptr %phi.363, ptr %value.285, ptr %value.216, ptr %value.320, ptr %value.322, ptr %value.318)
  %return.load.0.68 = load double, ptr %arg.23, align 8
  store double %return.load.0.68, ptr %out.0, align 8
  ret void
}

define void @engine_toy_ballistics__ballistics_run(ptr %buffers, ptr %extents) {
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
  %public.addr.14 = getelementptr ptr, ptr %buffers, i64 14
  %public.14 = load ptr, ptr %public.addr.14, align 8
  %public.addr.15 = getelementptr ptr, ptr %buffers, i64 15
  %public.15 = load ptr, ptr %public.addr.15, align 8
  %public.addr.16 = getelementptr ptr, ptr %buffers, i64 16
  %public.16 = load ptr, ptr %public.addr.16, align 8
  %public.addr.17 = getelementptr ptr, ptr %buffers, i64 17
  %public.17 = load ptr, ptr %public.addr.17, align 8
  %public.addr.18 = getelementptr ptr, ptr %buffers, i64 18
  %public.18 = load ptr, ptr %public.addr.18, align 8
  %public.addr.19 = getelementptr ptr, ptr %buffers, i64 19
  %public.19 = load ptr, ptr %public.addr.19, align 8
  %public.addr.20 = getelementptr ptr, ptr %buffers, i64 20
  %public.20 = load ptr, ptr %public.addr.20, align 8
  %public.addr.21 = getelementptr ptr, ptr %buffers, i64 21
  %public.21 = load ptr, ptr %public.addr.21, align 8
  %public.addr.22 = getelementptr ptr, ptr %buffers, i64 22
  %public.22 = load ptr, ptr %public.addr.22, align 8
  %public.addr.23 = getelementptr ptr, ptr %buffers, i64 23
  %public.23 = load ptr, ptr %public.addr.23, align 8
  call void @__ssa_engine_toy_ballistics__ballistics_run(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.23, ptr %extents)
  ret void
}
