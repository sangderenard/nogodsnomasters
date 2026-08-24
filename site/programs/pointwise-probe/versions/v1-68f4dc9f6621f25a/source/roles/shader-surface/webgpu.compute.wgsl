@group(0) @binding(0) var<storage, read> feed_4: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_1: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_2: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_3: array<f32>;


@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {
  let linear_index: u32 = gid.x + gid.y * grid.x * 32u + gid.z * grid.x * grid.y * 32u;
  if (linear_index >= 1u) { return; }
  let v_4: f32 = feed_4[linear_index];
  let v_8: f32 = cos(v_4);
  let v_9: f32 = (0.5f * v_8);
  let v_10: f32 = (0.5f + v_9);
  let v_14: f32 = (v_4 - 2.094395102393195f);
  let v_16: f32 = cos(v_14);
  let v_17: f32 = (0.5f * v_16);
  let v_18: f32 = (0.5f + v_17);
  let v_22: f32 = (v_4 - 4.18879020478639f);
  let v_24: f32 = cos(v_22);
  let v_25: f32 = (0.5f * v_24);
  let v_26: f32 = (0.5f + v_25);
  output_0[linear_index] = v_4;
  output_1[linear_index] = v_10;
  output_2[linear_index] = v_18;
  output_3[linear_index] = v_26;
}
