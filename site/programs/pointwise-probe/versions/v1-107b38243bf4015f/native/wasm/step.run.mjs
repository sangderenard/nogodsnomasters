
import { readFileSync } from "node:fs";
const [wasmPath, planPath] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(planPath, "utf-8"));
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const memory = instance.exports.memory;
if (plan.required_bytes > memory.buffer.byteLength) {
  memory.grow(Math.ceil((plan.required_bytes - memory.buffer.byteLength) / 65536));
}
const isI64 = plan.value_type === "i64";
const View =
  plan.value_type === "f32" ? Float32Array :
  plan.value_type === "i32" ? Int32Array :
  plan.value_type === "i64" ? BigInt64Array :
  Float64Array;
for (const feed of plan.feeds) {
  // A 64-bit integer view stores BigInt elements; the plan carries them as
  // plain JSON numbers (small, exact) and they are lifted to BigInt here.
  const data = isI64 ? feed.data.map(BigInt) : feed.data;
  new View(memory.buffer, feed.offset, feed.data.length).set(data);
}
instance.exports.run(plan.count, ...plan.run_offsets);
const outputs = plan.outputs.map(o => {
  const window = Array.from(new View(memory.buffer, o.offset, o.length));
  // BigInt has no JSON encoding, so a 64-bit result is emitted as decimal
  // strings and re-parsed to integers on the Python side.
  return isI64 ? window.map(String) : window;
});
console.log(JSON.stringify(outputs));
