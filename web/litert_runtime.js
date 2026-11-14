/**
 * Neri Web — Self-Contained LiteRT (.tflite FlatBuffer v3) Inference Runtime
 * Supports MegaDetector YOLO11s (md_v1000.0.0-sorrel.tflite) & MegaDetectorV6 (MDV6-yolov10-c.tflite)
 * Runs 100% locally in Browser (file:// or http://) and Node.js with zero external CDN dependencies.
 */
(function (root) {
  'use strict';

  // TFLite TensorType enum
  const TensorType = {
    FLOAT32: 0,
    FLOAT16: 1,
    INT32: 2,
    UINT8: 3,
    INT64: 4,
    STRING: 5,
    BOOL: 6,
    INT16: 7,
    COMPLEX64: 8,
    INT8: 9,
  };

  // TFLite BuiltinOperator enum
  const BuiltinOperator = {
    ADD: 0,
    CONCATENATION: 2,
    CONV_2D: 3,
    DEPTHWISE_CONV_2D: 4,
    DEQUANTIZE: 6,
    LOGISTIC: 14,
    MAX_POOL_2D: 17,
    MUL: 18,
    RESHAPE: 22,
    SOFTMAX: 25,
    PAD: 34,
    GATHER: 36,
    TRANSPOSE: 39,
    SUB: 41,
    STRIDED_SLICE: 45,
    TOPK_V2: 48,
    CAST: 53,
    LESS: 58,
    TILE: 69,
    REDUCE_MAX: 82,
    FLOOR_MOD: 95,
    RESIZE_NEAREST_NEIGHBOR: 97,
    GATHER_ND: 107,
    BATCH_MATMUL: 126,
  };

  // Precompute IEEE 754 float16 -> float32 lookup table (65536 entries)
  const FP16_TABLE = new Float32Array(65536);
  (function initFp16Table() {
    const buf = new ArrayBuffer(4);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    for (let i = 0; i < 65536; i++) {
      const s = (i & 0x8000) >> 15;
      const e = (i & 0x7c00) >> 10;
      const f = i & 0x03ff;
      if (e === 0) {
        if (f === 0) {
          u32[0] = s << 31;
        } else {
          let mant = f;
          let exp = -14;
          while ((mant & 0x0400) === 0) {
            mant <<= 1;
            exp--;
          }
          mant &= 0x03ff;
          u32[0] = (s << 31) | ((exp + 127) << 23) | (mant << 13);
        }
      } else if (e === 31) {
        u32[0] = (s << 31) | 0x7f800000 | (f << 13);
      } else {
        u32[0] = (s << 31) | ((e - 15 + 127) << 23) | (f << 13);
      }
      FP16_TABLE[i] = f32[0];
    }
  })();

  // Minimal FlatBuffer Reader for TFLite v3 schema
  class FlatBufferReader {
    constructor(arrayBuffer) {
      this.bytes = new Uint8Array(arrayBuffer);
      this.view = new DataView(
        this.bytes.buffer,
        this.bytes.byteOffset,
        this.bytes.byteLength
      );
    }
    readInt32(pos) {
      return this.view.getInt32(pos, true);
    }
    readUint32(pos) {
      return this.view.getUint32(pos, true);
    }
    readInt16(pos) {
      return this.view.getInt16(pos, true);
    }
    readUint16(pos) {
      return this.view.getUint16(pos, true);
    }
    readInt8(pos) {
      return this.view.getInt8(pos);
    }
    readUint8(pos) {
      return this.view.getUint8(pos);
    }
    readFloat32(pos) {
      return this.view.getFloat32(pos, true);
    }
    readBool(pos) {
      return this.view.getUint8(pos) !== 0;
    }
    // Return field offset in table or 0 if not present
    fieldOffset(tablePos, vtableOffset) {
      const vtablePos = tablePos - this.readInt32(tablePos);
      const vtableLen = this.readUint16(vtablePos);
      if (vtableOffset >= vtableLen) return 0;
      const off = this.readUint16(vtablePos + vtableOffset);
      return off === 0 ? 0 : tablePos + off;
    }
    indirect(pos) {
      return pos + this.readUint32(pos);
    }
    readString(pos) {
      const strPos = this.indirect(pos);
      const len = this.readUint32(strPos);
      const sub = this.bytes.subarray(strPos + 4, strPos + 4 + len);
      if (typeof TextDecoder !== 'undefined') {
        return new TextDecoder('utf-8').decode(sub);
      }
      let s = '';
      for (let i = 0; i < sub.length; i++) s += String.fromCharCode(sub[i]);
      return s;
    }
    vectorLen(tablePos, vtableOffset) {
      const fo = this.fieldOffset(tablePos, vtableOffset);
      if (!fo) return 0;
      const vecPos = this.indirect(fo);
      return this.readUint32(vecPos);
    }
    vectorDataPos(tablePos, vtableOffset) {
      const fo = this.fieldOffset(tablePos, vtableOffset);
      if (!fo) return 0;
      const vecPos = this.indirect(fo);
      return vecPos + 4;
    }
    readInt32Array(tablePos, vtableOffset) {
      const len = this.vectorLen(tablePos, vtableOffset);
      const start = this.vectorDataPos(tablePos, vtableOffset);
      const out = new Int32Array(len);
      for (let i = 0; i < len; i++) {
        out[i] = this.readInt32(start + i * 4);
      }
      return Array.from(out);
    }
  }

  function parseTfliteFlatBuffer(arrayBuffer) {
    const fb = new FlatBufferReader(arrayBuffer);
    const rootPos = fb.readUint32(0);

    // OperatorCodes (vtable 6)
    const numOpCodes = fb.vectorLen(rootPos, 6);
    const opCodesStart = fb.vectorDataPos(rootPos, 6);
    const operatorCodes = [];
    for (let i = 0; i < numOpCodes; i++) {
      const ocPos = fb.indirect(opCodesStart + i * 4);
      const depFo = fb.fieldOffset(ocPos, 4);
      const builtinFo = fb.fieldOffset(ocPos, 10);
      const depCode = depFo ? fb.readInt8(depFo) : 0;
      const builtinCode = builtinFo ? fb.readInt32(builtinFo) : depCode;
      operatorCodes.push(builtinCode || depCode);
    }

    // Buffers (vtable 12)
    const numBuffers = fb.vectorLen(rootPos, 12);
    const buffersStart = fb.vectorDataPos(rootPos, 12);
    const buffers = [];
    for (let i = 0; i < numBuffers; i++) {
      const bPos = fb.indirect(buffersStart + i * 4);
      const len = fb.vectorLen(bPos, 4);
      if (len > 0) {
        const dataPos = fb.vectorDataPos(bPos, 4);
        buffers.push(fb.bytes.subarray(dataPos, dataPos + len));
      } else {
        buffers.push(null);
      }
    }

    // Subgraphs (vtable 8) -> Subgraph 0
    const subgraphsStart = fb.vectorDataPos(rootPos, 8);
    const sgPos = fb.indirect(subgraphsStart);

    // Tensors (vtable 4)
    const numTensors = fb.vectorLen(sgPos, 4);
    const tensorsStart = fb.vectorDataPos(sgPos, 4);
    const tensors = [];
    for (let i = 0; i < numTensors; i++) {
      const tPos = fb.indirect(tensorsStart + i * 4);
      const shape = fb.readInt32Array(tPos, 4);
      const typeFo = fb.fieldOffset(tPos, 6);
      const ttype = typeFo ? fb.readInt8(typeFo) : 0;
      const bufFo = fb.fieldOffset(tPos, 8);
      const bufIdx = bufFo ? fb.readUint32(bufFo) : 0;
      const nameFo = fb.fieldOffset(tPos, 10);
      const name = nameFo ? fb.readString(nameFo) : `tensor_${i}`;
      tensors.push({ index: i, name, shape, type: ttype, bufferIndex: bufIdx });
    }

    const inputs = fb.readInt32Array(sgPos, 6);
    const outputs = fb.readInt32Array(sgPos, 8);

    // Operators (vtable 10)
    const numOps = fb.vectorLen(sgPos, 10);
    const opsStart = fb.vectorDataPos(sgPos, 10);
    const operators = [];
    for (let i = 0; i < numOps; i++) {
      const opPos = fb.indirect(opsStart + i * 4);
      const opcFo = fb.fieldOffset(opPos, 4);
      const opcodeIndex = opcFo ? fb.readUint32(opcFo) : 0;
      const builtinCode = operatorCodes[opcodeIndex];
      const opInputs = fb.readInt32Array(opPos, 6);
      const opOutputs = fb.readInt32Array(opPos, 8);
      const optPosFo = fb.fieldOffset(opPos, 12);
      const optPos = optPosFo ? fb.indirect(optPosFo) : 0;
      const options = parseOperatorOptions(fb, builtinCode, optPos);
      operators.push({ index: i, opcodeIndex, builtinCode, inputs: opInputs, outputs: opOutputs, options });
    }

    return { operatorCodes, buffers, tensors, inputs, outputs, operators };
  }

  function parseOperatorOptions(fb, builtinCode, optPos) {
    if (!optPos) return {};
    const readI32 = (vo, def = 0) => {
      const fo = fb.fieldOffset(optPos, vo);
      return fo ? fb.readInt32(fo) : def;
    };
    const readI8 = (vo, def = 0) => {
      const fo = fb.fieldOffset(optPos, vo);
      return fo ? fb.readInt8(fo) : def;
    };
    const readF32 = (vo, def = 0) => {
      const fo = fb.fieldOffset(optPos, vo);
      return fo ? fb.readFloat32(fo) : def;
    };
    const readBool = (vo, def = false) => {
      const fo = fb.fieldOffset(optPos, vo);
      return fo ? fb.readBool(fo) : def;
    };

    switch (builtinCode) {
      case BuiltinOperator.CONV_2D:
        return {
          padding: readI8(4, 0),
          strideW: readI32(6, 1),
          strideH: readI32(8, 1),
          fusedActivationFunction: readI8(10, 0),
          dilationWFactor: readI32(12, 1),
          dilationHFactor: readI32(14, 1),
        };
      case BuiltinOperator.DEPTHWISE_CONV_2D:
        return {
          padding: readI8(4, 0),
          strideW: readI32(6, 1),
          strideH: readI32(8, 1),
          depthMultiplier: readI32(10, 1),
          fusedActivationFunction: readI8(12, 0),
          dilationWFactor: readI32(14, 1),
          dilationHFactor: readI32(16, 1),
        };
      case BuiltinOperator.MAX_POOL_2D:
        return {
          padding: readI8(4, 0),
          strideW: readI32(6, 1),
          strideH: readI32(8, 1),
          filterWidth: readI32(10, 1),
          filterHeight: readI32(12, 1),
          fusedActivationFunction: readI8(14, 0),
        };
      case BuiltinOperator.CONCATENATION:
        return {
          axis: readI32(4, 0),
          fusedActivationFunction: readI8(6, 0),
        };
      case BuiltinOperator.STRIDED_SLICE:
        return {
          beginMask: readI32(4, 0),
          endMask: readI32(6, 0),
          ellipsisMask: readI32(8, 0),
          newAxisMask: readI32(10, 0),
          shrinkAxisMask: readI32(12, 0),
          offset: readBool(14, false),
        };
      case BuiltinOperator.SOFTMAX:
        return {
          beta: readF32(4, 1.0),
        };
      case BuiltinOperator.BATCH_MATMUL:
        return {
          adjX: readBool(4, false),
          adjY: readBool(6, false),
        };
      case BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
        return {
          alignCorners: readBool(4, false),
          halfPixelCenters: readBool(6, false),
        };
      case BuiltinOperator.GATHER:
        return {
          axis: readI32(4, 0),
          batchDims: readI32(6, 0),
        };
      case BuiltinOperator.REDUCE_MAX:
        return {
          keepDims: readBool(4, false),
        };
      default:
        return {};
    }
  }

  function numElements(shape) {
    if (!shape || shape.length === 0) return 1;
    let p = 1;
    for (let i = 0; i < shape.length; i++) p *= shape[i];
    return p;
  }

  function decodeBufferToTypedArray(u8, ttype, count) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    if (ttype === TensorType.FLOAT32) {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
      return out;
    }
    if (ttype === TensorType.FLOAT16) {
      const out = new Float32Array(count);
      for (let i = 0; i < count; i++) out[i] = FP16_TABLE[dv.getUint16(i * 2, true)];
      return out;
    }
    if (ttype === TensorType.INT32) {
      const out = new Int32Array(count);
      for (let i = 0; i < count; i++) out[i] = dv.getInt32(i * 4, true);
      return out;
    }
    if (ttype === TensorType.INT64) {
      const out = new Int32Array(count);
      for (let i = 0; i < count; i++) out[i] = dv.getInt32(i * 8, true);
      return out;
    }
    if (ttype === TensorType.UINT8 || ttype === TensorType.BOOL) {
      return new Uint8Array(u8.buffer, u8.byteOffset, count);
    }
    const out = new Float32Array(count);
    return out;
  }

  class LiteRTInterpreter {
    constructor(arrayBuffer) {
      const parsed = parseTfliteFlatBuffer(arrayBuffer);
      this.tensors = parsed.tensors;
      this.inputs = parsed.inputs;
      this.outputs = parsed.outputs;
      this.operators = parsed.operators;

      // Allocate constant buffers and pre-run constant DEQUANTIZE ops
      this.values = new Array(this.tensors.length).fill(null);
      this.isConstant = new Uint8Array(this.tensors.length);

      for (let i = 0; i < this.tensors.length; i++) {
        const t = this.tensors[i];
        const raw = parsed.buffers[t.bufferIndex];
        if (raw && raw.byteLength > 0) {
          const count = numElements(t.shape);
          this.values[i] = decodeBufferToTypedArray(raw, t.type, count);
          this.isConstant[i] = 1;
        }
      }

      // Filter out constant DEQUANTIZE ops after materializing them once
      const runtimeOps = [];
      for (let i = 0; i < this.operators.length; i++) {
        const op = this.operators[i];
        if (op.builtinCode === BuiltinOperator.DEQUANTIZE && this.isConstant[op.inputs[0]]) {
          this.values[op.outputs[0]] = this.values[op.inputs[0]];
          this.isConstant[op.outputs[0]] = 1;
          this.values[op.inputs[0]] = null;
          continue;
        }
        runtimeOps.push(op);
      }
      this.runtimeOps = runtimeOps;

      // Pre-pack CONV_2D weights from [C_out, kH, kW, C_in] -> [kH * kW * C_in, C_out] for contiguous L1-cache vectorization
      for (let i = 0; i < this.runtimeOps.length; i++) {
        const op = this.runtimeOps[i];
        if (op.builtinCode === BuiltinOperator.CONV_2D) {
          const wIdx = op.inputs[1];
          const wTensor = this.tensors[wIdx];
          const wData = this.values[wIdx];
          const xShape = this.tensors[op.inputs[0]].shape;
          const [cOut, kH, kW, cInGroup] = wTensor.shape;
          if (wData && cInGroup === xShape[3] && !wTensor._packedKC) {
            const K = kH * kW * cInGroup;
            const packed = new Float32Array(K * cOut);
            for (let co = 0; co < cOut; co++) {
              const coBase = co * K;
              for (let k = 0; k < K; k++) {
                packed[k * cOut + co] = wData[coBase + k];
              }
            }
            wTensor._packedKC = packed;
          }
        }
      }

      // Compute last consumer op index for each intermediate tensor so memory can be freed eagerly
      this.lastUsedAt = new Int32Array(this.tensors.length).fill(-1);
      for (let step = 0; step < this.runtimeOps.length; step++) {
        const op = this.runtimeOps[step];
        for (let j = 0; j < op.inputs.length; j++) {
          const tid = op.inputs[j];
          if (!this.isConstant[tid]) this.lastUsedAt[tid] = step;
        }
      }
      for (let j = 0; j < this.outputs.length; j++) {
        this.lastUsedAt[this.outputs[j]] = this.runtimeOps.length + 1;
      }
    }

    getInputDetails() {
      return this.inputs.map((idx) => this.tensors[idx]);
    }

    getOutputDetails() {
      return this.outputs.map((idx) => this.tensors[idx]);
    }

    predict(inputFloat32Array) {
      const inIdx = this.inputs[0];
      this.values[inIdx] = inputFloat32Array;

      for (let step = 0; step < this.runtimeOps.length; step++) {
        const op = this.runtimeOps[step];
        this.executeOp(op);

        // Release intermediate tensors no longer needed
        for (let j = 0; j < op.inputs.length; j++) {
          const tid = op.inputs[j];
          if (!this.isConstant[tid] && this.lastUsedAt[tid] === step) {
            this.values[tid] = null;
          }
        }
      }

      const outIdx = this.outputs[0];
      const outTensor = this.tensors[outIdx];
      const outData = this.values[outIdx];
      return {
        shape: outTensor.shape,
        data: () => Promise.resolve(outData),
        dataSync: () => outData,
      };
    }

    async predictAsync(inputFloat32Array) {
      const inIdx = this.inputs[0];
      this.values[inIdx] = inputFloat32Array;

      for (let step = 0; step < this.runtimeOps.length; step++) {
        const op = this.runtimeOps[step];
        this.executeOp(op);

        for (let j = 0; j < op.inputs.length; j++) {
          const tid = op.inputs[j];
          if (!this.isConstant[tid] && this.lastUsedAt[tid] === step) {
            this.values[tid] = null;
          }
        }

        if ((step & 15) === 15 && typeof window !== 'undefined') {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }

      const outIdx = this.outputs[0];
      const outTensor = this.tensors[outIdx];
      const outData = this.values[outIdx];
      return {
        shape: outTensor.shape,
        data: () => Promise.resolve(outData),
        dataSync: () => outData,
      };
    }

    executeOp(op) {
      const ins = op.inputs;
      const outs = op.outputs;
      const opts = op.options;
      const out0 = outs[0];
      const outShape = this.tensors[out0].shape;

      switch (op.builtinCode) {
        case BuiltinOperator.PAD: {
          const x = this.values[ins[0]];
          const xSh = this.tensors[ins[0]].shape;
          const pads = this.values[ins[1]]; // [4, 2] -> [pN0,pN1, pH0,pH1, pW0,pW1, pC0,pC1]
          const [N, H, W, C] = xSh;
          const [, , pH0, , pW0] = pads;
          const [, Hout, Wout] = outShape;
          const out = new Float32Array(N * Hout * Wout * C);
          const rowLen = W * C;
          for (let h = 0; h < H; h++) {
            const srcOff = h * rowLen;
            const dstOff = ((h + pH0) * Wout + pW0) * C;
            out.set(x.subarray(srcOff, srcOff + rowLen), dstOff);
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.CONV_2D: {
          const x = this.values[ins[0]];
          const xSh = this.tensors[ins[0]].shape;
          const wTensor = this.tensors[ins[1]];
          const bias = this.values[ins[2]];
          const [, H, W, Cin] = xSh;
          const [Cout, kH, kW, CinGroup] = wTensor.shape;
          const [, Hout, Wout] = outShape;
          const sH = opts.strideH || 1;
          const sW = opts.strideW || 1;
          let padTop = 0, padLeft = 0;
          if (opts.padding === 0) {
            const totalPadH = Math.max(0, (Hout - 1) * sH + kH - H);
            const totalPadW = Math.max(0, (Wout - 1) * sW + kW - W);
            padTop = totalPadH >> 1;
            padLeft = totalPadW >> 1;
          }

          // Grouped depthwise CONV_2D (CinGroup === 1 && Cin === Cout)
          if (CinGroup === 1 && Cin === Cout) {
            const wRaw = this.values[ins[1]]; // [C, kH, kW, 1]
            const out = new Float32Array(Hout * Wout * Cout);
            const kHW = kH * kW;
            for (let oh = 0; oh < Hout; oh++) {
              const ih0 = oh * sH - padTop;
              const khStart = ih0 < 0 ? -ih0 : 0;
              const khEnd = Math.min(kH, H - ih0);
              for (let ow = 0; ow < Wout; ow++) {
                const iw0 = ow * sW - padLeft;
                const kwStart = iw0 < 0 ? -iw0 : 0;
                const kwEnd = Math.min(kW, W - iw0);
                const outBase = (oh * Wout + ow) * Cout;
                out.set(bias, outBase);
                for (let kh = khStart; kh < khEnd; kh++) {
                  const inRowBase = (ih0 + kh) * W * Cin;
                  const khOff = kh * kW;
                  for (let kw = kwStart; kw < kwEnd; kw++) {
                    const inBase = inRowBase + (iw0 + kw) * Cin;
                    const kIdx = khOff + kw;
                    for (let c = 0; c < Cout; c++) {
                      out[outBase + c] += x[inBase + c] * wRaw[c * kHW + kIdx];
                    }
                  }
                }
              }
            }
            this.values[out0] = out;
            break;
          }

          const wKC = wTensor._packedKC;
          const M = Hout * Wout;
          const K = kH * kW * Cin;
          const out = new Float32Array(M * Cout);

          let A = x;
          if (kH === 1 && kW === 1 && sH === 1 && sW === 1 && padTop === 0 && padLeft === 0) {
            A = x;
          } else {
            A = new Float32Array(M * K);
            const kwCin = kW * Cin;
            let mIdx = 0;
            for (let oh = 0; oh < Hout; oh++) {
              const ih0 = oh * sH - padTop;
              const khStart = ih0 < 0 ? -ih0 : 0;
              const khEnd = Math.min(kH, H - ih0);
              for (let ow = 0; ow < Wout; ow++) {
                const iw0 = ow * sW - padLeft;
                const kwStart = iw0 < 0 ? -iw0 : 0;
                const kwEnd = Math.min(kW, W - iw0);
                const dstRow = mIdx * K;
                for (let kh = khStart; kh < khEnd; kh++) {
                  const inRowBase = (ih0 + kh) * W * Cin;
                  const dstKh = dstRow + kh * kwCin;
                  if (kwStart === 0 && kwEnd === kW) {
                    const srcStart = inRowBase + iw0 * Cin;
                    A.set(x.subarray(srcStart, srcStart + kwCin), dstKh);
                  } else {
                    for (let kw = kwStart; kw < kwEnd; kw++) {
                      const srcStart = inRowBase + (iw0 + kw) * Cin;
                      A.set(x.subarray(srcStart, srcStart + Cin), dstKh + kw * Cin);
                    }
                  }
                }
                mIdx++;
              }
            }
          }

          // Register-blocked 4x8 GEMM: out[M, Cout] = A[M, K] * wKC[K, Cout] + bias[Cout]
          const M4 = M - (M % 4);
          const N8 = Cout - (Cout % 8);
          for (let m = 0; m < M4; m += 4) {
            const aRow0 = m * K;
            const aRow1 = (m + 1) * K;
            const aRow2 = (m + 2) * K;
            const aRow3 = (m + 3) * K;
            const oRow0 = m * Cout;
            const oRow1 = (m + 1) * Cout;
            const oRow2 = (m + 2) * Cout;
            const oRow3 = (m + 3) * Cout;

            for (let n = 0; n < N8; n += 8) {
              const b0 = bias[n], b1 = bias[n + 1], b2 = bias[n + 2], b3 = bias[n + 3];
              const b4 = bias[n + 4], b5 = bias[n + 5], b6 = bias[n + 6], b7 = bias[n + 7];
              let c00 = b0, c01 = b1, c02 = b2, c03 = b3, c04 = b4, c05 = b5, c06 = b6, c07 = b7;
              let c10 = b0, c11 = b1, c12 = b2, c13 = b3, c14 = b4, c15 = b5, c16 = b6, c17 = b7;
              let c20 = b0, c21 = b1, c22 = b2, c23 = b3, c24 = b4, c25 = b5, c26 = b6, c27 = b7;
              let c30 = b0, c31 = b1, c32 = b2, c33 = b3, c34 = b4, c35 = b5, c36 = b6, c37 = b7;

              for (let k = 0; k < K; k++) {
                const a0 = A[aRow0 + k];
                const a1 = A[aRow1 + k];
                const a2 = A[aRow2 + k];
                const a3 = A[aRow3 + k];
                const wBase = k * Cout + n;
                const w0 = wKC[wBase], w1 = wKC[wBase + 1], w2 = wKC[wBase + 2], w3 = wKC[wBase + 3];
                const w4 = wKC[wBase + 4], w5 = wKC[wBase + 5], w6 = wKC[wBase + 6], w7 = wKC[wBase + 7];

                c00 += a0 * w0; c01 += a0 * w1; c02 += a0 * w2; c03 += a0 * w3;
                c04 += a0 * w4; c05 += a0 * w5; c06 += a0 * w6; c07 += a0 * w7;

                c10 += a1 * w0; c11 += a1 * w1; c12 += a1 * w2; c13 += a1 * w3;
                c14 += a1 * w4; c15 += a1 * w5; c16 += a1 * w6; c17 += a1 * w7;

                c20 += a2 * w0; c21 += a2 * w1; c22 += a2 * w2; c23 += a2 * w3;
                c24 += a2 * w4; c25 += a2 * w5; c26 += a2 * w6; c27 += a2 * w7;

                c30 += a3 * w0; c31 += a3 * w1; c32 += a3 * w2; c33 += a3 * w3;
                c34 += a3 * w4; c35 += a3 * w5; c36 += a3 * w6; c37 += a3 * w7;
              }

              out[oRow0 + n] = c00; out[oRow0 + n + 1] = c01; out[oRow0 + n + 2] = c02; out[oRow0 + n + 3] = c03;
              out[oRow0 + n + 4] = c04; out[oRow0 + n + 5] = c05; out[oRow0 + n + 6] = c06; out[oRow0 + n + 7] = c07;

              out[oRow1 + n] = c10; out[oRow1 + n + 1] = c11; out[oRow1 + n + 2] = c12; out[oRow1 + n + 3] = c13;
              out[oRow1 + n + 4] = c14; out[oRow1 + n + 5] = c15; out[oRow1 + n + 6] = c16; out[oRow1 + n + 7] = c17;

              out[oRow2 + n] = c20; out[oRow2 + n + 1] = c21; out[oRow2 + n + 2] = c22; out[oRow2 + n + 3] = c23;
              out[oRow2 + n + 4] = c24; out[oRow2 + n + 5] = c25; out[oRow2 + n + 6] = c26; out[oRow2 + n + 7] = c27;

              out[oRow3 + n] = c30; out[oRow3 + n + 1] = c31; out[oRow3 + n + 2] = c32; out[oRow3 + n + 3] = c33;
              out[oRow3 + n + 4] = c34; out[oRow3 + n + 5] = c35; out[oRow3 + n + 6] = c36; out[oRow3 + n + 7] = c37;
            }

            for (let n = N8; n < Cout; n++) {
              let c0 = bias[n], c1 = bias[n], c2 = bias[n], c3 = bias[n];
              for (let k = 0; k < K; k++) {
                const wv = wKC[k * Cout + n];
                c0 += A[aRow0 + k] * wv;
                c1 += A[aRow1 + k] * wv;
                c2 += A[aRow2 + k] * wv;
                c3 += A[aRow3 + k] * wv;
              }
              out[oRow0 + n] = c0;
              out[oRow1 + n] = c1;
              out[oRow2 + n] = c2;
              out[oRow3 + n] = c3;
            }
          }

          for (let m = M4; m < M; m++) {
            const aRow = m * K;
            const oRow = m * Cout;
            for (let n = 0; n < Cout; n++) {
              let sum = bias[n];
              for (let k = 0; k < K; k++) {
                sum += A[aRow + k] * wKC[k * Cout + n];
              }
              out[oRow + n] = sum;
            }
          }

          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.DEPTHWISE_CONV_2D: {
          const x = this.values[ins[0]];
          const xSh = this.tensors[ins[0]].shape;
          const wTensor = this.tensors[ins[1]];
          const w = this.values[ins[1]]; // [1, kH, kW, C]
          const bias = this.values[ins[2]];
          const [, H, W, C] = xSh;
          const [, kH, kW] = wTensor.shape;
          const [, Hout, Wout] = outShape;
          const sH = opts.strideH || 1;
          const sW = opts.strideW || 1;
          let padTop = 0, padLeft = 0;
          if (opts.padding === 0) {
            const totalPadH = Math.max(0, (Hout - 1) * sH + kH - H);
            const totalPadW = Math.max(0, (Wout - 1) * sW + kW - W);
            padTop = totalPadH >> 1;
            padLeft = totalPadW >> 1;
          }
          const out = new Float32Array(Hout * Wout * C);
          for (let oh = 0; oh < Hout; oh++) {
            const ih0 = oh * sH - padTop;
            const khStart = ih0 < 0 ? -ih0 : 0;
            const khEnd = Math.min(kH, H - ih0);
            for (let ow = 0; ow < Wout; ow++) {
              const iw0 = ow * sW - padLeft;
              const kwStart = iw0 < 0 ? -iw0 : 0;
              const kwEnd = Math.min(kW, W - iw0);
              const outBase = (oh * Wout + ow) * C;
              out.set(bias, outBase);
              for (let kh = khStart; kh < khEnd; kh++) {
                const inRowBase = (ih0 + kh) * W * C;
                const wKhBase = kh * kW * C;
                for (let kw = kwStart; kw < kwEnd; kw++) {
                  const inBase = inRowBase + (iw0 + kw) * C;
                  const wBase = wKhBase + kw * C;
                  for (let c = 0; c < C; c++) {
                    out[outBase + c] += x[inBase + c] * w[wBase + c];
                  }
                }
              }
            }
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.LOGISTIC: {
          const x = this.values[ins[0]];
          const n = x.length;
          const out = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            out[i] = 1.0 / (1.0 + Math.exp(-x[i]));
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.ADD:
        case BuiltinOperator.SUB:
        case BuiltinOperator.MUL: {
          const a = this.values[ins[0]];
          const b = this.values[ins[1]];
          const aSh = this.tensors[ins[0]].shape;
          const bSh = this.tensors[ins[1]].shape;
          const out = this.binaryBroadcastOp(a, aSh, b, bSh, outShape, op.builtinCode);
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.MAX_POOL_2D: {
          const x = this.values[ins[0]];
          const [, H, W, C] = this.tensors[ins[0]].shape;
          const [, Hout, Wout] = outShape;
          const kH = opts.filterHeight || 5;
          const kW = opts.filterWidth || 5;
          const sH = opts.strideH || 1;
          const sW = opts.strideW || 1;
          const padTop = opts.padding === 0 ? Math.max(0, (Hout - 1) * sH + kH - H) >> 1 : 0;
          const padLeft = opts.padding === 0 ? Math.max(0, (Wout - 1) * sW + kW - W) >> 1 : 0;
          const out = new Float32Array(Hout * Wout * C);
          out.fill(-Infinity);
          for (let oh = 0; oh < Hout; oh++) {
            const ih0 = oh * sH - padTop;
            const khStart = ih0 < 0 ? -ih0 : 0;
            const khEnd = Math.min(kH, H - ih0);
            for (let ow = 0; ow < Wout; ow++) {
              const iw0 = ow * sW - padLeft;
              const kwStart = iw0 < 0 ? -iw0 : 0;
              const kwEnd = Math.min(kW, W - iw0);
              const outBase = (oh * Wout + ow) * C;
              for (let kh = khStart; kh < khEnd; kh++) {
                const inRowBase = (ih0 + kh) * W * C;
                for (let kw = kwStart; kw < kwEnd; kw++) {
                  const inBase = inRowBase + (iw0 + kw) * C;
                  for (let c = 0; c < C; c++) {
                    const v = x[inBase + c];
                    if (v > out[outBase + c]) out[outBase + c] = v;
                  }
                }
              }
            }
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: {
          const x = this.values[ins[0]];
          const [, H, W, C] = this.tensors[ins[0]].shape;
          const [, Hout, Wout] = outShape;
          const out = new Float32Array(Hout * Wout * C);
          const scaleH = H / Hout;
          const scaleW = W / Wout;
          for (let oh = 0; oh < Hout; oh++) {
            const ih = Math.min(H - 1, Math.floor(oh * scaleH));
            for (let ow = 0; ow < Wout; ow++) {
              const iw = Math.min(W - 1, Math.floor(ow * scaleW));
              const srcOff = (ih * W + iw) * C;
              const dstOff = (oh * Wout + ow) * C;
              out.set(x.subarray(srcOff, srcOff + C), dstOff);
            }
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.CONCATENATION: {
          const rank = outShape.length;
          let axis = opts.axis < 0 ? opts.axis + rank : opts.axis;
          let outer = 1, inner = 1;
          for (let d = 0; d < axis; d++) outer *= outShape[d];
          for (let d = axis + 1; d < rank; d++) inner *= outShape[d];
          const outAxisDim = outShape[axis];
          const outType = this.tensors[out0].type;
          const ArrayCtor = (outType === TensorType.INT32 || outType === TensorType.INT64) ? Int32Array : Float32Array;
          const out = new ArrayCtor(outer * outAxisDim * inner);
          let axisOffset = 0;
          for (let k = 0; k < ins.length; k++) {
            const arr = this.values[ins[k]];
            const inSh = this.tensors[ins[k]].shape;
            const inAxisDim = inSh[axis];
            const chunk = inAxisDim * inner;
            for (let o = 0; o < outer; o++) {
              const srcOff = o * chunk;
              const dstOff = (o * outAxisDim + axisOffset) * inner;
              out.set(arr.subarray(srcOff, srcOff + chunk), dstOff);
            }
            axisOffset += inAxisDim;
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.STRIDED_SLICE: {
          const x = this.values[ins[0]];
          const xSh = this.tensors[ins[0]].shape;
          const beginArr = this.values[ins[1]];
          const endArr = this.values[ins[2]];
          const rank = xSh.length;
          const bMask = opts.beginMask || 0;
          const eMask = opts.endMask || 0;
          const starts = new Array(rank);
          const ends = new Array(rank);
          for (let d = 0; d < rank; d++) {
            let b = (bMask & (1 << d)) ? 0 : beginArr[d];
            let e = (eMask & (1 << d)) ? xSh[d] : endArr[d];
            if (b < 0) b += xSh[d];
            if (e < 0) e += xSh[d];
            starts[d] = Math.max(0, Math.min(xSh[d], b));
            ends[d] = Math.max(0, Math.min(xSh[d], e));
          }
          // Identify the single sliced axis (or general ND slice)
          let sliceAxis = 0;
          for (let d = 0; d < rank; d++) {
            if (starts[d] !== 0 || ends[d] !== xSh[d]) {
              sliceAxis = d;
              break;
            }
          }
          let outer = 1, inner = 1;
          for (let d = 0; d < sliceAxis; d++) outer *= xSh[d];
          for (let d = sliceAxis + 1; d < rank; d++) inner *= xSh[d];
          const inAxisDim = xSh[sliceAxis];
          const outAxisDim = ends[sliceAxis] - starts[sliceAxis];
          const out = new Float32Array(outer * outAxisDim * inner);
          const chunk = outAxisDim * inner;
          const startInner = starts[sliceAxis] * inner;
          for (let o = 0; o < outer; o++) {
            const srcOff = o * inAxisDim * inner + startInner;
            out.set(x.subarray(srcOff, srcOff + chunk), o * chunk);
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.RESHAPE: {
          this.values[out0] = this.values[ins[0]];
          break;
        }

        case BuiltinOperator.TRANSPOSE: {
          const x = this.values[ins[0]];
          const xSh = this.tensors[ins[0]].shape;
          const perm = this.values[ins[1]];
          const rank = xSh.length;
          const out = new Float32Array(x.length);
          if (rank === 4) {
            const [D0, D1, D2, D3] = xSh;
            const [O0, O1, O2, O3] = outShape;
            const inStrides = [D1 * D2 * D3, D2 * D3, D3, 1];
            const s0 = inStrides[perm[0]], s1 = inStrides[perm[1]], s2 = inStrides[perm[2]], s3 = inStrides[perm[3]];
            let dst = 0;
            for (let i0 = 0; i0 < O0; i0++) {
              const b0 = i0 * s0;
              for (let i1 = 0; i1 < O1; i1++) {
                const b1 = b0 + i1 * s1;
                for (let i2 = 0; i2 < O2; i2++) {
                  const b2 = b1 + i2 * s2;
                  for (let i3 = 0; i3 < O3; i3++) {
                    out[dst++] = x[b2 + i3 * s3];
                  }
                }
              }
            }
          } else if (rank === 3) {
            const [D0, D1, D2] = xSh;
            const [O0, O1, O2] = outShape;
            const inStrides = [D1 * D2, D2, 1];
            const s0 = inStrides[perm[0]], s1 = inStrides[perm[1]], s2 = inStrides[perm[2]];
            let dst = 0;
            for (let i0 = 0; i0 < O0; i0++) {
              for (let i1 = 0; i1 < O1; i1++) {
                const b1 = i0 * s0 + i1 * s1;
                for (let i2 = 0; i2 < O2; i2++) {
                  out[dst++] = x[b1 + i2 * s2];
                }
              }
            }
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.SOFTMAX: {
          const x = this.values[ins[0]];
          const lastDim = outShape[outShape.length - 1];
          const outer = x.length / lastDim;
          const beta = opts.beta || 1.0;
          const out = new Float32Array(x.length);
          for (let o = 0; o < outer; o++) {
            const base = o * lastDim;
            let maxVal = -Infinity;
            for (let i = 0; i < lastDim; i++) {
              const v = x[base + i];
              if (v > maxVal) maxVal = v;
            }
            let sum = 0;
            for (let i = 0; i < lastDim; i++) {
              const e = Math.exp((x[base + i] - maxVal) * beta);
              out[base + i] = e;
              sum += e;
            }
            const inv = 1.0 / sum;
            for (let i = 0; i < lastDim; i++) {
              out[base + i] *= inv;
            }
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.BATCH_MATMUL: {
          const A = this.values[ins[0]];
          const B = this.values[ins[1]];
          const aSh = this.tensors[ins[0]].shape;
          const bSh = this.tensors[ins[1]].shape;
          const adjX = !!opts.adjX;
          const adjY = !!opts.adjY;
          const aRows = aSh[aSh.length - 2];
          const aCols = aSh[aSh.length - 1];
          const bRows = bSh[bSh.length - 2];
          const bCols = bSh[bSh.length - 1];
          const M = adjX ? aCols : aRows;
          const K = adjX ? aRows : aCols;
          const N = adjY ? bRows : bCols;
          const batches = A.length / (aRows * aCols);
          const out = new Float32Array(batches * M * N);
          for (let b = 0; b < batches; b++) {
            const aBase = b * aRows * aCols;
            const bBase = b * bRows * bCols;
            const oBase = b * M * N;
            for (let m = 0; m < M; m++) {
              for (let n = 0; n < N; n++) {
                let sum = 0;
                for (let k = 0; k < K; k++) {
                  const av = adjX ? A[aBase + k * aCols + m] : A[aBase + m * aCols + k];
                  const bv = adjY ? B[bBase + n * bCols + k] : B[bBase + k * bCols + n];
                  sum += av * bv;
                }
                out[oBase + m * N + n] = sum;
              }
            }
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.REDUCE_MAX: {
          const x = this.values[ins[0]];
          const xSh = this.tensors[ins[0]].shape;
          const lastDim = xSh[xSh.length - 1];
          const outer = x.length / lastDim;
          const out = new Float32Array(outer);
          for (let o = 0; o < outer; o++) {
            let mv = -Infinity;
            const base = o * lastDim;
            for (let i = 0; i < lastDim; i++) {
              if (x[base + i] > mv) mv = x[base + i];
            }
            out[o] = mv;
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.TOPK_V2: {
          const x = this.values[ins[0]];
          const k = this.values[ins[1]][0];
          const n = x.length;
          const indices = new Int32Array(n);
          for (let i = 0; i < n; i++) indices[i] = i;
          indices.sort((a, b) => x[b] - x[a]);
          const topVals = new Float32Array(k);
          const topIdx = new Int32Array(k);
          for (let i = 0; i < k; i++) {
            topIdx[i] = indices[i];
            topVals[i] = x[indices[i]];
          }
          this.values[outs[0]] = topVals;
          this.values[outs[1]] = topIdx;
          break;
        }

        case BuiltinOperator.CAST: {
          const x = this.values[ins[0]];
          const dstType = this.tensors[out0].type;
          if (dstType === TensorType.FLOAT32) {
            const out = new Float32Array(x.length);
            for (let i = 0; i < x.length; i++) out[i] = x[i];
            this.values[out0] = out;
          } else {
            const out = new Int32Array(x.length);
            for (let i = 0; i < x.length; i++) out[i] = Math.trunc(x[i]);
            this.values[out0] = out;
          }
          break;
        }

        case BuiltinOperator.TILE: {
          const x = this.values[ins[0]];
          const reps = this.values[ins[1]];
          const lastRep = reps[reps.length - 1];
          const out = new Int32Array(x.length * lastRep);
          for (let i = 0; i < x.length; i++) {
            const v = x[i];
            const base = i * lastRep;
            for (let r = 0; r < lastRep; r++) out[base + r] = v;
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.FLOOR_MOD: {
          const a = this.values[ins[0]];
          const b = this.values[ins[1]][0];
          const out = new Int32Array(a.length);
          for (let i = 0; i < a.length; i++) {
            out[i] = ((a[i] % b) + b) % b;
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.GATHER_ND: {
          const params = this.values[ins[0]];
          const pSh = this.tensors[ins[0]].shape; // [D0, D1, D2]
          const indices = this.values[ins[1]]; // [N, 3]
          const numSlices = indices.length / 3;
          const out = new Float32Array(numSlices);
          const s0 = pSh[1] * pSh[2];
          const s1 = pSh[2];
          for (let i = 0; i < numSlices; i++) {
            let i0 = indices[i * 3];
            let i1 = indices[i * 3 + 1];
            let i2 = indices[i * 3 + 2];
            if (i0 < 0) i0 += pSh[0];
            if (i1 < 0) i1 += pSh[1];
            if (i2 < 0) i2 += pSh[2];
            out[i] = params[i0 * s0 + i1 * s1 + i2];
          }
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.LESS: {
          const a = this.values[ins[0]];
          const b = this.values[ins[1]][0];
          const out = new Uint8Array(a.length);
          for (let i = 0; i < a.length; i++) out[i] = a[i] < b ? 1 : 0;
          this.values[out0] = out;
          break;
        }

        case BuiltinOperator.GATHER: {
          const params = this.values[ins[0]];
          const pSh = this.tensors[ins[0]].shape; // [300, 4]
          const indices = this.values[ins[1]]; // [1, 300]
          const inner = pSh[1];
          const out = new Float32Array(indices.length * inner);
          for (let i = 0; i < indices.length; i++) {
            const idx = indices[i];
            for (let c = 0; c < inner; c++) {
              out[i * inner + c] = params[idx * inner + c];
            }
          }
          this.values[out0] = out;
          break;
        }

        default:
          throw new Error(`Unsupported LiteRT builtinCode: ${op.builtinCode}`);
      }
    }

    binaryBroadcastOp(a, aSh, b, bSh, outSh, code) {
      const n = numElements(outSh);
      const out = new Float32Array(n);
      if (a.length === b.length) {
        if (code === BuiltinOperator.ADD) {
          for (let i = 0; i < n; i++) out[i] = a[i] + b[i];
        } else if (code === BuiltinOperator.SUB) {
          for (let i = 0; i < n; i++) out[i] = a[i] - b[i];
        } else {
          for (let i = 0; i < n; i++) out[i] = a[i] * b[i];
        }
        return out;
      }
      if (b.length === 1) {
        const bv = b[0];
        if (code === BuiltinOperator.ADD) {
          for (let i = 0; i < n; i++) out[i] = a[i] + bv;
        } else if (code === BuiltinOperator.SUB) {
          for (let i = 0; i < n; i++) out[i] = a[i] - bv;
        } else {
          for (let i = 0; i < n; i++) out[i] = a[i] * bv;
        }
        return out;
      }
      // Broadcast [1, C, L] with [1, 1, L]
      const L = outSh[outSh.length - 1];
      for (let i = 0; i < n; i++) {
        const av = a.length === n ? a[i] : a[i % L];
        const bv = b.length === n ? b[i] : b[i % L];
        if (code === BuiltinOperator.ADD) out[i] = av + bv;
        else if (code === BuiltinOperator.SUB) out[i] = av - bv;
        else out[i] = av * bv;
      }
      return out;
    }
  }

  const LiteRTRuntime = {
    LiteRTInterpreter,
    loadTFLiteModel: async function (source) {
      if (source instanceof ArrayBuffer) {
        return new LiteRTInterpreter(source);
      }
      if (ArrayBuffer.isView(source)) {
        return new LiteRTInterpreter(
          source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength)
        );
      }
      if (typeof source === 'string') {
        const resp = await fetch(source);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${source}`);
        const buf = await resp.arrayBuffer();
        return new LiteRTInterpreter(buf);
      }
      throw new Error('Invalid LiteRT model source');
    },
  };

  root.LiteRTRuntime = LiteRTRuntime;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = LiteRTRuntime;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
