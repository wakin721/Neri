import os
import pickle
import numpy as np
from ai_edge_litert.tools import flatbuffer_utils as fb
from ai_edge_litert import schema_py_generated as schema_fb
from ai_edge_litert.interpreter import Interpreter

def build_sorrel_tflite(ir_path, out_tflite_path, fp16_weights=True):
    with open(ir_path, 'rb') as f:
        ir = pickle.load(f)

    onnx_shapes = ir['shapes']
    consts = ir['consts']
    nodes = ir['nodes']

    # Pre-populate folded shape constants in tail (nodes 326..338)
    consts['/model.23/Mul_output_0'] = np.array([2], dtype=np.int64)
    consts['/model.23/Mul_1_output_0'] = np.array([4], dtype=np.int64)
    skip_outputs = {
        '/model.23/Shape_output_0',
        '/model.23/Gather_output_0',
        '/model.23/Add_output_0',
        '/model.23/Div_output_0',
        '/model.23/Mul_output_0',
        '/model.23/Mul_1_output_0',
    }

    model = schema_fb.ModelT()
    model.version = 3
    model.description = b"MegaDetector v1000-sorrel (YOLO11s) LiteRT converted"
    model.buffers = []
    model.operatorCodes = []

    # Buffer 0 must be empty
    buf0 = schema_fb.BufferT()
    buf0.data = None
    model.buffers.append(buf0)

    sg = schema_fb.SubGraphT()
    sg.name = b"main"
    sg.tensors = []
    sg.operators = []
    model.subgraphs = [sg]

    opcode_map = {}
    def get_opcode_index(builtin_code, version=1):
        if builtin_code not in opcode_map:
            oc = schema_fb.OperatorCodeT()
            oc.builtinCode = builtin_code
            oc.deprecatedBuiltinCode = min(builtin_code, 127)
            oc.version = version
            idx = len(model.operatorCodes)
            model.operatorCodes.append(oc)
            opcode_map[builtin_code] = idx
        return opcode_map[builtin_code]

    def add_buffer(arr):
        if arr is None:
            return 0
        b = schema_fb.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8)
        idx = len(model.buffers)
        model.buffers.append(b)
        return idx

    def add_tensor(name, shape, ttype=schema_fb.TensorType.FLOAT32, arr=None):
        t = schema_fb.TensorT()
        t.name = name.encode('utf-8') if isinstance(name, str) else name
        t.shape = [int(x) for x in shape]
        t.shapeSignature = [int(x) for x in shape]
        t.type = ttype
        t.buffer = add_buffer(arr)
        t.isVariable = False
        idx = len(sg.tensors)
        sg.tensors.append(t)
        return idx

    def add_op(builtin_code, inputs, outputs, opt_type=schema_fb.BuiltinOptions.NONE, opts=None, version=1):
        op = schema_fb.OperatorT()
        op.opcodeIndex = get_opcode_index(builtin_code, version)
        op.inputs = [int(x) for x in inputs]
        op.outputs = [int(x) for x in outputs]
        op.builtinOptionsType = opt_type
        op.builtinOptions = opts
        sg.operators.append(op)

    def add_const_float(name, arr_f32):
        arr_f32 = np.ascontiguousarray(arr_f32, dtype=np.float32)
        if fp16_weights and arr_f32.size >= 16:
            arr_f16 = arr_f32.astype(np.float16)
            f16_idx = add_tensor(name + "_fp16", arr_f16.shape, schema_fb.TensorType.FLOAT16, arr_f16)
            f32_idx = add_tensor(name, arr_f32.shape, schema_fb.TensorType.FLOAT32, None)
            add_op(schema_fb.BuiltinOperator.DEQUANTIZE, [f16_idx], [f32_idx])
            return f32_idx
        else:
            return add_tensor(name, arr_f32.shape, schema_fb.TensorType.FLOAT32, arr_f32)

    def add_const_int32(name, arr_i32):
        arr_i32 = np.ascontiguousarray(arr_i32, dtype=np.int32)
        return add_tensor(name, arr_i32.shape, schema_fb.TensorType.INT32, arr_i32)

    # Track tensor index, current shape, and layout ('NHWC' or 'NCHW')
    tensor_map = {}
    tensor_shape = {}
    tensor_layout = {}

    # Input 0: images [1, 640, 640, 3] NHWC
    in_idx = add_tensor("images", [1, 640, 640, 3], schema_fb.TensorType.FLOAT32, None)
    sg.inputs = [in_idx]
    tensor_map["images"] = in_idx
    tensor_shape["images"] = [1, 640, 640, 3]
    tensor_layout["images"] = "NHWC"

    def emit_transpose(in_tid, in_sh, perm, name_hint):
        out_sh = [in_sh[p] for p in perm]
        perm_tid = add_const_int32(name_hint + "_perm", perm)
        out_tid = add_tensor(name_hint, out_sh, schema_fb.TensorType.FLOAT32, None)
        add_op(schema_fb.BuiltinOperator.TRANSPOSE, [in_tid, perm_tid], [out_tid])
        return out_tid, out_sh

    def emit_reshape(in_tid, target_sh, name_hint):
        sh_tid = add_const_int32(name_hint + "_shape", target_sh)
        out_tid = add_tensor(name_hint, target_sh, schema_fb.TensorType.FLOAT32, None)
        opts = schema_fb.ReshapeOptionsT()
        opts.newShape = [int(x) for x in target_sh]
        add_op(schema_fb.BuiltinOperator.RESHAPE, [in_tid, sh_tid], [out_tid], schema_fb.BuiltinOptions.ReshapeOptions, opts)
        return out_tid

    for n in nodes:
        op_type = n['op_type']
        ins = n['inputs']
        outs = n['outputs']
        attrs = n['attrs']
        out0 = outs[0]

        if out0 in skip_outputs:
            continue

        if op_type == 'Conv':
            x_name = ins[0]
            w_name = ins[1]
            b_name = ins[2] if len(ins) > 2 else None

            x_tid = tensor_map[x_name]
            x_sh = tensor_shape[x_name]
            w_arr = consts[w_name]  # [C_out, C_in_g, kH, kW]
            c_out, c_in_g, kH, kW = w_arr.shape
            b_arr = consts[b_name] if b_name else np.zeros((c_out,), dtype=np.float32)

            strides = attrs.get('strides', [1, 1])
            pads = attrs.get('pads', [0, 0, 0, 0])
            dilations = attrs.get('dilations', [1, 1])
            group = attrs.get('group', 1)

            # Handle padding
            if pads == [0, 0, 0, 0]:
                padding_code = schema_fb.Padding.VALID
                cur_tid, cur_sh = x_tid, x_sh
            elif strides == [1, 1] and pads[0] == pads[2] == kH // 2 and pads[1] == pads[3] == kW // 2:
                padding_code = schema_fb.Padding.SAME
                cur_tid, cur_sh = x_tid, x_sh
            else:
                # Explicit symmetric PAD followed by VALID conv
                pad_arr = np.array([[0, 0], [pads[0], pads[2]], [pads[1], pads[3]], [0, 0]], dtype=np.int32)
                pad_tid = add_const_int32(out0 + "_pad_const", pad_arr)
                cur_sh = [x_sh[0], x_sh[1] + pads[0] + pads[2], x_sh[2] + pads[1] + pads[3], x_sh[3]]
                cur_tid = add_tensor(out0 + "_padded", cur_sh, schema_fb.TensorType.FLOAT32, None)
                add_op(schema_fb.BuiltinOperator.PAD, [x_tid, pad_tid], [cur_tid], schema_fb.BuiltinOptions.PadOptions, schema_fb.PadOptionsT())
                padding_code = schema_fb.Padding.VALID

            if padding_code == schema_fb.Padding.SAME:
                h_out = (cur_sh[1] + strides[0] - 1) // strides[0]
                w_out = (cur_sh[2] + strides[1] - 1) // strides[1]
            else:
                h_out = (cur_sh[1] - kH) // strides[0] + 1
                w_out = (cur_sh[2] - kW) // strides[1] + 1
            out_sh = [cur_sh[0], h_out, w_out, c_out]

            if group == 1:
                w_tfl = np.transpose(w_arr, (0, 2, 3, 1))  # [C_out, kH, kW, C_in]
                w_tid = add_const_float(w_name + "_tfl", w_tfl)
                b_tid = add_const_float((b_name or out0 + "_bias") + "_tfl", b_arr)
                out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
                opts = schema_fb.Conv2DOptionsT()
                opts.padding = padding_code
                opts.strideH = int(strides[0])
                opts.strideW = int(strides[1])
                opts.dilationHFactor = int(dilations[0])
                opts.dilationWFactor = int(dilations[1])
                opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
                add_op(schema_fb.BuiltinOperator.CONV_2D, [cur_tid, w_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.Conv2DOptions, opts)
            elif group == c_out:
                w_tfl = np.transpose(w_arr, (1, 2, 3, 0))  # [1, kH, kW, C_out]
                w_tid = add_const_float(w_name + "_tfl", w_tfl)
                b_tid = add_const_float((b_name or out0 + "_bias") + "_tfl", b_arr)
                out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
                opts = schema_fb.DepthwiseConv2DOptionsT()
                opts.padding = padding_code
                opts.strideH = int(strides[0])
                opts.strideW = int(strides[1])
                opts.depthMultiplier = 1
                opts.dilationHFactor = int(dilations[0])
                opts.dilationWFactor = int(dilations[1])
                opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
                add_op(schema_fb.BuiltinOperator.DEPTHWISE_CONV_2D, [cur_tid, w_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.DepthwiseConv2DOptions, opts)
            else:
                raise ValueError(f"Unsupported conv group={group} c_out={c_out}")

            tensor_map[out0] = out_tid
            tensor_shape[out0] = out_sh
            tensor_layout[out0] = "NHWC"

        elif op_type == 'Sigmoid':
            x_tid = tensor_map[ins[0]]
            x_sh = tensor_shape[ins[0]]
            out_tid = add_tensor(out0, x_sh, schema_fb.TensorType.FLOAT32, None)
            add_op(schema_fb.BuiltinOperator.LOGISTIC, [x_tid], [out_tid])
            tensor_map[out0] = out_tid
            tensor_shape[out0] = x_sh
            tensor_layout[out0] = tensor_layout[ins[0]]

        elif op_type in ('Mul', 'Add', 'Sub', 'Div'):
            a_name, b_name = ins[0], ins[1]
            if op_type == 'Div' and b_name in consts:
                # Convert Div by constant into Mul by reciprocal
                recip = (1.0 / consts[b_name].astype(np.float32)).reshape(-1)
                b_tid = add_const_float(b_name + "_recip", recip)
                a_tid = tensor_map[a_name]
                out_sh = tensor_shape[a_name]
                out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
                opts = schema_fb.MulOptionsT()
                opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
                add_op(schema_fb.BuiltinOperator.MUL, [a_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.MulOptions, opts)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = out_sh
                tensor_layout[out0] = tensor_layout[a_name]
                continue

            def resolve_operand(name, ref_sh):
                if name in tensor_map:
                    return tensor_map[name], tensor_shape[name], tensor_layout[name]
                arr = consts[name].astype(np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                elif arr.ndim == 2 and len(ref_sh) == 3 and arr.shape == (1, ref_sh[2]):
                    arr = arr.reshape(1, 1, ref_sh[2])
                tid = add_const_float(name + "_const", arr)
                return tid, list(arr.shape), "NCHW"

            ref_sh = tensor_shape.get(a_name) or tensor_shape.get(b_name)
            a_tid, a_sh, a_lay = resolve_operand(a_name, ref_sh)
            b_tid, b_sh, b_lay = resolve_operand(b_name, ref_sh)
            out_sh = a_sh if len(a_sh) >= len(b_sh) and np.prod(a_sh) >= np.prod(b_sh) else b_sh
            out_lay = a_lay if a_name in tensor_map else b_lay
            out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)

            if op_type == 'Mul':
                opts = schema_fb.MulOptionsT()
                opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
                add_op(schema_fb.BuiltinOperator.MUL, [a_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.MulOptions, opts)
            elif op_type == 'Add':
                opts = schema_fb.AddOptionsT()
                opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
                add_op(schema_fb.BuiltinOperator.ADD, [a_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.AddOptions, opts)
            elif op_type == 'Sub':
                opts = schema_fb.SubOptionsT()
                opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
                add_op(schema_fb.BuiltinOperator.SUB, [a_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.SubOptions, opts)

            tensor_map[out0] = out_tid
            tensor_shape[out0] = out_sh
            tensor_layout[out0] = out_lay

        elif op_type == 'MaxPool':
            x_tid = tensor_map[ins[0]]
            x_sh = tensor_shape[ins[0]]
            kH, kW = attrs['kernel_shape']
            sH, sW = attrs['strides']
            out_tid = add_tensor(out0, x_sh, schema_fb.TensorType.FLOAT32, None)
            opts = schema_fb.Pool2DOptionsT()
            opts.padding = schema_fb.Padding.SAME
            opts.strideH = int(sH)
            opts.strideW = int(sW)
            opts.filterHeight = int(kH)
            opts.filterWidth = int(kW)
            opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
            add_op(schema_fb.BuiltinOperator.MAX_POOL_2D, [x_tid], [out_tid], schema_fb.BuiltinOptions.Pool2DOptions, opts)
            tensor_map[out0] = out_tid
            tensor_shape[out0] = x_sh
            tensor_layout[out0] = "NHWC"

        elif op_type == 'Resize':
            x_tid = tensor_map[ins[0]]
            x_sh = tensor_shape[ins[0]]
            onnx_out_sh = onnx_shapes[out0]  # [1, C, H_out, W_out]
            h_out, w_out = onnx_out_sh[2], onnx_out_sh[3]
            out_sh = [x_sh[0], h_out, w_out, x_sh[3]]
            sz_tid = add_const_int32(out0 + "_size", [h_out, w_out])
            out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
            opts = schema_fb.ResizeNearestNeighborOptionsT()
            opts.alignCorners = False
            opts.halfPixelCenters = False
            add_op(schema_fb.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR, [x_tid, sz_tid], [out_tid], schema_fb.BuiltinOptions.ResizeNearestNeighborOptions, opts)
            tensor_map[out0] = out_tid
            tensor_shape[out0] = out_sh
            tensor_layout[out0] = "NHWC"

        elif op_type == 'Concat':
            in_tids = [tensor_map[x] for x in ins]
            lay = tensor_layout[ins[0]]
            onnx_ax = attrs['axis']
            if onnx_ax < 0:
                onnx_ax += len(tensor_shape[ins[0]])
            ax = 3 if (lay == "NHWC" and len(tensor_shape[ins[0]]) == 4 and onnx_ax == 1) else onnx_ax
            out_sh = list(tensor_shape[ins[0]])
            out_sh[ax] = sum(tensor_shape[x][ax] for x in ins)
            out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
            opts = schema_fb.ConcatenationOptionsT()
            opts.axis = int(ax)
            opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
            add_op(schema_fb.BuiltinOperator.CONCATENATION, in_tids, [out_tid], schema_fb.BuiltinOptions.ConcatenationOptions, opts)
            tensor_map[out0] = out_tid
            tensor_shape[out0] = out_sh
            tensor_layout[out0] = lay

        elif op_type == 'Split':
            x_name = ins[0]
            x_tid = tensor_map[x_name]
            x_sh = tensor_shape[x_name]
            lay = tensor_layout[x_name]
            onnx_ax = attrs.get('axis', 0)
            if onnx_ax < 0:
                onnx_ax += len(x_sh)
            ax = 3 if (lay == "NHWC" and len(x_sh) == 4 and onnx_ax == 1) else onnx_ax
            splits = consts[ins[1]].tolist()
            cur = 0
            for idx_out, sp_len in zip(outs, splits):
                begin = [0] * len(x_sh)
                end = list(x_sh)
                strides = [1] * len(x_sh)
                begin[ax] = int(cur)
                end[ax] = int(cur + sp_len)
                out_sh = list(x_sh)
                out_sh[ax] = int(sp_len)
                b_tid = add_const_int32(idx_out + "_begin", begin)
                e_tid = add_const_int32(idx_out + "_end", end)
                s_tid = add_const_int32(idx_out + "_strides", strides)
                out_tid = add_tensor(idx_out, out_sh, schema_fb.TensorType.FLOAT32, None)
                opts = schema_fb.StridedSliceOptionsT()
                opts.beginMask = 0
                opts.endMask = 0
                opts.ellipsisMask = 0
                opts.newAxisMask = 0
                opts.shrinkAxisMask = 0
                opts.offset = False
                add_op(schema_fb.BuiltinOperator.STRIDED_SLICE, [x_tid, b_tid, e_tid, s_tid], [out_tid], schema_fb.BuiltinOptions.StridedSliceOptions, opts)
                tensor_map[idx_out] = out_tid
                tensor_shape[idx_out] = out_sh
                tensor_layout[idx_out] = lay
                cur += sp_len

        elif op_type == 'Slice':
            x_name = ins[0]
            x_tid = tensor_map[x_name]
            x_sh = tensor_shape[x_name]
            lay = tensor_layout[x_name]
            starts = int(consts[ins[1]][0])
            ends = int(consts[ins[2]][0])
            ax = int(consts[ins[3]][0])
            begin = [0] * len(x_sh)
            end = list(x_sh)
            strides = [1] * len(x_sh)
            begin[ax] = starts
            end[ax] = min(ends, x_sh[ax])
            out_sh = list(x_sh)
            out_sh[ax] = end[ax] - begin[ax]
            b_tid = add_const_int32(out0 + "_begin", begin)
            e_tid = add_const_int32(out0 + "_end", end)
            s_tid = add_const_int32(out0 + "_strides", strides)
            out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
            opts = schema_fb.StridedSliceOptionsT()
            opts.beginMask = 0
            opts.endMask = 0
            opts.ellipsisMask = 0
            opts.newAxisMask = 0
            opts.shrinkAxisMask = 0
            opts.offset = False
            add_op(schema_fb.BuiltinOperator.STRIDED_SLICE, [x_tid, b_tid, e_tid, s_tid], [out_tid], schema_fb.BuiltinOptions.StridedSliceOptions, opts)
            tensor_map[out0] = out_tid
            tensor_shape[out0] = out_sh
            tensor_layout[out0] = lay

        elif op_type == 'Reshape':
            x_name = ins[0]
            x_tid = tensor_map[x_name]
            x_sh = tensor_shape[x_name]
            lay = tensor_layout[x_name]
            target_sh = onnx_shapes[out0]

            if out0 == '/model.23/dfl/Reshape_1_output_0':
                # Input from DFL conv is [1, 4, 8400, 1] -> reshape directly to [1, 4, 8400]
                out_tid = emit_reshape(x_tid, [1, 4, 8400], out0)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = [1, 4, 8400]
                tensor_layout[out0] = "NCHW"
            elif lay == "NHWC" and len(x_sh) == 4:
                # Transpose NHWC [1, H, W, C] -> NCHW [1, C, H, W] before flattening H*W
                tr_tid, _ = emit_transpose(x_tid, x_sh, [0, 3, 1, 2], out0 + "_nhwc2nchw")
                out_tid = emit_reshape(tr_tid, target_sh, out0)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = target_sh
                tensor_layout[out0] = "NCHW"
            elif out0 in ('/model.10/m/m.0/attn/Reshape_1_output_0', '/model.10/m/m.0/attn/Reshape_2_output_0'):
                # Reshape [1, 4, 64, 400] -> [1, 256, 20, 20] NCHW, then transpose to [1, 20, 20, 256] NHWC
                rs_tid = emit_reshape(x_tid, target_sh, out0 + "_nchw")
                out_tid, out_sh = emit_transpose(rs_tid, target_sh, [0, 2, 3, 1], out0)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = out_sh
                tensor_layout[out0] = "NHWC"
            else:
                out_tid = emit_reshape(x_tid, target_sh, out0)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = target_sh
                tensor_layout[out0] = "NCHW"

        elif op_type == 'Transpose':
            x_name = ins[0]
            x_tid = tensor_map[x_name]
            x_sh = tensor_shape[x_name]
            if out0 == '/model.23/dfl/Transpose_output_0':
                # Transpose [1, 4, 16, 8400] -> [1, 4, 8400, 16] so Softmax(axis=-1) and Conv2D(NHWC) work directly
                out_tid, out_sh = emit_transpose(x_tid, x_sh, [0, 1, 3, 2], out0)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = out_sh
                tensor_layout[out0] = "NHWC"
            else:
                perm = attrs['perm']
                out_tid, out_sh = emit_transpose(x_tid, x_sh, perm, out0)
                tensor_map[out0] = out_tid
                tensor_shape[out0] = out_sh
                tensor_layout[out0] = tensor_layout[x_name]

        elif op_type == 'Softmax':
            x_tid = tensor_map[ins[0]]
            x_sh = tensor_shape[ins[0]]
            out_tid = add_tensor(out0, x_sh, schema_fb.TensorType.FLOAT32, None)
            opts = schema_fb.SoftmaxOptionsT()
            opts.beta = 1.0
            add_op(schema_fb.BuiltinOperator.SOFTMAX, [x_tid], [out_tid], schema_fb.BuiltinOptions.SoftmaxOptions, opts)
            tensor_map[out0] = out_tid
            tensor_shape[out0] = x_sh
            tensor_layout[out0] = tensor_layout[ins[0]]

        elif op_type == 'MatMul':
            a_tid = tensor_map[ins[0]]
            b_tid = tensor_map[ins[1]]
            a_sh = tensor_shape[ins[0]]
            b_sh = tensor_shape[ins[1]]
            out_sh = [a_sh[0], a_sh[1], a_sh[2], b_sh[3]]
            out_tid = add_tensor(out0, out_sh, schema_fb.TensorType.FLOAT32, None)
            opts = schema_fb.BatchMatMulOptionsT()
            opts.adjX = False
            opts.adjY = False
            opts.asymmetricQuantizeInputs = False
            add_op(schema_fb.BuiltinOperator.BATCH_MATMUL, [a_tid, b_tid], [out_tid], schema_fb.BuiltinOptions.BatchMatMulOptions, opts)
            tensor_map[out0] = out_tid
            tensor_shape[out0] = out_sh
            tensor_layout[out0] = "NCHW"

        else:
            raise ValueError(f"Unhandled op_type: {op_type}")

    sg.outputs = [tensor_map['output0']]
    os.makedirs(os.path.dirname(out_tflite_path), exist_ok=True)
    fb.write_model(model, out_tflite_path)
    print(f"Wrote LiteRT model to {out_tflite_path} ({os.path.getsize(out_tflite_path) / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    ir_path = r'E:\Files\Neri\temp\sorrel_ir.pkl'
    out_path = r'E:\Files\Neri\web\models\md_v1000.0.0-sorrel.tflite'
    build_sorrel_tflite(ir_path, out_path, fp16_weights=True)

    # Verify in LiteRT Interpreter!
    interp = Interpreter(model_path=out_path)
    interp.allocate_tensors()
    print("Input details:", interp.get_input_details())
    print("Output details:", interp.get_output_details())

    # Also generate bundled_models.js for offline/file:// execution in browser
    import base64
    models_dir = r'E:\Files\Neri\web\models'
    files = ['md_v1000.0.0-sorrel.tflite', 'MDV6-yolov10-c.tflite']
    out_js = os.path.join(models_dir, 'bundled_models.js')
    with open(out_js, 'w', encoding='utf-8') as f:
        f.write('// Auto-generated bundled LiteRT (.tflite) models for offline / file:// execution\n')
        f.write('(function(root){\n')
        f.write('  root.__NERI_BUNDLED_TFLITE_B64 = {\n')
        for fname in files:
            p = os.path.join(models_dir, fname)
            if os.path.exists(p):
                with open(p, 'rb') as bf:
                    b64 = base64.b64encode(bf.read()).decode('ascii')
                f.write(f'    {fname!r}: "{b64}",\n')
        f.write('  };\n')
        f.write('''  root.getBundledTfliteArrayBuffer = function(name) {
    const b64 = root.__NERI_BUNDLED_TFLITE_B64 && root.__NERI_BUNDLED_TFLITE_B64[name];
    if (!b64) return null;
    if (typeof Buffer !== 'undefined') {
      const buf = Buffer.from(b64, 'base64');
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
''')
    print(f"Generated {out_js} ({os.path.getsize(out_js) / (1024*1024):.2f} MB)")
