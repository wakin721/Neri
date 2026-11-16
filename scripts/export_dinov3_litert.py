"""Export Neri1.2preview (DINOv3 ViT-B/16) to LiteRT (.tflite) and benchmark CPU speed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any

import numpy as np
from ai_edge_litert import schema_py_generated as schema_fb
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.tools import flatbuffer_utils as fb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _LiteRtGraphBuilder:
    """Minimal FlatBuffer builder for DINOv3 ViT-B/16 float32/fp16-weight LiteRT models."""

    def __init__(self, description: str, *, fp16_weights: bool = False):
        self.fp16_weights = fp16_weights
        self.model = schema_fb.ModelT()
        self.model.version = 3
        self.model.description = description.encode("utf-8")
        self.model.buffers = [schema_fb.BufferT()]
        self.model.operatorCodes = []

        self.sg = schema_fb.SubGraphT()
        self.sg.name = b"main"
        self.sg.tensors = []
        self.sg.operators = []
        self.model.subgraphs = [self.sg]

        self._opcodes: dict[tuple[int, int], int] = {}

    def _opcode_index(self, builtin_code: int, version: int = 1) -> int:
        key = (int(builtin_code), int(version))
        if key not in self._opcodes:
            oc = schema_fb.OperatorCodeT()
            oc.builtinCode = builtin_code
            oc.deprecatedBuiltinCode = min(builtin_code, 127)
            oc.version = version
            self._opcodes[key] = len(self.model.operatorCodes)
            self.model.operatorCodes.append(oc)
        return self._opcodes[key]

    def _add_buffer(self, arr: np.ndarray | None) -> int:
        if arr is None:
            return 0
        buf = schema_fb.BufferT()
        buf.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8)
        self.model.buffers.append(buf)
        return len(self.model.buffers) - 1

    def tensor(
        self,
        name: str,
        shape: list[int],
        *,
        ttype: int = schema_fb.TensorType.FLOAT32,
        arr: np.ndarray | None = None,
        shape_signature: list[int] | None = None,
    ) -> int:
        t = schema_fb.TensorT()
        t.name = name.encode("utf-8")
        t.shape = [int(x) for x in shape]
        t.shapeSignature = [int(x) for x in (shape_signature if shape_signature is not None else shape)]
        t.type = ttype
        t.buffer = self._add_buffer(arr)
        t.isVariable = False
        self.sg.tensors.append(t)
        return len(self.sg.tensors) - 1

    def const_f32(self, name: str, arr: np.ndarray, *, allow_fp16: bool = True) -> int:
        arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
        if self.fp16_weights and allow_fp16 and arr_f32.size >= 64:
            arr_f16 = arr_f32.astype(np.float16)
            f16_tid = self.tensor(
                f"{name}_fp16",
                list(arr_f16.shape),
                ttype=schema_fb.TensorType.FLOAT16,
                arr=arr_f16,
            )
            f32_tid = self.tensor(name, list(arr_f32.shape), ttype=schema_fb.TensorType.FLOAT32)
            self.op(schema_fb.BuiltinOperator.DEQUANTIZE, [f16_tid], [f32_tid], version=1)
            return f32_tid
        return self.tensor(name, list(arr_f32.shape), ttype=schema_fb.TensorType.FLOAT32, arr=arr_f32)

    def const_i32(self, name: str, arr: np.ndarray | list[int] | int) -> int:
        arr_i32 = np.ascontiguousarray(np.asarray(arr, dtype=np.int32))
        return self.tensor(name, list(arr_i32.shape), ttype=schema_fb.TensorType.INT32, arr=arr_i32)

    def op(
        self,
        builtin_code: int,
        inputs: list[int],
        outputs: list[int],
        *,
        opt_type: int = schema_fb.BuiltinOptions.NONE,
        opts: Any = None,
        version: int = 1,
    ) -> None:
        operator = schema_fb.OperatorT()
        operator.opcodeIndex = self._opcode_index(builtin_code, version)
        operator.inputs = [int(x) for x in inputs]
        operator.outputs = [int(x) for x in outputs]
        operator.builtinOptionsType = opt_type
        operator.builtinOptions = opts
        self.sg.operators.append(operator)

    def binary(self, code: int, a: int, b: int, out_shape: list[int], name: str, out_sig: list[int] | None = None) -> int:
        out = self.tensor(name, out_shape, shape_signature=out_sig)
        opt_type = schema_fb.BuiltinOptions.NONE
        opts = None
        if code == schema_fb.BuiltinOperator.ADD:
            opt_type = schema_fb.BuiltinOptions.AddOptions
            opts = schema_fb.AddOptionsT()
        elif code == schema_fb.BuiltinOperator.SUB:
            opt_type = schema_fb.BuiltinOptions.SubOptions
            opts = schema_fb.SubOptionsT()
        elif code == schema_fb.BuiltinOperator.MUL:
            opt_type = schema_fb.BuiltinOptions.MulOptions
            opts = schema_fb.MulOptionsT()
        self.op(code, [a, b], [out], opt_type=opt_type, opts=opts, version=1)
        return out

    def reshape(self, x: int, new_shape: list[int], concrete_shape: list[int], name: str, sig: list[int] | None = None) -> int:
        sh_tid = self.const_i32(f"{name}_shape", new_shape)
        out = self.tensor(name, concrete_shape, shape_signature=sig)
        opts = schema_fb.ReshapeOptionsT()
        opts.newShape = [int(v) for v in new_shape]
        self.op(
            schema_fb.BuiltinOperator.RESHAPE,
            [x, sh_tid],
            [out],
            opt_type=schema_fb.BuiltinOptions.ReshapeOptions,
            opts=opts,
            version=1,
        )
        return out

    def transpose(self, x: int, perm: list[int], out_shape: list[int], name: str, sig: list[int] | None = None) -> int:
        perm_tid = self.const_i32(f"{name}_perm", perm)
        out = self.tensor(name, out_shape, shape_signature=sig)
        self.op(
            schema_fb.BuiltinOperator.TRANSPOSE,
            [x, perm_tid],
            [out],
            opt_type=schema_fb.BuiltinOptions.TransposeOptions,
            opts=schema_fb.TransposeOptionsT(),
            version=1,
        )
        return out

    def linear(
        self,
        x: int,
        weight: np.ndarray,
        bias: np.ndarray | None,
        out_shape: list[int],
        name: str,
        *,
        out_sig: list[int] | None = None,
    ) -> int:
        w_tid = self.const_f32(f"{name}_weight", weight, allow_fp16=True)
        if bias is not None:
            b_tid = self.const_f32(f"{name}_bias", bias, allow_fp16=False)
            ins = [x, w_tid, b_tid]
        else:
            ins = [x, w_tid, -1]
        out = self.tensor(name, out_shape, shape_signature=out_sig)
        opts = schema_fb.FullyConnectedOptionsT()
        opts.keepNumDims = True
        self.op(
            schema_fb.BuiltinOperator.FULLY_CONNECTED,
            ins,
            [out],
            opt_type=schema_fb.BuiltinOptions.FullyConnectedOptions,
            opts=opts,
            version=9,
        )
        return out

    def layer_norm(
        self,
        x: int,
        shape: list[int],
        sig: list[int],
        weight: np.ndarray,
        bias: np.ndarray,
        eps: float,
        name: str,
    ) -> int:
        axes_tid = self.const_i32(f"{name}_axes", [-1])
        reduced_shape = shape[:-1] + [1]
        reduced_sig = sig[:-1] + [1]
        red_opts = schema_fb.ReducerOptionsT()
        red_opts.keepDims = True

        mean = self.tensor(f"{name}_mean", reduced_shape, shape_signature=reduced_sig)
        self.op(
            schema_fb.BuiltinOperator.MEAN,
            [x, axes_tid],
            [mean],
            opt_type=schema_fb.BuiltinOptions.ReducerOptions,
            opts=red_opts,
            version=1,
        )

        diff = self.binary(schema_fb.BuiltinOperator.SUB, x, mean, shape, f"{name}_diff", sig)
        sq = self.binary(schema_fb.BuiltinOperator.MUL, diff, diff, shape, f"{name}_sq", sig)

        var = self.tensor(f"{name}_var", reduced_shape, shape_signature=reduced_sig)
        self.op(
            schema_fb.BuiltinOperator.MEAN,
            [sq, axes_tid],
            [var],
            opt_type=schema_fb.BuiltinOptions.ReducerOptions,
            opts=red_opts,
            version=1,
        )

        eps_tid = self.const_f32(f"{name}_eps", np.array([eps], dtype=np.float32), allow_fp16=False)
        var_eps = self.binary(
            schema_fb.BuiltinOperator.ADD,
            var,
            eps_tid,
            reduced_shape,
            f"{name}_var_eps",
            reduced_sig,
        )
        inv_std = self.tensor(f"{name}_inv_std", reduced_shape, shape_signature=reduced_sig)
        self.op(schema_fb.BuiltinOperator.RSQRT, [var_eps], [inv_std], version=1)

        normed = self.binary(schema_fb.BuiltinOperator.MUL, diff, inv_std, shape, f"{name}_normed", sig)
        gamma_tid = self.const_f32(f"{name}_gamma", weight, allow_fp16=False)
        beta_tid = self.const_f32(f"{name}_beta", bias, allow_fp16=False)
        scaled = self.binary(schema_fb.BuiltinOperator.MUL, normed, gamma_tid, shape, f"{name}_scaled", sig)
        return self.binary(schema_fb.BuiltinOperator.ADD, scaled, beta_tid, shape, f"{name}_out", sig)

    def split(self, x: int, axis: int, num_splits: int, out_shape: list[int], out_sig: list[int], name: str) -> list[int]:
        ax_tid = self.const_i32(f"{name}_axis", np.array(axis, dtype=np.int32))
        outs = [self.tensor(f"{name}_{i}", out_shape, shape_signature=out_sig) for i in range(num_splits)]
        opts = schema_fb.SplitOptionsT()
        opts.numSplits = int(num_splits)
        self.op(
            schema_fb.BuiltinOperator.SPLIT,
            [ax_tid, x],
            outs,
            opt_type=schema_fb.BuiltinOptions.SplitOptions,
            opts=opts,
            version=1,
        )
        return outs

    def slice(self, x: int, begin: list[int], size: list[int], out_shape: list[int], out_sig: list[int], name: str) -> int:
        b_tid = self.const_i32(f"{name}_begin", begin)
        s_tid = self.const_i32(f"{name}_size", size)
        out = self.tensor(name, out_shape, shape_signature=out_sig)
        self.op(schema_fb.BuiltinOperator.SLICE, [x, b_tid, s_tid], [out], version=1)
        return out

    def concat(
        self,
        inputs: list[int],
        axis: int,
        out_shape: list[int],
        out_sig: list[int],
        name: str,
        *,
        ttype: int = schema_fb.TensorType.FLOAT32,
    ) -> int:
        out = self.tensor(name, out_shape, ttype=ttype, shape_signature=out_sig)
        opts = schema_fb.ConcatenationOptionsT()
        opts.axis = int(axis)
        self.op(
            schema_fb.BuiltinOperator.CONCATENATION,
            inputs,
            [out],
            opt_type=schema_fb.BuiltinOptions.ConcatenationOptions,
            opts=opts,
            version=1,
        )
        return out


def build_dinov3_vitb16_tflite(
    weights_path: Path,
    encoder_code_dir: Path,
    out_tflite_path: Path,
    *,
    fp16_weights: bool = False,
) -> Path:
    """Convert official DINOv3 ViT-B/16 weights to a LiteRT (.tflite) model.

    Input:  images [B, 224, 224, 3] float32 (NHWC, ImageNet normalized)
    Output: x_norm_clstoken_l2 [B, 768] float32 (L2-normalized CLS embedding)
    """
    import torch

    sys.path.insert(0, str(encoder_code_dir))
    try:
        from dinov3.hub.backbones import dinov3_vitb16

        model = dinov3_vitb16(pretrained=False)
    finally:
        sys.path.remove(str(encoder_code_dir))

    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.float().eval().requires_grad_(False)

    if model.untie_cls_and_patch_norms or model.untie_global_and_local_cls_norm:
        raise ValueError("Untied CLS/patch norms are not expected for dinov3_vitb16")

    embed_dim = int(model.embed_dim)
    num_heads = int(model.num_heads)
    head_dim = embed_dim // num_heads
    n_storage = int(model.n_storage_tokens)
    n_prefix = 1 + n_storage
    grid = 224 // int(model.patch_size)
    n_patches = grid * grid
    n_tokens = n_prefix + n_patches

    with torch.inference_mode():
        sin_t, cos_t = model.rope_embed(H=grid, W=grid)
        rope_sin_arr = sin_t.detach().cpu().numpy().astype(np.float32).reshape(1, 1, n_patches, head_dim)
        rope_cos_arr = cos_t.detach().cpu().numpy().astype(np.float32).reshape(1, 1, n_patches, head_dim)
        cls_arr = model.cls_token.detach().cpu().numpy().astype(np.float32)
        if n_storage > 0:
            stor_arr = model.storage_tokens.detach().cpu().numpy().astype(np.float32)
            prefix_arr = np.concatenate([cls_arr, stor_arr], axis=1)
        else:
            prefix_arr = cls_arr

    precision_tag = "fp16-weights" if fp16_weights else "fp32"
    gb = _LiteRtGraphBuilder(
        f"Neri1.2preview DINOv3 ViT-B/16 encoder ({precision_tag}, NHWC [B,224,224,3] -> L2 CLS [B,768])",
        fp16_weights=fp16_weights,
    )

    in_tid = gb.tensor("images", [1, 224, 224, 3], shape_signature=[-1, 224, 224, 3])
    gb.sg.inputs = [in_tid]

    # PatchEmbed: Conv2d [768, 3, 16, 16] -> OHWI [768, 16, 16, 3]
    proj_w = model.patch_embed.proj.weight.detach().cpu().numpy().astype(np.float32)
    proj_w_ohwi = np.transpose(proj_w, (0, 2, 3, 1))
    proj_b = model.patch_embed.proj.bias.detach().cpu().numpy().astype(np.float32)
    w_tid = gb.const_f32("patch_embed_w", proj_w_ohwi, allow_fp16=True)
    b_tid = gb.const_f32("patch_embed_b", proj_b, allow_fp16=False)
    conv_out = gb.tensor("patch_conv", [1, grid, grid, embed_dim], shape_signature=[-1, grid, grid, embed_dim])
    conv_opts = schema_fb.Conv2DOptionsT()
    conv_opts.padding = schema_fb.Padding.VALID
    conv_opts.strideH = int(model.patch_size)
    conv_opts.strideW = int(model.patch_size)
    conv_opts.dilationHFactor = 1
    conv_opts.dilationWFactor = 1
    gb.op(
        schema_fb.BuiltinOperator.CONV_2D,
        [in_tid, w_tid, b_tid],
        [conv_out],
        opt_type=schema_fb.BuiltinOptions.Conv2DOptions,
        opts=conv_opts,
        version=1,
    )

    patches = gb.reshape(
        conv_out,
        [-1, n_patches, embed_dim],
        [1, n_patches, embed_dim],
        "patches",
        sig=[-1, n_patches, embed_dim],
    )

    # Dynamic-batch broadcast for [1, 5, 768] prefix tokens via TILE([B, 1, 1])
    patch_shape = gb.tensor("patch_shape", [3], ttype=schema_fb.TensorType.INT32)
    shape_opts = schema_fb.ShapeOptionsT()
    shape_opts.outType = schema_fb.TensorType.INT32
    gb.op(
        schema_fb.BuiltinOperator.SHAPE,
        [patches],
        [patch_shape],
        opt_type=schema_fb.BuiltinOptions.ShapeOptions,
        opts=shape_opts,
        version=1,
    )

    b_begin = gb.const_i32("b_begin", [0])
    b_end = gb.const_i32("b_end", [1])
    b_stride = gb.const_i32("b_stride", [1])
    batch_dim = gb.tensor("batch_dim", [1], ttype=schema_fb.TensorType.INT32)
    ss_opts = schema_fb.StridedSliceOptionsT()
    gb.op(
        schema_fb.BuiltinOperator.STRIDED_SLICE,
        [patch_shape, b_begin, b_end, b_stride],
        [batch_dim],
        opt_type=schema_fb.BuiltinOptions.StridedSliceOptions,
        opts=ss_opts,
        version=1,
    )
    ones_tail = gb.const_i32("tile_ones_tail", [1, 1])
    tile_mult = gb.concat([batch_dim, ones_tail], 0, [3], [3], "tile_mult", ttype=schema_fb.TensorType.INT32)
    prefix_const = gb.const_f32("prefix_tokens", prefix_arr, allow_fp16=False)
    prefix_b = gb.tensor("prefix_b", [1, n_prefix, embed_dim], shape_signature=[-1, n_prefix, embed_dim])
    gb.op(schema_fb.BuiltinOperator.TILE, [prefix_const, tile_mult], [prefix_b], version=1)

    x = gb.concat(
        [prefix_b, patches],
        1,
        [1, n_tokens, embed_dim],
        [-1, n_tokens, embed_dim],
        "tokens_in",
    )

    rope_sin = gb.const_f32("rope_sin", rope_sin_arr, allow_fp16=False)
    rope_cos = gb.const_f32("rope_cos", rope_cos_arr, allow_fp16=False)
    attn_scale = gb.const_f32("attn_scale", np.array([head_dim ** -0.5], dtype=np.float32), allow_fp16=False)

    tok_shape = [1, n_tokens, embed_dim]
    tok_sig = [-1, n_tokens, embed_dim]
    head_shape = [1, num_heads, n_tokens, head_dim]
    head_sig = [-1, num_heads, n_tokens, head_dim]
    prefix_head_shape = [1, num_heads, n_prefix, head_dim]
    prefix_head_sig = [-1, num_heads, n_prefix, head_dim]
    patch_head_shape = [1, num_heads, n_patches, head_dim]
    patch_head_sig = [-1, num_heads, n_patches, head_dim]
    half_patch_shape = [1, num_heads, n_patches, head_dim // 2]
    half_patch_sig = [-1, num_heads, n_patches, head_dim // 2]

    for idx, blk in enumerate(model.blocks):
        p = f"b{idx}"
        n1 = gb.layer_norm(
            x,
            tok_shape,
            tok_sig,
            blk.norm1.weight.detach().cpu().numpy(),
            blk.norm1.bias.detach().cpu().numpy(),
            float(blk.norm1.eps),
            f"{p}_norm1",
        )

        qkv_w = blk.attn.qkv.weight.detach().cpu().numpy().astype(np.float32)
        if blk.attn.qkv.bias is not None:
            qkv_b = blk.attn.qkv.bias.detach().cpu().numpy().astype(np.float32)
            if hasattr(blk.attn.qkv, "bias_mask") and blk.attn.qkv.bias_mask is not None:
                qkv_b = qkv_b * blk.attn.qkv.bias_mask.detach().cpu().numpy().astype(np.float32)
            if float(np.max(np.abs(qkv_b))) == 0.0:
                qkv_b = None
        else:
            qkv_b = None

        qkv = gb.linear(n1, qkv_w, qkv_b, [1, n_tokens, embed_dim * 3], f"{p}_qkv", out_sig=[-1, n_tokens, embed_dim * 3])
        q_flat, k_flat, v_flat = gb.split(qkv, -1, 3, tok_shape, tok_sig, f"{p}_qkv_split")

        def to_heads(tid: int, tag: str) -> int:
            r = gb.reshape(
                tid,
                [-1, n_tokens, num_heads, head_dim],
                [1, n_tokens, num_heads, head_dim],
                f"{p}_{tag}_4d",
                sig=[-1, n_tokens, num_heads, head_dim],
            )
            return gb.transpose(r, [0, 2, 1, 3], head_shape, f"{p}_{tag}_bhnd", sig=head_sig)

        q_h = to_heads(q_flat, "q")
        k_h = to_heads(k_flat, "k")
        v_h = to_heads(v_flat, "v")

        def apply_rope(tid: int, tag: str) -> int:
            t_pre = gb.slice(tid, [0, 0, 0, 0], [-1, num_heads, n_prefix, head_dim], prefix_head_shape, prefix_head_sig, f"{p}_{tag}_pre")
            t_pat = gb.slice(tid, [0, 0, n_prefix, 0], [-1, num_heads, n_patches, head_dim], patch_head_shape, patch_head_sig, f"{p}_{tag}_pat")
            t1, t2 = gb.split(t_pat, -1, 2, half_patch_shape, half_patch_sig, f"{p}_{tag}_half")
            neg_t2 = gb.tensor(f"{p}_{tag}_neg2", half_patch_shape, shape_signature=half_patch_sig)
            gb.op(schema_fb.BuiltinOperator.NEG, [t2], [neg_t2], version=1)
            t_rot = gb.concat([neg_t2, t1], -1, patch_head_shape, patch_head_sig, f"{p}_{tag}_rot")
            t_cos = gb.binary(schema_fb.BuiltinOperator.MUL, t_pat, rope_cos, patch_head_shape, f"{p}_{tag}_cos", patch_head_sig)
            t_sin = gb.binary(schema_fb.BuiltinOperator.MUL, t_rot, rope_sin, patch_head_shape, f"{p}_{tag}_sin", patch_head_sig)
            t_pat_rope = gb.binary(schema_fb.BuiltinOperator.ADD, t_cos, t_sin, patch_head_shape, f"{p}_{tag}_rope", patch_head_sig)
            return gb.concat([t_pre, t_pat_rope], 2, head_shape, head_sig, f"{p}_{tag}_full_rope")

        q_rope = apply_rope(q_h, "q")
        k_rope = apply_rope(k_h, "k")
        q_scaled = gb.binary(schema_fb.BuiltinOperator.MUL, q_rope, attn_scale, head_shape, f"{p}_q_scaled", head_sig)

        attn_logits = gb.tensor(f"{p}_attn_logits", [1, num_heads, n_tokens, n_tokens], shape_signature=[-1, num_heads, n_tokens, n_tokens])
        bmm_qk = schema_fb.BatchMatMulOptionsT()
        bmm_qk.adjX = False
        bmm_qk.adjY = True
        gb.op(
            schema_fb.BuiltinOperator.BATCH_MATMUL,
            [q_scaled, k_rope],
            [attn_logits],
            opt_type=schema_fb.BuiltinOptions.BatchMatMulOptions,
            opts=bmm_qk,
            version=1,
        )

        attn_probs = gb.tensor(f"{p}_attn_probs", [1, num_heads, n_tokens, n_tokens], shape_signature=[-1, num_heads, n_tokens, n_tokens])
        sm_opts = schema_fb.SoftmaxOptionsT()
        sm_opts.beta = 1.0
        gb.op(
            schema_fb.BuiltinOperator.SOFTMAX,
            [attn_logits],
            [attn_probs],
            opt_type=schema_fb.BuiltinOptions.SoftmaxOptions,
            opts=sm_opts,
            version=1,
        )

        attn_ctx = gb.tensor(f"{p}_attn_ctx", head_shape, shape_signature=head_sig)
        bmm_av = schema_fb.BatchMatMulOptionsT()
        bmm_av.adjX = False
        bmm_av.adjY = False
        gb.op(
            schema_fb.BuiltinOperator.BATCH_MATMUL,
            [attn_probs, v_h],
            [attn_ctx],
            opt_type=schema_fb.BuiltinOptions.BatchMatMulOptions,
            opts=bmm_av,
            version=1,
        )

        attn_4d = gb.transpose(attn_ctx, [0, 2, 1, 3], [1, n_tokens, num_heads, head_dim], f"{p}_attn_4d", sig=[-1, n_tokens, num_heads, head_dim])
        attn_flat = gb.reshape(attn_4d, [-1, n_tokens, embed_dim], tok_shape, f"{p}_attn_flat", sig=tok_sig)

        # Fold LayerScale ls1 into proj weights & bias
        ls1_gamma = blk.ls1.gamma.detach().cpu().numpy().astype(np.float32) if hasattr(blk.ls1, "gamma") else np.ones(embed_dim, dtype=np.float32)
        proj_w = blk.attn.proj.weight.detach().cpu().numpy().astype(np.float32) * ls1_gamma[:, None]
        proj_b = blk.attn.proj.bias.detach().cpu().numpy().astype(np.float32) * ls1_gamma if blk.attn.proj.bias is not None else None
        attn_out = gb.linear(attn_flat, proj_w, proj_b, tok_shape, f"{p}_proj", out_sig=tok_sig)
        x_attn = gb.binary(schema_fb.BuiltinOperator.ADD, x, attn_out, tok_shape, f"{p}_res1", tok_sig)

        # MLP block
        n2 = gb.layer_norm(
            x_attn,
            tok_shape,
            tok_sig,
            blk.norm2.weight.detach().cpu().numpy(),
            blk.norm2.bias.detach().cpu().numpy(),
            float(blk.norm2.eps),
            f"{p}_norm2",
        )
        fc1_w = blk.mlp.fc1.weight.detach().cpu().numpy().astype(np.float32)
        fc1_b = blk.mlp.fc1.bias.detach().cpu().numpy().astype(np.float32) if blk.mlp.fc1.bias is not None else None
        hidden_dim = int(fc1_w.shape[0])
        fc1_out = gb.linear(n2, fc1_w, fc1_b, [1, n_tokens, hidden_dim], f"{p}_fc1", out_sig=[-1, n_tokens, hidden_dim])

        gelu_out = gb.tensor(f"{p}_gelu", [1, n_tokens, hidden_dim], shape_signature=[-1, n_tokens, hidden_dim])
        gelu_opts = schema_fb.GeluOptionsT()
        gelu_opts.approximate = False
        gb.op(
            schema_fb.BuiltinOperator.GELU,
            [fc1_out],
            [gelu_out],
            opt_type=schema_fb.BuiltinOptions.GeluOptions,
            opts=gelu_opts,
            version=1,
        )

        ls2_gamma = blk.ls2.gamma.detach().cpu().numpy().astype(np.float32) if hasattr(blk.ls2, "gamma") else np.ones(embed_dim, dtype=np.float32)
        fc2_w = blk.mlp.fc2.weight.detach().cpu().numpy().astype(np.float32) * ls2_gamma[:, None]
        fc2_b = blk.mlp.fc2.bias.detach().cpu().numpy().astype(np.float32) * ls2_gamma if blk.mlp.fc2.bias is not None else None
        mlp_out = gb.linear(gelu_out, fc2_w, fc2_b, tok_shape, f"{p}_fc2", out_sig=tok_sig)
        x = gb.binary(schema_fb.BuiltinOperator.ADD, x_attn, mlp_out, tok_shape, f"{p}_res2", tok_sig)

    # Final CLS slice + LayerNorm + L2 normalization
    cls_3d = gb.slice(x, [0, 0, 0], [-1, 1, embed_dim], [1, 1, embed_dim], [-1, 1, embed_dim], "cls_slice")
    cls_2d = gb.reshape(cls_3d, [-1, embed_dim], [1, embed_dim], "cls_2d", sig=[-1, embed_dim])
    cls_normed = gb.layer_norm(
        cls_2d,
        [1, embed_dim],
        [-1, embed_dim],
        model.norm.weight.detach().cpu().numpy(),
        model.norm.bias.detach().cpu().numpy(),
        float(model.norm.eps),
        "final_norm",
    )
    cls_l2 = gb.tensor("x_norm_clstoken_l2", [1, embed_dim], shape_signature=[-1, embed_dim])
    l2_opts = schema_fb.L2NormOptionsT()
    gb.op(
        schema_fb.BuiltinOperator.L2_NORMALIZATION,
        [cls_normed],
        [cls_l2],
        opt_type=schema_fb.BuiltinOptions.L2NormOptions,
        opts=l2_opts,
        version=1,
    )
    gb.sg.outputs = [cls_l2]

    out_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    fb.write_model(gb.model, str(out_tflite_path))
    return out_tflite_path


def _load_demo_crops() -> list[tuple[str, Any]]:
    from PIL import Image

    demo_dir = REPO_ROOT / "res" / "demo"
    paths = sorted(p for p in demo_dir.glob("*.png") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No demo PNGs found in {demo_dir}")
    crops = []
    for p in paths:
        with Image.open(p) as im:
            crops.append((p.name, im.convert("RGB").copy()))
    return crops


def _preprocess_nhwc_batch(crops: list[Any]) -> np.ndarray:
    from system.dino.preprocess import IMAGENET_MEAN, IMAGENET_STD, _as_rgb_image, letterbox_image

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, None, :]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, None, :]
    rows = []
    for crop in crops:
        img = letterbox_image(_as_rgb_image(crop, array_color="rgb"))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
        rows.append(arr)
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)


class LiteRtDinoV3Encoder:
    """Drop-in LiteRT CPU encoder matching DinoV3Encoder's float32 CLS contract."""

    def __init__(self, tflite_path: Path, *, num_threads: int | None = None, batch_size: int = 16):
        self.tflite_path = Path(tflite_path).resolve()
        self.batch_size = int(batch_size)
        kwargs: dict[str, Any] = {"model_path": str(self.tflite_path)}
        if num_threads is not None:
            kwargs["num_threads"] = int(num_threads)
        self.interpreter = Interpreter(**kwargs)
        self._allocated_batch: int | None = None
        self._input_index = self.interpreter.get_input_details()[0]["index"]
        self._output_index = self.interpreter.get_output_details()[0]["index"]
        self._ensure_batch(1)

    def _ensure_batch(self, batch: int) -> None:
        if self._allocated_batch != batch:
            self.interpreter.resize_tensor_input(self._input_index, [batch, 224, 224, 3])
            self.interpreter.allocate_tensors()
            self._allocated_batch = batch

    def encode_nhwc(self, nhwc_batch: np.ndarray) -> np.ndarray:
        nhwc_batch = np.ascontiguousarray(nhwc_batch, dtype=np.float32)
        total = len(nhwc_batch)
        outs = []
        for begin in range(0, total, self.batch_size):
            chunk = nhwc_batch[begin : begin + self.batch_size]
            self._ensure_batch(len(chunk))
            self.interpreter.set_tensor(self._input_index, chunk)
            self.interpreter.invoke()
            outs.append(self.interpreter.get_tensor(self._output_index).copy())
        return np.concatenate(outs, axis=0).astype(np.float32, copy=False)

    def encode(self, crops: list[Any], *, array_color: str = "rgb") -> np.ndarray:
        if array_color != "rgb":
            raise ValueError("Use rgb crops for LiteRtDinoV3Encoder")
        return self.encode_nhwc(_preprocess_nhwc_batch(crops))


def _time_calls(fn, *, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "runs": runs,
    }


def export_and_benchmark(
    manifest_path: Path,
    out_dir: Path,
    *,
    num_threads: int,
    warmup: int,
    runs_single: int,
    runs_batch: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from system.dinov3.classifier import DinoV3Classifier
    from system.dinov3.encoder import DinoV3Encoder
    from system.dinov3.model import DinoV3RidgeWithinModel
    from system.dinov3.runtime import read_manifest, resolve_manifest_asset

    manifest = read_manifest(manifest_path)
    checkpoint_path = resolve_manifest_asset(manifest_path, manifest["checkpoint"])
    weights_path = resolve_manifest_asset(manifest_path, manifest["encoder_weights"])
    code_dir = resolve_manifest_asset(manifest_path, manifest["encoder_code"])
    cal_path = resolve_manifest_asset(manifest_path, manifest["calibration"]) if "calibration" in manifest else None

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "dinov3_vitb16_fp32.tflite"
    fp16_path = out_dir / "dinov3_vitb16_fp16_weights.tflite"

    print(f"[1/5] Exporting FP32 LiteRT model -> {fp32_path}")
    t0 = time.perf_counter()
    build_dinov3_vitb16_tflite(weights_path, code_dir, fp32_path, fp16_weights=False)
    fp32_export_s = time.perf_counter() - t0

    print(f"[2/5] Exporting FP16-weight LiteRT model -> {fp16_path}")
    t0 = time.perf_counter()
    build_dinov3_vitb16_tflite(weights_path, code_dir, fp16_path, fp16_weights=True)
    fp16_export_s = time.perf_counter() - t0

    shutil.copy2(checkpoint_path, out_dir / "model.neri-model")
    if cal_path is not None:
        shutil.copy2(cal_path, out_dir / "calibration.npz")
    litert_manifest = {
        "schema_version": 1,
        "display_name": f"{manifest.get('display_name', 'Neri1.2preview')} (LiteRT)",
        "source_manifest": str(manifest_path),
        "architecture": manifest["architecture"],
        "inference_recipe": manifest.get("inference_recipe"),
        "head_type": manifest.get("head_type"),
        "feature_dim": manifest["feature_dim"],
        "preprocessing": manifest["preprocessing"],
        "encoder_fp32": fp32_path.name,
        "encoder_fp32_sha256": _sha256(fp32_path),
        "encoder_fp16_weights": fp16_path.name,
        "encoder_fp16_weights_sha256": _sha256(fp16_path),
        "checkpoint": "model.neri-model",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "calibration": "calibration.npz" if cal_path is not None else None,
        "calibration_sha256": manifest.get("calibration_sha256"),
    }
    (out_dir / "Neri1.2preview-litert.json").write_text(
        json.dumps(litert_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[3/5] Loading PyTorch CPU (threads={num_threads}) & LiteRT CPU (threads={num_threads})...")
    torch.set_num_threads(num_threads)
    pt_encoder = DinoV3Encoder(
        weights_path,
        code_dir,
        manifest["encoder_sha256"],
        device="cpu",
        batch_size=batch_size,
    )
    head_model = DinoV3RidgeWithinModel.load(checkpoint_path)
    classifier = DinoV3Classifier(head_model)

    litert_fp32 = LiteRtDinoV3Encoder(fp32_path, num_threads=num_threads, batch_size=batch_size)
    litert_fp16 = LiteRtDinoV3Encoder(fp16_path, num_threads=num_threads, batch_size=batch_size)

    demo_items = _load_demo_crops()
    demo_names = [name for name, _ in demo_items]
    demo_crops = [img for _, img in demo_items]
    demo_nhwc = _preprocess_nhwc_batch(demo_crops)
    demo_nchw = torch.from_numpy(np.transpose(demo_nhwc, (0, 3, 1, 2)).copy())

    print("[4/5] Verifying numerical & Neri1.2preview recognition equivalence...")
    with torch.inference_mode():
        pt_emb = pt_encoder._encode_tensors(iter(demo_nchw), len(demo_nchw))
    lrt_fp32_emb = litert_fp32.encode_nhwc(demo_nhwc)
    lrt_fp16_emb = litert_fp16.encode_nhwc(demo_nhwc)

    def diff_metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
        cos = np.sum(ref * cand, axis=1)
        abs_diff = np.abs(ref - cand)
        return {
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "min_cosine_sim": float(cos.min()),
            "mean_cosine_sim": float(cos.mean()),
        }

    pt_preds = classifier.classify_features(pt_emb)
    fp32_preds = classifier.classify_features(lrt_fp32_emb)
    fp16_preds = classifier.classify_features(lrt_fp16_emb)

    def pred_summary(preds) -> list[dict[str, Any]]:
        return [
            {
                "image": demo_names[i],
                "species": p.species,
                "accepted": p.accepted,
                "known_score": round(p.known_score, 6),
                "threshold": round(p.threshold, 6),
                "ridge_score": round(p.ridge_score, 6),
                "within_score": round(p.within_score, 6),
            }
            for i, p in enumerate(preds)
        ]

    fp32_exact_species = all(a.species == b.species and a.accepted == b.accepted for a, b in zip(pt_preds, fp32_preds))
    fp16_exact_species = all(a.species == b.species and a.accepted == b.accepted for a, b in zip(pt_preds, fp16_preds))

    print("[5/5] Benchmarking CPU inference speed...")
    single_nhwc = demo_nhwc[:1]
    single_nchw = demo_nchw[:1]

    def pt_single():
        with torch.inference_mode():
            return torch.nn.functional.normalize(
                pt_encoder.model.forward_features(single_nchw)["x_norm_clstoken"].float(),
                dim=1,
            ).numpy()

    b1_pt = _time_calls(pt_single, warmup=warmup, runs=runs_single)
    b1_lrt32 = _time_calls(lambda: litert_fp32.encode_nhwc(single_nhwc), warmup=warmup, runs=runs_single)
    b1_lrt16 = _time_calls(lambda: litert_fp16.encode_nhwc(single_nhwc), warmup=warmup, runs=runs_single)

    rng = np.random.default_rng(42)
    batch_nhwc = rng.standard_normal((batch_size, 224, 224, 3), dtype=np.float32)
    batch_nchw = torch.from_numpy(np.transpose(batch_nhwc, (0, 3, 1, 2)).copy())

    def pt_batch():
        with torch.inference_mode():
            return torch.nn.functional.normalize(
                pt_encoder.model.forward_features(batch_nchw)["x_norm_clstoken"].float(),
                dim=1,
            ).numpy()

    bn_pt = _time_calls(pt_batch, warmup=max(2, warmup // 2), runs=runs_batch)
    bn_lrt32 = _time_calls(lambda: litert_fp32.encode_nhwc(batch_nhwc), warmup=max(2, warmup // 2), runs=runs_batch)
    bn_lrt16 = _time_calls(lambda: litert_fp16.encode_nhwc(batch_nhwc), warmup=max(2, warmup // 2), runs=runs_batch)

    e2e_pt = _time_calls(
        lambda: classifier.classify_features(pt_encoder.encode(demo_crops)),
        warmup=2,
        runs=runs_batch,
    )
    e2e_lrt32 = _time_calls(
        lambda: classifier.classify_features(litert_fp32.encode(demo_crops)),
        warmup=2,
        runs=runs_batch,
    )
    e2e_lrt16 = _time_calls(
        lambda: classifier.classify_features(litert_fp16.encode(demo_crops)),
        warmup=2,
        runs=runs_batch,
    )
    head_only = _time_calls(
        lambda: classifier.classify_features(pt_emb),
        warmup=5,
        runs=runs_single,
    )

    report = {
        "model": manifest.get("display_name", "Neri1.2preview"),
        "cpu_threads": num_threads,
        "os_cpu_count": os.cpu_count(),
        "batch_size": batch_size,
        "artifacts": {
            "out_dir": str(out_dir),
            "fp32_tflite": str(fp32_path),
            "fp32_size_mb": round(fp32_path.stat().st_size / (1024 * 1024), 2),
            "fp32_export_s": round(fp32_export_s, 2),
            "fp16_weights_tflite": str(fp16_path),
            "fp16_weights_size_mb": round(fp16_path.stat().st_size / (1024 * 1024), 2),
            "fp16_export_s": round(fp16_export_s, 2),
        },
        "accuracy_vs_pytorch_cpu": {
            "litert_fp32": diff_metrics(pt_emb, lrt_fp32_emb),
            "litert_fp16_weights": diff_metrics(pt_emb, lrt_fp16_emb),
            "fp32_classification_exact_match": fp32_exact_species,
            "fp16_classification_exact_match": fp16_exact_species,
            "pytorch_predictions": pred_summary(pt_preds),
            "litert_fp32_predictions": pred_summary(fp32_preds),
            "litert_fp16_predictions": pred_summary(fp16_preds),
        },
        "benchmark_encoder_batch1_ms": {
            "pytorch_cpu": b1_pt,
            "litert_fp32": b1_lrt32,
            "litert_fp16_weights": b1_lrt16,
        },
        "benchmark_encoder_batchN": {
            "batch_size": batch_size,
            "pytorch_cpu": {
                **bn_pt,
                "per_image_median_ms": bn_pt["median_ms"] / batch_size,
                "imgs_per_sec": 1000.0 * batch_size / bn_pt["median_ms"],
            },
            "litert_fp32": {
                **bn_lrt32,
                "per_image_median_ms": bn_lrt32["median_ms"] / batch_size,
                "imgs_per_sec": 1000.0 * batch_size / bn_lrt32["median_ms"],
            },
            "litert_fp16_weights": {
                **bn_lrt16,
                "per_image_median_ms": bn_lrt16["median_ms"] / batch_size,
                "imgs_per_sec": 1000.0 * batch_size / bn_lrt16["median_ms"],
            },
        },
        "benchmark_e2e_4_demo_images_ms": {
            "pytorch_cpu": e2e_pt,
            "litert_fp32": e2e_lrt32,
            "litert_fp16_weights": e2e_lrt16,
            "head_only_4_images": head_only,
        },
    }
    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "res/model/cls/user/dinov3-experiment-local.neri.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/Neri1.2preview-litert",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs-single", type=int, default=20)
    parser.add_argument("--runs-batch", type=int, default=10)
    args = parser.parse_args()

    report = export_and_benchmark(
        args.manifest.resolve(),
        args.out_dir.resolve(),
        num_threads=args.threads,
        warmup=args.warmup,
        runs_single=args.runs_single,
        runs_batch=args.runs_batch,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
