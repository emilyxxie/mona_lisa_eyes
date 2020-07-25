/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/// <reference types="webgl2" />
import { PixelData, TypedArray } from '@tensorflow/tfjs-core';
import { TextureConfig } from './tex_util';
export declare function createVertexShader(gl: WebGLRenderingContext, debug: boolean): WebGLShader;
export declare function createVertexBuffer(gl: WebGLRenderingContext, debug: boolean): WebGLBuffer;
export declare function createIndexBuffer(gl: WebGLRenderingContext, debug: boolean): WebGLBuffer;
export declare function createFloat32MatrixTexture(gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): WebGLTexture;
export declare function createFloat16MatrixTexture(gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): WebGLTexture;
export declare function createUnsignedBytesMatrixTexture(gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): WebGLTexture;
export declare function createPackedMatrixTexture(gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): WebGLTexture;
export declare function createFloat16PackedMatrixTexture(gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): WebGLTexture;
export declare function bindVertexProgramAttributeStreams(gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram, vertexBuffer: WebGLBuffer): boolean;
export declare function uploadDenseMatrixToTexture(gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture, width: number, height: number, data: TypedArray, textureConfig: TextureConfig): void;
export declare function uploadPixelDataToTexture(gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture, pixels: PixelData | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): void;
export declare function createBufferFromOutputTexture(gl2: WebGL2RenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): WebGLBuffer;
export declare function downloadFloat32MatrixFromBuffer(gl: WebGLRenderingContext, buffer: WebGLBuffer, size: number): Float32Array;
export declare function downloadByteEncodedFloatMatrixFromOutputTexture(gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number, textureConfig: TextureConfig): Float32Array;
export declare function downloadPackedMatrixFromBuffer(gl: WebGLRenderingContext, buffer: WebGLBuffer, batch: number, rows: number, cols: number, physicalRows: number, physicalCols: number, textureConfig: TextureConfig): Float32Array;
export declare function downloadMatrixFromPackedOutputTexture(gl: WebGLRenderingContext, debug: boolean, physicalRows: number, physicalCols: number): Float32Array;
