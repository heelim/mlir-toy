//===- LinalgToLLVM.h - Utils to convert from the linalg dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_ToyToHello_ToyToHello_H_
#define MLIR_CONVERSION_ToyToHello_ToyToHello_H_
#include "Pass/Passes.h"
#include "Dialect/Toy/IR/ToyOps.hpp"
#include "Dialect/Hello/IR/HelloOps.hpp"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Populate the given list with patterns that convert from Toy to Hello.
void populateLoweringToyPrintOpToHelloPatterns(
    RewritePatternSet &patterns, MLIRContext *context);
} // namespace mlir

#endif // MLIR_CONVERSION_ToyToHello_ToyToHello_H_
