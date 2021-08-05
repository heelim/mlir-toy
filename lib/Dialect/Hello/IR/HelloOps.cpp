//===--------- HelloAPIOps.cpp - HelloAPI dialect ops ---------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <queue>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "Dialect/Hello/IR/HelloOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Hello dialect.
//===----------------------------------------------------------------------===//

HelloOpsDialect::HelloOpsDialect(MLIRContext *context)
  : Dialect(getDialectNamespace(), context, TypeID::get<HelloOpsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Hello/IR/HelloOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/Hello/IR/HelloOps.cpp.inc"

//===----------------------------------------------------------------------===//
// HelloWorldOp
/*
void HelloWorldOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input});
}
*/

//===----------------------------------------------------------------------===//
// HelloConvertOp

void F32ToF64TensorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value input) {
	state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
	state.addOperands({input});
}

//===----------------------------------------------------------------------===//
// TaskOp

void TaskOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::IntegerAttr task_id)
{
	state.addAttribute("task_id", task_id);
	Region *region = state.addRegion();
	TaskOp::ensureTerminator(*region, builder, state.location);
}

//===----------------------------------------------------------------------===//
// AddPartOp

void AddPartOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		mlir::Value lhs, mlir::Value rhs) {
	state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
	state.addOperands({lhs, rhs});
}
